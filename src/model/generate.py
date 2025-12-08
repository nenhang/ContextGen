import os

import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from src.flux.attention_processor import FluxAttnProcessor2_0_Attention
from src.flux.pipeline_flux_kontext import FluxKontextPipeline
from src.flux.pipeline_forward import fluxkontext_pipeline_forward
from src.model.model import ContextGenModel
from src.model.pipeline_preprocess import (
    prepare_instance_img_pe,
    prepare_mig_attn_mask,
    prepare_reference_latents,
)
from src.utils.file_utils import get_config
from src.utils.image_process import (
    uniform_resize_16x,
)
from src.utils.text_process import get_kontext_edit_template


def load_model_from_config(config_path: str, device: str | torch.device = "cuda") -> FluxKontextPipeline:
    config = get_config(config_path)
    training_config = config["train"]
    model = ContextGenModel(
        flux_pipe_id=config["flux_path"],
        save_dir=os.path.join(training_config["save_path"], training_config["model_name"]),
        training_config=training_config,
        dtype=getattr(torch, config["dtype"]),
    )
    model.setup(stage="predict")
    model.to(device)
    return model.flux_pipe


def load_model(
    flux_path: str, adapter_path: str, adapter_name: str = "flux_mig", device: str | torch.device = "cuda"
) -> FluxKontextPipeline:
    flux_pipe = FluxKontextPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16)
    if not hasattr(flux_pipe, "mask_processor"):
        flux_pipe.mask_processor = VaeImageProcessor(
            vae_scale_factor=flux_pipe.vae_scale_factor * 2,
            vae_latent_channels=flux_pipe.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
    adapter_files = [f for f in os.listdir(adapter_path) if f.endswith(".safetensors")]
    if not adapter_files:
        raise FileNotFoundError(f"No .safetensors files found in {adapter_path}")
    adapter_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True)
    adapter_filename = adapter_files[0]
    print(f"Loading adapter weights from {os.path.join(adapter_path, adapter_filename)}")
    flux_pipe.load_lora_weights(
        pretrained_model_name_or_path_or_dict=adapter_path,
        adapter_name=adapter_name,
        weight_name=adapter_filename,
    )
    flux_pipe.transformer.set_attn_processor(FluxAttnProcessor2_0_Attention())
    flux_pipe.transformer.eval()
    flux_pipe.to(device)
    return flux_pipe


def check_inputs(reference_info, base_size=None):
    if not isinstance(reference_info, list):
        raise ValueError(f"Reference info must be a list. Got {type(reference_info)}.")

    def load_img(im, mode="RGB"):
        if isinstance(im, str):
            im = Image.open(im)
        if not isinstance(im, Image.Image):
            raise ValueError(f"Unsupported image type: {type(im)}. Expected str or PIL.Image.")
        return im.convert(mode)

    new_infos = []
    for ref in reference_info:
        if isinstance(ref, dict):
            ref_image = ref.get("image")
            bbox = ref.get("bbox")
            instance_mask = ref.get("mask")
        else:
            raise ValueError(f"Unsupported reference info type: {type(ref)}. Expected dict or tuple.")

        if instance_mask is None:
            if isinstance(ref_image, str):
                image_pil = Image.open(ref_image)
            elif isinstance(ref_image, Image.Image):
                image_pil = ref_image
            else:
                raise ValueError(f"Unsupported image type: {type(ref_image)}")
            if image_pil.mode == "RGBA":
                instance_mask = image_pil.split()[-1]
                image_pil = image_pil.convert("RGB")
            elif image_pil.mode == "P" and "transparency" in image_pil.info:
                transparency = image_pil.info["transparency"]
                instance_mask = image_pil.convert("L").point(lambda p: 255 if p == transparency else 0)
                image_pil = image_pil.convert("RGB")
            else:
                raise ValueError(
                    f"Make sure the image is in RGBA mode or P mode with transparency info. Got {image_pil.mode}."
                )
            ref_image = image_pil
        else:
            ref_image = load_img(ref_image, "RGB")
            instance_mask = load_img(instance_mask, "L")

        if base_size is not None:
            target_size = uniform_resize_16x(ref_image.size, base_size)
            ref_image = ref_image.resize(target_size)
            instance_mask = instance_mask.resize(target_size)

        if isinstance(bbox, list):
            bbox = torch.tensor(bbox, dtype=torch.int32)

        new_infos.append({"image": ref_image, "bbox": bbox, "mask": instance_mask})

    return new_infos


def process_layout_image(layout_info, size=None):
    if isinstance(layout_info, list) and all(
        isinstance(
            layout,
            (str, Image.Image)
            or (
                isinstance(layout, (tuple, list))
                and len(layout) == 2
                and all(isinstance(x, (str, Image.Image)) for x in layout)
            ),
        )
        for layout in layout_info
    ):
        # layout_info is a list of Image or (Image, mask) pairs
        layout_images = []
        for layout in layout_info:
            if isinstance(layout, (tuple, list)) and len(layout) == 2:
                item, mask = layout
                if isinstance(item, str):
                    item = Image.open(item)
                if isinstance(mask, str):
                    mask = Image.open(mask)
                item.convert("RGB")
                mask.convert("L")
            elif isinstance(layout, (Image.Image, str)):
                item = Image.open(layout) if isinstance(layout, str) else layout
                if isinstance(item, Image.Image) and item.mode == "RGBA":
                    mask = item.split()[-1]  # Get the alpha channel as mask
                    item = item.convert("RGB")
                else:
                    raise ValueError(f"Make Sure the image is in RGBA mode, got {item.mode}.")
            else:
                raise ValueError(
                    f"Unsupported layout type: {type(layout)}. Expected str, PIL.Image, or (Image, mask) tuple."
                )
            if size is not None:
                item = item.resize(size, resample=Image.LANCZOS)
                mask = mask.resize(size, resample=Image.NEAREST)
            layout_images.append(
                {
                    "image": item,
                    "mask": mask,
                }
            )

        return layout_images

    else:
        raise ValueError("layout_info must be a list of images or (image, mask) pairs.")


@torch.inference_mode()
def generate(
    flux_pipe: FluxKontextPipeline,
    prompts: list,
    reference_info: list,
    height: int,
    width: int,
    layout_image: list | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: int | list | None = None,
    resize_pattern="instance-fit",
    debug_refs: bool = False,
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
):
    assert len(prompts) == len(reference_info), "Length of prompts and reference_info must match."
    bs = len(prompts)
    prompts = [get_kontext_edit_template(p) for p in prompts]

    shorter_side = min(height, width)
    if shorter_side <= 384:
        base_size = 256
    elif shorter_side <= 512:
        base_size = 384
    elif shorter_side <= 768:
        base_size = 512
    elif shorter_side <= 1024:
        base_size = 768
    else:
        base_size = 1024

    reference_info = [check_inputs(ref, base_size=base_size) for ref in reference_info]

    if seed is not None:
        if isinstance(seed, int):
            generator = torch.Generator(device=device).manual_seed(seed)
        elif isinstance(seed, list):
            assert len(seed) == bs, "Length of seed list must match batch size."
            generator = [torch.Generator(device=device).manual_seed(s) for s in seed]
        else:
            raise ValueError("Seed must be an int or a list of ints.")
    else:
        generator = None

    # Prepare image input
    assert height % 16 == 0 and width % 16 == 0, "Image height and width must be multiples of 16."
    height_in_token, width_in_token = height // 16, width // 16
    main_token_len = height_in_token * width_in_token

    if layout_image is not None:
        layout_image = process_layout_image(layout_image, size=(width, height))

    layout_instance_dict = prepare_reference_latents(
        flux_pipe=flux_pipe,
        instance_info=reference_info,
        given_layout_images=layout_image,
        resize_pattern=resize_pattern,
        width=width,
        height=height,
        device=device,
        debug_visualize=debug_refs,
    )
    layout_instance_latents = layout_instance_dict["layout_instance_latents"]
    batch_instance_token_1d_idx_list = layout_instance_dict["batch_instance_token_1d_idx_list"]
    batch_instance_bbox_2d_idx_list = layout_instance_dict["batch_instance_bbox_2d_idx_list"]
    batch_instance_token_2d_size_list = layout_instance_dict["batch_instance_token_2d_size_list"]
    batch_instance_mask_list = layout_instance_dict["batch_instance_mask_list"]
    debug_instance_info = layout_instance_dict["debug_instance_info"]

    rotary_embs = prepare_instance_img_pe(
        batch_instance_token_2d_size_list,
        flux_pipe=flux_pipe,
        device=device,
        dtype=dtype,
    )

    attn_masks = prepare_mig_attn_mask(
        width_in_token,
        height_in_token,
        batch_instance_token_1d_idx_list,
        batch_instance_bbox_2d_idx_list,
        batch_instance_mask_list=batch_instance_mask_list,
        device=device,
        debug_dir=False,
        image_instance_token_2d_size_list=batch_instance_token_2d_size_list,
    )

    # prepare reference_dict
    reference_dict = {
        "main_token_len": main_token_len,
        "instance_latents": layout_instance_latents,
        **rotary_embs,
        **attn_masks,
    }

    # Forward pass
    image_out = fluxkontext_pipeline_forward(
        self=flux_pipe,
        prompt=prompts,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        reference_dict=reference_dict,
        generator=generator,
    ).images

    if debug_refs:
        return image_out, debug_instance_info

    return image_out
