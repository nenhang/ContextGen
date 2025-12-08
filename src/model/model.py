import os
import time

import lightning as L
import prodigyopt
import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from lightning.pytorch.utilities import rank_zero_only
from peft import LoraConfig, get_peft_model_state_dict

from ..flux.attention_processor import FluxAttnProcessor2_0_Attention
from ..flux.pipeline_flux_kontext import FluxKontextPipeline
from ..flux.pipeline_tools import decode_images, encode_images, prepare_text_input
from ..flux.transformer_flux import FluxTransformer2DModel_forward
from ..utils.file_utils import find_latest_checkpoint
from ..utils.text_process import get_kontext_edit_template
from .pipeline_preprocess import prepare_instance_img_pe, prepare_mig_attn_mask, prepare_reference_latents


class ContextGenModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        training_config: dict = {},
        save_dir: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        process_visualize: bool = False,
        gradient_checkpointing: bool = False,
        train_method: str = "sft",
        debug_dir: str | None = "./debug",
    ):
        # Initialize the LightningModule
        super().__init__()
        self.flux_pipe_id = flux_pipe_id
        self.lora_config = training_config["lora_config"]
        self.dpo_config = training_config["dpo_config"]
        self.optimizer_config = training_config["optimizer"]
        self.gradient_checkpointing = gradient_checkpointing
        self.process_visualize = process_visualize
        self.ckpt_dir = save_dir
        assert train_method in ["sft", "dpo"], "train_method must be 'sft' or 'dpo'"
        self.train_method = train_method
        self.target_dtype = dtype
        self.total_steps = 0
        self.debug_dir = os.path.join(debug_dir, time.strftime("%Y%m%d-%H%M%S")) if debug_dir else None
        if self.debug_dir and rank_zero_only.rank == 0:
            os.makedirs(self.debug_dir, exist_ok=True)

    def setup(self, stage):
        self.flux_pipe: FluxKontextPipeline = FluxKontextPipeline.from_pretrained(
            self.flux_pipe_id, torch_dtype=self.target_dtype
        )

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        if not hasattr(self.flux_pipe, "mask_processor"):
            self.flux_pipe.mask_processor = VaeImageProcessor(
                vae_scale_factor=self.flux_pipe.vae_scale_factor * 2,
                vae_latent_channels=self.flux_pipe.latent_channels,
                do_normalize=False,
                do_binarize=True,
                do_convert_grayscale=True,
            )

        self.text_encoder = self.flux_pipe.text_encoder
        self.text_encoder_2 = self.flux_pipe.text_encoder_2
        self.vae = self.flux_pipe.vae
        self.transformer = self.flux_pipe.transformer

        self.transformer.set_attn_processor(FluxAttnProcessor2_0_Attention())

        if self.gradient_checkpointing:
            from diffusers.utils import is_torch_version

            def my_gradient_checkpointing_func(call_func, *args):
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                return torch.utils.checkpoint.checkpoint(
                    call_func,
                    *args,
                    **ckpt_kwargs,
                )

            self.transformer.enable_gradient_checkpointing(my_gradient_checkpointing_func)

        if self.train_method == "dpo":
            # 从 subfolder 加载 ref transformer
            self.ref_transformer: FluxTransformer2DModel = FluxTransformer2DModel.from_pretrained(
                self.flux_pipe_id,
                subfolder="transformer",
                torch_dtype=self.target_dtype,
            )
            self.ref_transformer.set_attn_processor(FluxAttnProcessor2_0_Attention())
            self.ref_transformer.requires_grad_(False).eval()
            print(f"✅ DPO beta={self.dpo_config['beta']}, freeze reference transformer.")

        self.transformer.train()
        self.prepare_model(self.lora_config)
        self.to(dtype=self.target_dtype)

    def prepare_model(self, lora_config: dict, adapter_name: str = "flux_mig"):
        if self.ckpt_dir is not None:
            latest_checkpoint = find_latest_checkpoint(
                self.ckpt_dir,
                prefix="transformer_",
            )
            if latest_checkpoint:
                print(f"✅ Loading latest checkpoint: {latest_checkpoint}")
                last_ckpt_path, self.total_steps = latest_checkpoint
                self.flux_pipe.load_lora_weights(
                    pretrained_model_name_or_path_or_dict=os.path.dirname(last_ckpt_path),
                    adapter_name=adapter_name,
                    weight_name=os.path.basename(last_ckpt_path),
                )
                if self.train_method == "dpo":
                    self.ref_transformer.load_lora_adapter(
                        pretrained_model_name_or_path_or_dict=os.path.dirname(last_ckpt_path),
                        adapter_name=adapter_name,
                        weight_name=os.path.basename(last_ckpt_path),
                    )
                    self.ref_transformer.fuse_lora()
                    self.ref_transformer.unload_lora()
                    self.ref_transformer.requires_grad_(False).eval()
            else:
                print("⚠️ No checkpoint found, training from scratch.")
                self.transformer.add_adapter(adapter_config=LoraConfig(**lora_config), adapter_name=adapter_name)
                if self.train_method == "dpo":
                    # do nothing, keep the ref transformer as the original pre-trained model
                    pass

    def get_optimize_parameters(self):
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def save_optimize_parameters(
        self, run_name: str, total_steps: int, adapter_name: str = "flux_mig", remove_old: bool = True
    ):
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        save_dir = os.path.join(self.ckpt_dir, run_name, "ckpt")

        os.makedirs(save_dir, exist_ok=True)
        if remove_old:
            for file in os.listdir(save_dir):
                if file.endswith(".safetensors"):
                    os.remove(os.path.join(save_dir, file))

        weight_name = f"transformer_{total_steps:05d}.safetensors"
        transformer_lora_layers = get_peft_model_state_dict(
            self.transformer,
            adapter_name=adapter_name,
        )
        FluxKontextPipeline.save_lora_weights(
            save_directory=save_dir,
            transformer_lora_layers=transformer_lora_layers,
            safe_serialization=True,
            weight_name=weight_name,
        )

    def configure_optimizers(self):
        opt_config = self.optimizer_config

        self.trainable_params = self.get_optimize_parameters()

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item() if not hasattr(self, "log_loss") else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        with torch.no_grad():
            target_images = batch["target_images"]
            prompts = batch["captions"]
            instance_info = batch["reference_images"]

            prompts = [get_kontext_edit_template(p) for p in prompts]

            # Prepare image input
            target_images = self.flux_pipe.image_processor.preprocess(target_images)
            bs, channel, height, width = target_images.shape
            height_in_token, width_in_token = height // 16, width // 16
            main_token_len = height_in_token * width_in_token

            # Prepare text input
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(self.flux_pipe, prompts)
            x_0 = encode_images(self.flux_pipe, target_images)

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((bs,), device=self.device))
            # mean_shift = 1.5  # 右偏程度
            # std_scale = 1.0  # 增加标准差使分布更分散
            # t = torch.sigmoid(std_scale * torch.randn((bs,), device=self.device) + mean_shift)

            x_1 = torch.randn_like(x_0, device=self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            img_ids = self.flux_pipe._prepare_latent_image_ids(
                bs,
                height // 16,
                width // 16,
                self.device,
                self.dtype,
            )

            # Prepare guidance
            guidance = torch.ones_like(t).to(self.device) if self.flux_pipe.transformer.config.guidance_embeds else None

            reference_info_dict = prepare_reference_latents(
                flux_pipe=self.flux_pipe,
                instance_info=instance_info,
                height=height,
                width=width,
                device=self.device,
                debug_visualize=self.process_visualize and rank_zero_only.rank == 0,
            )

            layout_instance_latents = reference_info_dict["layout_instance_latents"]
            layout_latents = reference_info_dict["layout_latents"]
            batch_instance_token_1d_idx_list = reference_info_dict["batch_instance_token_1d_idx_list"]
            batch_instance_bbox_2d_idx_list = reference_info_dict["batch_instance_bbox_2d_idx_list"]
            batch_instance_token_2d_size_list = reference_info_dict["batch_instance_token_2d_size_list"]
            batch_instance_mask_list = reference_info_dict["batch_instance_mask_list"]
            debug_instance_info = reference_info_dict["debug_instance_info"]

            rotary_embs = prepare_instance_img_pe(
                batch_instance_token_2d_size_list,
                flux_pipe=self.flux_pipe,
                device=self.device,
                dtype=self.dtype,
            )

            attn_masks = prepare_mig_attn_mask(
                width_in_token,
                height_in_token,
                batch_instance_token_1d_idx_list=batch_instance_token_1d_idx_list,
                batch_instance_bbox_2d_idx_list=batch_instance_bbox_2d_idx_list,
                batch_instance_mask_list=batch_instance_mask_list,
                device=self.device,
                debug_dir=os.path.join(self.debug_dir, "attn_masks")
                if self.process_visualize and self.debug_dir and rank_zero_only.rank == 0
                else None,
                image_instance_token_2d_size_list=batch_instance_token_2d_size_list,
            )

            # prepare reference_dict
            reference_dict = {
                "main_token_len": main_token_len,
                **attn_masks,
                **rotary_embs,
            }

            hidden_states_input = torch.cat([x_t, layout_instance_latents], dim=1)

            if self.train_method == "dpo":
                # set reference_layouts_latents as less preferred
                x_0_l = layout_latents
                x_t_l = ((1 - t_) * x_0_l + t_ * x_1).to(self.dtype)
                hidden_states_input_l = torch.cat([x_t_l, layout_instance_latents], dim=1)
                hidden_states_input = torch.cat([hidden_states_input, hidden_states_input_l], dim=0)
                x_0 = torch.cat([x_0, x_0_l], dim=0)
                x_t = torch.cat([x_t, x_t_l], dim=0)
                x_1 = x_1.repeat(2, 1, 1)
                # repeat all the other inputs
                prompt_embeds = prompt_embeds.repeat(2, 1, 1)
                pooled_prompt_embeds = pooled_prompt_embeds.repeat(2, 1)
                t = t.repeat(2)
                guidance = guidance.repeat(2)

                # 遍历 reference_dict 的每个张量，并重复
                for key in reference_dict:
                    if isinstance(reference_dict[key], torch.Tensor) and reference_dict[key].shape[0] == bs:
                        reference_dict[key] = reference_dict[key].repeat(2, *[1] * (reference_dict[key].ndim - 1))
                    elif (
                        key.endswith("rotary_emb")
                        and isinstance(reference_dict[key], tuple)
                        and len(reference_dict[key]) == 2
                        and reference_dict[key][0].shape[0] == bs
                    ):
                        reference_dict[key] = (
                            reference_dict[key][0].repeat(2, *[1] * (reference_dict[key][0].ndim - 1)),
                            reference_dict[key][1].repeat(2, *[1] * (reference_dict[key][1].ndim - 1)),
                        )

            target = x_1 - x_0

        # Forward pass
        transformer_out = FluxTransformer2DModel_forward(
            self=self.flux_pipe.transformer,
            hidden_states=hidden_states_input,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
            reference_dict=reference_dict,
        )
        pred = transformer_out[0][:, :main_token_len, :]

        # Compute loss
        if self.train_method == "sft":
            loss = F.mse_loss(pred, target, reduction="mean")
        elif self.train_method == "dpo":
            # 1st half of tensors is preferred (y_w), second half is unpreferred
            model_losses = (pred - target).pow(2).mean(dim=[1, 2])
            model_losses_w, model_losses_l = model_losses.chunk(2)
            model_diff = model_losses_w - model_losses_l  # These are both LBS (as is t)

            with torch.no_grad():  # Get the reference policy (unet) prediction
                ref_transformer_out = FluxTransformer2DModel_forward(
                    self=self.ref_transformer,
                    hidden_states=hidden_states_input,
                    timestep=t,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                    reference_dict=reference_dict,
                )
                ref_pred = ref_transformer_out[0][:, :main_token_len, :]
                ref_losses = (ref_pred - target).pow(2).mean(dim=[1, 2])
                ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                ref_diff = ref_losses_w - ref_losses_l

            scale_term = -0.5 * self.dpo_config["beta"]
            inside_term = scale_term * (model_diff - ref_diff)
            loss = -1 * F.logsigmoid(inside_term).mean()

        # for visualization
        if self.process_visualize and self.debug_dir and rank_zero_only.rank == 0:
            with torch.no_grad():
                from .callbacks import annotate

                debug_image_save_dir = os.path.join(self.debug_dir, "train_debug_images")
                os.makedirs(debug_image_save_dir, exist_ok=True)

                res = (x_t - pred * t_).to(self.dtype)
                res_images = decode_images(self.flux_pipe, res, height, width)
                gt_images = decode_images(self.flux_pipe, x_0, height, width)
                gt_noised_images = decode_images(self.flux_pipe, x_t, height, width)
                for i, instances in enumerate(debug_instance_info):
                    bbox_for_annotation = []
                    phrases_for_annotation = []
                    for j, instance in enumerate(instances["instances"]):
                        bbox_for_annotation.append(instance["bbox"])
                        phrases_for_annotation.append(f"{j + 1}")
                    annotate(res_images[i], bbox_for_annotation, phrases_for_annotation).save(
                        os.path.join(debug_image_save_dir, f"vis_{i}_res.png")
                    )
                    annotate(gt_images[i], bbox_for_annotation, phrases_for_annotation).save(
                        os.path.join(debug_image_save_dir, f"vis_{i}_gt.png")
                    )
                    annotate(gt_noised_images[i], bbox_for_annotation, phrases_for_annotation).save(
                        os.path.join(debug_image_save_dir, f"vis_{i}_gt_noised.png")
                    )
                    annotate(instances["layout"], bbox_for_annotation, phrases_for_annotation).save(
                        os.path.join(debug_image_save_dir, f"vis_{i}_layout.png")
                    )

                    for j, instance in enumerate(instances["instances"]):
                        instance_image_pil = instance["masked_image"]
                        instance_image_pil.save(os.path.join(debug_image_save_dir, f"vis_{i}_ref_{j}.png"))

                    if self.train_method == "dpo":
                        annotate(res_images[i + bs], bbox_for_annotation, phrases_for_annotation).save(
                            os.path.join(debug_image_save_dir, f"vis_{i}_res_less_preferred.png")
                        )
                        annotate(gt_images[i + bs], bbox_for_annotation, phrases_for_annotation).save(
                            os.path.join(debug_image_save_dir, f"vis_{i}_gt_less_preferred.png")
                        )
                        annotate(gt_noised_images[i + bs], bbox_for_annotation, phrases_for_annotation).save(
                            os.path.join(debug_image_save_dir, f"vis_{i}_gt_noised_less_preferred.png")
                        )

        self.last_t = t.mean().item()
        # self.check_param_update()
        return loss
