import torch
from PIL import Image
from diffusers.pipelines.flux.pipeline_flux import logger
from diffusers.utils import logging
from torch import Tensor


def encode_images(pipeline, images: Tensor | Image.Image) -> Tensor:
    if not isinstance(images, Tensor):
        images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (images - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    return images_tokens


def decode_images(pipeline, latents: Tensor, height: int, width: int, output_type: str = "pil"):
    latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    image = pipeline.vae.decode(latents, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type=output_type)
    return image


def prepare_text_input(pipeline, prompts, max_sequence_length=512):
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def encode_images_fill(pipeline, image: Tensor, mask_image: Tensor) -> Tensor:
    height, width = image.shape[-2:]
    if not isinstance(image, Tensor):
        image = pipeline.image_processor.preprocess(image, height=height, width=width)
    if not isinstance(mask_image, Tensor):
        mask_image = pipeline.mask_processor.preprocess(mask_image, height=height, width=width)
    masked_image = image * (1 - mask_image)
    masked_image = masked_image.to(device=pipeline.device, dtype=pipeline.dtype)
    num_channels_latents = pipeline.vae.config.latent_channels
    mask, masked_image_latents = pipeline.prepare_mask_latents(
        mask_image,
        masked_image,
        batch_size=image.shape[0],
        num_channels_latents=num_channels_latents,
        num_images_per_prompt=1,
        height=height,
        width=width,
        dtype=pipeline.dtype,
        device=pipeline.device,
        generator=None,
    )
    masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)
    return masked_image_latents
