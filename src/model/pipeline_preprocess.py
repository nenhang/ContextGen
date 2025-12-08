import os

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from src.flux.pipeline_tools import decode_images, encode_images
from src.utils.image_process import (
    determine_mask_sequence,
    place_instance_image_and_mask,
    preprocess,
    stack_images_with_masks,
)


def prepare_mig_attn_mask(
    width_in_tokens: int,
    height_in_tokens: int,
    batch_instance_token_1d_idx_list: list,
    batch_instance_bbox_2d_idx_list: list,
    batch_instance_mask_list: list,
    txt_token_len: int = 512,
    device=torch.device("cuda"),
    # for visualization
    debug_dir=None,
    image_instance_token_2d_size_list=None,
):
    main_token_len = width_in_tokens * height_in_tokens
    bs = len(batch_instance_bbox_2d_idx_list)
    k_len = max(instance_token_1d_idx_list[-1] for instance_token_1d_idx_list in batch_instance_token_1d_idx_list)
    attn_mask = torch.zeros(bs, k_len, k_len, dtype=torch.bool, device=device)
    # full_k_mask = torch.ones(bs, k_len, dtype=torch.bool, device=device)

    # 确保 batch 中每个 token 序列的前两个 instance 分别是 main image 和 layout image，且两者的 token 数量相同
    assert all(batch_instance_token_1d_idx_list[i][1] == 2 * main_token_len for i in range(bs))

    for i, instance_bbox_2d_idx_list in enumerate(batch_instance_bbox_2d_idx_list):
        num_bboxes = len(instance_bbox_2d_idx_list)
        bbox_mask_2d = torch.zeros(num_bboxes, height_in_tokens, width_in_tokens, dtype=torch.bool, device=device)
        for j, bbox in enumerate(instance_bbox_2d_idx_list):
            # if j == 0 and bbox[0] == bbox[1] == 0 and bbox[2] == width_in_tokens and bbox[3] == height_in_tokens:
            #     continue  # layout image 不参与 bbox wise attention mask 的计算
            bbox_mask_2d[j, bbox[1] : bbox[3], bbox[0] : bbox[2]] = True  # bbox 部分填充 True
        bbox_mask_1d = torch.zeros(num_bboxes, k_len, dtype=torch.bool, device=device)
        bbox_mask_1d[:, :main_token_len] = rearrange(bbox_mask_2d, "n h w -> n (h w)")
        bbox_mask_union = bbox_mask_1d.any(dim=0)
        bbox_mask_union[main_token_len:] = True  # 后面拼上去的 refs 部分不能关注 main image 的部分
        assert (
            batch_instance_token_1d_idx_list[i][0] == main_token_len
            and batch_instance_token_1d_idx_list[i][1] == 2 * main_token_len
            and torch.all(batch_instance_mask_list[i][0])
        )
        attn_mask[i, ~bbox_mask_union, : 2 * main_token_len] = True  # bbox 外部分关注 main image 和 layout image
        for j, bbox_mask_1d_j in enumerate(bbox_mask_1d):
            k_attention_mask = bbox_mask_1d_j.clone()
            instance_start_idx = batch_instance_token_1d_idx_list[i][j]
            instance_end_idx = batch_instance_token_1d_idx_list[i][j + 1]
            k_attention_mask[instance_start_idx:instance_end_idx] = batch_instance_mask_list[i][j]
            attn_mask[i, bbox_mask_1d_j == 1] |= k_attention_mask  # bbox 内部关注自己
            if j == 0:
                attn_mask[i, instance_start_idx:instance_end_idx, 0:instance_end_idx] = True
            else:
                attn_mask[i, instance_start_idx:instance_end_idx, instance_start_idx:instance_end_idx] = (
                    batch_instance_mask_list[i][j]
                )
            # full_k_mask[i, instance_start_idx:instance_end_idx] = batch_instance_mask_list[i][j]

    attn_mask_txt_k = torch.ones(bs, k_len, txt_token_len, dtype=torch.bool, device=device)  # 512 是默认的文本长度
    attn_mask_with_txt = torch.cat([attn_mask_txt_k, attn_mask], dim=2)  # 先拼接文本的 mask
    # attn_mask_txt_q = torch.ones(bs, txt_token_len, k_len + txt_token_len, dtype=torch.bool, device=device)
    # attn_mask_txt_q[:, :, txt_token_len:] = full_k_mask.unsqueeze(1)
    attn_mask_txt_q = torch.zeros(bs, txt_token_len, k_len + txt_token_len, dtype=torch.bool, device=device)
    attn_mask_txt_q[:, :, : txt_token_len + 2 * main_token_len] = True
    attn_mask_with_txt = torch.cat([attn_mask_txt_q, attn_mask_with_txt], dim=1)  # 再拼接文本的 mask

    layout_only_attn_mask = attn_mask.clone()
    layout_only_attn_mask[:, : 2 * main_token_len, : 2 * main_token_len] = True
    layout_only_attn_mask[:, : 2 * main_token_len, 2 * main_token_len :] = False
    layout_only_attn_mask_txt_q = torch.zeros(bs, txt_token_len, k_len + txt_token_len, dtype=torch.bool, device=device)
    layout_only_attn_mask_txt_q[:, :, : txt_token_len + 2 * main_token_len] = True
    layout_only_attn_mask_with_txt = torch.cat([attn_mask_txt_k, layout_only_attn_mask], dim=2)
    layout_only_attn_mask_with_txt = torch.cat([layout_only_attn_mask_txt_q, layout_only_attn_mask_with_txt], dim=1)

    if debug_dir:

        def visualize_attn_mask(instance_token_2d_size_list, attn_mask_1d, txt_len):
            total_width = sum([width for width, _ in instance_token_2d_size_list])
            total_height = max([height for _, height in instance_token_2d_size_list])
            attn_mask_2d = torch.zeros(total_height, total_width, dtype=torch.bool, device=device)
            for k, instance_size in enumerate(instance_token_2d_size_list):
                ins_w, ins_h = instance_size
                instance_1d_begin_idx = batch_instance_token_1d_idx_list[i][k - 1] + txt_len if k > 0 else txt_len
                instance_2d_begin_x = sum([width for width, _ in instance_token_2d_size_list[:k]])
                for row in range(ins_h):
                    row_1d_begin_idx = row * ins_w + instance_1d_begin_idx
                    attn_mask_2d[row, instance_2d_begin_x : instance_2d_begin_x + ins_w] = attn_mask_1d[
                        row_1d_begin_idx : row_1d_begin_idx + ins_w
                    ]

            attn_mask_2d_uint8 = (attn_mask_2d.to(torch.uint8) * 255).cpu().numpy()
            attn_mask_2d_image = Image.fromarray(attn_mask_2d_uint8).convert("1")
            return attn_mask_2d_image

        assert image_instance_token_2d_size_list is not None, (
            "image_instance_token_2d_size_list must be provided for visualization."
        )
        layout_only_masks_save_dir = os.path.join(debug_dir, "layout_only_attn_masks")
        ref_aware_masks_save_dir = os.path.join(debug_dir, "ref_aware_attn_masks")
        os.makedirs(layout_only_masks_save_dir, exist_ok=True)
        os.makedirs(ref_aware_masks_save_dir, exist_ok=True)
        for i, instance_bbox_2d_idx_list in enumerate(batch_instance_bbox_2d_idx_list):
            txt_token_mask = visualize_attn_mask(
                image_instance_token_2d_size_list[i],
                attn_mask_with_txt[i, 0],
                txt_token_len,
            )
            txt_token_mask.save(os.path.join(ref_aware_masks_save_dir, f"{i}_txt_token_mask.png"))
            layout_only_txt_token_mask = visualize_attn_mask(
                image_instance_token_2d_size_list[i],
                layout_only_attn_mask_with_txt[i, 0],
                txt_token_len,
            )
            layout_only_txt_token_mask.save(os.path.join(layout_only_masks_save_dir, f"{i}_txt_token_mask.png"))
            left_up_token_mask = visualize_attn_mask(
                image_instance_token_2d_size_list[i],
                attn_mask_with_txt[i, txt_token_len],
                txt_token_len,
            )
            left_up_token_mask.save(os.path.join(ref_aware_masks_save_dir, f"{i}_left_up_token_mask.png"))
            layout_only_left_up_token_mask = visualize_attn_mask(
                image_instance_token_2d_size_list[i],
                layout_only_attn_mask_with_txt[i, txt_token_len],
                txt_token_len,
            )
            layout_only_left_up_token_mask.save(os.path.join(layout_only_masks_save_dir, f"{i}_left_up_token_mask.png"))
            right_down_token_mask = visualize_attn_mask(
                image_instance_token_2d_size_list[i],
                attn_mask_with_txt[i, txt_token_len + main_token_len - 1],
                txt_token_len,
            )
            right_down_token_mask.save(os.path.join(ref_aware_masks_save_dir, f"{i}_right_down_token_mask.png"))
            layout_only_right_down_token_mask = visualize_attn_mask(
                image_instance_token_2d_size_list[i],
                layout_only_attn_mask_with_txt[i, txt_token_len + main_token_len - 1],
                txt_token_len,
            )
            layout_only_right_down_token_mask.save(
                os.path.join(layout_only_masks_save_dir, f"{i}_right_down_token_mask.png")
            )
            for j, bbox in enumerate(instance_bbox_2d_idx_list):
                bbox_start_idx = bbox[1] * width_in_tokens + bbox[0] + txt_token_len
                bbox_mask = visualize_attn_mask(
                    image_instance_token_2d_size_list[i],
                    attn_mask_with_txt[i, bbox_start_idx],
                    txt_token_len,
                )
                bbox_mask.save(os.path.join(ref_aware_masks_save_dir, f"{i}_bbox_{j}_mask.png"))
                layout_only_bbox_mask = visualize_attn_mask(
                    image_instance_token_2d_size_list[i],
                    layout_only_attn_mask_with_txt[i, bbox_start_idx],
                    txt_token_len,
                )
                layout_only_bbox_mask.save(os.path.join(layout_only_masks_save_dir, f"{i}_bbox_{j}_mask.png"))
                refs_start_idx = batch_instance_token_1d_idx_list[i][j] + txt_token_len
                ref_mask = visualize_attn_mask(
                    image_instance_token_2d_size_list[i],
                    attn_mask_with_txt[i, refs_start_idx],
                    txt_token_len,
                )
                ref_mask.save(os.path.join(ref_aware_masks_save_dir, f"{i}_ref_{j}_mask.png"))
                layout_only_ref_mask = visualize_attn_mask(
                    image_instance_token_2d_size_list[i],
                    layout_only_attn_mask_with_txt[i, refs_start_idx],
                    txt_token_len,
                )
                layout_only_ref_mask.save(os.path.join(layout_only_masks_save_dir, f"{i}_ref_{j}_mask.png"))

    return {
        "attn_mask": attn_mask_with_txt.unsqueeze(1),
        "layout_only_attn_mask": layout_only_attn_mask_with_txt.unsqueeze(1),
    }


def prepare_instance_img_pe(
    batch_instance_token_2d_size_list: list,  # in (width, height) format
    flux_pipe,
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
):
    def prepare_latent_image_ids_with_offset(batch_size, height, width, device, dtype, start_y=0, start_x=0):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None] + start_y
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :] + start_x

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        return latent_image_ids.to(device=device, dtype=dtype)

    text_ids = torch.zeros(512, 3).to(device=device, dtype=dtype)
    instance_pe_cos_list = []
    instance_pe_sin_list = []
    main_pe_cos_list = []
    main_pe_sin_list = []
    for i, instance_token_2d_size_list in enumerate(batch_instance_token_2d_size_list):
        instances_ids = text_ids  # text 拼在前面
        for j, (ins_w, ins_h) in enumerate(instance_token_2d_size_list):
            instance_2d_begin_x = sum([width for width, _ in instance_token_2d_size_list[1:j]]) if j >= 2 else 0
            instance_2d_begin_y = sum([height for _, height in instance_token_2d_size_list[1:j]]) if j >= 2 else 0
            instance_ids = prepare_latent_image_ids_with_offset(
                batch_size=1,
                height=ins_h,
                width=ins_w,
                device=device,
                dtype=dtype,
                start_y=instance_2d_begin_y,  # instance 的左上角 y 坐标
                start_x=instance_2d_begin_x,  # instance 的左上角 x 坐标
            )
            # 只有 main image token 的 ids 的第一列是 0，其他的都是 1
            if j >= 1:
                instance_ids[..., 0] = 1
            instances_ids = torch.cat([instances_ids, instance_ids], dim=0)
            if j == 0:
                main_token_ids = instances_ids
        instances_pe_cos, instances_pe_sin = flux_pipe.transformer.pos_embed(instances_ids)
        instance_pe_cos_list.append(instances_pe_cos)
        instance_pe_sin_list.append(instances_pe_sin)
        main_pe_cos, main_pe_sin = flux_pipe.transformer.pos_embed(main_token_ids)
        main_pe_cos_list.append(main_pe_cos)
        main_pe_sin_list.append(main_pe_sin)
    instance_pe_out_cos = torch.nn.utils.rnn.pad_sequence(instance_pe_cos_list, batch_first=True).unsqueeze(1)
    instance_pe_out_sin = torch.nn.utils.rnn.pad_sequence(instance_pe_sin_list, batch_first=True).unsqueeze(1)
    main_pe_out_cos = torch.stack(main_pe_cos_list, dim=0).unsqueeze(1)
    main_pe_out_sin = torch.stack(main_pe_sin_list, dim=0).unsqueeze(1)
    return {
        "rotary_emb": (main_pe_out_cos, main_pe_out_sin),
        "ctx_rotary_emb": (instance_pe_out_cos, instance_pe_out_sin),
    }


def prepare_reference_latents(
    flux_pipe,
    instance_info: list,
    width: int,
    height: int,
    given_layout_images=None,
    resize_pattern=None,
    device=torch.device("cuda"),
    debug_visualize=False,
):
    bs = len(instance_info)
    width_in_token = width // 16
    height_in_token = height // 16
    main_token_len = width_in_token * height_in_token

    instance_latent_list = [None] * bs

    batch_instance_token_1d_idx_list = []
    batch_instance_bbox_2d_idx_list = []
    batch_instance_token_2d_size_list = []
    batch_instance_mask_list = []

    stacked_instance_image_list = []

    debug_instance_list = None if not debug_visualize else [{"instances": []} for _ in range(bs)]

    for i, image_instance_info in enumerate(instance_info):
        instance_token_1d_idx_list = [main_token_len, 2 * main_token_len]
        instance_bbox_2d_idx_list = [(0, 0, width_in_token, height_in_token)]
        instance_token_2d_size_list = [(width_in_token, height_in_token), (width_in_token, height_in_token)]

        instance_mask_list = [torch.ones(main_token_len, dtype=torch.bool, device=device)]

        instance_image_layer_list = []
        instance_mask_layer_list = []

        for j, instance in enumerate(image_instance_info):
            instance_image_pil = instance["image"]
            bbox = instance["bbox"]
            instance_mask_pil = instance["mask"]
            instance_image = flux_pipe.image_processor.preprocess(instance_image_pil)
            instance_mask = flux_pipe.mask_processor.preprocess(instance_mask_pil)
            instance_image = instance_image * instance_mask

            ins_h, ins_w = instance_image.shape[2], instance_image.shape[3]
            ins_token_h, ins_token_w = ins_h // 16, ins_w // 16
            instance_token_2d_size_list.append((ins_token_w, ins_token_h))  # 注意是 w, h

            instance_token_len = ins_token_h * ins_token_w
            instance_token_1d_idx_list.append(instance_token_1d_idx_list[-1] + instance_token_len)

            x1, y1, x2, y2 = bbox
            x1_ = int(torch.floor(x1 / 16))
            y1_ = int(torch.floor(y1 / 16))
            x2_ = int(torch.ceil(x2 / 16))
            y2_ = int(torch.ceil(y2 / 16))

            instance_bbox_2d_idx_list.append((x1_, y1_, x2_, y2_))

            instance_image_0 = encode_images(flux_pipe, instance_image)
            instance_latent_list[i] = (
                instance_image_0
                if instance_latent_list[i] is None
                else torch.cat([instance_latent_list[i], instance_image_0], dim=1)
            )

            instance_mask = np.array(instance_mask_pil.resize((ins_token_w, ins_token_h)))
            instance_mask = torch.from_numpy(instance_mask).to(device)
            instance_mask = (instance_mask > 0).reshape(instance_token_len)
            instance_mask_list.append(instance_mask)

            if debug_visualize:
                debug_instance_list[i]["instances"].append(
                    {
                        "image": flux_pipe.image_processor.postprocess(instance_image)[0],
                        "bbox": bbox,
                        "masked_image": decode_images(
                            flux_pipe,
                            instance_image_0 * instance_mask.unsqueeze(0).unsqueeze(2),
                            ins_h,
                            ins_w,
                        )[0],
                    }
                )

            if given_layout_images is not None:
                continue

            enhanced = instance.get("enhanced", True)
            valid_instance_bbox = instance.get("valid_instance_bbox", None)
            loc_point = instance.get("loc_point", None)
            instance_image_orig = preprocess(flux_pipe.image_processor, instance_image_pil).to(device)
            instance_mask_orig = preprocess(flux_pipe.mask_processor, instance_mask_pil).to(device)
            instance_image_layer, instance_mask_layer = place_instance_image_and_mask(
                instance_image_orig,
                instance_mask_orig,
                full_size=(height, width),
                bbox=bbox,
                resize_pattern=resize_pattern  # if resize_pattern is not None, use it, otherwise use the default value
                if resize_pattern is not None
                else "instance-fit"
                if (enhanced and valid_instance_bbox is not None)
                else "image-fit",
                valid_part_bbox=valid_instance_bbox,
                loc_point=loc_point,
                device=device,
            )
            instance_image_layer_list.append(instance_image_layer)
            instance_mask_layer_list.append(instance_mask_layer)
        if given_layout_images is None:
            image_sequence = determine_mask_sequence(instance_mask_layer_list)
            instance_image_layers = [instance_image_layer_list[i] for i in image_sequence]
            instance_mask_layers = [instance_mask_layer_list[i] for i in image_sequence]
            stacked_instance_image, _ = stack_images_with_masks(instance_image_layers, instance_mask_layers, 0.0)
        else:
            layout_image_i = flux_pipe.image_processor.preprocess(given_layout_images[i]["image"]).to(device)
            layout_mask_i = flux_pipe.mask_processor.preprocess(given_layout_images[i]["mask"]).to(device)
            stacked_instance_image = layout_image_i * layout_mask_i

        stacked_instance_image_list.append(stacked_instance_image.squeeze(0))

        batch_instance_token_1d_idx_list.append(instance_token_1d_idx_list)
        batch_instance_bbox_2d_idx_list.append(instance_bbox_2d_idx_list)
        batch_instance_token_2d_size_list.append(instance_token_2d_size_list)
        batch_instance_mask_list.append(instance_mask_list)

    instance_latent_list = [instance_x_0.squeeze(0) for instance_x_0 in instance_latent_list]
    instance_latents = torch.nn.utils.rnn.pad_sequence(instance_latent_list, batch_first=True)

    layout_images = torch.stack(stacked_instance_image_list, dim=0).to(device)
    layout_latents = encode_images(flux_pipe, layout_images)
    layout_instance_latents = torch.cat([layout_latents, instance_latents], dim=1)

    if debug_visualize:
        layout_images_pil = decode_images(flux_pipe, layout_latents, height, width)
        for i in range(bs):
            debug_instance_list[i]["layout"] = layout_images_pil[i]

    return {
        "layout_instance_latents": layout_instance_latents,
        "layout_latents": layout_latents,  # for dpo
        "batch_instance_token_1d_idx_list": batch_instance_token_1d_idx_list,
        "batch_instance_bbox_2d_idx_list": batch_instance_bbox_2d_idx_list,
        "batch_instance_token_2d_size_list": batch_instance_token_2d_size_list,
        "batch_instance_mask_list": batch_instance_mask_list,
        "debug_instance_info": debug_instance_list,
    }
