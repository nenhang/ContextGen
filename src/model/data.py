import json
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils.image_process import round_to_upper_16x


class MigDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        text_drop_prob: int = 0,
        main_image_width: int | None = None,
        using_enhance_rate: float = 0.5,
        reference_image_width: float | None = None,
        data_mix_ratio: dict | None = None,
        using_aligned_face_ratio: float = 0.75,
    ):
        self.dataset_root = dataset_root
        self.text_drop_prob = text_drop_prob
        assert main_image_width is None or main_image_width % 16 == 0, (
            f"main_image_width {main_image_width} must be a multiple of 16"
        )
        self.main_image_width = main_image_width
        self.using_enhance_rate = using_enhance_rate
        self.reference_image_area = reference_image_width**2 if reference_image_width is not None else None
        self.using_aligned_face_ratio = using_aligned_face_ratio

        self.data = []
        self.sub_dataset_dirs = {}

        def get_base_dataset():
            sub_datasets = ["imig-basic", "imig-complex"]
            subdataset_data = []
            for sub_dataset in sub_datasets:
                sub_dataset_root = os.path.join(self.dataset_root, sub_dataset)
                with open(os.path.join(sub_dataset_root, "filtered_prompts.json")) as f:
                    sub_data = json.load(f)
                for data in sub_data:
                    data["sub_dataset"] = sub_dataset
                subdataset_data.extend(sub_data)
                self.sub_dataset_dirs[sub_dataset] = {
                    "image_root": os.path.join(sub_dataset_root, "data"),
                    "instance_image_root": os.path.join(sub_dataset_root, "instance_data"),
                    "masked_instance_root": os.path.join(sub_dataset_root, "masked_instance_data"),
                    "repainted_image_root": os.path.join(sub_dataset_root, "kontext_data"),
                    "masked_repainted_root": os.path.join(sub_dataset_root, "masked_kontext_data"),
                }
            return subdataset_data

        def get_composite_dataset():
            sub_datasets = ["imig-composite", "imig-multicomposite"]
            subdataset_data = []
            for sub_dataset in sub_datasets:
                sub_dataset_root = os.path.join(self.dataset_root, sub_dataset)
                with open(os.path.join(sub_dataset_root, "filtered_prompts.json")) as f:
                    sub_data = json.load(f)
                for data in sub_data:
                    data["sub_dataset"] = sub_dataset
                subdataset_data.extend(sub_data)
                self.sub_dataset_dirs[sub_dataset] = {
                    "image_root": os.path.join(sub_dataset_root, "composite_images"),
                    "reference_image_root": os.path.join(sub_dataset_root, "reference_masks"),
                    "reference_mask_root": os.path.join(sub_dataset_root, "reference_masks"),
                    "aligned_face_root": os.path.join(sub_dataset_root, "aligned_faces"),
                }
            return subdataset_data

        base_data = get_base_dataset()
        composite_data = get_composite_dataset()
        if data_mix_ratio is None:
            self.data = base_data + composite_data
        else:
            self.data = []
            base_ratio = data_mix_ratio.get("base", 0.5)
            composite_ratio = data_mix_ratio.get("composite", 0.5)
            assert not (base_ratio == 0 and composite_ratio == 0), "Both base_ratio and composite_ratio cannot be zero."
            normed_base_ratio = base_ratio / (base_ratio + composite_ratio)
            normed_composite_ratio = composite_ratio / (base_ratio + composite_ratio)
            if normed_base_ratio != 0 and normed_composite_ratio != 0:
                base_composite_ratio = len(base_data) / len(composite_data)
                if normed_base_ratio / normed_composite_ratio < base_composite_ratio:
                    # make full use of composite data and sample base data
                    composite_len = len(composite_data)
                    base_len = round(composite_len * normed_base_ratio / normed_composite_ratio)
                else:
                    # make full use of base data and sample composite data
                    base_len = len(base_data)
                    composite_len = round(base_len * normed_composite_ratio / normed_base_ratio)

                import random

                base_sampled = random.sample(base_data, base_len)
                composite_sampled = random.sample(composite_data, composite_len)
                self.data = base_sampled + composite_sampled
            elif normed_base_ratio != 0:
                self.data = base_data
                base_sampled = base_data
                composite_sampled = []
            else:
                self.data = composite_data
                base_sampled = []
                composite_sampled = composite_data
            print(
                f"Dataset mixed: {len(base_sampled)} base + {len(composite_sampled)} composite = {len(self.data)} total",
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if item["sub_dataset"] in ["imig-basic", "imig-complex"]:
            image_path = os.path.join(
                self.sub_dataset_dirs[item["sub_dataset"]]["image_root"],
                f"{item['index']:06d}.jpg",
            )
            if torch.rand(1) < self.text_drop_prob:
                caption = ""
            else:
                caption = item["prompt"]
            reference_images = item["reference_images"]

            main_image_orig = Image.open(image_path).convert("RGB")
            scale_factor = None

            main_image = main_image_orig
            if self.main_image_width is not None:
                new_h, new_w = round_to_upper_16x(self.main_image_width, self.main_image_width)
                main_image = main_image_orig.resize((new_w, new_h))
                scale_factor = new_h / main_image_orig.height

            reference_data = []
            for i, ref in enumerate(reference_images):
                bbox = (
                    torch.tensor(ref["bbox"], dtype=torch.int32)
                    if scale_factor is None
                    else torch.round(torch.tensor(ref["bbox"], dtype=torch.float32) * scale_factor).int()
                )

                enhanced = False
                valid_instance_bbox = None
                if torch.rand(1) < self.using_enhance_rate:
                    ref_image_path = os.path.join(
                        self.sub_dataset_dirs[item["sub_dataset"]]["repainted_image_root"],
                        f"{item['index']:06d}_{ref['index']}.jpg",
                    )
                    mask_image_path = os.path.join(
                        self.sub_dataset_dirs[item["sub_dataset"]]["masked_repainted_root"],
                        f"{item['index']:06d}_{ref['index']}_mask.png",
                    )
                    enhanced = True
                    valid_instance_bbox = ref.get("valid_bbox", None)
                else:
                    ref_image_path = os.path.join(
                        self.sub_dataset_dirs[item["sub_dataset"]]["instance_image_root"],
                        f"{item['index']:06d}_{ref['index']}.jpg",
                    )
                    mask_image_path = os.path.join(
                        self.sub_dataset_dirs[item["sub_dataset"]]["masked_instance_root"],
                        f"{item['index']:06d}_{ref['index']}_mask.png",
                    )

                ref_image = Image.open(ref_image_path).convert("RGB")
                mask_image = Image.open(mask_image_path).convert("L")
                assert ref_image.size == mask_image.size, (
                    f"Reference image {ref_image} size {ref_image.size} does not match mask image {mask_image} size {mask_image.size}"
                )
                if valid_instance_bbox is not None:
                    valid_instance_bbox = torch.tensor(valid_instance_bbox, dtype=torch.float32) / torch.tensor(
                        [ref_image.width, ref_image.height, ref_image.width, ref_image.height],
                        dtype=torch.float32,
                    )
                if self.reference_image_area is not None:
                    aspect_ratio = ref_image.width / ref_image.height
                    new_size = (
                        round((self.reference_image_area * aspect_ratio) ** 0.5),
                        round((self.reference_image_area / aspect_ratio) ** 0.5),
                    )
                    ref_image = ref_image.resize(new_size, Image.LANCZOS)
                    mask_image = mask_image.resize(new_size, Image.LANCZOS)

                elif scale_factor is not None:
                    new_w = round(ref_image.width * scale_factor)
                    new_h = round(ref_image.height * scale_factor)

                    ref_image = ref_image.resize((new_w, new_h))
                    mask_image = mask_image.resize((new_w, new_h))

                reference_data.append(
                    {
                        "image": ref_image,
                        "bbox": bbox,
                        "mask": mask_image,
                        "phrase": ref["phrase"],
                        "enhanced": enhanced,
                        "valid_instance_bbox": valid_instance_bbox,
                        "contain_face": False,
                    },
                )

        elif item["sub_dataset"] in ["imig-composite", "imig-multicomposite"]:
            image_path = os.path.join(
                self.sub_dataset_dirs[item["sub_dataset"]]["image_root"],
                f"{item['index']:06d}.jpg",
            )
            caption = item["prompt"]
            reference_images = item["instance"]

            main_image_orig = Image.open(image_path).convert("RGB")
            scale_factor = None

            main_image = main_image_orig
            if self.main_image_width is not None:
                new_h, new_w = round_to_upper_16x(self.main_image_width, self.main_image_width)
                main_image = main_image_orig.resize((new_w, new_h))
                scale_factor = new_h / main_image_orig.height

            reference_data = []
            indices_mapping = item["indices"]
            for i, phrase in enumerate(reference_images):
                contain_face = False
                if (
                    item["face_bbox"][i] is not None
                    and item["face_bbox"][i]["aligned_face_bbox"] is not None
                    and (item["bbox"][i] is None or torch.rand(1) < self.using_aligned_face_ratio)
                ):
                    target_face_bbox = (
                        torch.tensor(item["face_bbox"][i]["target"], dtype=torch.float32)
                        if scale_factor is None
                        else torch.tensor(item["face_bbox"][i]["target"], dtype=torch.float32) * scale_factor
                    )
                    target_face_bbox_center = (
                        (target_face_bbox[1] + target_face_bbox[3]) / 2,
                        (target_face_bbox[0] + target_face_bbox[2]) / 2,
                    )
                    face_bbox_normed_center = (
                        target_face_bbox_center[0] / main_image.height,
                        target_face_bbox_center[1] / main_image.width,
                    )
                    aligned_face_bbox = torch.tensor(item["face_bbox"][i]["aligned_face_bbox"], dtype=torch.float32)
                    aligned_face_bbox_center = (
                        (aligned_face_bbox[1] + aligned_face_bbox[3]) / 2,
                        (aligned_face_bbox[0] + aligned_face_bbox[2]) / 2,
                    )

                    aligned_face_bbox_width = aligned_face_bbox[2] - aligned_face_bbox[0]
                    aligned_face_bbox_height = aligned_face_bbox[3] - aligned_face_bbox[1]
                    aligned_face_bbox_area = aligned_face_bbox_width * aligned_face_bbox_height

                    target_face_bbox_width = target_face_bbox[2] - target_face_bbox[0]
                    target_face_bbox_height = target_face_bbox[3] - target_face_bbox[1]

                    aligned_face_path = os.path.join(
                        self.sub_dataset_dirs[item["sub_dataset"]]["aligned_face_root"],
                        f"{item['index']:06d}_{indices_mapping[i]}_aligned_face.png",
                    )
                    aligned_face = Image.open(aligned_face_path).convert("RGBA")
                    aligned_face_mask = aligned_face.split()[-1].convert("L")

                    aligned_face_image_area = aligned_face.width * aligned_face.height
                    face_scale_factor = (aligned_face_image_area / aligned_face_bbox_area) ** 0.5

                    # scale the target bbox around its center
                    target_face_bbox_center_x, target_face_bbox_center_y = target_face_bbox_center
                    target_face_bbox_width_scaled = target_face_bbox_width * face_scale_factor
                    target_face_bbox_height_scaled = target_face_bbox_height * face_scale_factor
                    new_target_face_bbox = torch.tensor(
                        [
                            torch.clamp(target_face_bbox_center_y - target_face_bbox_width_scaled / 2, min=0).item(),
                            torch.clamp(target_face_bbox_center_x - target_face_bbox_height_scaled / 2, min=0).item(),
                            torch.clamp(
                                target_face_bbox_center_y + target_face_bbox_width_scaled / 2,
                                max=main_image.width,
                            ).item(),
                            torch.clamp(
                                target_face_bbox_center_x + target_face_bbox_height_scaled / 2,
                                max=main_image.height,
                            ).item(),
                        ],
                        dtype=torch.int32,
                    )

                    bbox = new_target_face_bbox
                    ref_image = aligned_face.convert("RGB")
                    mask_image = aligned_face_mask
                    ref_face_bbox_normed_center = (
                        aligned_face_bbox_center[0] / aligned_face.width,
                        aligned_face_bbox_center[1] / aligned_face.height,
                    )
                    contain_face = True
                else:
                    # use obj bbox as bbox, and provide the loc point
                    bbox = (
                        torch.tensor(item["bbox"][i]["bbox"], dtype=torch.int32)
                        if scale_factor is None
                        else torch.round(
                            torch.tensor(item["bbox"][i]["bbox"], dtype=torch.float32) * scale_factor,
                        ).int()
                    )

                    ref_image_path = os.path.join(
                        self.sub_dataset_dirs[item["sub_dataset"]]["reference_image_root"],
                        f"{item['index']:06d}_{indices_mapping[i]}.jpg",
                    )
                    mask_image_path = os.path.join(
                        self.sub_dataset_dirs[item["sub_dataset"]]["reference_mask_root"],
                        f"{item['index']:06d}_{indices_mapping[i]}_mask.png",
                    )

                    ref_image = Image.open(ref_image_path).convert("RGB")
                    mask_image = Image.open(mask_image_path).convert("L")

                    if item["face_bbox"][i] is not None:
                        face_bbox = torch.tensor(item["face_bbox"][i]["target"], dtype=torch.float32)
                        face_bbox_normed = face_bbox / torch.tensor(
                            [
                                main_image_orig.width,
                                main_image_orig.height,
                                main_image_orig.width,
                                main_image_orig.height,
                            ],
                            dtype=torch.float32,
                        )
                        face_bbox_normed_center = (
                            (face_bbox_normed[1] + face_bbox_normed[3]) / 2,
                            (face_bbox_normed[0] + face_bbox_normed[2]) / 2,
                        )
                        ref_face_bbox = torch.tensor(item["face_bbox"][i]["ref"], dtype=torch.float32)
                        ref_face_bbox_normed = ref_face_bbox / torch.tensor(
                            [ref_image.width, ref_image.height, ref_image.width, ref_image.height],
                            dtype=torch.float32,
                        )
                        ref_face_bbox_normed_center = (
                            (ref_face_bbox_normed[1] + ref_face_bbox_normed[3]) / 2,
                            (ref_face_bbox_normed[0] + ref_face_bbox_normed[2]) / 2,
                        )
                        contain_face = True
                    else:
                        ref_face_bbox_normed_center = None
                        face_bbox_normed_center = None

                if self.reference_image_area is not None:
                    # keep aspect ratio
                    aspect_ratio = ref_image.width / ref_image.height
                    new_size = (
                        round((self.reference_image_area * aspect_ratio) ** 0.5),
                        round((self.reference_image_area / aspect_ratio) ** 0.5),
                    )
                    ref_image = ref_image.resize(new_size, Image.LANCZOS)
                    mask_image = mask_image.resize(new_size, Image.LANCZOS)

                elif scale_factor is not None:
                    new_w = round(ref_image.width * scale_factor)
                    new_h = round(ref_image.height * scale_factor)

                    ref_image = ref_image.resize((new_w, new_h))
                    mask_image = mask_image.resize((new_w, new_h))

                reference_data.append(
                    {
                        "image": ref_image,
                        "bbox": bbox,
                        "mask": mask_image,
                        "phrase": phrase,
                        "loc_point": (ref_face_bbox_normed_center, face_bbox_normed_center)
                        if ref_face_bbox_normed_center is not None and face_bbox_normed_center is not None
                        else None,
                        "contain_face": contain_face,
                    },
                )

        else:
            raise ValueError(f"Unknown sub-dataset type: {item['sub_dataset']}")

        return {
            "target_image": main_image,
            "caption": caption,
            "reference_image": reference_data,
            "image_path": image_path,
        }

    def collate_fn(self, batch):
        main_images = []
        captions = []
        reference_images = []
        image_paths = []
        for item in batch:
            if item is None:
                continue
            main_image = item["target_image"]
            caption = item["caption"]
            reference_image = item["reference_image"]
            image_path = item["image_path"]
            main_images.append(main_image)
            captions.append(caption)
            reference_images.append(reference_image)
            image_paths.append(image_path)
        if len(main_images) == 0:
            return None
        return {
            "target_images": main_images,
            "captions": captions,
            "reference_images": reference_images,
            "image_paths": image_paths,
        }

    def get_dataloader(self, batch_size, num_workers, shuffle=True, sampler=None):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            sampler=sampler,
        )
