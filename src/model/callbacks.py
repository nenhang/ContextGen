import json
import os

import lightning as L
import torch

import wandb

from ..utils.image_process import annotate
from .generate import generate


class TrainingCallback(L.Callback):
    def __init__(self, run_name, sample_data_loader=None, training_config=None):
        self.run_name = run_name
        self.training_config = training_config or {}
        self.sample_data_loader = sample_data_loader

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = os.path.join(
            training_config.get("save_path", "./output"), training_config.get("model_name", "main")
        )

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = wandb is not None
        self.current_steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        pl_module.total_steps += 1
        self.current_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "batch": batch_idx,
                "steps": pl_module.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict, step=pl_module.total_steps)

        if self.current_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {pl_module.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.current_steps % self.save_interval == 0:
            print(f"Epoch: {trainer.current_epoch}, Steps: {pl_module.total_steps} - Saving LoRA weights")
            pl_module.save_optimize_parameters(
                run_name=self.run_name, total_steps=pl_module.total_steps, remove_old=True
            )

        # Generate and save a sample image at specified intervals
        if self.current_steps % self.sample_interval == 0 or self.current_steps == 1:
            print(f"Epoch: {trainer.current_epoch}, Steps: {pl_module.total_steps} - Generating a sample")
            self.generate_sample(
                pl_module,
                f"{self.save_path}/{self.run_name}/samples",
                f"{pl_module.total_steps:05d}",
            )

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Save LoRA weights at the end of each epoch
        print(f"Epoch: {trainer.current_epoch} - Saving LoRA weights at epoch end")
        pl_module.save_optimize_parameters(run_name=self.run_name, total_steps=pl_module.total_steps, remove_old=False)

        # Generate and save a sample image at the end of each epoch
        print(f"Epoch: {trainer.current_epoch} - Generating a sample at epoch end")
        self.generate_sample(
            pl_module,
            f"{self.save_path}/{self.run_name}/samples",
            f"epoch_{trainer.current_epoch}",
        )
        return super().on_train_epoch_end(trainer, pl_module)

    @torch.no_grad()
    def generate_sample(self, pl_module, save_path, file_name_prefix, mode: str = "fixed_instances", seed=42):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if mode == "fixed_instances":
            fix_width = 768
            num_samples_per_image = 4
            with open("./images/input/segment_info.json") as f:
                segment_info = json.load(f)
            for i, item in enumerate(segment_info):
                for sample_id in range(num_samples_per_image):
                    prompt = item["caption"]
                    width = item["width"]
                    reference_info = []
                    layout_image = item.get("layout_image", None)
                    if layout_image is not None:
                        layout_image = os.path.join("./images/input", layout_image)
                        assert os.path.exists(layout_image), f"Layout image {layout_image} does not exist."
                    for instance in item["instances"]:
                        bbox = (torch.tensor(instance["bbox"]) * fix_width / width).to(dtype=torch.int32)
                        instance_image_path = os.path.join("./images/input", instance["image"])
                        assert os.path.exists(instance_image_path), f"Image {instance_image_path} does not exist."
                        if "mask" in instance:
                            instance_mask_path = os.path.join("./images/input", instance["mask"])
                            assert os.path.exists(instance_mask_path), f"Mask {instance_mask_path} does not exist."
                        else:
                            instance_mask_path = None
                        reference_info.append(
                            {
                                "image": instance_image_path,
                                "bbox": bbox,
                                "mask": instance_mask_path,
                            }
                        )
                    bbox_for_annotation = [info["bbox"] for info in reference_info]
                    labels_for_annotation = [segment["label"] for segment in item["instances"]]
                    seed_ = seed + sample_id
                    res_image = generate(
                        flux_pipe=pl_module.flux_pipe,
                        prompts=[prompt],
                        reference_info=[reference_info],
                        width=fix_width,
                        height=fix_width,
                        layout_image=[layout_image] if layout_image is not None else None,
                        seed=seed_,
                    )[0]
                    sample_save_path = os.path.join(save_path, f"{sample_id:02d}")
                    os.makedirs(sample_save_path, exist_ok=True)
                    res_image.save(
                        os.path.join(sample_save_path, f"{file_name_prefix}_{i:02d}_{sample_id}_{seed_}.png")
                    )
                    res_image_annotated = annotate(res_image, bbox_for_annotation, labels_for_annotation)
                    res_image_annotated.save(
                        os.path.join(sample_save_path, f"{file_name_prefix}_{i:02d}_{sample_id}_{seed_}_annotated.png")
                    )

        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'sample_dataset' and 'fixed'.")
