import re
from itertools import chain
from pathlib import Path

import yaml


def get_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def check_param_update(module_wrapper, sub_module):
    if not hasattr(module_wrapper, "prev_params"):
        module_wrapper.prev_params = None
    if module_wrapper.prev_params is None:
        module_wrapper.prev_params = {
            k: (param, param.detach().clone())
            for k, param in chain(
                sub_module.named_parameters(),
            )
            if param.requires_grad
        }
    else:
        post_step_params = {
            k: (param, param.detach().clone())
            for k, param in chain(
                sub_module.named_parameters(),
            )
            if param.requires_grad
        }
        num_optimize_params = len(post_step_params)
        num_unupdated_params = 0
        for k, (param, current_val) in post_step_params.items():
            old_param, old_val = module_wrapper.prev_params[k]
            assert param is old_param, "Parameter object changed unexpectedly!"

            delta = current_val - old_val

            if delta.norm(2).item() == 0:
                if param.grad is None:
                    print(f"❌ No Gradient: {k}")

                elif param.grad.norm(2).item() == 0:
                    print(f"❌ Zero Gradient: {k}")

                else:
                    grad_norm = param.grad.norm(2).item()
                    weight_norm = old_val.norm(2).item()
                    ratio = grad_norm / (weight_norm + 1e-8)
                    if ratio < 1e-6:
                        print(f"⚠️ Tiny Grad: {k} grad/w={ratio:.1e} (g={grad_norm:.1e}, w={weight_norm:.1e})")
                    else:
                        print(f"❓ Unupdated: {k} grad/w={ratio:.1e} (g={grad_norm:.1e}, w={weight_norm:.1e})")

                num_unupdated_params += 1

            module_wrapper.prev_params[k] = (param, current_val)

        print(f"⚠️ {num_unupdated_params}/{num_optimize_params} params not updated")


def find_latest_checkpoint(ckpt_dir: str, prefix: str = "", postfix: str = ".safetensors") -> tuple[str, int] | None:
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists() or not ckpt_path.is_dir():
        return None

    # YYYYMMDD-HHMMSS
    run_dir_pattern = re.compile(r".*\d{8}-\d{6}$")

    # 获取所有符合条件的run目录，并按名称逆序（时间从新到旧）
    run_dirs = sorted(
        [d for d in ckpt_path.iterdir() if d.is_dir() and run_dir_pattern.match(d.name)],
        key=lambda x: x.name,
        reverse=True,
    )

    for run_dir in run_dirs:
        ckpt_subdir = run_dir / "ckpt"

        if not ckpt_subdir.exists() or not ckpt_subdir.is_dir():
            continue

        max_step = -1
        latest_checkpoint = None

        for pth_file in ckpt_subdir.glob(f"{prefix}*{postfix}"):
            try:
                step_str = pth_file.stem[len(prefix) :]
                step = int(step_str) if step_str.isdigit() else -1  # 如果不是数字则设为 -1
                if step > max_step:
                    max_step = step
                    latest_checkpoint = pth_file
            except ValueError:
                continue  # 跳过文件名不符合数字格式的文件

        if latest_checkpoint:
            return str(latest_checkpoint), max_step

    return None
