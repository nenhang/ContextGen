import base64
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class VQAScorer:
    """Loads Qwen3-VL and scores yes/no answers (1.0 / 0.0)."""

    def __init__(
        self,
        vqa_model_path: str,
        *,
        dtype: str | torch.dtype = "auto",
        device: str | None = None,
        device_map: str | dict | None = None,
        attn_implementation: str | None = None,
        max_new_tokens: int = 128,
    ):
        if device is not None:
            dm: str | dict = {"": device}
        elif device_map is not None:
            dm = device_map
        else:
            dm = "auto"

        load_kw: dict = {
            "dtype": dtype,
            "device_map": dm,
            "trust_remote_code": True,
        }
        if attn_implementation is not None:
            load_kw["attn_implementation"] = attn_implementation

        self._model = AutoModelForImageTextToText.from_pretrained(vqa_model_path, **load_kw)
        self._processor = AutoProcessor.from_pretrained(vqa_model_path, trust_remote_code=True)
        self._model.eval()
        self._max_new_tokens = max_new_tokens

    @staticmethod
    def _image_file_to_base64_str(image_path: str) -> str:
        path = Path(image_path).expanduser().resolve()
        return base64.standard_b64encode(path.read_bytes()).decode("ascii")

    def score(self, image_path: str, question: str) -> float:
        image_b64 = self._image_file_to_base64_str(image_path)
        prompt = f"Based on the image, answer my question with only one word 'yes' or 'no'. Question: {question}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_b64},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)

        infer_device = next(self._model.parameters()).device
        inputs = inputs.to(infer_device)

        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answer = output_text[0].strip().lower()
        if answer.startswith("yes"):
            return 1.0
        if answer.startswith("no"):
            return 0.0
        print(f"Uncertain answer: {answer!r}. Assigning score of 0.0")
        return 0.0
