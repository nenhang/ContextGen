import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25


class GroundedSAM2:
    def __init__(self, sam2_ckpt, groundingdino_ckpt, device="cuda"):
        self.device = device
        self.sam2_model = Sam2Model.from_pretrained(sam2_ckpt).to(device)
        self.sam2_processor = Sam2Processor.from_pretrained(sam2_ckpt)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(groundingdino_ckpt).to(device)
        self.grounding_processor = AutoProcessor.from_pretrained(groundingdino_ckpt)
        self.device = device

    def predict(self, img_path, text_prompt):
        # image_source, image = load_image(img_path)
        image_pil = Image.open(img_path).convert("RGB")
        grounding_inputs = self.grounding_processor(
            images=image_pil, text=text_prompt.lower() + ".", return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            grounding_outputs = self.grounding_model(**grounding_inputs)

        grounding_results = self.grounding_processor.post_process_grounded_object_detection(
            grounding_outputs,
            grounding_inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image_pil.size[::-1]],
        )
        boxes = grounding_results[0]["boxes"].cpu().numpy()
        if boxes is None or len(boxes) == 0:
            return None

        sam2_inputs = self.sam2_processor(images=image_pil, input_boxes=boxes[None, :], return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            sam2_outputs = self.sam2_model(**sam2_inputs, multimask_output=False)

        masks = self.sam2_processor.post_process_masks(sam2_outputs.pred_masks.cpu(), sam2_inputs["original_sizes"])[0]

        results = [
            {
                "bbox": box,
                "segmentation": mask,
            }
            for box, mask in zip(boxes, masks.squeeze(1).cpu().numpy())
        ]

        return results[0]
