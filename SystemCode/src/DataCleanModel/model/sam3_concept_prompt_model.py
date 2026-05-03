# SAM3 conecept prompt inference using the exemplers from different images
import argparse
import hashlib
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from gradio_image_prompter import ImagePrompter  # type: ignore
from PIL import Image, ImageDraw, ImageOps, ImageChops

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import (
    Sam3CrossImageProcessor,
    Sam3Processor,
)
import cv2
from shapely.geometry import Point, Polygon


# TODO 1: manage ref_state in processor here to avoid recreate the ref_state
# TODO 2: support ref_state from different reference images

class Sam3ConceptPromptInference:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
    ):
        self.device = device
        self.model_name = model_name
        self.target_size = 1008

        # Build model
        self.model = build_sam3_image_model(
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            device=self.device,
            cross_image=True,
            checkpoint_path=self.model_name,
            enable_inst_interactivity=True)
        self.model.to(device=self.device)
        self.model.eval()

        # Build processor
        self.processor = Sam3CrossImageProcessor(
            self.model,
            device=self.device,
            confidence_threshold=0.5,
        )

    def _bbox_from_corners(self,
    x0: float, y0: float, x1: float, y1: float
    ) -> Dict[str, float]:
        return {
            "x": float(min(x0, x1)),
            "y": float(min(y0, y1)),
            "width": max(abs(float(x1) - float(x0)), 1.0),
            "height": max(abs(float(y1) - float(y0)), 1.0),
        }

    def scale_bbox_to_target(self,
    bbox: Dict[str, float],
    source_size: Tuple[int, int],
    target_size: int,
    ) -> Dict[str, float]:
        src_w, src_h = source_size
        scale_x = target_size / float(max(src_w, 1))
        scale_y = target_size / float(max(src_h, 1))
        return {
            "x": bbox["x"] * scale_x,
            "y": bbox["y"] * scale_y,
            "width": max(bbox["width"] * scale_x, 1.0),
            "height": max(bbox["height"] * scale_y, 1.0),
        }

    def scale_point_to_target(self,
        point: Dict[str, float],
        source_size: Tuple[int, int],
        target_size: int,
    ) -> Dict[str, float]:
        src_w, src_h = source_size
        scale_x = target_size / float(max(src_w, 1))
        scale_y = target_size / float(max(src_h, 1))
        return {
            "x": point["x"] * scale_x,
            "y": point["y"] * scale_y,
            "label": point["label"],
        }
    
    def bbox_to_normalized_cxcywh(self,
    bbox: Dict[str, float], base_w: float, base_h: float
    ) -> list[float]:
        cx = bbox["x"] + bbox["width"] / 2.0
        cy = bbox["y"] + bbox["height"] / 2.0
        return [
            cx / base_w,
            cy / base_h,
            bbox["width"] / base_w,
            bbox["height"] / base_h,
        ]
    
    def resize_for_model(self,
        image: np.ndarray,
        target_size: int,
    ) -> np.ndarray:
        return cv2.resize(
            image,
            (target_size, target_size),
            interpolation=cv2.INTER_LINEAR,
        )
    
    def annotate_output(self, 
                        image: Image.Image, 
                        result: Dict, 
                        mask_thresh: float):
        if not result or "masks_logits" not in result:
            return (image, [])

        prob_masks = result["masks_logits"]
        if prob_masks.ndim == 4:
            prob_masks = prob_masks.squeeze(1)
        masks = prob_masks > float(mask_thresh)

        annotations = []
        for idx, (mask, score) in enumerate(zip(masks, result.get("scores", []))):
            mask_np = mask.detach().cpu().numpy().astype(np.float32)
            annotations.append((mask_np, f"inst {idx + 1} ({float(score):.2f})", float(score)))

        return image, annotations

    @torch.inference_mode()
    def prepare_reference_state(self,
        reference_image: np.ndarray,
        boxes: list[Dict[str, float]],
        box_labels: list[bool],
        text_prompt: Optional[str],
        encode_prompt:bool
    ):
        ref_resized = self.resize_for_model(reference_image, self.target_size)
        ref_resized_pil = Image.fromarray(ref_resized.astype(np.uint8)).convert("RGB")
        scaled_boxes = [
            self.scale_bbox_to_target(box, (reference_image.shape[1], reference_image.shape[0]), self.target_size) for box in boxes
        ]
        norm_boxes = [
            self.bbox_to_normalized_cxcywh(box, float(self.target_size), float(self.target_size))
            for box in scaled_boxes
        ]
        ref_state = self.processor.prepare_reference_state(
            image=ref_resized_pil,
            boxes=norm_boxes,
            box_labels=box_labels,
            text_prompt=text_prompt,
            state=None,
            encode_prompt=encode_prompt
        )
        return ref_state
    
    @torch.inference_mode()
    def run_cross_inference(self,
        target_image: np.ndarray,
        ref_state:Dict,
        det_thresh: float,
        mask_thresh: float,
    ):
        tgt_resized = self.resize_for_model(target_image, self.target_size)

        self.processor.set_confidence_threshold(float(det_thresh))

        try:
            # convert ref_sized to PIL image
            tgt_resized_pil = Image.fromarray(tgt_resized.astype(np.uint8)).convert("RGB")
            reference_results, target_results = self.processor.run_cross_image(
                reference_image=None,
                target_image=tgt_resized_pil,    
                boxes=None,
                box_labels=None,
                text_prompt=None,
                use_reference_as_target=False,
                ref_state=ref_state,
            )
        except ValueError as exc:
            raise RuntimeError("Inference failed. Please check the input prompts.") from exc

        annotated_target = self.annotate_output(tgt_resized_pil, target_results, mask_thresh)

        tgt_count = len(annotated_target[1])
        status = (
            f"Target: {tgt_count} instance(s) | "
            f"det ≥ {float(det_thresh):.2f}, mask ≥ {float(mask_thresh):.2f}"
        )
        return annotated_target, status
    
    @torch.inference_mode()
    def get_image_embedding(self, image: np.ndarray):
        resized_image = self.resize_for_model(image, self.target_size)
        resized_image_pil = Image.fromarray(resized_image.astype(np.uint8)).convert("RGB")
        state = self.processor.set_image(resized_image_pil)
        vision_features = state['backbone_out']['vision_features']
        # average pool the vision features to get a single vector representation for the image
        state['image_embedding'] = torch.mean(vision_features, dim=[2, 3])
        return state['image_embedding']






