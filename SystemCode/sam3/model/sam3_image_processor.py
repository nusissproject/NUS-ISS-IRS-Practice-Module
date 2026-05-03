# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch

from sam3.model import box_ops
from sam3.model.sam3_image import Sam3CrossImage

from sam3.model.data_misc import FindStage, interpolate
from torchvision.transforms import v2


class Sam3Processor:
    """ """

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image = v2.functional.to_image(image).to(self.device)
        image = self.transform(image).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(
            images[0], PIL.Image.Image
        ), "Images must be a list of PIL images"

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        images = [
            self.transform(v2.functional.to_image(image).to(self.device))
            for image in images
        ]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        """Sets the text prompt and run the inference"""

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state


class Sam3CrossImageProcessor(Sam3Processor):
    """Processor that reuses prompts from a reference image on a target image."""

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        super().__init__(model, resolution, device, confidence_threshold)
        if not isinstance(model, Sam3CrossImage):
            raise ValueError(
                "Sam3CrossImageProcessor expects a Sam3CrossImage model instance."
            )

    def _ensure_text_features(self, text_prompt: str, state: Dict) -> Dict:
        text_prompt_to_use = text_prompt if text_prompt else "visual"
        text_outputs = self.model.backbone.forward_text(
            [text_prompt_to_use], device=self.device
        )
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        return state

    def _add_geometric_prompts(
        self, boxes: List[List[float]], box_labels: List[bool], state: Dict
    ) -> Dict:
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        geo_prompt = state["geometric_prompt"]
        if len(box_labels) < len(boxes):
            box_labels = [True for _ in boxes]

        for box, label in zip(boxes, box_labels):
            boxes_tensor = torch.tensor(
                box, device=self.device, dtype=torch.float32
            ).view(1, 1, 4)
            labels_tensor = torch.tensor(
                [label], device=self.device, dtype=torch.bool
            ).view(1, 1)
            geo_prompt.append_boxes(boxes_tensor, labels_tensor)
        return state

    def _encode_prompt_from_state(self, state: Dict) -> Dict:
        prompt, prompt_mask, backbone_out = self.model._encode_prompt(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
        )
        state["backbone_out"] = backbone_out
        state["prompt"] = prompt
        state["prompt_mask"] = prompt_mask
        return state

    @torch.inference_mode()
    def prepare_reference_state(
        self,
        image=None,
        boxes: Optional[List[List[float]]] = None,
        box_labels: Optional[List[bool]] = None,
        text_prompt: str = "",
        state: Optional[Dict] = None,
        encode_prompt: bool = True,
    ) -> Dict:
        """Prepare (or update) a reference state for cross-image prompting."""
        if state is None:
            state = {}

        if image is not None:
            state = self.set_image(image, state=state)
        if "backbone_out" not in state:
            raise ValueError("Reference state must contain backbone_out or provide image.")

        if "language_features" not in state["backbone_out"] or text_prompt:
            state = self._ensure_text_features(text_prompt, state)

        if boxes is not None:
            if box_labels is None:
                box_labels = [True for _ in boxes]
            state["geometric_prompt"] = self.model._get_dummy_prompt()
            if boxes:
                state = self._add_geometric_prompts(boxes, box_labels, state)
        elif "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        if encode_prompt:
            state = self._encode_prompt_from_state(state)
        return state

    @torch.inference_mode()
    def run_with_prompt_from_state(
        self, state: Dict, prompt: torch.Tensor, prompt_mask: torch.Tensor
    ) -> Dict:
        """Run cross-image forward on a prepared state with an external prompt."""
        if "backbone_out" not in state:
            raise ValueError("State must contain backbone_out. Call set_image first.")
        if "original_height" not in state or "original_width" not in state:
            raise ValueError(
                "State must contain original image size. Call set_image first."
            )

        outputs = self.model.forward_with_prompt(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        return self._postprocess_outputs(
            outputs,
            state["original_height"],
            state["original_width"],
        )

    def _postprocess_outputs(
        self, outputs: Dict, original_height: int, original_width: int
    ) -> Dict:
        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs.get("presence_logit_dec")
        if presence_score is not None:
            out_probs = out_probs * presence_score.sigmoid().unsqueeze(1)
        out_probs = out_probs.squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor(
            [original_width, original_height, original_width, original_height],
            device=self.device,
        )
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (original_height, original_width),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        return {
            "masks_logits": out_masks,
            "masks": out_masks > 0.5,
            "boxes": boxes,
            "scores": out_probs,
        }

    @torch.inference_mode()
    def run_cross_image(
        self,
        reference_image=None,
        target_image=None,
        boxes: Optional[List[List[float]]] = None,
        box_labels: Optional[List[bool]] = None,
        text_prompt: str = "",
        use_reference_as_target: bool = False,
        ref_state: Optional[Dict] = None,
        target_state: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        if ref_state is None:
            ref_state = {}
        needs_ref_refresh = (
            reference_image is not None
            or boxes is not None
            or bool(text_prompt)
            or "prompt" not in ref_state
            or "prompt_mask" not in ref_state
        )
        if needs_ref_refresh:
            ref_state = self.prepare_reference_state(
                image=reference_image,
                boxes=boxes,
                box_labels=box_labels,
                text_prompt=text_prompt,
                state=ref_state,
                encode_prompt=True,
            )

        reference_processed = self.run_with_prompt_from_state(
            ref_state,
            prompt=ref_state["prompt"],
            prompt_mask=ref_state["prompt_mask"],
        )

        if use_reference_as_target:
            target_processed = reference_processed
            return reference_processed, target_processed

        if target_state is None:
            target_state = {}
        if target_image is not None:
            target_state = self.set_image(target_image, state=target_state)
        if "backbone_out" not in target_state:
            raise ValueError(
                "Target image is required when target_state has no backbone_out."
            )

        target_processed = self.run_with_prompt_from_state(
            target_state,
            prompt=ref_state["prompt"],
            prompt_mask=ref_state["prompt_mask"],
        )
        return reference_processed, target_processed
