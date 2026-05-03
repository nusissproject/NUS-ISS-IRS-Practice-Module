import logging
import os
import re
import warnings
from argparse import ArgumentParser
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, List, Optional

import sam3.model_builder as mb
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model

from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.sam3_image import Sam3Image
from sam3.train.utils.checkpoint_utils import unix_pattern_to_parameter_names
from sam3.train.utils.train_utils import register_omegaconf_resolvers

logger = logging.getLogger(__name__)


class Sam3ImageFinetune(Sam3Image):
    """
    Sam3Image with optional LoRA layers and trainable parameter overrides.
    """

    def __init__(
        self,
        *args,
        lora_config: Optional[DictConfig | dict] = None,
        mode: str = "train",
        trainable_param_patterns: Optional[List[str]] = None,
        **kwargs,
    ):
        self.mode = mode
        self.orig_name_prefix: Optional[str] = None
        self.ignore_missing_keys: Optional[List[str]] = None
        self.lora_model_ptr = None
        self.trainable_param_patterns = trainable_param_patterns

        super().__init__(*args, **kwargs)

        checkpoint_path = mb.download_ckpt_from_hf()
        mb._load_checkpoint(self, checkpoint_path)

        if (
            isinstance(lora_config, (dict, DictConfig))
            and lora_config.get("target_names") is not None
            and len(lora_config["target_names"]) > 0
        ):
            self.setup_lora(lora_config)

    def _apply_trainable_patterns(
        self, patterns: Optional[List[str]]
    ) -> List[str]:
        """
        Apply trainable parameter patterns to unfreeze specific parameters.
        """
        if patterns is None or len(patterns) == 0:
            return []

        all_param_names = [n for n, _ in self.named_parameters()]
        try:
            trainable_names = unix_pattern_to_parameter_names(
                patterns, all_param_names
            )
        except AssertionError as exc:
            warnings.warn(f"Pattern matching failed: {exc}", stacklevel=2)
            return []

        for name, param in self.named_parameters():
            if name in trainable_names:
                param.requires_grad = True

        return list(trainable_names)

    def _log_trainable_params(self) -> None:
        """
        Log a summary of trainable vs frozen parameters grouped by component.
        """
        components = {
            "vision_backbone": [],
            "text_backbone": [],
            "transformer": [],
            "segmentation_head": [],
            "geometry_encoder": [],
            "dot_prod_scoring": [],
            "lora": [],
            "other": [],
        }

        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params

            if not param.requires_grad:
                continue

            trainable_params += num_params
            if "lora" in name.lower():
                bucket = "lora"
            elif name.startswith("backbone.visual"):
                bucket = "vision_backbone"
            elif name.startswith("backbone.text"):
                bucket = "text_backbone"
            elif name.startswith("transformer"):
                bucket = "transformer"
            elif name.startswith("segmentation_head"):
                bucket = "segmentation_head"
            elif name.startswith("geometry_encoder"):
                bucket = "geometry_encoder"
            elif "dot_prod_scoring" in name:
                bucket = "dot_prod_scoring"
            else:
                bucket = "other"
            components[bucket].append((name, num_params))

        logger.info("=" * 80)
        logger.info("TRAINABLE PARAMETERS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(
            "Trainable parameters: %s (%.2f%%)",
            f"{trainable_params:,}",
            100 * trainable_params / max(total_params, 1),
        )
        logger.info(
            "Frozen parameters: %s (%.2f%%)",
            f"{total_params - trainable_params:,}",
            100 * (total_params - trainable_params) / max(total_params, 1),
        )
        logger.info("-" * 80)

        for component_name, params in components.items():
            if len(params) == 0:
                continue
            component_total = sum(param_count for _, param_count in params)
            logger.info(
                "%s: %d trainable params, %s elements",
                component_name.upper(),
                len(params),
                f"{component_total:,}",
            )
            for param_name, param_count in params:
                logger.info("  - %s: %s", param_name, f"{param_count:,}")

        logger.info("=" * 80)

    def setup_lora(self, lora_config: DictConfig | dict) -> None:
        config_dict = (
            OmegaConf.to_container(lora_config, resolve=True)
            if isinstance(lora_config, DictConfig)
            else deepcopy(lora_config)
        )
        target_names = config_dict.pop("target_names", None)
        if target_names is None or len(target_names) == 0:
            warnings.warn(
                "LoRA config provided without target_names; skipping LoRA setup.",
                stacklevel=2,
            )
            return

        pattern = re.compile("|".join(target_names))
        target_modules = [
            name for name, _ in self.named_modules() if pattern.search(name)
        ]
        if len(target_modules) == 0:
            warnings.warn(
                f"No modules matched LoRA target_names pattern {target_names}",
                stacklevel=2,
            )
            return

        config_dict["target_modules"] = target_modules
        config = LoraConfig(**config_dict)
        lora_model = get_peft_model(self, config)

        trainable_names = self._apply_trainable_patterns(
            self.trainable_param_patterns
        )
        if len(trainable_names) > 0:
            logger.info(
                "Unfroze %d parameters matching trainable_param_patterns",
                len(trainable_names),
            )

        self._log_trainable_params()

        if self.mode == "train":
            self.ignore_missing_keys = ["*lora*"]
            self.orig_name_prefix = "base_layer."
        else:
            self.lora_model_ptr = lora_model

    def _remove_lora_prefix(self, state_dict: Mapping[str, Any]):
        missing_keys = []
        for name, _ in self.named_parameters():
            if self.orig_name_prefix is None or self.orig_name_prefix not in name:
                continue

            orig_name = name.replace(self.orig_name_prefix, "")
            if orig_name not in state_dict:
                missing_keys.append(orig_name)
                continue
            state_dict[name] = state_dict[orig_name]
            state_dict.pop(orig_name)

        if missing_keys:
            warnings.warn(
                "Possibly resuming training, cannot find the following "
                f"keys in pretrained weights: \n\n{missing_keys}\n\n",
                stacklevel=2,
            )

        return state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        if self.orig_name_prefix is not None:
            state_dict = self._remove_lora_prefix(state_dict)

        temp = self.lora_model_ptr
        self.lora_model_ptr = None
        try:
            missing_keys, unexpected_keys = super().load_state_dict(
                state_dict, strict, assign
            )
        finally:
            self.lora_model_ptr = temp

        if self.ignore_missing_keys and missing_keys:
            try:
                ignored_keys = unix_pattern_to_parameter_names(
                    self.ignore_missing_keys, missing_keys
                )
                missing_keys = [
                    key for key in missing_keys if key not in ignored_keys
                ]
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Possibly resuming training, {exc}", stacklevel=2)

        return missing_keys, unexpected_keys

    def deploy(self, ckpt_path: str) -> None:
        from collections import OrderedDict

        state = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(state["model"], strict=True)

        if self.lora_model_ptr is not None:
            self.lora_model_ptr.merge_and_unload(progressbar=True)
            self.lora_model_ptr = None

        state["model"] = OrderedDict(
            ("detector." + k, v) for k, v in self.state_dict().items()
        )
        deploy_path = os.path.join(
            *ckpt_path.split(os.path.sep)[:-1], "final.pt"
        )
        torch.save(state, deploy_path)


def build_sam3_image_finetune_model(
    bpe_path: Optional[str] = None,
    eval_mode: bool = True,
    enable_segmentation: bool = True,
    enable_inst_interactivity: bool = False,
    compile: bool = False,
    lora_config: Optional[DictConfig | dict] = None,
    trainable_param_patterns: Optional[List[str]] = None,
    mode: str = "train",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Build a Sam3ImageFinetune model with optional LoRA layers.
    """
    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )

    compile_mode = "default" if compile else None
    vision_encoder = mb._create_vision_backbone(
        compile_mode=compile_mode, enable_inst_interactivity=enable_inst_interactivity
    )
    text_encoder = mb._create_text_encoder(bpe_path)
    backbone = mb._create_vl_backbone(vision_encoder, text_encoder)
    transformer = mb._create_sam3_transformer()
    dot_prod_scoring = mb._create_dot_product_scoring()

    segmentation_head = (
        mb._create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )
    input_geometry_encoder = mb._create_geometry_encoder()
    if enable_inst_interactivity:
        sam3_pvs_base = mb.build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None

    matcher = None
    if not eval_mode:
        from sam3.train.matcher import BinaryHungarianMatcherV2

        matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )

    model = Sam3ImageFinetune(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head,
        dot_prod_scoring=dot_prod_scoring,
        num_feature_levels=1,
        o2m_mask_predict=True,
        use_instance_query=False,
        multimask_output=True,
        inst_interactive_predictor=inst_predictor,
        matcher=matcher,
        lora_config=lora_config,
        mode=mode,
        trainable_param_patterns=trainable_param_patterns,
    )

    model = mb._setup_device_and_mode(model, device, eval_mode)

    return model


def _load_cfg(cfg_path: str):
    register_omegaconf_resolvers()
    with initialize_config_module(
        version_base=None, config_module="sam3.train.configs"
    ):
        cfg = compose(config_name=cfg_path)
    return cfg


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help=(
            "path to config file (e.g. "
            "train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml)"
        ),
    )
    parser.add_argument(
        "-p",
        "--ckpt",
        required=True,
        type=str,
        help="path to checkpoint (.pt) file",
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    OmegaConf.set_struct(cfg.trainer.model, False)
    cfg.trainer.model.setdefault("mode", "eval")
    cfg.trainer.model["_target_"] = (
        "sam3.model.sam3_finetune.build_sam3_image_finetune_model"
    )

    sam3: Sam3ImageFinetune = instantiate(cfg.trainer.model, _recursive_=True)
    sam3.deploy(args.ckpt)


if __name__ == "__main__":
    main()
