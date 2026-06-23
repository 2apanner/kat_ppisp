"""
Nerfstudio splatfacto method with PPISP photometric correction.

Registers `splatfacto-ppisp` via pyproject entry point for `ns-train`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

import torch
from torch import Tensor

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from ppisp import PPISP, PPISPConfig


def _resolve_frame_idx(camera: Cameras, *, training: bool) -> int:
    if not training:
        return -1
    meta = camera.metadata or {}
    if "image_idx" in meta:
        return int(meta["image_idx"])
    if "cam_idx" in meta:
        return int(meta["cam_idx"])
    return 0


def _resolve_camera_idx(camera: Cameras) -> int:
    meta = camera.metadata or {}
    return int(meta.get("cam_idx", 0))


class SplatfactoPPISPModel(SplatfactoModel):
    """Splatfacto with physically-plausible ISP post-processing on rendered RGB."""

    config: SplatfactoPPISPModelConfig

    def populate_modules(self) -> None:
        super().populate_modules()
        ppisp_config = PPISPConfig(
            controller_activation_ratio=self.config.ppisp_controller_activation_ratio,
            controller_distillation=self.config.ppisp_controller_distillation,
        )
        self.ppisp = PPISP(
            num_cameras=self.config.num_cameras,
            num_frames=max(1, int(self.num_train_data)),
            config=ppisp_config,
        )
        self._ppisp_optimizers: list[torch.optim.Optimizer] | None = None
        self._ppisp_schedulers: list[torch.optim.lr_scheduler.LRScheduler] | None = None
        self._scene_frozen = False
        max_iters = max(1, int(self.config.ppisp_max_optimization_iters))
        self._ppisp_optimizers = self.ppisp.create_optimizers()
        self._ppisp_schedulers = self.ppisp.create_schedulers(self._ppisp_optimizers, max_iters)
        self._ppisp_activation_step = int(
            ppisp_config.controller_activation_ratio * max_iters
        )

    def _maybe_freeze_scene(self, step: int) -> None:
        if self._scene_frozen:
            return
        if step < self._ppisp_activation_step:
            return
        if not self.ppisp.config.controller_distillation:
            return
        for name, param in self.named_parameters():
            if name.startswith("ppisp."):
                continue
            param.requires_grad = False
        self._scene_frozen = True

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[Tensor, List]]:
        outputs = super().get_outputs(camera)
        if not isinstance(camera, Cameras) or "rgb" not in outputs:
            return outputs

        rgb = outputs["rgb"]
        if rgb.dim() != 3:
            return outputs

        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        camera_idx = _resolve_camera_idx(camera)
        frame_idx = _resolve_frame_idx(camera, training=self.training)

        outputs["rgb_raw"] = rgb
        outputs["rgb"] = self.ppisp(
            rgb,
            resolution=(width, height),
            camera_idx=camera_idx,
            frame_idx=frame_idx,
        )
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        loss_dict["ppisp_reg"] = self.ppisp.get_regularization_loss()
        return loss_dict

    def get_training_callbacks(
        self,
        training_callback_attributes: TrainingCallbackAttributes,
    ) -> List[TrainingCallback]:
        callbacks = list(super().get_training_callbacks(training_callback_attributes))

        def after_train(step: int) -> None:
            if self._ppisp_optimizers is None:
                return
            self._maybe_freeze_scene(step)
            for optimizer in self._ppisp_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if self._ppisp_schedulers:
                for scheduler in self._ppisp_schedulers:
                    scheduler.step()

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=after_train,
            )
        )
        return callbacks


@dataclass
class SplatfactoPPISPModelConfig(SplatfactoModelConfig):
    """Splatfacto config with PPISP photometric correction."""

    _target: Type = field(default_factory=lambda: SplatfactoPPISPModel)
    num_cameras: int = 1
    """Number of physical cameras (1 for single-drone captures)."""
    ppisp_controller_activation_ratio: float = 0.8
    """Train PPISP controller after this fraction of total iterations."""
    ppisp_controller_distillation: bool = True
    """Freeze scene + PPISP params when training the controller."""
    ppisp_max_optimization_iters: int = 7500
    """Must match ns-train --max-num-iterations (set by Colab runner)."""


splatfacto_ppisp_method = MethodSpecification(
    config=TrainerConfig(
        method_name="splatfacto-ppisp",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=SplatfactoPPISPModelConfig(
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=30000),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=30000),
            },
        },
        viewer=ViewerConfig(quit_on_train_completion=True),
    ),
    description="Gaussian Splatting with PPISP photometric correction (Katada pilot)",
)
