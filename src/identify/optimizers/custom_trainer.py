from __future__ import annotations

import math
import time
import warnings
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from ultralytics.engine.trainer import RANK
from ultralytics.models.yolo.detect import train as detect_train
from ultralytics.utils import DEFAULT_CFG, LOGGER, TQDM, colorstr
from ultralytics.utils.torch_utils import autocast, unwrap_model

from .optimizers import Lion, SAM


class CustomDetectionTrainer(detect_train.DetectionTrainer):
    """Detection trainer that plugs in custom Lion and SAM optimizers."""

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
    ):
        # Split the custom optimizer args before BaseTrainer validates overrides.
        overrides = dict(overrides or {})
        extra_args = {
            "sam_rho": overrides.pop("sam_rho", None),
            "lion_beta1": overrides.pop("lion_beta1", None),
            "lion_beta2": overrides.pop("lion_beta2", None),
        }
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        optimizer_name = str(getattr(self.args, "optimizer", "auto")).lower()
        self._optimizer_name = optimizer_name
        self.use_sam_optimizer = optimizer_name == "sam"
        self.use_lion_optimizer = optimizer_name == "lion"
        # Fall back to Ultralytics defaults when the user does not supply overrides.
        self.sam_rho = float(
            extra_args["sam_rho"]
            if extra_args["sam_rho"] is not None
            else getattr(self.args, "sam_rho", 0.05)
        )
        self.lion_beta1 = float(
            extra_args["lion_beta1"]
            if extra_args["lion_beta1"] is not None
            else getattr(self.args, "lion_beta1", 0.9)
        )
        self.lion_beta2 = float(
            extra_args["lion_beta2"]
            if extra_args["lion_beta2"] is not None
            else getattr(self.args, "lion_beta2", 0.99)
        )
        self.sam_grad_clip = float(getattr(self.args, "sam_grad_clip", 5.0))
        self.sam_grad_log_interval = max(1, int(getattr(self.args, "sam_grad_log_interval", 200)))

    # --------------------------------------------------------------------- #
    # Optimizer integration
    # --------------------------------------------------------------------- #
    def build_optimizer(
        self,
        model,
        name: str = "auto",
        lr: float = 0.001,
        momentum: float = 0.9,
        decay: float = 1e-5,
        iterations: float = 1e5,
    ):
        name = (name or "auto").lower()
        if name not in {"sam", "lion"}:
            return super().build_optimizer(model, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations)

        # Reuse Ultralytics' parameter bucketing so weight decay behavior matches upstream.
        g = [], [], []
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:
                    g[1].append(param)
                else:
                    g[0].append(param)

        param_groups = [
            {"params": g[2], "weight_decay": 0.0},
            {"params": g[0], "weight_decay": decay},
            {"params": g[1], "weight_decay": 0.0},
        ]

        if name == "lion":
            optimizer = Lion(
                param_groups,
                lr=lr,
                betas=(self.lion_beta1, self.lion_beta2),
                weight_decay=decay,
            )
            LOGGER.info(
                f"{colorstr('optimizer:')} Lion(lr={lr}, betas=({self.lion_beta1}, {self.lion_beta2}), weight_decay={decay})"
            )
            return optimizer

        optimizer = SAM(
            param_groups,
            base_optimizer=optim.AdamW,
            rho=self.sam_rho,
            lr=lr,
            betas=(momentum, 0.999),
        )
        LOGGER.info(
            f"{colorstr('optimizer:')} SAM(base=AdamW, lr={lr}, rho={self.sam_rho}, betas=({momentum}, 0.999), "
            f"weight_decay groups=[0.0,{decay},0.0])"
        )
        return optimizer

    # --------------------------------------------------------------------- #
    # Training loop overrides
    # --------------------------------------------------------------------- #
    def _do_train(self):
        if not self.use_sam_optimizer:
            return super()._do_train()

        # SAM is sensitive to gradient accumulation; force one optimizer step per batch.
        if self.args.nbs != self.args.batch:
            LOGGER.info(
                f"{colorstr('optimizer:')} SAM requires per-step updates; overriding nbs={self.args.nbs} -> batch={self.args.batch}"
            )
            self.args.nbs = self.args.batch

        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()
        self.accumulate = 1

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for "
            + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    if self.args.compile:
                        preds = self.model(batch["img"])
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        loss, self.loss_items = self.model(batch)
                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= self.world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                self.scaler.scale(self.loss).backward()

                if ni - last_opt_step >= 1:
                    # Complete the adversarial SAM step immediately instead of accumulating.
                    self._sam_optimizer_step(batch)
                    last_opt_step = ni
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(threshold=0.5)
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)

            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1

        if RANK in {-1, 0}:
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")

    def _sam_optimizer_step(self, batch):
        optimizer: SAM = self.optimizer
        base_opt = optimizer.base

        # Step 1: perturb weights in the gradient direction.
        self.scaler.unscale_(base_opt)
        step_idx = getattr(self, "_sam_step_index", 0) + 1
        self._sam_step_index = step_idx
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.sam_grad_clip)
        if not math.isfinite(grad_norm):
            LOGGER.warning(
                f"{colorstr('optimizer:')} SAM grad norm is non-finite ({grad_norm}); skipping step to avoid NaNs"
            )
            optimizer.zero_grad()
            return
        self._last_sam_grad_norm = float(grad_norm)
        if RANK in {-1, 0} and step_idx % self.sam_grad_log_interval == 0:
            LOGGER.info(
                f"{colorstr('optimizer:')} SAM grad norm (pre-perturb) {self._last_sam_grad_norm:.3f} "
                f"(clip={self.sam_grad_clip})"
            )
        optimizer.first_step(zero_grad=True)

        # Step 2: recompute loss at the perturbed point.
        loss_items_backup = self.loss_items
        with autocast(self.amp):
            if self.args.compile:
                preds = self.model(batch["img"])
                second_loss, _ = unwrap_model(self.model).loss(batch, preds)
            else:
                second_loss, _ = self.model(batch)
            second_loss = second_loss.sum()
        self.scaler.scale(second_loss).backward()
        self.loss_items = loss_items_backup

        # Step 3: restore weights, apply the base optimizer step, and sync EMA.
        self.scaler.unscale_(base_opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.sam_grad_clip)
        optimizer.second_step(zero_grad=False)
        self.scaler.step(base_opt)
        self.scaler.update()
        optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

