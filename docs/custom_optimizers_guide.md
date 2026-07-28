## Custom Optimizers for YOLOv8

This workspace now includes first-class SAM and Lion optimizer support for the WIDER FACE YOLOv8 training script.

### Overview

- `yolo/optimizers/optimizers.py` defines portable implementations of SAM and Lion. SAM exposes `first_step`/`second_step` and a `base` optimizer handle so the trainer can drive the two-phase update under AMP.
- `yolo/optimizers/custom_trainer.py` subclasses Ultralytics' `DetectionTrainer`, wiring in the new optimizers. Lion slots directly into the stock training loop; SAM overrides `_do_train` and issues the extra adversarial step per batch while remaining compatible with GradScaler, EMA, and callbacks.
- `yolo/train.py` adds CLI flags (`--optimizer`, `--sam-rho`, `--lion-beta1`, `--lion-beta2`) and conditionally swaps in the custom trainer when you request `sam` or `lion`. SAM automatically forces `nbs=batch` so accumulation stays at one step per update.
- Training run directories inherit the optimizer tag (`train_sam_...`, `train_lion_...`) when you pass a non-auto optimizer, making it easy to compare experiments.
- SAM-specific loop tweaks: we override `_do_train` only when SAM is selected, disable gradient accumulation (`nbs=batch`), and call a helper that performs the three SAM phases (perturb weights, recompute loss at `w+ε`, restore weights + `base.step()`) every batch, so the math lines up with the paper while still using Ultralytics’ logging/EMA/scheduler plumbing.

### Usage

1. Lion (single forward/backward per batch):
   ```bash
   source venv/bin/activate
   python yolo/train.py --optimizer lion --lion-beta1 0.95 --lion-beta2 0.99 --epochs 1 --fraction 0.001
   ```
2. SAM (double forward/backward per batch):
   ```bash
   python yolo/train.py --optimizer sam --sam-rho 0.05 --epochs 1 --fraction 0.001
   ```
   Expect ~2× step time and higher VRAM requirements versus standard optimizers.

Hyperparameters you do _not_ set fall back to the defaults baked into `train.py`. All other Ultralytics arguments remain available; the trainer quietly strips the custom options before passing overrides into the base class, so there are no parser conflicts.

### Under the Hood: Custom Trainer

The lion path is simple: `CustomDetectionTrainer` inherits the stock `_do_train`, so once `build_optimizer` returns the Lion instance everything else follows Ultralytics’ normal flow (GradScaler → `optimizer.step()` → EMA).

SAM needs a bespoke loop, so `_do_train` swaps in three key behaviors:

1. **Disable accumulation** – we force `nbs == batch == accumulate == 1` to guarantee one optimizer step per dataloader batch. Accumulated gradients would break the SAM perturbation math.
2. **Two-pass batches** – every iteration does the standard forward/backward, then immediately calls `_sam_optimizer_step` instead of waiting for Ultralytics’ `optimizer_step`.
3. **Scheduler/EMA consistency** – outside the modified sections we keep the base trainer’s learning-rate warmup, EMA updates, validation cadence, callbacks, and logging.

`_sam_optimizer_step` itself:

1. Unscale & clip gradients, run `SAM.first_step(zero_grad=True)` to add the perturbation vector `ρ g/||g||` to each parameter, and clear grads.
2. Re-run the model (with AMP/compile support) at the perturbed weights to obtain `loss(w + ε)` and its gradients, storing the original loss items for logging.
3. Unscale & clip again, call `SAM.second_step()` to subtract the stored perturbation, execute the base optimizer step (`AdamW`), update the GradScaler, zero grads, and tick EMA.

The net effect: SAM follows the paper’s min–max approximation, but we still get Ultralytics’ diagnostics and checkpoint handling for free.

### Notes

- SAM uses `optim.AdamW` as its base optimizer. If you need an alternate base (e.g., SGD), adjust `CustomDetectionTrainer.build_optimizer`.
- Because SAM now delegates the actual `.step()` to the training loop, do not reuse `SAM` directly elsewhere without reproducing the two-step call pattern (`first_step` → loss recompute → `second_step` + `base.step()`).
- SAM’s helper (`_sam_optimizer_step`) works like this:
  1. Unscale/clip gradients and call `first_step` to push parameters along the current gradient direction.
  2. Forward+backward again at the perturbed weights to get `∇L(w+ε)`.
  3. Unscale/clip, call `second_step` to subtract the perturbation, run the base optimizer step, update GradScaler/EMA, and zero grads.

Feel free to extend `optimizers/custom_trainer.py` with additional experimental optimizers following the same pattern.
