## Lion Training Configuration

Command used:

```bash
python train.py \
  --fraction 1.0 \
  --epochs 30 \
  --batch-size 16 \
  --workers 4 \
  --imgsz 640 \
  --optimizer lion \
  --patience 0 \
  --weight-decay 0.02 \
  --lr0 0.004 \
  --warmup-epochs 2
```

### Parameter Justification

- `--fraction 1.0`: train on the full WIDER FACE dataset to keep the comparison fair with the baseline SGD run (no subsampling noise).
- `--epochs 30`: match the original experiment length so any performance delta comes from the optimizer, not training duration.
- `--batch-size 16`: halves the previous batch (32) to stay within GPU memory headroom when Lion + large lr are used.
- `--workers 4`: keeps data loading parallelism modest, which is sufficient for 16-image batches and avoids oversubscribing CPU cores.
- `--imgsz 640`: standard YOLOv8 input resolution; changing it would alter both compute and accuracy, so we hold it constant.
- `--optimizer lion`: the experimental optimizer under test (vs. baseline SGD/AdamW) — all other changes are made to make this comparison meaningful.
- `--patience 0`: disables early stopping so Lion always runs the full 30 epochs; otherwise it might stop early due to short-term plateaus and skew the comparison.
- `--weight-decay 0.02`: aligns with the Lion paper’s recommended regularization level for vision models, improving stability with sign-based updates.
- `--lr0 0.004`: keeps the nominal learning rate close to the baseline SGD value (0.01) but slightly lower to avoid divergence with Lion’s sign steps.
- `--warmup-epochs 2`: reintroduces a short warmup so the optimizer ramps from a tiny LR to 0.005 over two epochs, preventing the large initial step that previously produced NaNs.

## SAM Training Configuration

```bash
python train.py \
  --fraction 1.0 \
  --epochs 30 \
  --batch-size 16 \
  --workers 4 \
  --imgsz 640 \
  --optimizer sam \
  --sam-rho 0.03 \
  --patience 0 \
  --lr0 0.001 \
  --weight-decay 0.0005 \
  --warmup-epochs 2
```

### Parameter Justification

- `--batch-size 16`: matches the Lion run so comparisons isolate optimizer behavior now that SAM memory usage is under control.
- `--optimizer sam` / `--sam-rho 0.03`: enables SAM with a slightly smaller perturbation radius to keep gradients finite on early epochs.
- `--lr0 0.001` & `--weight-decay 0.0005`: conservative AdamW defaults that play nicely with the SAM perturb step.
- `--patience 0`: keep the full 30-epoch schedule for fair comparison.
- `--warmup-epochs 2`: warmup smooths the transition into the two-step SAM updates without wasting too many iterations.

### Stability Changes Applied

- Disabled AMP whenever `optimizer=sam`, so both the base step and the adversarial step run in FP32 (`train.py`).
- Added gradient clipping/logging inside `CustomDetectionTrainer._sam_optimizer_step` to cap the norm before/after each SAM step and skip updates when the norm becomes non-finite.
- Logged the clipped gradient norms every N SAM steps to watch for divergence without digging into TensorBoard.
