"""WIDER FACE validation: the official protocol, plus recall-first operating points (E1).

AP easy / medium / hard follows the official evaluation, ported from the
MIT-licensed Python version in `biubug6/Pytorch_Retinaface`
(`widerface_evaluate/evaluation.py`, itself from `wondervictor/WiderFace-
Evaluation`, MIT):
- scores min-max normalized over the whole split;
- IoU 0.5 with the `+1` pixel convention;
- ground truth outside a setting's keep list is ignored;
- 1000 thresholds;
- VOC AP.

The ground-truth `.mat` files come from the official `eval_tools`. If they
are missing they are downloaded from that repository's `ground_truth/`.

Added for this project's recall-first priority, on the `hard` set (every
annotated face):
- precision / recall / F1 at a fixed confidence (raw score);
- recall at a fixed number of false positives per image (FPPI);
- latency per image and peak memory, measured on a subset after a warm-up;
- the parameter count, read from the ONNX initializers when the model is an
  ONNX file.

Usage:
    python -m src.eval.widerface --model scrfd-10gf --out results/e1/scrfd-10gf \\
        [--conf-at 0.3] [--fppi 0.01 0.1] [--det-size 640] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONFIG  # noqa: E402

from ..pipeline.phase1_detect.detector import FaceDetector  # noqa: E402
from ..pipeline.phase1_detect.models import DEFAULT_WEIGHTS_FILENAME  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GT_URL = "https://github.com/biubug6/Pytorch_Retinaface/raw/master/widerface_evaluate/ground_truth/{}"
GT_FILES = ("wider_face_val.mat", "wider_easy_val.mat", "wider_medium_val.mat", "wider_hard_val.mat")
SETTINGS = ("easy", "medium", "hard")
THRESH_NUM = 1000


def ground_truth(gt_dir: Path):
    from scipy.io import loadmat

    gt_dir.mkdir(parents=True, exist_ok=True)
    for name in GT_FILES:
        if not (gt_dir / name).is_file():
            logger.info(f"downloading {name}")
            urllib.request.urlretrieve(GT_URL.format(name), gt_dir / name)
    full = loadmat(str(gt_dir / "wider_face_val.mat"))
    keep = {s: loadmat(str(gt_dir / f"wider_{s}_val.mat"))["gt_list"] for s in SETTINGS}
    return full["face_bbx_list"], full["event_list"], full["file_list"], keep


def overlaps(boxes: np.ndarray, query: np.ndarray) -> np.ndarray:
    """IoU, (N,4) x (K,4), x1y1x2y2, with the official +1 pixel convention."""
    area_b = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    area_q = (query[:, 2] - query[:, 0] + 1) * (query[:, 3] - query[:, 1] + 1)
    iw = np.minimum(boxes[:, None, 2], query[None, :, 2]) - np.maximum(boxes[:, None, 0], query[None, :, 0]) + 1
    ih = np.minimum(boxes[:, None, 3], query[None, :, 3]) - np.maximum(boxes[:, None, 1], query[None, :, 1]) + 1
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    return inter / (area_b[:, None] + area_q[None, :] - inter)


def image_eval(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray, iou: float):
    _pred, _gt = pred.copy(), gt.copy().astype(np.float64)
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])
    _pred[:, 2] += _pred[:, 0]
    _pred[:, 3] += _pred[:, 1]
    _gt[:, 2] += _gt[:, 0]
    _gt[:, 3] += _gt[:, 1]
    ov = overlaps(_pred[:, :4], _gt)
    for h in range(_pred.shape[0]):
        idx = int(ov[h].argmax())
        if ov[h, idx] >= iou:
            if ignore[idx] == 0:
                recall_list[idx] = -1
                proposal_list[h] = -1
            elif recall_list[idx] == 0:
                recall_list[idx] = 1
        pred_recall[h] = int((recall_list == 1).sum())
    return pred_recall, proposal_list


def img_pr_info(pred: np.ndarray, proposal_list: np.ndarray, pred_recall: np.ndarray) -> np.ndarray:
    pr = np.zeros((THRESH_NUM, 2))
    for t in range(THRESH_NUM):
        r = np.where(pred[:, 4] >= 1 - (t + 1) / THRESH_NUM)[0]
        if len(r):
            r = r[-1]
            pr[t, 0] = int((proposal_list[:r + 1] == 1).sum())
            pr[t, 1] = pred_recall[r]
    return pr


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def official_ap(preds: dict, gt, iou: float = 0.5) -> dict:
    """preds: event -> image -> (N, 5) [x, y, w, h, raw score], sorted by score desc."""
    boxes_all, events, files, keep = gt
    scores = [v[:, 4] for ev in preds.values() for v in ev.values() if len(v)]
    lo, hi = (float(np.min(np.concatenate(scores))), float(np.max(np.concatenate(scores)))) if scores else (0, 1)
    norm = {e: {k: np.hstack([v[:, :4], ((v[:, 4:] - lo) / max(hi - lo, 1e-12))]) if len(v) else v
                for k, v in ev.items()} for e, ev in preds.items()}
    out = {}
    for setting in SETTINGS:
        pr_curve, count_face = np.zeros((THRESH_NUM, 2)), 0
        for i in range(len(events)):
            event = str(events[i][0][0])
            for j in range(len(files[i][0])):
                name = str(files[i][0][j][0][0])
                gt_boxes = boxes_all[i][0][j][0].astype(np.float64)
                keep_idx = keep[setting][i][0][j][0]
                count_face += len(keep_idx)
                pred = norm.get(event, {}).get(name, np.zeros((0, 5)))
                if len(gt_boxes) == 0 or len(pred) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_idx):
                    ignore[keep_idx.ravel() - 1] = 1
                pred_recall, proposal_list = image_eval(pred, gt_boxes, ignore, iou)
                pr_curve += img_pr_info(pred, proposal_list, pred_recall)
        precision = pr_curve[:, 1] / np.maximum(pr_curve[:, 0], 1e-12)
        recall = pr_curve[:, 1] / max(count_face, 1)
        out[setting] = round(voc_ap(recall, precision), 4)
    return out


def operating_points(preds: dict, gt, conf_at: float, fppi: list[float], iou: float = 0.5) -> dict:
    """Hard set, raw scores: P/R/F1 at `conf_at`; recall at each FPPI target."""
    boxes_all, events, files, keep = gt
    matches = []   # (score, 1 if true positive, 0 if false positive); predictions on ignored faces dropped
    n_faces = n_images = 0
    for i in range(len(events)):
        event = str(events[i][0][0])
        for j in range(len(files[i][0])):
            n_images += 1
            gt_boxes = boxes_all[i][0][j][0].astype(np.float64)
            keep_idx = keep["hard"][i][0][j][0].ravel() - 1
            valid = np.zeros(len(gt_boxes), bool)
            valid[keep_idx] = True
            n_faces += int(valid.sum())
            pred = preds.get(event, {}).get(str(files[i][0][j][0][0]), np.zeros((0, 5)))
            if len(pred) == 0:
                continue
            g = gt_boxes.copy()
            g[:, 2:] += g[:, :2]
            p = pred[:, :4].copy()
            p[:, 2:] += p[:, :2]
            ov = overlaps(p, g) if len(g) else np.zeros((len(p), 0))
            used = np.zeros(len(g), bool)
            for h in range(len(p)):   # score order
                cand = np.where((ov[h] >= iou) & ~used)[0] if len(g) else []
                if len(cand):
                    k = cand[np.argmax(ov[h, cand])]
                    used[k] = True
                    if valid[k]:
                        matches.append((pred[h, 4], 1))
                    # a match to an ignored face is neither TP nor FP
                else:
                    matches.append((pred[h, 4], 0))
    m = np.array(sorted(matches, key=lambda x: -x[0])) if matches else np.zeros((0, 2))
    tp, fp = np.cumsum(m[:, 1]), np.cumsum(1 - m[:, 1])
    at = m[:, 0] >= conf_at
    k = int(at.sum())
    precision = float(tp[k - 1] / k) if k else 0.0
    recall = float(tp[k - 1] / n_faces) if k else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    recall_at = {}
    for target in fppi:
        ok = np.where(fp / n_images <= target)[0]
        recall_at[str(target)] = round(float(tp[ok[-1]] / n_faces), 4) if len(ok) else 0.0
    return {"conf_at": conf_at, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "recall_at_fppi": recall_at, "faces": n_faces, "images": n_images}


def gpu_memory_mib(pid: int) -> Optional[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    used = [int(m) for p, m in (line.split(",") for line in out.strip().splitlines() if "," in line) if int(p) == pid]
    return sum(used) if used else None


def onnx_params(model: str, weights: Optional[Path]) -> Optional[int]:
    path = weights
    if path is None and model == "scrfd-10gf":
        path = Path.home() / ".insightface" / "models" / "buffalo_l" / "det_10g.onnx"
    if path is None or path.suffix != ".onnx" or not path.is_file():
        return None
    import onnx

    return int(sum(np.prod(t.dims) for t in onnx.load(str(path)).graph.initializer))


def main() -> None:
    import os

    p = argparse.ArgumentParser(description="WIDER FACE val: official AP + recall-first operating points")
    p.add_argument("--model", required=True)
    p.add_argument("--weights", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--root", default=None, help="WIDER FACE root (default CONFIG.widerface_root)")
    p.add_argument("--gt-dir", default=None, help="ground-truth .mat dir (default <root>/eval_tools/ground_truth)")
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--score-floor", type=float, default=0.02, help="detector threshold for the AP sweep")
    p.add_argument("--conf-at", type=float, default=0.3)
    p.add_argument("--fppi", type=float, nargs="+", default=[0.01, 0.1])
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--latency-warmup", type=int, default=20)
    p.add_argument("--latency-images", type=int, default=500)
    p.add_argument("--ctx-id", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="first N images only (testing; AP not comparable)")
    args = p.parse_args()

    root = Path(args.root) if args.root else CONFIG.widerface_root
    images = root / "WIDER_val" / "images"
    if not images.is_dir():
        p.error(f"no WIDER FACE val images at {images}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = ground_truth(Path(args.gt_dir) if args.gt_dir else root / "eval_tools" / "ground_truth")
    _, events, files, _ = gt

    weights = Path(args.weights) if args.weights else (
        CONFIG.weights_dir / DEFAULT_WEIGHTS_FILENAME[args.model] if args.model in DEFAULT_WEIGHTS_FILENAME else None)
    detector = FaceDetector(model=args.model, weights=weights, conf_threshold=args.score_floor,
                            det_size=(args.det_size, args.det_size), ctx_id=args.ctx_id)

    preds: dict[str, dict[str, np.ndarray]] = {}
    times, n = [], 0
    for i in range(len(events)):
        event = str(events[i][0][0])
        for j in range(len(files[i][0])):
            if args.limit is not None and n >= args.limit:
                break
            name = str(files[i][0][j][0][0])
            img = cv2.imread(str(images / event / f"{name}.jpg"))
            t = time.perf_counter()
            dets = detector.detect(img)
            dt = time.perf_counter() - t
            if args.latency_warmup <= n < args.latency_warmup + args.latency_images:
                times.append(dt)
            preds.setdefault(event, {})[name] = np.array(
                [[d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1, d.confidence] for d in dets], dtype=np.float64
            ).reshape(-1, 5)
            n += 1
            if n % 500 == 0:
                logger.info(f"widerface {args.model}: {n} images")

    result = {
        "model": args.model, "det_size": args.det_size, "images": n, "score_floor": args.score_floor,
        "ap": official_ap(preds, gt, args.iou) if args.limit is None else None,
        "operating_points": operating_points(preds, gt, args.conf_at, args.fppi, args.iou),
        "latency_ms": {"median": round(1000 * float(np.median(times)), 2) if times else None,
                       "p90": round(1000 * float(np.percentile(times, 90)), 2) if times else None,
                       "images": len(times)},
        "gpu_memory_mib": gpu_memory_mib(os.getpid()),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "params": onnx_params(args.model, weights),
    }
    (out / "widerface.json").write_text(json.dumps(result, indent=2) + "\n")
    logger.info(json.dumps(result))


if __name__ == "__main__":
    main()
