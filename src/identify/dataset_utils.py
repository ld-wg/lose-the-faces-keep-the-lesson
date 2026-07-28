#!/usr/bin/env python3
"""Dataset prep & checks for WIDER FACE → YOLOv8 (minimal)."""

from __future__ import annotations
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Lightweight image-size readers
try:
    from PIL import Image  # preferred
except Exception:  # pragma: no cover
    Image = None
try:
    import cv2  # fallback
except Exception:  # pragma: no cover
    cv2 = None


def _img_size(p: Path) -> tuple[int, int]:
    if Image is not None:
        with Image.open(p) as im:
            w, h = im.size
            return int(w), int(h)
    if cv2 is not None:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(p)
        h, w = img.shape[:2]
        return int(w), int(h)
    raise RuntimeError("Install pillow or opencv-python to read image sizes")


def prepare_dataset(fraction: float, data_path: Path, dataset_dir: Path) -> Dict[str, int]:
    """Create YOLO-style tree with images/{train,val} and labels/{train,val}."""
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    train_img = dataset_dir / "images" / "train"
    val_img   = dataset_dir / "images" / "val"
    train_lab = dataset_dir / "labels" / "train"
    val_lab   = dataset_dir / "labels" / "val"
    for d in (train_img, val_img, train_lab, val_lab):
        d.mkdir(parents=True, exist_ok=True)

    train_count = convert_annotations(
        data_path / "wider_face_split" / "wider_face_train_bbx_gt.txt",
        data_path / "WIDER_train" / "images",
        train_img,
        train_lab,
        fraction,
        "train",
    )
    val_count = convert_annotations(
        data_path / "wider_face_split" / "wider_face_val_bbx_gt.txt",
        data_path / "WIDER_val" / "images",
        val_img,
        val_lab,
        fraction,
        "val",
    )
    logger.info(f"dataset: {train_count} train / {val_count} val")
    return {"train_count": train_count, "val_count": val_count}


def convert_annotations(
    annotation_file: Path,
    src_img_dir: Path,
    dest_img_dir: Path,
    dest_lbl_dir: Path,
    fraction: float,
    split_name: str,
) -> int:
    """WIDER txt → YOLO txt + image symlinks, preserving subfolders."""
    if not annotation_file.exists():
        logger.error(f"missing annotations: {annotation_file}")
        return 0

    lines = annotation_file.read_text().splitlines()
    i = 0
    records: dict[str, list[str]] = {}

    while i < len(lines):
        rel = lines[i].strip()
        if not rel.endswith(".jpg"):
            i += 1
            continue
        i += 1
        if i >= len(lines):
            break
        try:
            n = int(lines[i].strip())
        except ValueError:
            break
        i += 1

        img_path = src_img_dir / rel
        if not img_path.exists():
            i += n
            continue
        try:
            img_w, img_h = _img_size(img_path)
        except Exception:
            i += n
            continue

        ylines: list[str] = []
        for k in range(n):
            if i + k >= len(lines):
                break
            parts = lines[i + k].split()
            if len(parts) < 8:
                continue
            x1, y1, w, h = map(float, parts[:4])
            invalid = int(parts[7])
            if invalid or w <= 0 or h <= 0:
                continue
            # clip box into image bounds
            x2 = x1 + w
            y2 = y1 + h
            x1 = max(0.0, min(x1, img_w))
            y1 = max(0.0, min(y1, img_h))
            x2 = max(0.0, min(x2, img_w))
            y2 = max(0.0, min(y2, img_h))
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw == 0 or bh == 0:
                continue
            xc = (x1 + x2) / (2 * img_w)
            yc = (y1 + y2) / (2 * img_h)
            wn = bw / img_w
            hn = bh / img_h
            # keep only boxes fully inside after clipping
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < wn <= 1 and 0 < hn <= 1):
                continue
            ylines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        i += n

        if ylines:
            records[rel] = ylines

    # deterministic sampling
    total = len(records)
    if fraction < 1.0 and total:
        k = max(1, int(total * fraction))
        rnd = random.Random(42)
        keep = set(rnd.sample(list(records.keys()), k))
        records = {k: records[k] for k in keep}

    # write labels + symlink images (preserve subfolders)
    made = 0
    for rel, ylines in records.items():
        img_src = src_img_dir / rel
        lbl_rel = Path(rel).with_suffix(".txt")
        img_dst = dest_img_dir / rel
        lbl_dst = dest_lbl_dir / lbl_rel
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.write_text("\n".join(ylines))
        try:
            img_dst.symlink_to(os.path.relpath(img_src, img_dst.parent))
        except OSError:
            try:
                img_dst.symlink_to(img_src)
            except OSError as e:
                logger.warning(f"symlink fail: {rel} → {e}")
                continue
        made += 1

    logger.info(f"{split_name}: {made} pairs")
    return made


def check_dataset_integrity(image_dir: Path, label_dir: Path) -> Dict[str, Any]:
    """Report mismatches/empties using relative stems (avoids name collisions)."""
    issues = {
        "image_without_label": [],
        "label_without_image": [],
        "empty_labels": [],
    }
    if not image_dir.exists() or not label_dir.exists():
        return issues

    img_stems = set(
        (p.relative_to(image_dir).with_suffix("") ).as_posix()
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    lbl_stems = set(
        (p.relative_to(label_dir).with_suffix("")).as_posix()
        for p in label_dir.rglob("*.txt")
    )

    issues["image_without_label"] = sorted(img_stems - lbl_stems)
    issues["label_without_image"] = sorted(lbl_stems - img_stems)

    for p in label_dir.rglob("*.txt"):
        if p.stat().st_size == 0:
            issues["empty_labels"].append(p.relative_to(label_dir).as_posix())

    return issues


def validate_yolo_labels(label_dir: Path) -> Dict[str, int]:
    """Count total/empty/malformed/valid YOLO files (class 0 only)."""
    out = {"total_files": 0, "empty_files": 0, "malformed_files": 0, "valid_files": 0}
    if not label_dir.exists():
        return out

    for f in label_dir.rglob("*.txt"):
        out["total_files"] += 1
        try:
            lines = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
            if not lines:
                out["empty_files"] += 1
                continue
            ok = True
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    ok = False
                    break
                c, x, y, w, h = parts
                try:
                    c = int(float(c))
                    x = float(x); y = float(y); w = float(w); h = float(h)
                except ValueError:
                    ok = False; break
                if c != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    ok = False; break
            if ok:
                out["valid_files"] += 1
            else:
                out["malformed_files"] += 1
        except Exception:
            out["malformed_files"] += 1
    return out
