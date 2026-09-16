# Vendored, TRIMMED subset of Krasjet-Yu/YOLO-FaceV2's utils/general.py.
# Source: https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/2c125edc693a2fbb9d225a023a9514abef9ecb59/utils/general.py
# No LICENSE file upstream — see ../../NOTICE.md before any public release.
#
# Trim (documented, not a logic change): upstream's utils/general.py is a
# 700-line grab-bag of training/dataset/logging/NMS helpers. `vendor/` is
# used only by `convert.py` (one-time checkpoint -> ONNX export) — inference
# postprocessing (NMS, letterbox, coordinate rescaling) lives in
# `../backend.py` instead, reimplemented in plain numpy so the inference
# path has no vendored-code or `torch`/`torchvision` dependency at all.
# That leaves only the three names `models/yolo.py` imports at module load
# time, needed just to construct the model for tracing: `make_divisible`,
# `check_file`, `set_logging`. Every kept function body is verbatim.

import glob
import logging
import math
import os


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def check_file(file):
    # Search for file if not found
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        return files[0]  # return file


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor
