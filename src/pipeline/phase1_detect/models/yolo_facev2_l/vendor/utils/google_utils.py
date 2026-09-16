# Vendored, TRIMMED replacement for Krasjet-Yu/YOLO-FaceV2's utils/google_utils.py.
# Source: https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/2c125edc693a2fbb9d225a023a9514abef9ecb59/utils/google_utils.py
# No LICENSE file upstream — see ../NOTICE.md before any public release.
#
# Trim (documented, NOT a verbatim port — this is the one file in the
# vendored slice with rewritten behavior, not just rewritten imports):
# upstream's `attempt_download()` silently reaches out to GitHub
# releases / Google Cloud Storage over the network to fetch a missing
# `yolov5*.pt` by name, using `requests` + `torch.hub`. That's the wrong
# behavior for this pipeline for two reasons: (1) `yolo-facev2l-preweight.pt`
# isn't one of the auto-downloadable `ultralytics/yolov5` release assets
# upstream's logic matches against anyway, so the real download branch
# was already dead code for our checkpoint; (2) `convert.py` (the only
# caller of this file — see ../NOTICE.md) validates the checkpoint file
# exists *before* ever calling `attempt_load()`, so silent network access
# here would only ever mask a bug. This replacement keeps the same
# name/signature `attempt_load()` calls, but no-ops if the file exists and
# raises a clear `FileNotFoundError` if it doesn't, instead of reaching out
# to the network. Avoids a hard dependency on `requests` too.

from pathlib import Path


def attempt_download(file, repo='ultralytics/yolov5'):
    # See module docstring: no network fallback. Callers (convert.py) are
    # expected to have already validated the checkpoint file exists.
    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        raise FileNotFoundError(
            f"YOLO-FaceV2 weights not found: {file}\n"
            "Download 'yolo-facev2l-preweight.pt' from "
            "https://github.com/Krasjet-Yu/YOLO-FaceV2/releases and place it at this path "
            "(this vendored attempt_download() does not auto-download, unlike upstream's)."
        )
