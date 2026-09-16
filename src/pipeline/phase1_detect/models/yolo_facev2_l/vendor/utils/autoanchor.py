# Vendored, TRIMMED subset of Krasjet-Yu/YOLO-FaceV2's utils/autoanchor.py —
# just `check_anchor_order()`, called once from models/yolo.py's Model.__init__.
# Source: https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/2c125edc693a2fbb9d225a023a9514abef9ecb59/utils/autoanchor.py
# No LICENSE file upstream — see ../NOTICE.md before any public release.
#
# Trim (documented, not a logic change): upstream's utils/autoanchor.py
# also defines `check_anchors()` and `kmean_anchors()` — training-time
# anchor-fitting helpers whose module-level imports (`scipy.cluster.vq`,
# `tqdm`) would otherwise become hard dependencies of this inference-only
# backend just to import a function that's never called. Only
# `check_anchor_order()` is kept, verbatim; it needs no other imports.


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)
