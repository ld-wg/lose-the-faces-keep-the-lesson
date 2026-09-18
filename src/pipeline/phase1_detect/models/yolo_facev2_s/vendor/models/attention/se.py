# Vendored verbatim from Krasjet-Yu/YOLO-FaceV2 @ 2c125edc693a2fbb9d225a023a9514abef9ecb59
# https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/2c125edc693a2fbb9d225a023a9514abef9ecb59/models/attention/se.py
# No LICENSE file upstream — see ../../../NOTICE.md before any public release.
#
import torch
import torch.nn as nn

class SE(nn.Module):
    def __init__(self, c1, c2, r=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // r, c1, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        # print(x.size())
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)