import torch
import torch.nn as nn
from ..builder import BACKBONES
from mmcv.runner import BaseModule

def conv_bn_relu(in_c, out_c, kernel, stride=1, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True)
    )

class TakuBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super(TakuBlock, self).__init__()
        mid_c = out_c // 2
        self.conv1 = conv_bn_relu(in_c, mid_c, 1)
        self.conv2 = conv_bn_relu(mid_c, mid_c, 3, stride, 1, groups=mid_c)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = stride == 1 and in_c == out_c

    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        return out + x if self.shortcut else out

@BACKBONES.register_module()
class TakuNet(BaseModule):
    def __init__(self, in_channels=3, out_indices=(0, 1, 2, 3), init_cfg=None):
        super(TakuNet, self).__init__(init_cfg)
        self.out_indices = out_indices
        
        # Stem
        self.stem = conv_bn_relu(in_channels, 16, 3, stride=2, padding=1)
        
        # Stages (Simplification of TakuNet architecture)
        # Stage output channels: 32, 64, 128, 256
        self.stage1 = nn.Sequential(TakuBlock(16, 32, 2))
        self.stage2 = nn.Sequential(TakuBlock(32, 64, 2))
        self.stage3 = nn.Sequential(TakuBlock(64, 128, 2))
        self.stage4 = nn.Sequential(TakuBlock(128, 256, 2))

    def forward(self, x):
        outs = []
        x = self.stem(x) # 1/2
        
        x = self.stage1(x) # 1/4
        if 0 in self.out_indices: outs.append(x)
        
        x = self.stage2(x) # 1/8
        if 1 in self.out_indices: outs.append(x)
        
        x = self.stage3(x) # 1/16
        if 2 in self.out_indices: outs.append(x)
        
        x = self.stage4(x) # 1/32
        if 3 in self.out_indices: outs.append(x)
        
        return tuple(outs)