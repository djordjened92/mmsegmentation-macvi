import torch
from torch import nn
import logging
from ..builder import BACKBONES
from mmcv.runner import BaseModule

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Stem(nn.Module):
    def __init__(self, img_width:int, in_channels: int, out_channels: int, reduction:int=1) -> None:
        super(Stem, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        self.img_width = img_width

        self.stride1 = 2 if reduction % 2 == 0 else 1
        self.stride2 = 2 if reduction % 4 == 0 else 1

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride1, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=self.stride2, padding=2, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

    def get_output_img_width(self) -> int:
        return (((self.img_width - 1) // self.stride1) + 1) // self.stride2 + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return x

class GroupedDilationBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dilations):
        super(GroupedDilationBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations
        self.groups = len(self.dilations)

        assert self.in_channels % self.groups == 0
        assert self.out_channels % self.groups == 0
        self.group_size = self.in_channels // self.groups
        self.out_group_size = self.out_channels // self.groups

        convs = []
        for d in self.dilations:
            # pad = ((kernel_size - 1) * d) // 2 # for onnx export purpose
            convs.append(nn.Conv2d(self.group_size,
                                   self.out_group_size,
                                   kernel_size,
                                   padding='same',
                                   dilation=d,
                                   groups=self.group_size if self.group_size==self.out_group_size else 1))
        self.convs = nn.ModuleList(convs)

        self.skip_conn = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else nn.Identity()
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU6()

    def forward(self, x):
        skip = self.skip_conn(x)

        x_reshaped = x.view(x.shape[0],
                            self.groups,
                            self.group_size,
                            x.shape[2],
                            x.shape[3])
        out_shape = list(x.shape)
        out_shape[1] = self.out_channels
        out = torch.empty(out_shape, device=x.device)
        for i, conv in enumerate(self.convs):
            out[:, i * self.group_size:(i + 1) * self.group_size] = conv(x_reshaped[:, i])

        out = self.activation(self.bn(out))
        out = out + skip

        return out

class AggDownSample(nn.Module):
    def __init__(self, resolution: int,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 dilations_len: int,
                 kernel_size: int,
                 stride: int,
                 pooling:nn.Module=None) -> None:
        super(AggDownSample, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilations_len = dilations_len

        self.dense_channels = in_channels + hidden_channels
        self.downsampler_channels = out_channels

        self.dense_fc = nn.Conv2d(self.dense_channels, out_channels, kernel_size=1, stride=1, groups=self.dense_channels // (2 * self.dilations_len))
        self.activation = nn.ReLU6()
        self.batch_norm = nn.BatchNorm2d(self.out_channels)

        self.downsampler = None
        if pooling is None or pooling == nn.Identity:
            self.downsampler = nn.Identity()
        elif pooling == nn.Conv2d:
            self.downsampler = nn.Conv2d(self.downsampler_channels, out_channels, kernel_size=2, stride=2, groups=out_channels, bias=False)
        elif pooling == nn.MaxPool2d or pooling == nn.AvgPool2d:
            self.downsampler = pooling(kernel_size, stride)#, padding=kernel_size//2)
        
        self.grn = GRN(self.out_channels)

        logging.info(f"Setting DownSampler {hidden_channels} -> {out_channels} with downsampler type {type(self.downsampler)} and dense_fc {type(self.dense_fc)}")

    def get_output_img_width(self) -> int:
        if self.pooling == nn.MaxPool2d or self.pooling == nn.AvgPool2d:
            output_resolution = (self.resolution - self.kernel_size) // self.stride + 1
        else:
            raise NotImplementedError(f"Pooling layer {self.pooling} not implemented")
        return output_resolution

    def aggregate(self, x: torch.Tensor, dense_x: torch.Tensor) -> torch.Tensor:
        # Recombine channel dimension to have
        # fmaps from different dilations grouped
        B, C, H, W = x.shape
        group_size = C // self.dilations_len
        x = x.view(B, self.dilations_len, group_size, H, W)
        x.swapaxes_(1, 2)
        x = x.reshape(B, C, H, W)

        # Mix Concatenation
        b, c, h, w = x.size()
        x = torch.cat([x.view(b, -1, 1, h, w), dense_x.view(b, -1, 1, h, w)], dim=2)
        x = x.reshape(b, -1, h, w)
        x = self.activation(self.batch_norm(self.dense_fc(x)))

        return x

    def forward(self, x: torch.Tensor, dense_x: torch.Tensor) -> torch.Tensor:
        x = self.aggregate(x, dense_x)
        x = self.grn(self.downsampler(x))
        return x

class GroupedDilationStage(nn.Module):
    def __init__(self, module: nn.Module, img_width: int, in_channels: int, hidden_channels: int, out_channels: int, depth: int, dilations: list, pooling: nn.Module=None) -> None:
        super(GroupedDilationStage, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(depth):
            cur_in_channels = in_channels if i == 0 else hidden_channels
            self.layers.append(module(cur_in_channels,
                                      hidden_channels,
                                      kernel_size=3,
                                      stride=1,
                                      dilations=dilations))

        self.stage = nn.Sequential(*self.layers)
        self.aggregate_downsample = AggDownSample(img_width, in_channels, hidden_channels, out_channels, len(dilations), kernel_size=2, stride=2, pooling=pooling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stage(x)
        out = self.aggregate_downsample(out, dense_x=x)
        return out

@BACKBONES.register_module()
class GlimmerNet(BaseModule):
    r""" GlimmerNet
    """
    def __init__(self,
                img_width,
                out_indices=(0, 1, 2, 3),
                init_cfg=None
                ):
        super(GlimmerNet, self).__init__(init_cfg)
        self.out_indices = out_indices

        self.input_channels = 3
        self.depths = [4, 4, 4, 1]
        self.widths = [40, 80, 160, 240]
        self.dilations = [[1, 2, 2, 3], [1, 2, 2, 3], [1, 2, 2, 3], [1, 2, 2, 3]]
        self.poolings = [nn.MaxPool2d, nn.MaxPool2d, nn.MaxPool2d, nn.AvgPool2d]
        self.net_modules = [GroupedDilationBlock, GroupedDilationBlock, GroupedDilationBlock, GroupedDilationBlock]
        self.reduction = 2
        self.img_width = img_width
        
        assert len(self.depths) == len(self.widths) == len(self.dilations), "depths, dilations and widths must have the same length"

        self.stages = nn.ModuleList()
        self.stem = Stem(self.img_width, self.input_channels, self.widths[0], reduction=self.reduction)
        curr_img_width = self.stem.get_output_img_width()
        prev_channel_dim = self.widths[0]
        for i in range(len(self.depths)):
            hidden_channels = self.widths[i]
            out_channels = self.widths[i + 1] if i < len(self.depths) - 1 else self.widths[i]
            
            prev_channel_dim = self.widths[i]
            self.stages.append(GroupedDilationStage(self.net_modules[i],
                                                    curr_img_width,
                                                    prev_channel_dim,
                                                    hidden_channels,
                                                    out_channels,
                                                    self.depths[i],
                                                    self.dilations[i],
                                                    self.poolings[i]))
            curr_img_width = self.stages[i].aggregate_downsample.get_output_img_width()

    def forward(self, x: torch.Tensor):
        outs = []
        x = self.stem(x) # 1/2

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)