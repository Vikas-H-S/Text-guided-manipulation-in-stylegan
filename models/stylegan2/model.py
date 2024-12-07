import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from models.stylegan2.op import CustomActivation, custom_leaky_relu, adaptive_filter

class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, data):
        return data * torch.rsqrt(torch.mean(data ** 2, dim=1, keepdim=True) + 1e-8)

def build_kernel(matrix):
    matrix = torch.tensor(matrix, dtype=torch.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :] * matrix[:, None]
    return matrix / matrix.sum()

class Upscale(nn.Module):
    def __init__(self, kernel, scale=2):
        super().__init__()
        self.scale = scale
        kernel = build_kernel(kernel) * (scale ** 2)
        self.register_buffer('kernel', kernel)
        padding = kernel.shape[0] - scale
        self.pad = ((padding + 1) // 2 + scale - 1, padding // 2)

    def forward(self, data):
        return adaptive_filter(data, self.kernel, up=self.scale, pad=self.pad)

class Downscale(nn.Module):
    def __init__(self, kernel, scale=2):
        super().__init__()
        self.scale = scale
        kernel = build_kernel(kernel)
        self.register_buffer('kernel', kernel)
        padding = kernel.shape[0] - scale
        self.pad = ((padding + 1) // 2, padding // 2)

    def forward(self, data):
        return adaptive_filter(data, self.kernel, down=self.scale, pad=self.pad)

class BlurFilter(nn.Module):
    def __init__(self, kernel, padding, upsampling_factor=1):
        super().__init__()
        kernel = build_kernel(kernel)
        if upsampling_factor > 1:
            kernel = kernel * (upsampling_factor ** 2)
        self.register_buffer('kernel', kernel)
        self.pad = padding

    def forward(self, data):
        return adaptive_filter(data, self.kernel, pad=self.pad)

class Conv2DAdaptive(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.kernel_weights = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.scaling_factor = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, data):
        return F.conv2d(
            data,
            self.kernel_weights * self.scaling_factor,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.kernel_weights.shape[1]}, {self.kernel_weights.shape[0]},"
            f" {self.kernel_weights.shape[2]}, stride={self.stride}, padding={self.padding})"
        )