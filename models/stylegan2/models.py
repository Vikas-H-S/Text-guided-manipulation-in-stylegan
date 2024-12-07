import math
import random
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from .op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

class NormalizedPixel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp_tensor):
        return inp_tensor * torch.rsqrt(torch.mean(inp_tensor ** 2, dim=1, keepdim=True) + 1e-8)

def create_kernel(kernel_vals):
    kernel_vals = torch.tensor(kernel_vals, dtype=torch.float32)

    if kernel_vals.ndim == 1:
        kernel_vals = kernel_vals[None, :] * kernel_vals[:, None]

    kernel_vals /= kernel_vals.sum()

    return kernel_vals

class Upscale(nn.Module):
    def __init__(self, kernel_vals, factor=2):
        super().__init__()

        self.scale_factor = factor
        kernel_vals = create_kernel(kernel_vals) * (factor ** 2)
        self.register_buffer("kernel_vals", kernel_vals)

        p = kernel_vals.shape[0] - factor

        pad_0 = (p + 1) // 2 + factor - 1
        pad_1 = p // 2

        self.padding = (pad_0, pad_1)

    def forward(self, inp_tensor):
        output = upfirdn2d(inp_tensor, self.kernel_vals, up=self.scale_factor, down=1, pad=self.padding)
        return output

class Downscale(nn.Module):
    def __init__(self, kernel_vals, factor=2):
        super().__init__()

        self.scale_factor = factor
        kernel_vals = create_kernel(kernel_vals)
        self.register_buffer("kernel_vals", kernel_vals)

        p = kernel_vals.shape[0] - factor

        pad_0 = (p + 1) // 2
        pad_1 = p // 2

        self.padding = (pad_0, pad_1)

    def forward(self, inp_tensor):
        output = upfirdn2d(inp_tensor, self.kernel_vals, up=1, down=self.scale_factor, pad=self.padding)
        return output

class GaussianBlur(nn.Module):
    def __init__(self, kernel_vals, pad, scale_factor=1):
        super().__init__()

        kernel_vals = create_kernel(kernel_vals)

        if scale_factor > 1:
            kernel_vals = kernel_vals * (scale_factor ** 2)

        self.register_buffer("kernel_vals", kernel_vals)
        self.padding = pad

    def forward(self, inp_tensor):
        output = upfirdn2d(inp_tensor, self.kernel_vals, pad=self.padding)
        return output

class WeightScaledConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.scale_factor = 1 / math.sqrt(in_channels * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, inp_tensor):
        output = F.conv2d(
            inp_tensor,
            self.weight * self.scale_factor,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return output

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )

class WeightScaledLinear(nn.Module):
    def __init__(
        self, input_dim, output_dim, bias=True, bias_init=0, lr_mult=1, activation_fn=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(output_dim, input_dim).div_(lr_mult))

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation_fn = activation_fn

        self.scale_factor = (1 / math.sqrt(input_dim)) * lr_mult
        self.lr_mult = lr_mult

    def forward(self, inp_tensor):
        if self.activation_fn:
            output = F.linear(inp_tensor, self.weight * self.scale_factor)
            output = fused_leaky_relu(output, self.bias * self.lr_mult)
        else:
            output = F.linear(
                inp_tensor, self.weight * self.scale_factor, bias=self.bias * self.lr_mult
            )

        return output

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )
    
class AdaptiveLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, inp_tensor):
        output = F.leaky_relu(inp_tensor, negative_slope=self.negative_slope)

        return output * math.sqrt(2)