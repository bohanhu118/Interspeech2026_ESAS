import torch
import torch.nn as nn
from typing import Optional
from torchvision.ops.misc import ConvNormActivation


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class GRN(nn.Module):
    """
    global response normalization as introduced in https://arxiv.org/pdf/2301.00808.pdf
    """

    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)

        x = self.gamma * (x * nx) + self.beta + x
        return x


class CPMobileBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            expansion_rate,
            stride
    ):
        super().__init__()
        exp_channels = make_divisible(in_channels * expansion_rate, 8)

        # create the three factorized convs that make up our block
        exp_conv = ConvNormActivation(in_channels,
                                      exp_channels,
                                      kernel_size=1,
                                      stride=1,
                                      norm_layer=nn.BatchNorm2d,
                                      activation_layer=nn.ReLU,
                                      inplace=False
                                      )

        # depthwise convolution with possible stride
        depth_conv = ConvNormActivation(exp_channels,
                                        exp_channels,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,
                                        groups=exp_channels,
                                        norm_layer=nn.BatchNorm2d,
                                        activation_layer=nn.ReLU,
                                        inplace=False
                                        )

        proj_conv = ConvNormActivation(exp_channels,
                                       out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       norm_layer=nn.BatchNorm2d,
                                       activation_layer=None,
                                       inplace=False
                                       )

        self.after_block_norm = GRN()
        self.after_block_activation = nn.ReLU()

        if in_channels == out_channels:
            self.use_shortcut = True
            if stride == 1 or stride == (1, 1):
                self.shortcut = nn.Sequential()
            else:
                # average pooling required for shortcut
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.Sequential()
                )
        else:
            self.use_shortcut = False

        self.block = nn.Sequential(
            exp_conv,
            depth_conv,
            proj_conv
        )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_shortcut:
            x = self.skip_add.add(self.block(x), self.shortcut(x))
        else:
            x = self.block(x)
        x = self.after_block_norm(x)
        x = self.after_block_activation(x)
        return x

