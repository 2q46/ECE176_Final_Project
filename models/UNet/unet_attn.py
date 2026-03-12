import torch
import torch.nn as nn

from blocks.conv_relu import ConvReLUBlock
from params import UNetParams

class AttentionUNet(nn.Module):

    def __init__(self, params : UNetParams):

        super(AttentionUNet, self).__init__()

        self.down_conv1 = ConvReLUBlock(in_channels=params.encoder1_in, out_channels=params.encoder1_out, params=params)
        self.down_conv2 = ConvReLUBlock(in_channels=params.encoder2_in, out_channels=params.encoder2_out, params=params)
        self.down_conv3 = ConvReLUBlock(in_channels=params.encoder3_in, out_channels=params.encoder3_out, params=params)
        self.down_conv4 = ConvReLUBlock(in_channels=params.encoder4_in, out_channels=params.encoder4_out, params=params)

        self.bottlenck = ConvReLUBlock(in_channels=params.bottleneck_in, out_channels=params.bottleneck_out, params=params)

        self.maxpool = nn.MaxPool3d(kernel_size=params.max_pool_kernel)

        self.final_conv = nn.Conv3d(
            in_channels=params.final_conv_in, 
            kernel_size=params.final_kernel,
            out_channels=params.final_conv_out
        )

    def forward(self, x):


        return x