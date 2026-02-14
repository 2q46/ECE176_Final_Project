import torch
import torch.nn as nn
import torch.nn.functional as F

from models.UNet.blocks.conv_relu import ConvReLUBlock
from models.UNet.blocks.upsample_conv import TransposedConvBlock
from models.UNet.params import UNetParams

class UNet(nn.Module):

    def __init__(self, params : UNetParams):

        super(UNet, self).__init__()

        self.down_conv1 = ConvReLUBlock(in_channels=params.encoder1_in, out_channels=params.encoder1_out, params=params)
        self.down_conv2 = ConvReLUBlock(in_channels=params.encoder2_in, out_channels=params.encoder2_out, params=params)
        self.down_conv3 = ConvReLUBlock(in_channels=params.encoder3_in, out_channels=params.encoder3_out, params=params)
        self.down_conv4 = ConvReLUBlock(in_channels=params.encoder4_in, out_channels=params.encoder4_out, params=params)

        self.bottlenck = ConvReLUBlock(in_channels=params.bottleneck_in, out_channels=params.bottleneck_out, params=params)

        self.up_conv1 = TransposedConvBlock(in_channels=params.decoder1_in, out_channels=params.decoder1_out, params=params)
        self.up_conv2 = TransposedConvBlock(in_channels=params.decoder2_in, out_channels=params.decoder2_out, params=params)
        self.up_conv3 = TransposedConvBlock(in_channels=params.decoder3_in, out_channels=params.decoder3_out, params=params)
        self.up_conv4 = TransposedConvBlock(in_channels=params.decoder4_in, out_channels=params.decoder4_out, params=params)
        
        self.maxpool = nn.MaxPool3d(kernel_size=params.max_pool_kernel)
        self.softmax = nn.Softmax(dim=1)

        self.final_conv = nn.Conv3d(
            in_channels=params.final_conv_in, 
            kernel_size=params.final_kernel,
            out_channels=params.final_conv_out
        )

    def forward(self, x):

        e1 = self.down_conv1(x)

        e2 = self.maxpool(e1)
        e2 = self.down_conv2(e2)

        e3 = self.maxpool(e2)
        e3 = self.down_conv3(e3)

        e4 = self.maxpool(e3)
        e4 = self.down_conv4(e4)

        b1 = self.bottlenck(self.maxpool(e4))

        out = self.up_conv1(b1, e4)
        out = self.up_conv2(out, e3)
        out = self.up_conv3(out, e2)
        out = self.up_conv4(out, e1)

        out = self.final_conv(out)
        self.softmax(out)

        return out