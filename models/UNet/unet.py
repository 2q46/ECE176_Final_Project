import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.conv_relu import ConvReLUBlock
from blocks.upsample_conv import TransposeConvBlock


from params import UNetParams

class UNet(nn.Module):

    def __init__(self, params : UNetParams):

        super().__init__(UNet, self)

        self.down_conv1 = ConvReLUBlock(in_channels=params.encoder1_in, out_channels=params.encoder1_out)
        self.down_conv2 = ConvReLUBlock(in_channels=params.encoder2_in, out_channels=params.encoder2_out)
        self.down_conv3 = ConvReLUBlock(in_channels=params.encoder3_in, out_channels=params.encoder3_out)
        self.down_conv4 = ConvReLUBlock(in_channels=params.encoder4_in, out_channels=params.encoder4_out)

        self.bottlenck = ConvReLUBlock(in_channels=params.bottleneck_in, out_channels=params.bottleneck_out)

        self.up_conv1 = TransposeConvBlock(in_channels=params.deocder1_in, out_channels=params.decoder1_out)
        self.up_conv2 = TransposeConvBlock(in_channels=params.deocder2_in, out_channels=params.decoder2_out)
        self.up_conv3 = TransposeConvBlock(in_channels=params.deocder3_in, out_channels=params.decoder3_out)
        self.up_conv4 = TransposeConvBlock(in_channels=params.deocder4_in, out_channels=params.decoder4_out)
        
        self.maxpool = nn.MaxPool3d(kernel_size=params.max_pool_kernel, stride=params.max_pool_stride)
        self.softmax = nn.Softmax(dim=1)

        self.final_conv = nn.Conv3d(
            in_channels=params.final_conv_in, 
            kernel_size=params.final_kernel,
            out_channels=params.final_conv_out
        )

    def forward(self, x):

        e1 = self.down_conv1(x)
        m_e1 = self.maxpool(e1)

        e2 = self.down_conv2(m_e1)
        m_e2 = self.maxpool(e2)

        e3 = self.down_conv3(m_e2)
        m_e3 = self.maxpool(e3)

        e4 = self.down_conv4(m_e3)
        m_e4 = self.maxpool(e4)

        b1 = self.bottlenck(m_e4)

        out = self.up_conv1(b1, m_e4)
        out = self.up_conv2(out, m_e3)
        out = self.up_conv3(out, m_e2)
        out = self.up_conv4(out, m_e1)

        out = self.final_conv(out)
        self.softmax(out)

        return out