import torch
import torch.nn as nn
import torch.nn.functional as F

from models.UNet.params import UNetParams
from models.UNet.blocks.conv_relu import ConvReLUBlock

class TransposedConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, params: UNetParams):
        '''
        Class that represents the upsample convolutional layers in the U-net paper. 
        The architecture is one 3D Transpose conv layer followed by a ConvReluBlock.
        The input channels, and output channels are passed to the class through the constructor.
        '''
        super(TransposedConvBlock, self).__init__()

        self.conv3d_transposed = nn.ConvTranspose3d (
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=params.decoder_kernel,
            stride=params.decoder_stride
        )

        self.conv3d_block = ConvReLUBlock (
            in_channels=in_channels,
            out_channels=out_channels,
            params=params
        )

        self.reset_parameters()

    def reset_parameters(self):

        self.conv3d_transposed.reset_parameters()
        self.conv3d_block.reset_parameters()

    def forward(self, x, skip_connection):
        
        x = self.conv3d_transposed(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv3d_block(x)
        #print(x.shape)
        return x
