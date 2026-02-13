import torch
import torch.nn as nn
import torch.nn.functional as F

from params import UNetParams

class TransposeConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, params : UNetParams):
        '''
        Class that represents the 2X2X2 up-convolutional layers in the U-net paper. 
        The architecture is two 3D transposed conv layer followed by a ReLU activation function
        in series with each other. The input channels, dense channels, and output channels 
        are passed to the class through the constructor.
        '''
        
        super().__init__(TransposeConvBlock, self)

        self.transposed_conv3d_1 = nn.ConvTranspose3d(
            in_channels=in_channels,
            kernel_size=params.up_kernel_size,
            out_channels=out_channels
        )

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            kernel_size=params.up_kernel_size,
            out_channels=out_channels
        )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.init_params()

    def init_params(self):

        self.transposed_conv3d_1.reset_parameters()
        self.conv3d.reset_parameters()

    def forward(self, x, skip_conntection):

        x = torch.cat([x, skip_conntection], dim=1) # concat across the feature dim.
        x = self.transposed_conv3d_1(x)
        x = self.relu1(x)
        x = self.conv3d(x)
        x = self.relu2(x)

        return x
