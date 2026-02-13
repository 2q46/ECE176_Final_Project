import torch
import torch.nn as nn
import torch.nn.functional as F

from params import UNetParams

class ConvReLUBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, params : UNetParams):
        '''
        Class that represents the 3X3X3 convolutional layers in the U-net paper. 
        The architecture is two 3D conv layer followed by a ReLU activation function
        in series with each other. The input channels, dense channels, and output channels 
        are passed to the class through the constructor.
        '''

        super(ConvReLUBlock, self).__init__()

        self.conv3d_1 = nn.Conv3d (
            in_channels = in_channels, 
            out_channels = out_channels,
            kernel_size = params.down_kernel_size,
            stride = params.stride
        )
        
        self.conv3d_2 = nn.Conv3d (
            in_channels = out_channels, 
            out_channels = out_channels,
            kernel_size = params.down_kernel_size,
            stride = params.stride
        )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.init_params()

    def init_params(self):

        self.conv3d_1.reset_parameters()
        self.conv3d_2.reset_parameters()

    def forward(self, x):

        x = self.conv3d_1(x)
        x = self.relu1(x)
        x = self.conv3d_2(x)
        x = self.relu2(x)
        
        return x