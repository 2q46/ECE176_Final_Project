import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from params import UNetParams

class ConvReLU(nn.Module):

    def __init__(self, in_channels: int, dense_channels : int, out_channels: int, params : UNetParams):
        '''
        Class that represents the 3X3X3 convolutional layers in the U-net paper. 
        The architecture is two 3D conv layer followed by a ReLU activation function
        in series with each other. The input channels, dense channels, and output channels 
        are passed to the class through the constructor.
        '''

        super(ConvReLU, self).__init__()

        self.conv3d_1 = nn.Conv3d (
            in_channels = in_channels, 
            out_channels = dense_channels,
            kernel_size = params.kernel_size,
            stride = params.stride
        )
        
        self.conv3d_2 = nn.Conv3d (
            in_channels = dense_channels, 
            out_channels = out_channels,
            kernel_size = params.kernel_size,
            stride = params.stride
        )

        self.relu = F.relu()

    def init_params(self):

        init.xavier_normal_(self.conv3d_1)
        init.xavier_normal_(self.conv3d_2)

    def forward(self, x):

        x = self.conv3d_1(x)
        x = self.relu(x)
        x = self.conv3d_2(x)
        x = self.relu(x)
        
        return x