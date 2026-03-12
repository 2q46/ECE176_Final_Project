import torch
import torch.nn as nn

from params import UNetParams
from blocks.conv_relu import ConvReLUBlock

class ConvTransposeAttn(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, params: UNetParam):

        super(ConvTransposeAttn, self).__init__()

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

        self.attn_applied = AttnModule(out_channels)

        self.groupnorm = nn.GroupNorm(8, in_channels, affine=False)

        self.reset_parameters()

    def reset_parameters(self):

        self.conv3d_transposed.reset_parameters()
        self.conv3d_block.reset_parameters()

    def forward(self, x, skip_connection):

        x = self.conv3d_transposed(x)
        x = self.attn_applied(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.groupnorm(x)
        x = self.conv3d_block(x)
        return x

class AttnModule(nn.Module):

    def __init__(self, channels: int):
        '''
        Purpose is to figure would which portions of the skip connection are 
        important before passing them to the decoder. This is achieved through 
        learned parameters. 
        '''

        super(AttnModule, self).__init__()

        self.guidance_signal = nn.Sequential(
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 1, 1),
            ),
            nn.GroupNorm(8, channels, affine=True)
        )

        self.skip_signal = nn.Sequential(
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 1, 1)
            ),
            nn.GroupNorm(8, channels, affine=False)
        )

        self.attn_map = nn.Sequential(
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 1, 1)
            ),
            nn.Sigmoid(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, skip_connection):

        x_gs = self.guidance_signal(x)
        skip_connection = self.skip_signal(skip_connection)
        combined_signal = self.relu(x_gs + skip_connection)
        attn_scores = self.attn_map(combined_signal)
        return x * attn_scores # element wise multiply
