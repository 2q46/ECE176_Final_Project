import torch
import torch.nn as nn
import torch.nn.functional as F

from params import UNetParams

class Decoder(nn.Module):

    def __init__(self, params : UNetParams):
        
        super().__init__(Decoder, self)

        self.cache = []

    def forward(self, x):

        pass