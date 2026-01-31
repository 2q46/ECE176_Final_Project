import torch
import torch.nn as nn
import torch.nn.functional as F

from params import UNetParams

class BottleNeck(nn.Module):

    def __init__(self, params : UNetParams):
        
        super().__init__(BottleNeck, self)

        self.cache = []

    def forward(self, x):

        pass
