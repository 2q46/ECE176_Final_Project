import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpandingModule(nn.Module):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)


    