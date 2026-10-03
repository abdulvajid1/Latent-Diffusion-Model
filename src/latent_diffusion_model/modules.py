from typing import Any

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        pass
    

class ResidualAttentionLayer(nn.Module):
    def __init__(self, use_attention: bool = False) -> None:
        super().__init__()
    
    def forward(self, x):
        pass
    

class UpsampleLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass