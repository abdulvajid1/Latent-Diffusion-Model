import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import AutoencoderKLConfig

from .lpips import LPIPS
from .patchgan import NLDiscriminator


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        pass
    
class Encoder(nn.Module):
    def __init__(self, factor):
        super().__init__()
        pass
    
    def forward(self, x):
        pass

class Decoder(nn.Module):
    def __init__(self, factor):
        super().__init__()
        pass
    
    def forward(self, x):
        pass

class DiagonalGuassianDistribution(nn.Module):
    pass

class AutoencoderKL(nn.Module):
    def __init__(self, config: AutoencoderKLConfig):
        super().__init__()
        factor = config.factor
        self.encoder = Encoder(factor)
        self.decoder = Decoder(factor)
        self.bottleneck = DiagonalGuassianDistribution()
        
        self.mse_loss = nn.MSELoss()
        self.lpips_loss = LPIPS()
        self.patchgan_loss = NLDiscriminator(img_channels=config.img_channel, start_dim=config.patchgan_start_dim)
        
    def encode(self, x):
        return x
    
    def decode(self, x):
        return x
    
    def forward(self, x: torch.Tensor):
        original_x = x
        x = self.encode(x)
        x, kl_loss = self.bottleneck(x)
        x = self.decode(x)
        
        # Reconstructoin Loss
        rec_loss = self.mse_loss(x, original_x)    
        # LpipsLoss
        lpips_loss = self.lpips_loss(x, original_x)
        # PatchGan Loss
        patchgan_loss = self.patchgan_loss(original_x, x)
        
        return x, rec_loss, kl_loss, lpips_loss, patchgan_loss
        
        