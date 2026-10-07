import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import AutoencoderKLConfig

from .lpips import LPIPS
from .patchgan import NLDiscriminator
from .modules import ResidualAttentionLayer
from .modules import UpsampleLayer
from .modules import DownsampleLayer


class VAEEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        pass
    
class VAEEncoder(nn.Module):
    def __init__(self, factor):
        super().__init__()
        pass
    
    def forward(self, x):
        pass

class VAEDecoder(nn.Module):
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
        batch_size = x.shape[0]
        original_x = x
        x = self.encode(x)
        x, kl_loss = self.bottleneck(x)
        generated_x = self.decode(x)
        
        # # Reconstructoin Loss
        # rec_loss = self.mse_loss(generated_x, original_x)    
        
        # # LpipsLoss
        # lpips_loss = self.lpips_loss(generated_x, original_x)
                
        # # PatchGan Loss
        # patch_gan_loss = self.patchgan_loss(generated_x).mean() # using mean for avg of patches
         

        
        
        
        
        # total_loss = rec_loss + self.config.lpips_weight * lpips_loss + patchgan_loss
        
        return generated_x
        
        