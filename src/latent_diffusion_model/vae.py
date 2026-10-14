import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import AutoencoderKLConfig

from .lpips import LPIPS
from .patchgan import NLDiscriminator
from .modules import AttentionLayer, ResidualLayer
from .modules import UpsampleLayer
from .modules import DownsampleLayer


class VAEEncoderBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_res_blocks,
                 groupnorm_groups=32,
                 dropout_p=0.0,
                 add_downsample=True):
        super().__init__()
        self.resblocks = nn.ModuleList()
        for i in range(num_res_blocks):
            conv_in_channels = in_channels if i == 0 else out_channels
            self.resblocks.append(
                ResidualLayer(
                    in_channels=conv_in_channels,
                    out_channels=out_channels,
                    num_group_groupnorm=groupnorm_groups,
                    dropout_p=dropout_p,
                    time_embed_dim=None,
                    class_embed_dim=None
                )
            )
        
        self.downsample = DownsampleLayer(in_channels=out_channels) if add_downsample else nn.Identity()
    
    def forward(self, x):
        for module in self.resblocks:
            x = module(x)
        
        return self.downsample(x)
        
    
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


class ResidualAttentionLayer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 num_layers, 
                 attention_num_heads=None,
                 attention_residual_connection: bool=True,
                 groupnorm_groups=32,
                 time_emb_dim=None, 
                 class_emb_dim=None,
                 ) -> None:
        super().__init__()
        self.resnet_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.resnet_blocks.append(ResidualLayer(in_channels, in_channels, num_group_groupnorm=groupnorm_groups, time_embed_dim=time_emb_dim, class_embed_dim=class_emb_dim))
        
        for _ in range(num_layers):
            self.attn_blocks.append(
                AttentionLayer(embedding_dim=in_channels, 
                               num_heads=attention_num_heads, 
                               num_groups_groupnorm=groupnorm_groups,
                               residual_connection=attention_residual_connection,
                               return_dim='2D')
                )
            
            self.resnet_blocks.append(
                ResidualLayer(in_channels, 
                              in_channels, 
                              num_group_groupnorm=groupnorm_groups, 
                              time_embed_dim=time_emb_dim, 
                              class_embed_dim=class_emb_dim
                              )
            )
        
    def forward(self, x, time_embed=None, class_embed=None):
        x = self.resnet_blocks[0](x, time_embed, class_embed)
        
        for attn, resnet in zip(self.attn_blocks, self.resnet_blocks[1:]):
            x = attn(x)
            x = resnet(x, time_embed, class_embed)
        
        return x

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
        
    

if __name__ == "__main__":
    x = torch.randn(1, 32, 32, 32)
    model = ResidualAttentionLayer(
        in_channels=32,
        num_layers=2,
        attention_num_heads=4,
        attention_residual_connection=True,
        time_emb_dim=None,
        class_emb_dim=None,
        groupnorm_groups=4
    )
    print('='*50)
    print("ResidualAttentionLayer")
    print('='*50)
    print(model(x).shape)
    
    print('='*50)
    print("ResidualAttentionLayer with time_embedding")
    print('='*50)
    time_embedding = torch.randn(1, 100)
    model = ResidualAttentionLayer(
        in_channels=32,
        num_layers=2,
        attention_num_heads=4,
        attention_residual_connection=True,
        time_emb_dim=100,
        class_emb_dim=None,
        groupnorm_groups=4,
    )
    print(model(x, time_embedding).shape)
    
    print('='*50)
    print("Encoder Block")
    model = VAEEncoderBlock(in_channels=32, out_channels=32, num_res_blocks=2, groupnorm_groups=4, dropout_p=0.0, add_downsample=True)
    print(model(x).shape)
    print('='*50) 
    
    