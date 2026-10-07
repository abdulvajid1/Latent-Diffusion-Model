from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AutoencoderKLConfig


class AttentionLayer(nn.Module):
    def __init__(self, config: AutoencoderKLConfig):
        super().__init__()
        self.is_cross_attn = config.is_cross_attn
        self.dropout = config.attn_dropout_p
        self.emb_dim = config.emb_dim
        self.head_dim = config.head_dim
        assert self.emb_dim % self.head_dim == 0, "Embedding size should be divisible by head_dim"
        self.num_heads = self.emb_dim//self.head_dim
        
        self.enable_gqa = config.enable_gqa
        self.num_kv_heads = config.num_kv_heads if self.enable_gqa else self.num_heads
        self.kv_emb_dim = self.num_kv_heads * self.head_dim
        
        if self.enable_gqa is None:
            assert self.kv_emb_dim == self.emb_dim, "if gqa"
        self.training = config.is_training
        
        
        self.qw = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.kw = nn.Linear(self.emb_dim, self.kv_emb_dim, bias=False)
        self.vw = nn.Linear(self.emb_dim, self.kv_emb_dim, bias=False)
               
        self.ow = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
    
    def forward(self, x, cross_emb=None, mask=None):
        batch_size, seq_len, emb_dim = x.shape
        num_heads = emb_dim // self.head_dim
        
        q = self.qw(x).reshape(batch_size, seq_len, num_heads, self.head_dim).transpose(-2, -3)
        
        if cross_emb is not None:
            cross_seq_len = cross_emb.shape[1]
            k = self.kw(cross_emb).reshape(batch_size, cross_seq_len, self.num_kv_heads, self.head_dim).transpose(-2, -3)
            v = self.vw(cross_emb).reshape(batch_size, cross_seq_len, self.num_kv_heads, self.head_dim).transpose(-2, -3)
        else:
            print(self.kw(x).shape)
            k = self.kw(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(-2, -3)
            v = self.vw(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(-2, -3)
        
        attn_emb = F.scaled_dot_product_attention(q, k, v, 
                                                  attn_mask=mask, 
                                                  dropout_p=(self.dropout if self.training else 0.0), 
                                                  enable_gqa=self.enable_gqa)
        
        attn_emb = attn_emb.transpose(-2, -3).contiguous().view(batch_size, seq_len, num_heads * self.head_dim)
        return self.ow(attn_emb)
    

class ResidualAttentionLayer(nn.Module):
    def __init__(self, use_attention: bool = False) -> None:
        super().__init__()
    
    def forward(self, x):
        pass
    

class UpsampleLayer(nn.Module):
    def __init__(self, in_channel, with_conv):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding='same') \
            if with_conv else nn.Identity()
            )
    
    def forward(self, x):
        return self.model(x)

class DownsampleLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        return self.pool(x)
    
    


if __name__ == "__main__":
    print("Sampling input: 64x64")
    print(DownsampleLayer()(torch.randn(1, 1, 64, 64)).shape)
    print(UpsampleLayer(in_channel=1, with_conv=False)(torch.randn(1, 1, 64, 64)).shape)
    print(UpsampleLayer(in_channel=1, with_conv=True)(torch.randn(1, 1, 64, 64)).shape)
    
    config = AutoencoderKLConfig(emb_dim=64, 
                                 is_cross_attn=False, 
                                 attn_dropout_p=0.0, 
                                 is_training=True, 
                                 num_kv_heads=2, 
                                 enable_gqa=False)
    
    print('attention input: (2, 10, 64)')
    print("attention working", AttentionLayer(config)(torch.randn(2, 10, 64)).shape)