from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AutoencoderKLConfig

def img2seq(x):
    batch, channels, height, width = x.shape
    return x.view(batch, channels, height*width).transpose(-1, -2) # b, c, h*w => b, h*w , c.

def seq2img(x):
    print(x.shape)
    batch, seq, hidden_dim = x.shape
    height, width = int(seq**0.5), int(seq**0.5) # 5 * 5 = 25, sqrt(25) == 5
    print(height, width)
    return x.transpose(-1, -2).reshape(batch, hidden_dim, height, width)    

def _check_input_type(x):
    if len(x.shape) == 4:
        return "image"
    else:
        return "seq"
    


class AttentionLayer(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 num_heads,
                 num_groups_groupnorm=None,
                 residual_connection=None,
                 cross_embedding_dim=None, 
                 dropout=0.0,
                 return_dim='1D'):
        
        super().__init__()
        self.residual_connection=residual_connection
        self.return_dim = return_dim
        self.cross_embedding_dim = cross_embedding_dim
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_groups_groupnorm = num_groups_groupnorm
        
        assert self.embedding_dim % self.num_heads == 0, "Embedding size should be divisible by num_heads"
        self.head_dim = self.embedding_dim//self.num_heads
        print(f"Head dim will be :{self.head_dim}")
        
        self.enable_gqa = config.enable_gqa
        self.kv_emb_dim = cross_embedding_dim if cross_embedding_dim else embedding_dim
            
        if num_groups_groupnorm is not None:
            self.groupnorm = nn.GroupNorm(num_channels=self.embedding_dim, num_groups=self.num_groups_groupnorm) # pyright: ignore[reportArgumentType]
            
        
        
        self.qw = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.kw = nn.Linear(self.kv_emb_dim, self.embedding_dim, bias=False)
        self.vw = nn.Linear(self.kv_emb_dim, self.embedding_dim, bias=False)
               
        self.ow = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
    def forward_cross_emb(self, x, cross_emb, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        _, kv_seq_len, kv_embed_dim = cross_emb.shape
        q = self.qw(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)
        
        # may add gqa, so specific kv_head_num, but for now going with same as query num heads not gqa
        k = self.kw(cross_emb).reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(-2, -3)
        v = self.vw(cross_emb).reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(-2, -3)

        attn_emb = F.scaled_dot_product_attention(q, k, v, 
                                                  attn_mask=mask, 
                                                  dropout_p=(self.dropout if self.training else 0.0), 
                                                #   enable_gqa=self.enable_gqa
                                                  )
        attn_emb = attn_emb.transpose(-2, -3).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        out = self.ow(attn_emb)
        
        return out

    def forward_self(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        q = self.qw(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)
        k = self.kw(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)
        v = self.vw(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)
        attn_emb = F.scaled_dot_product_attention(q, k, v, 
                                                   attn_mask=mask, 
                                                   dropout_p=(self.dropout if self.training else 0.0), 
                                                   enable_gqa=self.enable_gqa)
         
        attn_emb = attn_emb.transpose(-2, -3).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        out = self.ow(attn_emb)
        
        return out

        
        
    def forward(self, x, cross_emb=None, mask=None):
        
        residual = x 
        
        if self.num_groups_groupnorm is not None:
            x = self.groupnorm(x)
        
        if _check_input_type(x) == 'image':    
            x = img2seq(x)
            residual = img2seq(residual)
        
        # attention    
        if cross_emb is not None:
            out = self.forward_cross_emb(x, cross_emb, mask)
        else:
            out = self.forward_self(x, mask)
        
        if self.residual_connection:
            out = out + residual

        if self.return_dim == "2D":
            out = seq2img(out)
        
        return out
    

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
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                              kernel_size=3, padding=1, stride=2)
    def forward(self, x):
        return self.down(x)
    
    


if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)
    print(f"Downsampling from {x.shape} =>", DownsampleLayer(in_channels=3)(x).shape)
    print(f"UPsampling from {x.shape} =>",UpsampleLayer(in_channel=3, with_conv=False)(x).shape)
    print(f"UPsampling from {x.shape} =>",UpsampleLayer(in_channel=3, with_conv=True)(x).shape)
    
    config = AutoencoderKLConfig(emb_dim=64, 
                                 is_cross_attn=False, 
                                 attn_dropout_p=0.0, 
                                 is_training=True, 
                                 num_kv_heads=2, 
                                 enable_gqa=False)
    
    x = torch.randn(1, 16, 32, 32)
    cross = torch.randn(1, 10, 16)
    print(f'Attention input Size: {x.shape}')
    print("Attention working", AttentionLayer(embedding_dim=16, num_heads=4, return_dim='2D', num_groups_groupnorm=2)(x).shape)
    print("Cross attention working", AttentionLayer(embedding_dim=16, cross_embedding_dim=16, num_heads=4, return_dim='1D', residual_connection=True)(x, cross).shape)