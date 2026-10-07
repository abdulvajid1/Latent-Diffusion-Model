from dataclasses import dataclass, asdict
from typing import Literal



@dataclass
class AutoencoderKLConfig:
    
    # original_resolution / factor = latent_resolutoin
    factor: Literal[2, 4, 8, 16, 32] = 8
    img_channel: int = 3
    patchgan_start_dim: int = 64
    
    # attention config
    is_cross_attn: bool = False
    attn_dropout_p: float = 0.1
    is_training: bool = True
    emb_dim: int = 64
    num_kv_heads: int = 2
    enable_gqa: bool = True
    head_dim: int = 16
    

@dataclass
class LpipsConfig:
    pretrained_model: Literal['vgg', 'resnet', 'alexnet'] = 'vgg'
    feature_extract_layers = (3, 8, 13, 20, 25)
    use_dropout: bool = False
    p_loss_weight: float = 1.
    
    
    
    