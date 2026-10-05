from dataclasses import dataclass, asdict
from typing import Literal



@dataclass
class AutoencoderKLConfig:
    
    # original_resolution / factor = latent_resolutoin
    factor: Literal[2, 4, 8, 16, 32] = 8
    img_channel: int = 3
    patchgan_start_dim: int = 64
    

@dataclass
class LpipsConfig:
    pretrained_model: Literal['vgg', 'resnet', 'alexnet'] = 'vgg'
    feature_extract_layers = (3, 8, 13, 20, 25)
    use_dropout: bool = False
    
    
    
    