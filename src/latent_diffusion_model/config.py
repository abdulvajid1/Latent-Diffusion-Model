from dataclasses import dataclass, asdict
from typing import Literal



@dataclass
class AutoencoderKLConfig:
    
    # original_resolution / factor = latent_resolutoin
    factor: Literal[2, 4, 8, 16, 32] = 8
    
    
    