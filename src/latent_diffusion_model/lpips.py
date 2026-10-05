import torch
import torchvision
import torch.nn as nn 
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import alexnet, AlexNet_Weights

from .config import AutoencoderKLConfig, LpipsConfig


class LPIPS(nn.Module):
    def __init__(self, pretrained_model='vgg'):
        super().__init__()
        # Load pretrained model
        if pretrained_model == "vgg":
            self.pretrained_model = vgg16(VGG16_Weights).features
        elif pretrained_model == 'resnet':
            self.pretrained_model = resnet50(ResNet50_Weights).features
        else:
            self.pretrained_model = alexnet(AlexNet_Weights).features
        
        # disable gradient computation of pretrained model
        for param in self.pretrained_model.parameters(): #type: ignore
            param.requires_grad_(False)
        
        # slices the feature extractor lyaers
        self.feature_extract_layers = nn.ModuleDict()
        start_dim = 0
        for i, n in enumerate(config.feature_extract_layers):
            self.feature_extract_layers[f'section_{i}_feature_extractor'] = self.pretrained_model[start_dim : n] #type: ignore
            start_dim = n
    
        
        # project Layers
        project_layers_channels = [64, 128, 256, 512, 512]
        self.feature_project_layers = nn.ModuleDict()
        for i, in_chnl in enumerate(project_layers_channels):
            layers = [nn.Dropout2d()] if config.use_dropout else []
            layers += [
                nn.Conv2d(in_channels=in_chnl, out_channels=1, kernel_size=1, stride=1)
            ]
            
            self.feature_project_layers[f'section_{i}_feature_projector'] = nn.Sequential(*layers)
        
        # we remove any negative weight that destroy the distance measure
        for params in self.feature_project_layers.parameters():
            params.data = params.data.clamp(0)
        
        self.spatial_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.register_buffer('mean', (torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1) * 2) - 1)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1) * 2)
        
    def vgg_forward(self, x):
        extracted_features = []
        for key in self.feature_extract_layers.keys():
            x = self.feature_extract_layers[key](x)
            extracted_features.append(x)  
        return extracted_features

    def _unit_norm(self, x, eps=1e-8):
        return torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-8)
    
    def imagenet_normalize(self, x, eps=1e-8):
        return (x - self.mean) / (self.std + eps) # normalize on channels dim
    
    def forward(self, x0, x1):
        # normalize vgg style
        x0, x1 = self.imagenet_normalize(x0), self.imagenet_normalize(x1) 
        
        
        val = []
        # extract features
        x0_features = self.vgg_forward(x0)
        x1_features = self.vgg_forward(x1)
        
        for i, (x0, x1) in enumerate(zip(x0_features, x1_features)):
            
            # unit norm vgg fetures
            x0, x1 = self._unit_norm(x=x0), self._unit_norm(x=x1)
            
            # calc similarity/distance
            delta = (x0 - x1)**2
            
            # project layers
            delta = self.feature_project_layers[f"section_{i}_feature_projector"](delta)
            
            avg_delta = self.spatial_avg_pool(delta)
            
            val.append(avg_delta)    
             
        # print('avg', avg_distances)
        total_avg_distance = 0
        for i in val:
            total_avg_distance += i
    
        
        return total_avg_distance # shape = (b)

if __name__ == "__main__":
    x = torch.randn(2, 3, 64, 64)
    x1 = torch.randn(2, 3, 64, 64)
    config = LpipsConfig()
    lpips = LPIPS(config.pretrained_model, config.feature_extract_layers)
    print(lpips(x, x1).mean())