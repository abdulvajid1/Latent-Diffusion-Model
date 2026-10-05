import torch
import torch.nn as nn
import torch.nn.functional as F


class NLDiscriminator(nn.Module):
    def __init__(self, 
                 img_channels, 
                 start_dim, 
                 kernel_size=4, 
                 depth=3,
                 padding=1,
                 leaky_relu_slop=0.2):
        super().__init__()
        
        
        curr_out_channel = start_dim
        self.model = nn.ModuleList()
        self.model.append(
            nn.Conv2d(in_channels=img_channels, out_channels=curr_out_channel, kernel_size=kernel_size, stride=2, padding=padding)
        )
        self.model.append(
            nn.LeakyReLU(leaky_relu_slop)
        )
        
        for i in range(depth):
            
            stride = 1 if i == depth - 1 else 2
            self.model.append(
                nn.Conv2d(in_channels=curr_out_channel, out_channels=curr_out_channel*2, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            self.model.append(
                nn.BatchNorm2d(num_features=curr_out_channel*2)
            )
            self.model.append(
                nn.LeakyReLU(0.2)
            )
            curr_out_channel *= 2
        
        self.model.append(
            nn.Conv2d(in_channels=curr_out_channel, out_channels=1, kernel_size=kernel_size, stride=1, padding=padding)
        )
        
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x)
    
    
if __name__ == "__main__":
    model = NLDiscriminator(img_channels=3, start_dim=64)
    x = torch.randn(1, 3, 128, 128)
    print(model(x).shape)
    