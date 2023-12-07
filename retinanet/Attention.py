import torch
import torch.nn as nn
import torch.nn.functional as F
from .AdaIN import *
import fvcore.nn.weight_init as weight_init
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttentionBlock, self).__init__()
        
        self.g = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, stride=1, padding=0)
        for layer in [self.g, self.theta, self.phi, self.W]:
            weight_init.c2_xavier_fill(layer)
        
        self.adain = AdaIN_block()
        
    def forward(self, x):
        batch_size, C, height, width = x[0].size()

        g_x = self.g(x[0]).view(batch_size, -1, height * width).permute(0, 2, 1)

        theta_x = self.theta(x[1]).view(batch_size, -1, height * width)
        phi_x = self.phi(x[1]).view(batch_size, -1, height * width).permute(0, 2, 1)
        
        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x).view(batch_size, C // 8, height, width)
        
        W_y = self.W(y)
        z = self.adain(x[0], W_y)
        return z