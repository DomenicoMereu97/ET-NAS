from copy import deepcopy
from GenNAS.utils.tricks import *



import torch.nn as nn
import torch.nn.functional as F

from argparse import Namespace




class NB201Wrapper(nn.Module):
    """
    wrap the NB201 model for regression
    e.g. arch: [[0],[0,1],[0,1,2]] -> arch_str: '|none~0|+|none~0|skip_connect~1|+|none~0|skip_connect~1|nor_conv_1x1~2|'
    """
    def __init__(self, model, init_channels, last_channels, output_size = 8, num_labels = 120):
        super(NB201Wrapper, self).__init__()
        self.init_channels = init_channels
        self.last_channels = last_channels
        self.output_size = output_size
        self.num_labels = num_labels
        self.model = model
        self.out0 = nn.Sequential(nn.BatchNorm2d(init_channels),nn.Conv2d(init_channels,last_channels[0],1))
        self.out1 = nn.Sequential(nn.BatchNorm2d(init_channels*2),nn.Conv2d(init_channels*2,last_channels[1],1))
        self.out2 = nn.Sequential(nn.BatchNorm2d(init_channels*4),nn.Conv2d(init_channels*4,last_channels[2],1))
        self.stack_cell1 = nn.Sequential(*[model.cells[0] for i in range(5)])
        self.reduction1 = model.cells[5]
        self.stack_cell2 = nn.Sequential(*[model.cells[6] for i in range(5)])
        self.reduction2 = model.cells[11]
        self.stack_cell3 = nn.Sequential(*[model.cells[12] for i in range(5)])  
    def forward(self, x):
        x = self.model.stem(x)        
        x = self.stack_cell1(x)
        x0 = self.out0(F.interpolate((x),(self.output_size,self.output_size)))
        x = self.reduction1(x)
        x = self.stack_cell2(x)
        x1 = self.out1(F.interpolate((x),(self.output_size,self.output_size)))
        x = self.reduction2(x)
        x = self.stack_cell3(x)
        x2 = self.out2(F.interpolate((x),(self.output_size,self.output_size)))
        return [x0,x1,x2]


        