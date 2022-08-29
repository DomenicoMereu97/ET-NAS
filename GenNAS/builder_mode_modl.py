#from fbnas import NDS
#from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from GenNAS.model_wrapper import *
#from pycls.models.nas.genotypes import GENOTYPES, Genotype
import numpy as np

class ModelBuilder():
    def __init__(self, args):
        self.device = args.device
        self.output_size = args.output_size
        self.config = args.config
        self.last_channels = np.asarray([self.config['last_channel_l0']*args.last_channels, self.config['last_channel_l1']*args.last_channels, self.config['last_channel_l2']*args.last_channels]).astype(int)
        
        if args.init_channels:
            self.init_channels = args.init_channels

        
    def get_model_wrapped(self, model):
        model = NB201Wrapper(model, self.init_channels, self.last_channels, self.output_size)
        return model


        
