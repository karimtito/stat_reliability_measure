""""Script to configure the Importance Sampling (IS) method. """
from stat_reliability_measure.config import SamplerConfig
import numpy as np
import torch

class IS_CONFIG(SamplerConfig):
    default_dict = { 'config_name': "IS",
        'method_name': "IS",
        'requires_gen':True,
        'requires_h':True,
        'N':1000,
        'N_range':[],
        'batch_size':100,
        'batch_size_range':[],
        'track_advs':False,
        
       }
  