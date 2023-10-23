""""Script to configure the Importance Sampling (IS) method. """
from stat_reliability_measure.config import SamplerConfig
import numpy as np
import torch

class IS_Config(SamplerConfig):
    default_dict = { 'config_name': "ISconfig",
        'method_name': "IS",

        'requires_gen':True,
        'requires_h':True,
        'N':1000,
        'N_range':[],
        'batch_size':100,
        'batch_size_range':[],
        'track_advs':False,
        'alpha_CI':0.05,
        'sigma_bias':1.,
        'zero_latent':None,
        'requires_gradG':True,
        'requires_G':True,
        'requires_model':True,
        'requires_x_clean':True,
        'search_method':'mpp_search',

        
        
       }
  