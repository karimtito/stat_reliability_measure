from stat_reliability_measure.home import ROOT_DIR
import numpy as np
from stat_reliability_measure.config import SamplerConfig

class MLS_SMC_Config(SamplerConfig):
    default_dict = { 'config_name': "AMLS",
        'method_name': "MLS_SMC",
        'requires_gen':True,
        'requires_h':True,
        'N_range':[],
        'N':40,
        'T_range':[],
        'T':1,
        'ratio_range':[],
        'ratio':0.1,
        'clip_s':True,'s_min':8e-3,'s_max':3,
        's':1.,
        'n_max':2000,
        'track_finish':False,
        'track_accept':False,
        'track_calls':True,
        'track_levels':False,
        'track_s':False,
        'track_dt':False,
        'track_advs':False,
        'track_X':False,
        's_range': [], 
        'n_max':2000,
        'last_particle':False}
    def __init__(self, config_dict=default_dict, **kwargs):
        vars(self).update(config_dict)
        super().__init__(**kwargs)
        
    

   

    def update(self):
        if len(self.N_range)==0:
            self.N_range=[self.N]
      
        if len(self.T_range)==0:
            self.T_range=[self.T]
        
        if len(self.ratio_range)==0:
            self.ratio_range=[self.ratio]
        
        if len(self.s_range)==0:
            self.s_range=[self.s]
        return 
    
    