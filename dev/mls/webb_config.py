from stat_reliability_measure.home import ROOT_DIR
import numpy as np
from stat_reliability_measure.config import SamplerConfig

class MLS_Webb_Config(SamplerConfig):
    default_dict = { 'config_name': "MLS_Webb",
        'method_name': "MLS_Webb",
        'requires_score':True,
        'requires_x_clean':True,
        'requires_epsilon':True, 

        'N_range':[32,64,128,256,512,1024],
        'N':40,
        'T':1,
        'T_range':[1,10,20,50,100,200,500,1000],
        'ratio_range':[0.1],
        'ratio':0.1,
        
        'save_x':False,
        'track_X':False,
      
        'save_levels':True,
        'track_finish':False,
        'track_accept':False,
            }
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
        
        return 
    
    
        
