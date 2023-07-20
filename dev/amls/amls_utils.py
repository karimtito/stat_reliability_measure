from stat_reliability_measure.home import ROOT_DIR
import numpy as np
from stat_reliability_measure.config import SamplerConfig

class MLS_SMC_Config(SamplerConfig):
    default_dict = { 'config_name': "AMLS",
        'method_name': "MLS_SMC",

                    'N_range':[32,64,128,256,512,1024],'N':40,'T':1,
                    'T_range':[1,10,20,50,100,200,500,1000],
                    'ratio_range':[0.1],
                    'verbose':0,'min_rate':0.51,'clip_s':True,'s_min':8e-3,'s_max':3,
                    'n_max':2000,'allow_zero_est':True,
                    'ratio':0.1,
                    'n_max':2000,
                    's':1,'track_finish':False,
                    'track_accept':False,
                    's_range': [1.], 
                    'n_max':2000,
                    'allow_zero_est':True,
                    'last_particle':False}
    def __init__(self, config_dict=default_dict, **kwargs):
        vars(self).update(config_dict)
        self.K=int(self.N*self.ratio)
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
        self.K=int(self.N*self.ratio)
        return 
    
    