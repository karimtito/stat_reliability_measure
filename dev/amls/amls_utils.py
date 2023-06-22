from stat_reliability_measure.home import ROOT_DIR
import numpy as np
from stat_reliability_measure.config import SamplerConfig

class MLS_SMC_Config(SamplerConfig):
    default_dict = { 'method_name': "MLS_SMC",
                    'N_range':[32,64,128,256,512,1024],'N':40,'T':1,
                    'T_range':[1,10,20,50,100,200,500,1000],
                    'ratio_range':[0.1],
                    'verbose':0,'min_rate':0.51,'clip_s':True,'s_min':8e-3,'s_max':3,
                    'n_max':2000,'allow_zero_est':True,
                    
                    'ratio':0,
                    'n_max':2000,
                    's':1,
                    's_range': [1.], 
                    'n_max':2000,
                
                    'allow_zero_est':True,
                
                    'last_particle':False}
    def __init__(self, config_dict=default_dict, **kwargs):
        vars(self).update(config_dict)
        self.K=int(self.N*self.ratio)
        super().__init__(**kwargs)
    

   

    def update_config(self):
        if len(self.N_range)==0:
            self.N_range=[self.N]
        nb_runs*=len(self.N_range)
        if len(self.T_range)==0:
            self.T_range=[self.T]
        nb_runs*=len(self.T_range)
        if len(self.ratio_range)==0:
            self.ratio_range=[self.ratio]
        nb_runs*=len(self.ratio_range)
        if len(self.s_range)==0:
            self.s_range=[self.s]
        nb_runs*=len(self.s_range)
        
        return 