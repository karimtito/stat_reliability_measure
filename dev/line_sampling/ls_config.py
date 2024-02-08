""""Script to configure the Importance Sampling (IS) method. """
from stat_reliability_measure.config import SamplerConfig


class LS_Config(SamplerConfig):
    default_dict = { 'config_name': "LSconfig",
        'method_name': "LS",
        
        'requires_gen':True,
        'requires_h':True,
        'requires_gradG':True,
        'requires_G':True,
        'requires_model':True,
        'requires_x_clean':True,
        'requires_y_clean':True,
        'requires_t_transform':True,
        'requires_u_mpp':True,
        'requires_num_classes':True,
        'N':int(1E4),
        'N_range':[],
        'batch_size':int(1E3),
        'batch_size_range':[],
        'save_rare':False,
        'alpha_CI':0.05,
        'default_params':True,
        'zero_latent':None,
        'save_mpp':False,
        'search_method':'hlrf',
        
        'stepsize':1E-2,
        'stepsize_range':[],
        'num_iter':20,
        'num_iter_range':[],
        'random_init':False,
        'sigma_init':0.1,
        'steps':100,
        'gamma':0.5,
        'real_mpp':True,
        
       }
    
    
    def __init__(self, config_dict=default_dict, **kwargs):
        vars(self).update(config_dict)
        super().__init__(**kwargs)
        
    def update(self):
        if len(self.N_range)==0:
            self.N_range=[self.N]
      
        if len(self.batch_size_range)==0:
            self.batch_size_range=[self.batch_size]
        if len(self.stepsize_range)==0:
            self.stepsize_range=[self.stepsize]
        if len(self.num_iter_range)==0:
            self.num_iter_range=[self.num_iter]   

        return 
  