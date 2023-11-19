""""Script to configure the Importance Sampling (IS) method. """
from stat_reliability_measure.config import SamplerConfig


class IS_Config(SamplerConfig):
    default_dict = { 'config_name': "ISconfig",
        'method_name': "IS",

        'requires_gen':True,
        'requires_h':True,
        'N':1000,
        'N_range':[],
        'batch_size':1000,
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
        'requires_normal_cdf_layer':True,
        'requires_low':False,
        'requires_high':False,
        

       }
    
    
    def __init__(self, config_dict=default_dict, **kwargs):
        vars(self).update(config_dict)
        super().__init__(**kwargs)
        
    def update(self):
        if len(self.N_range)==0:
            self.N_range=[self.N]
      
        if len(self.batch_size_range)==0:
            self.batch_size_range=[self.T]

        return 
  