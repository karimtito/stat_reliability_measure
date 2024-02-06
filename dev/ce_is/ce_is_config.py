""""Script to configure the Importance Sampling (IS) method. """
from stat_reliability_measure.config import SamplerConfig


class CE_IS_Config(SamplerConfig):
    default_dict = { 'config_name': "CE-ISconfig",
        'method_name': "CE-IS",
        'rho':0.5,
        'rho_range':[],
        't_max':100,
        'ce_masri':False,
        'theta_0':None,
        'save_theta':False,
        'save_sigma':False,
        'save_thetas':False,
        'save_sigmas':False,
        'save_rare':False,
        'estimate_var':False,
        'estimate_covar':False,
        'requires_gen':True,
        'requires_h':True,
        'requires_gradG':False,
        'requires_G':False,
        'requires_model':False,
        'requires_t_transform':True,
        'requires_u_mpp':True,
        'N':int(1E4),
        'N_ce':int(1E4),
        'N_range':[],
        'batch_size':int(1E3),
        'batch_size_range':[],
        'track_advs':False,
        'alpha_CI':0.05,
        'sigma_bias':1.,
        'zero_latent':None,
        'search_method':'mpp_search',
        'stepsize':1E-2,
        'stepsize_range':[],
        'num_iter':20,
        'num_iter_range':[],
        'random_init':False,
        'sigma_init':0.1,
        'steps':100,
        'gamma':0.5,
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
        if len(self.rho_range)==0:
            self.rho_range=[self.rho]

        return 
  