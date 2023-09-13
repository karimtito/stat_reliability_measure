from stat_reliability_measure.config import SamplerConfig

class CrudeMC_Config(SamplerConfig):
    default_dict = { 'config_name': "MC",
        'method_name': "CrudeMC",
        'requires_gen':True,
        'requires_h':True,
        'N':1000,
        'N_range':[],
        'batch_size':100,
        'batch_size_range':[],
        'track_advs':False
       }
    def __init__(self, config_dict=default_dict, **kwargs):
        vars(self).update(config_dict)
        super().__init__(**kwargs)
        
    

   

    def update(self):
        if len(self.N_range)==0:
            self.N_range=[self.N]
      
        if len(self.batch_size_range)==0:
            self.batch_size_range=[self.batch_size]
        return 
