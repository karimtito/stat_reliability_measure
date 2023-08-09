from stat_reliability_measure.config import Config
from stat_reliability_measure.dev.utils import valid_pars_type

class FORM_config(Config):
    
    default_dict={"config_name":"FORM",
    "method_name":'FORM',
    'requires_V':False,
    'requires_model':True,
    "requires_x_clean":True,
    "requires_y_clean":True,
    "requires_epsilon":True,
    "requires_x_min":True,
    "requires_x_max":True,
    "requires_noise_dist":True,
    "requires_sigma":True,
    "steps":10,
    "steps_range" : [],
    "tol":1e-3,
    "num_iter":100,
    "num_iter_range":[] ,
    "stepsize":0.1,
    "stepsize_range":[],
    "random_init":False,
    "sigma_init":0.1,
    "search_method":'CarliniWagner'}
    def __init__(self,config_dict=default_dict):
        vars(self).update(config_dict)
        super().__init__()

    def add_parsargs(self,parser):
    
        for key in vars(self).keys():
            if ('_range' in key) or  ('_list' in key):
                # if the key is a range or list and the value is empty 
                # then the type is the type of the first element of the list
                if len(vars(self)[key])==0:
                    ptype,valid = valid_pars_type([vars(self)[key.replace('_range','').replace('_list','')]])
                else:
                    ptype,valid = valid_pars_type(vars(self)[key])
            elif 'pars' in key or 'config' in key:
                continue
            else:
                # else the type is the type of the default value
                ptype,valid = valid_pars_type(vars(self)[key])
            if valid:
                parser.add_argument('--'+key,type=ptype,default=vars(self)[key])
    
        return parser
    def update(self):
        if len(self.steps_range)==0:
            self.steps_range= [self.steps]
        if len(self.num_iter_range)==0:
            self.num_iter_range= [self.num_iter]
        if len(self.stepsize_range)==0:
            self.stepsize_range= [self.stepsize]
            

    

