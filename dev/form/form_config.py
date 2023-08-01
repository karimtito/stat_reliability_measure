from stat_reliability_measure.config import Config
class FORM_config(Config):
    
    default_dict={"config_name":"FORM",
    "method_name":'FORM',
    'requires_V':False,,
    'requires_model':True,

    "optim_steps":10
    "optim_steps_range" : []
    "tol":1e-3
    "max_iter":100
    "max_iter_range":[]
    random_init=False}
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
        if len(self.optim_steps_range)==0:
            self.optim_steps_range= [self.optim_steps]
        if len(self.max_iter_range)==0:
            self.max_iter_range= [self.max_iter]
            

    

