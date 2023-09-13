from stat_reliability_measure.config import Config
from stat_reliability_measure.dev.utils import valid_pars_type
from stat_reliability_measure.dev.smc.smc_utils import nextBetaESS,nextBetaSimpESS,SimpAdaptBetaPyt

class HMLS_Config(Config):

    default_dict={"config_name":"HMLS",
        "method_name":'HMLS',
        'requires_V':True,
        'requires_gradV':True,
        'requires_gen':True,
        'N_range': [],
        'N':100,
        'T_range':[],
        'T':1,
        'ratio_range':[],
        'ratio':0.75,
        'alpha_range':[],
        'alpha':0.25,
        'L_range':[],
        'L':1,
        'GV_opt':False,
        

     }
    
    default_dict.update({
    'n_max':10000,
    'track_accept':True,
    'track_calls':True,
    'track_levels':False    ,
    'track_X':False,
    'track_advs':False,
    'adapt_dt':False,
    
    'target_accept':0.574,
    'accept_spread':0.1,
    'dt_decay':0.999,
    'dt_gain':-1.,
    'dt_min':1e-5,
    'dt_max':0.7,
    'gaussian':True,
    
    'only_duplicated':True,
   
    'track_levels':True,

    'kappa_opt':True,
    'adapt_step':True,
    'FT':True,
    'sig_dt':0.02,
    'L_min':1,
    'skip_mh':False,
   })

        
    
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
        if len(self.ratio_range)==0:
            self.ratio_range= [self.ratio]
        if len(self.N_range)==0:
            self.N_range= [self.N]
        if len(self.T_range)==0:
            self.T_range= [self.T]
        if len(self.L_range)==0:
            self.L_range= [self.L]
        if len(self.alpha_range)==0:
            self.alpha_range= [self.alpha]
        if self.dt_gain==-1:
            self.dt_gain=1/self.dt_decay

    