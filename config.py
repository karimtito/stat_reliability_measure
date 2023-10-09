from stat_reliability_measure.home import ROOT_DIR
from pathlib import Path
from stat_reliability_measure.dev.utils import CustomEncoder
from stat_reliability_measure.dev.utils import valid_pars_type,simple_vars
import stat_reliability_measure.dev.torch_utils as t_u
import argparse
import git
import os
from time import time
from datetime import datetime
import numpy as np
import torch
import json
from stat_reliability_measure.dev.utils import float_to_file_float

model_dir = ROOT_DIR+'/models/'

class Config:
    def __init__(self,config_dict={}):
        self.config_name='default'
        if len(config_dict)> 0:
            self.__dict__.update(config_dict)
        config_path=f'{self.config_name}.json'
        
        self.build_parser()
    def __str__(self):
        str_ = str("config(")

        l = list(simple_vars(self).keys())
        l.sort()
        for key in l:
            if 'parser' in key:
                continue
            if isinstance(vars(self)[key],dict):
                continue
            if isinstance(vars(self)[key],torch.Tensor):
                continue
            if isinstance(vars(self)[key],np.ndarray) or isinstance(vars(self)[key],list):
                if len(vars(self)[key])>10:
                    continue
            elif type(vars(self)[key]) in [float,np.float32,np.float64]:
                str_+=f" {key}={float_to_file_float(vars(self)[key])},"
            else:
                str_+=f" {key}={vars(self)[key]},"
        str_+=")"
        return str_
    def __repr__(self):
        str_ = str("config(")

        l = list(simple_vars(self).keys())
        l.sort()
        for key in l:
            if 'parser' in key:
                continue
            if isinstance(vars(self)[key],dict):
                continue
            elif type(vars(self)[key]) in [float,np.float32,np.float64]:
                str_+=f" {key}={float_to_file_float(vars(self)[key])},"
            else:
                str_+=f" {key}={vars(self)[key]},"
        str_+=")"
        return str_
    def get_parser(self):
        return self.parser
    def build_parser(self,description='configuration'):
        parser = argparse.ArgumentParser(description=description)
        for key in vars(self).keys():
            if ('_range' in key) or  ('_list' in key):
                # if the key is a range or list and the value is empty 
                # then the type is the type of the first element of the list
                if len(vars(self)[key])==0:
                    place_holder_list=[vars(self)[key.replace('_range','').replace('_list','')]]
                    ptype,valid = valid_pars_type(place_holder_list)
                else:
                    ptype,valid = valid_pars_type(vars(self)[key])
            else:
                # else the type is the type of the default value
                ptype,valid= valid_pars_type(vars(self)[key])
            if valid:
                parser.add_argument('--'+key,type=ptype,default=vars(self)[key])
        self.parser = parser
    
    def print_config(self):
 
        print(f"{self.config_name} configuration: /n {self}")

    def to_json(self,path=None):
        if path is not None:
            self.config_path=path
        with open(self.config_path,'w') as f:
            f.write(json.dumps(simple_vars(self), indent = 4, cls=CustomEncoder))


class ExpConfig(Config):
    default_dict={'config_name':'Experiment','torch_seed':-1,'np_seed':-1,
                  'verbose':0,'aggr_res_path':'',
                  'update_aggr_res':True,'save_config':False,
                  'print_config_opt':True,'track_finish':False,
                  'track_gpu':False,'allow_multi_gpu':True,
                  'track_cpu':False,'save_img':False,
                  'save_text':False,'device':'',
                  'tqdm_opt':True,'clip_min':0.,
                  'clip_max':1.,'force_train':False,
                  'repeat_exp':False,'data_dir':ROOT_DIR+"/data",
                  'noise_scale':1.,'exp_name':'',
                  'notebook':False,'device':''}

    def __init__(self,config_dict=default_dict):   
        vars(self).update(config_dict)
        super().__init__()
        if self.torch_seed==-1:
            """if the torch seed is not set, then it is set to the current time"""
            self.torch_seed=int(time())
        torch.manual_seed(seed=self.torch_seed)
        if len(self.device)==0:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if self.verbose>=5:
                print(self.device)
        if 'cuda' in self.device: 
            torch.cuda.manual_seed_all(seed=self.torch_seed)

        if self.np_seed==-1:
            """if the numpy seed is not set, then it is set to the current time"""
            self.np_seed=int(time())
        np.random.seed(seed=self.np_seed)
        # if not self.allow_multi_gpu:
        #     os.environ["CUDA_VISIBLE_DEVICES"]="0"
        if self.track_gpu:
            import GPUtil
            gpus=GPUtil.getGPUs()
            if len(gpus)>1:
                print("Multi gpus detected, only the first GPU will be tracked.")
            self.gpu_name=gpus[0].name

        if self.track_cpu:
            import cpuinfo
            self.cpu_name=cpuinfo.get_cpu_info()[[key for key in cpuinfo.get_cpu_info().keys() if 'brand' in key][0]]
            self.cores_number=os.cpu_count()
        
    
        

score_functions = {'linear':t_u.build_h_lin,'quadratic':t_u.build_h_quad,'cosinus':t_u.build_h_cos,}
grad_score_functions = {'linear':t_u.build_gradh_lin,'quadratic':t_u.build_gradh_quad,'cosinus':t_u.build_gradh_cos,}
V_functions = {'linear':t_u.build_V_lin,'quadratic':t_u.build_V_quad,'cosinus':t_u.build_V_cos,}
gradV_functions = {'linear':t_u.build_gradV_lin,'quadratic':t_u.build_gradV_quad,'cosinus':t_u.build_gradV_cos,}

class ExpToyConfig(ExpConfig):
    default_dict={'config_name':'ToyExperimentConfig','dim':2,
                'log_dir':ROOT_DIR+"/logs/toy_exp",'n_rep':100,
                'noise_dist':'gaussian','noise_dist':'gaussian',
                'p_target_range':[],'p_target':1e-3,
                'score_name':'linear'}
    #p_ref_compute = False
    def __init__(self,config_dict=default_dict,p_target_range=[],score_name='',
                aggr_res_path='',verbose=0.,method_name='',**kwargs):
        super().__init__()
        vars(self).update(config_dict)
        if len(score_name)>0:
            self.score_name=score_name
        self.p_target_range=p_target_range
        self.aggr_res_path=aggr_res_path
        self.verbose=verbose
        vars(self).update(kwargs)
        self.method_name = method_name
        self.commit= git.Repo(path=ROOT_DIR).head.commit.hexsha
        if len(self.p_target_range)==0:
            if hasattr(self,'p_target'):
                self.p_target_range=[self.p_target]
            else:
                self.p_target_range=[1e-3]
        else:
            self.p_target_range = [float(p) for p in self.p_target_range]
     
        

        if len(self.log_dir)==0:
            self.log_dir=os.path.join(ROOT_DIR+'/logs','toy_exp_'+self.score_name)
        if not os.path.exists(ROOT_DIR+'/logs'):
            os.mkdir(ROOT_DIR+'/logs')  
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if len(self.aggr_res_path)==0:
            self.aggr_res_path=os.path.join(self.log_dir,'aggr_res.csv')
       
        self.raw_logs = os.path.join(self.log_dir,'raw_logs/')
        if not os.path.exists(self.raw_logs):
            os.mkdir(self.raw_logs)
        raw_logs_path=os.path.join(self.log_dir,'raw_logs/'+self.method_name)
        if not os.path.exists(raw_logs_path):
            os.mkdir(raw_logs_path)
        self.raw_logs_path=raw_logs_path
        self.loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
        
        
        self.d= self.dim
        
        if self.noise_dist=='gaussian':
            self.gen= lambda N: torch.randn(size=(N,self.d),device=torch.device(self.device))
        elif self.noise_dist=='uniform':
            self.gen = lambda N: (2*torch.rand(size=(N,self.d), device=torch.device(self.device) )-1)
           
        self.normal_dist = torch.distributions.normal.Normal(loc=0.,scale=1.)
        self.x_clean = torch.zeros(size=(self.dim,),device=torch.device(self.device))
        self.input_shape = (self.dim,)

    def update(self):
        """ update the score and V functions """
        log_name=self.method_name+'_'+self.loc_time
        exp_log_path=os.path.join(self.raw_logs_path,log_name)
        log_path_prop = exp_log_path
        k=1
        index_level=0
        while os.path.exists(log_path_prop):
            if k==10: 
                exp_log_path= log_path_prop.replace('9','1')
                k=1
                index_level+=1
            log_path_prop = exp_log_path+'_'+str(k) 
            if index_level >5:
                raise RuntimeError("Error the log naming system has failed")
            k+=1 
        self.exp_log_path =  log_path_prop                  
        os.mkdir(path=exp_log_path)
        self.V,_ = V_functions[self.score_name](p_target=self.p_target, device=self.device, dim=self.dim, verbose=self.verbose)
        self.gradV,_ = gradV_functions[self.score_name](p_target=self.p_target, device=self.device, dim=self.dim, verbose=self.verbose)
        self.h,self.thresh = score_functions[self.score_name](p_target=self.p_target, device=self.device, dim=self.dim, verbose=self.verbose)
        self.grad_h = grad_score_functions[self.score_name](p_target=self.p_target, device=self.device, dim=self.dim, verbose=self.verbose)
        return 
    
    

    
class ExpModelConfig(ExpConfig):
    default_dict={'config_name':'ModelExperimentConfig','dataset':'',
                  'data_dir':ROOT_DIR+"/data",'model':None,
                'log_dir':"",'low':None,'high':None,
                'model_arch':'','model_dir':'','epsilon_range':[],'eps_max':0.3,'eps_min':0.2,
                'eps_num':5,'epsilon':-1.,'input_start':0,'input_stop':-1,'n_rep':100,'model_path':'',
                'export_to_onnx':False,'use_attack':False,'attack':'PGD','X':None,'y':None,
                'lirpa_bounds':False,'download':True,'train_model_epochs':10,'method_name':'',
                'from_gaussian':True,'noise_dist':'uniform','x_min':None,'mask_opt':False,
                'sigma':1.,'x_max':None,'x_mean':None,'x_std':1.,'lirpa_cert':False,'robust_model':False,
                'robust_eps':0.1,'load_batch_size':128,'nb_epochs': 15,'adversarial_every':1,}
    #p_ref_compute = False
    def __init__(self,config_dict=default_dict,X=None,y=None,model=None,epsilon_range=[],
                dataset_name='mnist',aggr_res_path='',model_name='',verbose=0.,
                method_name='',x_mean=None,x_std=None,**kwargs):
        super().__init__()
        vars(self).update(config_dict)
        self.X = X
        self.y = y
        self.model = model
        self.epsilon_range=epsilon_range
        self.dataset=dataset_name
        self.aggr_res_path=aggr_res_path
        self.verbose=verbose
        vars(self).update(kwargs)
        self.normal_dist = torch.distributions.normal.Normal(loc=0.,scale=1.)
        self.commit= git.Repo(path=ROOT_DIR).head.commit.hexsha
        if len(self.model_dir)==0:
            self.model_dir=os.path.join(ROOT_DIR+"/models/",self.dataset)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        #color_dataset=self.dataset in ('cifar10','cifar100','imagenet')
        #prblm_str=self.dataset
        if self.input_stop==-1:
            self.input_stop=self.input_start+1
        else:
            assert self.input_start<self.input_stop,"/!\ input start must be strictly lower than input stop"
        if len(self.noise_dist)==0:
            self.noise_dist=self.noise_dist.lower()
        if self.noise_dist not in ['uniform','gaussian']:
            raise NotImplementedError("Only uniform and Gaussian distributions are implemented.")
        # if not self.allow_multi_gpu:
        #     os.environ["CUDA_VISIBLE_DEVICES"]="0"
        if len(self.log_dir)==0:
            self.log_dir=os.path.join(ROOT_DIR+'/logs','exp_model_'+self.dataset)
        if not os.path.exists(ROOT_DIR+'/logs'):
            os.mkdir(ROOT_DIR+'/logs')  
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if len(self.aggr_res_path)==0:
            self.aggr_res_path=os.path.join(self.log_dir,'aggr_res.csv')
       
        
        self.raw_logs = os.path.join(self.log_dir,'raw_logs/')
        if not os.path.exists(self.raw_logs):
            os.mkdir(self.raw_logs)
        
        
        
        if len(self.epsilon_range)==0:
            log_min,log_max=np.log(self.eps_min),np.log(self.eps_max)
            log_line=np.linspace(start=log_min,stop=log_max,num=self.eps_num)
            self.epsilon_range=np.exp(log_line)
        else:
            self.eps_num=len(self.epsilon_range)
            self.epsilon_range = [float(eps) for eps in self.epsilon_range]
            self.eps_min,self.eps_max=np.min(self.epsilon_range),np.max(self.epsilon_range)
        if self.epsilon==-1.:
            self.epsilon = self.epsilon_range[0]
        standard_dataset = self.dataset in t_u.supported_datasets
        data_provided = (self.X is not None) and self.y is not None 
        model_provided = hasattr(self,"model") and self.model is not None
        assert standard_dataset or (data_provided and model_provided)
        if standard_dataset:
            test_loader = t_u.get_loader(train=False,data_dir=self.data_dir,download=self.download
                ,dataset=self.dataset,batch_size=self.load_batch_size,
                    x_mean=None,x_std=None)
            if not model_provided:
                if len(self.model_arch)==0:
                    self.model_arch = t_u.datasets_default_arch[self.dataset]
                self.model, self.model_shape,self.model_name=t_u.get_model(self.model_arch, robust_model=self.robust_model, robust_eps=self.robust_eps,
                    nb_epochs=self.nb_epochs,model_dir=self.model_dir,data_dir=self.data_dir,test_loader=test_loader,device=self.device,
                    download=self.download,dataset=self.dataset, force_train=self.force_train,
                )
        
            if not data_provided:
                self.X,self.y,self.sample_accuracy=t_u.get_correct_x_y(data_loader=test_loader,device=self.device,model=self.model)
            else:
                self.sample_accuracy=t_u.get_model_accuracy(model=self.model,X=self.X,y=self.y)
            self.num_classes=t_u.datasets_num_c[self.dataset.lower()]
            self.x_mean=t_u.datasets_means[self.dataset]
            self.x_std=t_u.datasets_stds[self.dataset]
                
            
                

        else:
            
            """compute the accuracy of the model on the sample batch"""
            self.sample_accuracy=t_u.get_model_accuracy(model=self.model,X=self.X[self.input_start:self.input_stop],y=self.y[self.input_start:self.input_stop])
        if (not hasattr(self,"model_name")) or self.model_name=='':
                self.model_name = self.dataset + "_model"
        if (not hasattr(self,"model_arch")) or self.model_arch=='':
            self.model_arch = 'custom'
        self.d= np.prod(self.X.shape[1:])
        if self.verbose >=1:
            print(f"model accuracy on sample batch:{self.sample_accuracy}")
        if self.use_attack:
            import foolbox as fb
            fmodel = fb.PyTorchModel(self.model, bounds=(0,1),device=self.device)
            attack = fb.attacks.LinfPGD()
            #un-normalize data before performing attack
            _, advs, self.attack_success = attack(fmodel, self.X[self.input_start:self.input_stop], 
            self.y[self.input_start:self.input_stop], epsilon_range=self.epsilon_range)
        inp_indices=np.arange(start=self.input_start,stop=self.input_stop)
        if self.verbose>=5:
            print (f"input indices: {inp_indices}")
        self.nb_inputs=len(inp_indices)
        
        if self.from_gaussian:
            self.gen= lambda N: torch.randn(size=(N,self.d),device=torch.device(self.device))
        else:
            self.gen = lambda N: (2*torch.rand(size=(N,self.d), device=torch.device(self.device) )-1)
        self.x_clean=self.X[0]
        self.input_shape = self.x_clean.shape
        self.y_clean=self.y[0]

    def update(self, method_name=''):
        """ """
        self.method_name = method_name
        raw_logs_path=os.path.join(self.log_dir,'raw_logs/'+method_name)
        if not os.path.exists(raw_logs_path):
            os.mkdir(raw_logs_path)
        self.raw_logs_path = raw_logs_path
        self.loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
        self.log_name=method_name+'_'+self.loc_time
        exp_log_path=os.path.join(self.raw_logs_path,self.log_name)
        log_path_prop = exp_log_path
        k=1
        index_level=0
        while os.path.exists(log_path_prop):
            if k==10: 
                exp_log_path= log_path_prop.replace('9','1')
                k=1
                index_level+=1
            log_path_prop = exp_log_path+'_'+str(k) 
            if index_level >5:
                raise RuntimeError("Error the log naming system has failed")
            k+=1 

        self.exp_log_path =  log_path_prop                  
        
        os.mkdir(path=self.exp_log_path)
        if self.x_min is not None:
            if not isinstance(self.x_min,torch.Tensor):
                self.x_min = self.x_min*torch.ones_like(self.x_clean)
            if self.low is None:
                self.low=torch.maximum(self.x_clean-self.epsilon, self.x_min.view(-1,*self.input_shape).to(self.device))
        elif self.low is None:
            self.low=torch.maximum(self.x_clean-self.epsilon, torch.zeros_like(self.x_clean))
        if self.x_max is not None:
            if not isinstance(self.x_max,torch.Tensor):
                self.x_max = self.x_max*torch.ones_like(self.x_clean)
            if self.high is None:
                self.high=torch.minimum(self.x_clean+self.epsilon, self.x_max.view(-1,*self.input_shape).to(self.device))
        elif self.high is None:
            self.high=self.x_clean+self.epsilon
        if self.mask_opt:
            try:
                if self.mask_cond is not None:
                    idx = self.mask_cond(self.x_clean)
                    self.low[idx]=self.x_clean[idx]
                    self.high[idx]=self.x_clean[idx]
            except AttributeError:
                pass
            try:
                if self.mask_idx is not None:
                    self.low[self.mask_idx]=self.x_clean[self.mask_idx]
                    self.high[self.mask_idx]=self.x_clean[self.mask_idx]
            except AttributeError:
                pass
        
        return 
    
    def score(self,x):
            y = self.model(x)
            y_diff = torch.cat((y[:,:self.y_clean], y[:,(self.y_clean+1):]),dim=1) - y[:,self.y_clean].unsqueeze(-1)
            y_diff, _ = y_diff.max(dim=1)
            return y_diff #.max(dim=1)
    
    def gaussian_to_image(self, gaussian_sample, ):
        image = self.low+(self.high-self.low)*self.normal_dist.cdf(gaussian_sample).view(-1,*self.input_shape)
        return image
    
    def h(self,X):
        return t_u.h_pyt(X,x_clean=self.x_clean,model=self.model,low=self.low,high=self.high,target_class=self.y_clean
                ,from_gaussian=self.from_gaussian,noise_dist=self.noise_dist,
                noise_scale=self.noise_scale)
    def V(self,X):
        return t_u.V_pyt(X,self.x_clean,model=self.model,low=self.low,high=self.high,target_class=self.y_clean
                ,from_gaussian=self.from_gaussian,noise_dist=self.noise_dist,
                noise_scale=self.noise_scale)   
    def gradV(self,X):
        return t_u.gradV_pyt(X,self.x_clean,model=self.model,low=self.low,high=self.high,target_class=self.y_clean
                ,from_gaussian=self.from_gaussian,noise_dist=self.noise_dist,
                noise_scale=self.noise_scale)
    



class Exp3Config(ExpConfig):
    default_dict={'config_name':'Experiment 2','dataset':'imagenet',
                  'data_dir':ROOT_DIR+"/data",
                'log_dir':ROOT_DIR+"/logs/exp_3_imagenet",
                'model_arch':'','model_dir':'','epsilon_range':[],'eps_max':0.3,'eps_min':0.2,
                'eps_num':5,'epsilon':-1.,'input_start':0,'input_stop':-1,'n_rep':100,'model_path':'',
                'export_to_onnx':False,'use_attack':False,'attack':'PGD',
                'lirpa_bounds':False,'download':True,'train_model_epochs':10,
                'from_gaussian':True,'noise_dist':'uniform','x_min':0.,'mask_opt':False,
                'sigma':1.,'x_max':1.,'x_mean':0.,'x_std':1.,'lirpa_cert':False,'robust_model':False,
                'robust_eps':0.1,'load_batch_size':128,'nb_epochs': 15,'adversarial_every':1,}
    #p_ref_compute = False
    def __init__(self,config_dict=default_dict,X=None,y=None,model=None,epsilon_range=[],
                dataset_name='mnist',aggr_res_path='',model_name='',verbose=0.,**kwargs):
        if X is not None:
            self.X=X
        if y is not None:
            self.y=y
        if model is not None:
            self.model=model   
        vars(self).update(config_dict)
        self.epsilon_range=epsilon_range
        self.dataset=dataset_name
        self.aggr_res_path=aggr_res_path
        self.verbose=verbose
        vars(self).update(kwargs)
        super().__init__()

    def update(self, method_name=''):
        super().update()
        self.method_name = method_name
        self.commit= git.Repo(path=ROOT_DIR).head.commit.hexsha
        if len(self.model_dir)==0:
            self.model_dir=os.path.join(ROOT_DIR+"/models/",self.dataset)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        #color_dataset=self.dataset in ('cifar10','cifar100','imagenet')
        #prblm_str=self.dataset
        if self.input_stop==-1:
            self.input_stop=self.input_start+1
        else:
            assert self.input_start<self.input_stop,"/!\ input start must be strictly lower than input stop"
        if len(self.noise_dist)==0:
            self.noise_dist=self.noise_dist.lower()
        if self.noise_dist not in ['uniform','gaussian']:
            raise NotImplementedError("Only uniform and Gaussian distributions are implemented.")
        # if not self.allow_multi_gpu:
        #     os.environ["CUDA_VISIBLE_DEVICES"]="0"
        if len(self.device)==0:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if self.verbose>=5:
                print(self.device)


        if len(self.log_dir)==0:
            self.log_dir=os.path.join(ROOT_DIR+'/logs','exp_2_'+self.dataset)
        if not os.path.exists(ROOT_DIR+'/logs'):
            os.mkdir(ROOT_DIR+'/logs')  
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if len(self.aggr_res_path)==0:
            self.aggr_res_path=os.path.join(self.log_dir,'aggr_res.csv')
       
        
        self.raw_logs = os.path.join(self.log_dir,'raw_logs/')
        if not os.path.exists(self.raw_logs):
            os.mkdir(self.raw_logs)
        raw_logs_path=os.path.join(self.log_dir,'raw_logs/'+self.method_name)
        if not os.path.exists(raw_logs_path):
            os.mkdir(raw_logs_path)
        self.loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
        
        log_name=self.method_name+'_'+self.loc_time
        exp_log_path=os.path.join(raw_logs_path,log_name)
        log_path_prop = exp_log_path
        k=1
        index_level=0
        while os.path.exists(log_path_prop):
            if k==10: 
                exp_log_path= log_path_prop.replace('9','1')
                k=1
                index_level+=1
            log_path_prop = exp_log_path+'_'+str(k) 
            if index_level >5:
                raise RuntimeError("Error the log naming system has failed")
            k+=1 

        self.exp_log_path =  log_path_prop                  
        
        os.mkdir(path=exp_log_path)
        if len(self.epsilon_range)==0:
            log_min,log_max=np.log(self.eps_min),np.log(self.eps_max)
            log_line=np.linspace(start=log_min,stop=log_max,num=self.eps_num)
            self.epsilon_range=np.exp(log_line)
        else:
            self.eps_num=len(self.epsilon_range)
            self.epsilon_range = [float(eps) for eps in self.epsilon_range]
            self.eps_min,self.eps_max=np.min(self.epsilon_range),np.max(self.epsilon_range)
        if self.epsilon==-1.:
            self.epsilon = self.epsilon_range[0]
        
        test_loader = t_u.get_loader(train=False,data_dir=self.data_dir,download=self.download
        ,dataset=self.dataset,batch_size=self.load_batch_size,
            x_mean=None,x_std=None)
        if not hasattr(self,"model"):
            if len(self.model_arch)==0:
                self.model_arch = t_u.datasets_default_arch[self.dataset]
            self.model, self.model_shape,self.model_name=t_u.get_model_imagenet(self.model_arch, 
                model_dir=self.model_dir,
                )
        else:
            if (not hasattr(self,"model_name")) or self.model_name=='':
                self.model_name = self.dataset + "_model"
            if (not hasattr(self,"model_arch")) or self.model_arch=='':
                
                self.model_arch = 'custom'
        
            
        
        self.num_classes=t_u.datasets_num_c[self.dataset.lower()]
        self.x_mean=t_u.datasets_means[self.dataset]
        self.x_std=t_u.datasets_stds[self.dataset]
        if not hasattr(self,"X"):
            self.X,self.y,self.sample_accuracy=t_u.get_correct_x_y(data_loader=test_loader,
                device=self.device,model=self.model)

        else:
            """compute the accuracy of the model on the sample batch"""
            self.sample_accuracy=t_u.get_model_accuracy(model=self.model,
            X=self.X[self.input_start:self.input_stop],y=self.y[self.input_start:self.input_stop])
        self.d= np.prod(self.X.shape[1:])
        if self.verbose >=1:
            print(f"model accuracy on sample batch:{self.sample_accuracy}")
        if self.use_attack:
            import foolbox as fb
            fmodel = fb.PyTorchModel(self.model, bounds=(0,1),device=self.device)
            attack = fb.attacks.LinfPGD()
            #un-normalize data before performing attack
            _, advs, self.attack_success = attack(fmodel, self.X[self.input_start:self.input_stop], 
            self.y[self.input_start:self.input_stop], epsilon_range=self.epsilon_range)
        inp_indices=np.arange(start=self.input_start,stop=self.input_stop)
        if self.verbose>=5:
            print (f"input indices: {inp_indices}")
        self.nb_inputs=len(inp_indices)
        
        if self.from_gaussian:
            self.gen= lambda N: torch.randn(size=(N,self.d),device=torch.device(self.device))
        else:
            self.gen = lambda N: (2*torch.rand(size=(N,self.d), device=torch.device(self.device) )-1)
        self.x_clean=self.X[0]
        self.y_clean=self.y[0]
        self.low=torch.max(self.x_clean-self.epsilon, torch.tensor([self.x_min]).to(self.device))
        self.high=torch.min(self.x_clean+self.epsilon, torch.tensor([self.x_max]).to(self.device))
        if self.mask_opt:
            try:
                if self.mask_cond is not None:
                    idx = self.mask_cond(self.x_clean)
                    self.low[idx]=self.x_clean[idx]
                    self.high[idx]=self.x_clean[idx]
            except AttributeError:
                pass
            try:
                if self.mask_idx is not None:
                    self.low[self.mask_idx]=self.x_clean[self.mask_idx]
                    self.high[self.mask_idx]=self.x_clean[self.mask_idx]
            except AttributeError:
                pass
        return 
    
    def score(self,x):
            y = self.model(x)
            y_diff = torch.cat((y[:,:self.y_clean], y[:,(self.y_clean+1):]),dim=1) - y[:,self.y_clean].unsqueeze(-1)
            y_diff, _ = y_diff.max(dim=1)
            return y_diff #.max(dim=1)
    def h(self,X):
        return t_u.h_pyt(X,x_clean=self.x_clean,model=self.model,low=self.low,high=self.high,
                         target_class=self.y_clean
                ,from_gaussian=self.from_gaussian,noise_dist=self.noise_dist,
                noise_scale=self.noise_scale)
    def V(self,X):
        return t_u.V_pyt(X,self.x_clean,model=self.model,low=self.low,high=self.high,
                         target_class=self.y_clean,
                from_gaussian=self.from_gaussian,noise_dist=self.noise_dist,
                noise_scale=self.noise_scale)   
    def gradV(self,X):
        return t_u.gradV_pyt(X,self.x_clean,model=self.model,low=self.low,high=self.high,
                             target_class=self.y_clean,
                from_gaussian=self.from_gaussian,noise_dist=self.noise_dist,
                noise_scale=self.noise_scale)
      
    
    
class SamplerConfig(Config):
    def __init__(self,):
        super().__init__()
        return
    



        
    
    
    





