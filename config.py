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
                  'noise_scale':0.1,'exp_name':'',
                  'notebook':False,}

    def __init__(self,config_dict=default_dict):   
        vars(self).update(config_dict)
        super().__init__()
        
    def update(self):
        if self.torch_seed==-1:
            """if the torch seed is not set, then it is set to the current time"""
            self.torch_seed=int(time())
        torch.manual_seed(seed=self.torch_seed)
        if torch.cuda.is_available(): 
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
    
class Exp1config(ExpConfig):
    default_dict={'config_name':'Experiment 2',
                'log_dir':ROOT_DIR+"logs/exp_2_mnist",
                
                
                }
    #p_ref_compute = False
    def __init__(self,config_dict=default_dict):   
        vars(self).update(config_dict)
        super().__init__()
    def update(self):
        self.commit= git.Repo(path=ROOT_DIR).head.commit.hexsha
        if len(self.model_dir)==0:
            self.model_dir=os.path.join(ROOT_DIR+"/models/",self.dataset)

        if len(self.model_arch)==0:
            self.model_arch = t_u.datasets_default_arch[self.dataset]

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        
        self.d = t_u.datasets_dims[self.dataset]
        #color_dataset=self.dataset in ('cifar10','cifar100','imagenet')
        #prblm_str=self.dataset
        if self.input_stop==-1:
            self.input_stop=self.input_start+1
        else:
            assert self.input_start<self.input_stop,"/!\ input start must be strictly lower than input stop"

        if len(self.noise_dist)==0:
            self.noise_dist=self.noise_dist.lower()

        if self.noise_dist not in ['uniform','gaussian']:
            raise NotImplementedError("Only uniform and Gaussian noise distributions are implemented.")
        
        if len(self.device)==0:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if self.verbose>=1:
                print(f"PyTorch running on device: {self.device}")
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
        if len(self.epsilon_range)==0:
            log_min,log_max=np.log(self.eps_min),np.log(self.eps_max)
            log_line=np.linspace(start=log_min,stop=log_max,num=self.eps_num)
            self.epsilon_range=np.exp(log_line)
        else:
            self.eps_num=len(self.epsilon_range)
            self.epsilon_range = [float(eps) for eps in self.epsilon_range]
            self.eps_min,self.eps_max=np.min(self.epsilon_range),np.max(self.epsilon_range)
        
        test_loader = t_u.get_loader(train=False,data_dir=self.data_dir,download=self.download
        ,dataset=self.dataset,batch_size=self.load_batch_size,
            x_mean=None,x_std=None)
        model, self.model_shape,self.model_name=t_u.get_model(self.model_arch, robust_model=self.robust_model, robust_eps=self.robust_eps,
            nb_epochs=self.nb_epochs,model_dir=self.model_dir,data_dir=self.data_dir,test_loader=test_loader,device=self.device,
            download=self.download,dataset=self.dataset, force_train=self.force_train,
            )
        self.num_classes=t_u.datasets_num_c[self.dataset.lower()]
        self.x_mean=t_u.datasets_means[self.dataset]
        self.x_std=t_u.datasets_stds[self.dataset]

        X_correct,label_correct,accuracy=t_u.get_correct_x_y(data_loader=test_loader,device=self.device,model=model)
        if self.verbose>=2:
            print(f"model accuracy on test batch:{accuracy}")
        if self.use_attack:
            import foolbox as fb
            fmodel = fb.PyTorchModel(self.model, bounds=(0,1),device=self.device)
            attack=fb.attacks.LinfPGD()
            #un-normalize data before performing attack
            _, advs, self.attack_success = attack(fmodel, X_correct[self.input_start:self.input_stop], 
            label_correct[self.input_start:self.input_stop], epsilon_range=self.epsilon_range)
        inp_indices=np.arange(start=self.input_start,stop=self.input_stop)
        self.nb_inputs=len(inp_indices)
        X_test,y_test = X_correct[inp_indices],label_correct[inp_indices]

        return X_test,y_test,model

    
class Exp2Config(ExpConfig):
    default_dict={'config_name':'Experiment 2','dataset':'mnist',
                  'data_dir':ROOT_DIR+"/data",
                'log_dir':ROOT_DIR+"/logs/exp_2_mnist",
                'model_arch':'','model_dir':'','epsilon_range':[],'eps_max':0.3,'eps_min':0.2,
                'eps_num':5,'epsilon':-1.,'input_start':0,'input_stop':-1,'n_rep':100,'model_path':'',
                'export_to_onnx':False,'use_attack':False,'attack':'PGD',
                'lirpa_bounds':False,'download':True,'train_model_epochs':10,
                'gaussian_latent':True,'noise_dist':'uniform','x_min':0.,'mask_opt':False,
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
            self.model, self.model_shape,self.model_name=t_u.get_model(self.model_arch, robust_model=self.robust_model, robust_eps=self.robust_eps,
                nb_epochs=self.nb_epochs,model_dir=self.model_dir,data_dir=self.data_dir,test_loader=test_loader,device=self.device,
                download=self.download,dataset=self.dataset, force_train=self.force_train,
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
            self.X,self.y,self.sample_accuracy=t_u.get_correct_x_y(data_loader=test_loader,device=self.device,model=self.model)

        else:
            """compute the accuracy of the model on the sample batch"""
            self.sample_accuracy=t_u.get_model_accuracy(model=self.model,X=self.X[self.input_start:self.input_stop],y=self.y[self.input_start:self.input_stop])
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
        
        if self.gaussian_latent:
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
        return t_u.h_pyt(X,x_clean=self.x_clean,model=self.model,low=self.low,high=self.high,target_class=self.y_clean
                ,gaussian_latent=self.gaussian_latent,noise_dist=self.noise_dist,noise_scale=self.noise_scale)

    def V(self,X):
        return t_u.V_pyt(X,x_clean=self.x_clean,model=self.model,low=self.low,high=self.high,target_class=self.y_clean
                ,gaussian_latent=self.gaussian_latent,noise_dist=self.noise_dist,noise_scale=self.noise_scale)   
    def gradV(self,X):
        return t_u.gradV_pyt(X,x_clean=self.x_clean,model=self.model,low=self.low,high=self.high,target_class=self.y_clean
                ,gaussian_latent=self.gaussian_latent,noise_dist=self.noise_dist,noise_scale=self.noise_scale)




class Exp3Config(ExpConfig):
    default_dict={'config_name':'Experiment 2','dataset':'imagenet',
                  'data_dir':ROOT_DIR+"/data",
                'log_dir':ROOT_DIR+"/logs/exp_3_imagenet",
                'model_arch':'','model_dir':'','epsilon_range':[],'eps_max':0.3,'eps_min':0.2,
                'eps_num':5,'epsilon':-1.,'input_start':0,'input_stop':-1,'n_rep':100,'model_path':'',
                'export_to_onnx':False,'use_attack':False,'attack':'PGD',
                'lirpa_bounds':False,'download':True,'train_model_epochs':10,
                'gaussian_latent':True,'noise_dist':'uniform','x_min':0.,'mask_opt':False,
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
        
        if self.gaussian_latent:
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
        return t_u.h_pyt(X,x_clean=self.x_clean,model=self.model,low=self.low,high=self.high,target_class=self.y_clean
                ,gaussian_latent=self.gaussian_latent,noise_dist=self.noise_dist,noise_scale=self.noise_scale)

    def V(self,X):
        return t_u.V_pyt(X,x_clean=self.x_clean,model=self.model,low=self.low,high=self.high,target_class=self.y_clean
                ,gaussian_latent=self.gaussian_latent,noise_dist=self.noise_dist,noise_scale=self.noise_scale)   
    def gradV(self,X):
        return t_u.gradV_pyt(X,x_clean=self.x_clean,model=self.model,low=self.low,high=self.high,target_class=self.y_clean
                ,gaussian_latent=self.gaussian_latent,noise_dist=self.noise_dist,noise_scale=self.noise_scale)
      
    
    
class SamplerConfig(Config):
    def __init__(self,):
        super().__init__()
        return
    



        
    
    
    





