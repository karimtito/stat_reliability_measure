from stat_reliability_measure.home import ROOT_DIR
from pathlib import Path
from stat_reliability_measure.dev.utils import str2bool,str2floatList,str2intList,float_to_file_float
from stat_reliability_measure.dev.utils import clean_attr_dict,valid_pars_type
import stat_reliability_measure.dev.torch_utils as t_u
import argparse
import git
import os
from time import time
import numpy as np
import torch

class Config():
    config_name='Default'
    def __init__(self,config_dict=None):
        if config_dict is not None:
            self.__dict__.update(config_dict)
        self.clean_dict = clean_attr_dict(self)
        self.build_parser()
    def __str__(self):
        str_ = str("Config(")
        self.clean_dict = clean_attr_dict(self)
        for key in self.clean_dict.keys():
            str_+=f"{key}={self.clean_dict[key]},"
        str_+=")"
        return str_
    def __repr__(self):
        str_ = str("Config(")
        self.clean_dict = clean_attr_dict(self)
        for key in self.clean_dict.keys():
            str_+=f"{key}={self.clean_dict[key]},"
        str_+=")"
        return str_
    
    def print_config(self):
        print(f"{self.config_name} configuration:/n {self}")
    def get_parser(self):
        return self.parser
    def build_parser(self):
        parser = argparse.ArgumentParser(description='Experiment 2')
        for key in self.clean_dict.keys():
            if ('_range' or '_list' in key) and len(self.clean_dict[key])==0:
                # if the key is a range or list and the value is empty 
                # then the type is the type of the first element of the list
                ptype = pars_type([self.clean_dict[key.strip('_range').strip('_list')]])
            else:
                # else the type is the type of the default value
                ptype = pars_type(self.clean_dict[key])
            parser.add_argument('--'+key,type=ptype,default=self.clean_dict[key])
        self.parser = parser

class ExpConfig(Config):
    default_dict={'config_name':'Experiment','torch_seed':-1,'np_seed':-1,
                  'verbose':0,'aggr_res_path':'',
                  'update_aggr_res':True,'save_config':False,
                  'print_config':True,'track_finish':False,
                  'track_gpu':False,'allow_multi_gpu':False,
                  'track_cpu':False,'save_img':False,
                  'save_text':False,'device':'cpu',
                  'tqdm_opt':True,'clip_min':0.,
                  'clip_max':1.,'force_train':False,
                  'repeat_exp':False,'data_dir':ROOT_DIR+"/data"}
    
    torch_seed=0
    np_seed=0
    verbose=0
    aggr_res_path = ''
    update_aggr_res=True
    save_config=False 
    print_config=True
    track_finish=False
    track_gpu=False
    allow_multi_gpu=False
    track_cpu=False
    save_img=False
    save_text=False
    device='cpu'
    tqdm_opt=True
    clip_min=0.
    clip_max=1.
    force_train = False
    repeat_exp = False
    data_dir=ROOT_DIR+"/data"

    def __init__(self,config_dict=None):        
        super(self.__class__, self).__init__(config_dict)
    def __repr__(self):
        str_ = str("ExpConfig(")
        self.clean_dict = clean_attr_dict(self)
        for key in self.clean_dict.keys():
            str_+=f"{key}={self.clean_dict[key]},"
        str_+=")"
        return str_
    
class Exp2Config(ExpConfig):
    config_name='exp_2'
    dataset='mnist'
    log_dir=Path(ROOT_DIR,"logs/exp_2_mnist")
    model_arch=''  
    model_dir='' 
    epsilons = [0.15]
    eps_max=0.3
    eps_min=0.2
    eps_num=5
    input_start=0
    input_stop=1
    n_rep=100
    model_path=''
    #export_to_onnx=False
    use_attack=False
    #attack='PGD'
    lirpa_bounds=False
    download=True
    #train_model_epochs=10
    gaussian_latent=True
    noise_dist='uniform'
    x_min=0.
    x_max=1.
    #x_mean=0
    #x_std=1
    
    #track_finish=False
    lirpa_cert=False
    robust_model=False
    robust_eps=0.1
    load_batch_size=100 
    nb_epochs= 15
    adversarial_every=1
    
    #p_ref_compute = False
    def to_json(self,path=None):
        if path is None:
            path = os.path.join(self.self.exp_logs_path,'exp_config.json')
        with open(path,'w') as f:
            f.write(json.dumps(self.cl, indent = 4, cls=utils.CustomEncoder))
    def update(self,method_name='relibability_method'):
        self.method_name=method_name
        self.commit= git.Repo(path=ROOT_DIR).head.commit.hexsha
        if self.model_dir is None:
            self.model_dir=os.path.join(ROOT_DIR+"/models/",self.dataset)

        if self.model_arch is None:
            self.model_arch = t_u.datasets_default_arch[self.dataset]

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        
        self.d = t_u.datasets_dims[self.dataset]
        #color_dataset=self.dataset in ('cifar10','cifar100','imagenet')
        #prblm_str=self.dataset
        if self.input_stop is None:
            self.input_stop=self.input_start+1
        else:
            assert self.input_start<self.input_stop,"/!\ input start must be strictly lower than input stop"

        if self.noise_dist is not None:
            self.noise_dist=self.noise_dist.lower()

        if self.noise_dist not in ['uniform','gaussian']:
            raise NotImplementedError("Only uniform and Gaussian distributions are implemented.")
        if not self.allow_multi_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"]="0"

        if self.torch_seed is None:
            self.torch_seed=int(time())
        torch.manual_seed(seed=self.torch_seed)

        if self.np_seed is None:
            self.np_seed=int(time())
        torch.manual_seed(seed=self.np_seed)

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


        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if self.verbose>=5:
                print(self.device)


        if self.log_dir is None:
            self.log_dir=os.path.join(ROOT_DIR+'/logs','exp_2_'+self.dataset)
        if not os.path.exists(ROOT_DIR+'/logs'):
            os.mkdir(ROOT_DIR+'/logs')  
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if self.aggr_res_path is None:
            self.aggr_res_path=os.path.join(self.log_dir,'aggr_res.csv')
       
        
        self.raw_logs = os.path.join(self.log_dir,'raw_logs/')
        if not os.path.exists(self.raw_logs):
            os.mkdir(self.raw_logs)
        if self.epsilons is None:
            log_min,log_max=np.log(self.eps_min),np.log(self.eps_max)
            log_line=np.linspace(start=log_min,stop=log_max,num=self.eps_num)
            self.epsilons=np.exp(log_line)
        
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

        X_correct,label_correct,accuracy=t_u.get_correct_x_y(data_loader=test_loader,device=exp_config.device,model=model)
        if self.verbose>=2:
            print(f"model accuracy on test batch:{accuracy}")
        if self.use_attack:
            import foolbox as fb
            fmodel = fb.PyTorchModel(self.model, bounds=(0,1),device=self.device)
            attack=fb.attacks.LinfPGD()
            #un-normalize data before performing attack
            _, advs, self.attack_success = attack(fmodel, X_correct[self.input_start:self.input_stop], 
            label_correct[self.input_start:self.input_stop], epsilons=self.epsilons)
        inp_indices=np.arange(start=self.input_start,stop=self.input_stop)
        self.nb_inputs=len(inp_indices)
        X_test,y_test = X_correct[inp_indices],label_correct[inp_indices]
        self.raw_logs_path=os.path.join(exp_config.log_dir,'raw_logs/'+method_name)
        if not os.path.exists(self.raw_logs_path):
            os.mkdir(self.raw_logs_path)
        loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
        log_name=method_name+'_'+loc_time
        self.exp_logs_path=os.path.join(self.raw_logs_path,log_name)
        if os.path.exists(path=self.exp_logs_path):
            self.exp_logs_path = self.exp_logs_path+'_'+str(np.random.randint(low=0,high=99))
        os.mkdir(path=self.exp_logs_path)

        return X_test,y_test,model
        


        
    
    
    





