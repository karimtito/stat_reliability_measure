import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from stat_reliability_measure.dev.utils import dichotomic_search,str2bool,str2list
from scipy.special import betainc
import stat_reliability_measure.dev.langevin_base.langevin_base_pyt as smc_pyt
from importlib import reload
from time import time
import cpuinfo
import GPUtil
import argparse
from stat_reliability_measure.dev.torch_utils import project_ball_pyt, projected_langevin_kernel_pyt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

str2floatList=lambda x: str2list(in_str=x, type_out=float)
str2intList=lambda x: str2list(in_str=x, type_out=int)

import os
if os.path.exists(ROOT_DIR+'/logs/linear_tests/torch_results.csv'):
    results=pd.read_csv(ROOT_DIR+'/logs/linear_tests/torch_results.csv')
else:
    results=pd.DataFrame(columns=['p_t','mean_est','mean_time','mean_err','std_time','std_est','T','N','rho','alpha','n_rep','min_rate'])

reload(smc_pyt)
class config:
    N_range=[10, 50,100,]
    T_range=[1,5]
    min_rate=0.51
    rho_range=[80,100,150]
    alpha_range=[0.008,0.01,0.02]
    p_range=[1e-15,1e-10,1e-5]
    n_rep=10
    verbose=0
    d=1024
    epsilon=1.
    save_config=True
    tqdm_opt=True
    update_agg_res=True
    save_every=10
    track_cpu=True
    track_gpu=True
    g_target=None
    allow_multi_gpu=False
    gaussian_latent=False
    n_max=5000
    aggr_res_path=ROOT_DIR+'/logs/linear_tests/torch_results.csv'
    gpu_name=None
    cpu_name=None
    print_config=True
    cpu_count=None
    log_dir=ROOT_DIR+'/logs/linear_tests'
    json=None

parser=argparse.ArgumentParser()
parser.add_argument('--log_dir',default=config.log_dir)
parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--N_range',type= str2intList,default=config.N_range)
parser.add_argument('--verbose',type=float,default=config.verbose)
parser.add_argument('--d',type=int,default=config.d)
parser.add_argument('--p_range',type=float,default=config.p_range)
parser.add_argument('--min_rate',type=float,default=config.min_rate)
parser.add_argument('--alpha_range',type=str2floatList,default=config.alpha_range)
parser.add_argument('--n_max',type=int,default=config.n_max)
parser.add_argument('--epsilon',type=float, default=config.epsilon)
parser.add_argument('--tqdm_opt',type=str2bool,default=config.tqdm_opt)
parser.add_argument('--T_range',type=str2intList,default=config.T_range)
parser.add_argument('--save_config',type=str2bool, default=config.save_config)
parser.add_argument('--update_agg_res',type=str2bool,default=True)
parser.add_argument('--rho_range',type=str2floatList,default=config.rho_range)
parser.add_argument('--gaussian_latent',type=str2bool, default=config.gaussian_latent)
parser.add_argument('--allow_multi_gpu',type=str2bool,default=config.allow_multi_gpu)
parser.add_argument('--g_target',type=float,default=config.g_target)
parser.add_argument('--track_gpu',type=str2bool,default=config.track_gpu)
parser.add_argument('--track_cpu',type=str2bool,default=config.track_cpu)
parser.add_argument('--aggr_res_path',type=str, default=config.aggr_res_path)
args=parser.parse_args()

param_lists = [config.N_range,config.T_range,config.rho_range,config.alpha_range,config.p_range]
param_lens = np.array([len(l) for l in param_lists])
nb_runs = np.prod(param_lens)

for k,v in vars(args).items():
    setattr(config, k, v)

if not config.allow_multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

if config.track_gpu:
    gpus=GPUtil.getGPUs()
    if len(gpus)>1:
        print("Multi gpus detected, only the first GPU will be tracked.")
    config.gpu_name=gpus[0].name

if config.track_cpu:
    config.cpu_name=cpuinfo.get_cpu_info()[[key for key in cpuinfo.get_cpu_info().keys() if 'brand' in key][0]]
    config.cpu_count=os.cpu_count()

d,epsilon=config.d,config.epsilon

p_target_f=lambda h: 0.5*betainc(0.5*(d+1),0.5,(2*epsilon*h-h**2)/(epsilon**2))
config.json=vars(args)
if config.print_config:
    print(config.json)
#get_c= lambda p:1-dichotomic_search(f=p_target_f,a=0,b=1,thresh=p,n_max=100)[0]
run_nb=0
param_lists = [config.N_range,config.T_range,config.rho_range,config.alpha_range,config.p_range]
param_lens = np.array([len(l) for l in param_lists])
nb_runs = np.prod(param_lens)
for p_t in config.p_range:
    h,P_target = dichotomic_search(f=p_target_f,a=0,b=1,thresh=p_t,n_max=100)
    c=1-h
    if config.verbose>=3:
        print(f'c:{c}')
    e_1= torch.Tensor([1]+[0]*(d-1)).to(device)
    V = lambda X: torch.clamp(input=c-X[:,0], min=0, max=None)
    gradV= lambda X: -torch.transpose(e_1[:,None]*(X[:,0]<c),dim0=1,dim1=0)
    big_gen= lambda N: torch.randn(size=(N,d+2)).to(device)
    norm_and_select= lambda X: (X/torch.norm(X,dim=1)[:,None])[:,:d] 
    uniform_ball_gen_pyt = lambda N: epsilon*norm_and_select(big_gen(N))
    prjct_epsilon = lambda X: project_ball_pyt(X, R=epsilon)
    prjct_epsilon_langevin_kernel = lambda X, gradV, delta_t,beta: projected_langevin_kernel_pyt(X,gradV,delta_t,beta, projection=prjct_epsilon)
    for rho in config.rho_range:
        for T in config.T_range:
            for alpha in config.alpha_range:       
                for N in config.N_range:
                    run_nb+=1
                    print(f'Run {run_nb}/{nb_runs}')
                    iterator= tqdm(range(config.n_rep)) if config.tqdm_opt else range(config._n_rep)
                    times=[]
                    ests = []
                    for i in iterator:
                        t=time()
                        p_est,finish_flag=smc_pyt.LangevinSMCBasePyt(gen=uniform_ball_gen_pyt,l_kernel=prjct_epsilon_langevin_kernel,N=N,min_rate=config.min_rate,
                        rho=rho,alpha=alpha,T=T,V= V,gradV=gradV,verbose=config.verbose,n_max=config.n_max)
                        t1=time()-t
                        times.append(t1)
                        ests.append(p_est)
                    times=np.array(times)  
                    ests=np.array(ests)
                    errs=np.abs(ests-p_t)
                    results=results.append({'p_t':p_t,'mean_est':ests.mean(),'mean_err':errs.mean(),
                    'mean_time':times.mean(),'std_time':times.std(),'std_est':ests.std(),'T':T,'N':N,
                    'rho':rho,'alpha':alpha,'n_rep':config.n_rep,'min_rate':config.min_rate,
                    'GPU':config.gpu_name,'CPU':config.cpu_name,'cpu_count':config.cpu_count},ignore_index=True)
                    if run_nb%config.save_every==0 and config.update_agg_res:
                        results.to_csv(config.aggr_res_path,index=False)
                    
                    
                    

