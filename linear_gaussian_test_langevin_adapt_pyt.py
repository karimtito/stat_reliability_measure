import dev.torch_utils as t_u 
import dev.langevin_adapt_pyt as smc_pyt
import scipy.stats as stat
import numpy as np
from tqdm import tqdm
from time import time
import os
import torch
import GPUtil
import cpuinfo
import pandas as pd
import argparse
from dev.utils import str2bool,str2floatList,str2intList


method_name="langevin_adapt_pyt"

#gaussian_linear
class config:
    N_range=[100]
    T_range=[1]
    min_rate=0.90
    rho_range=[]
    alpha_range=[0.1]
    g_target=0.9
    g_range=[]
    p_range=[]
    p_t=1e-15
    n_rep=10
    
    save_config=False 
    d=5
    verbose=1
    log_dir='./logs/linear_gaussian_tests'
    sigma=1
    v1_kernel=True
    torch_seed=0
    gpu_name=None
    cpu_name=None
    cores_number=None
    track_gpu=True
    track_cpu=True
    device=None
    n_max=10000 
    allow_multi_gpu=False
    tqdm_opt=True
    allow_zero_est=True
    track_accept=True
    track_calls=False
    mh_opt=False
    adapt_d_t=False
    target_accept=0.574
    accept_spread=0.1
    d_t_decay=0.999
    d_t_gain=1/d_t_decay

parser=argparse.ArgumentParser()
parser.add_argument('--log_dir',default=config.log_dir)
parser.add_argument('--n_rep',type=int,default=config.n_rep)
#parser.add_argument('--N',type=int,default=config.N)
parser.add_argument('--verbose',type=float,default=config.verbose)
parser.add_argument('--d',type=int,default=config.d)

parser.add_argument('--min_rate',type=float,default=config.min_rate)
#parser.add_argument('--alpha',type=float,default=config.alpha)
parser.add_argument('--n_max',type=int,default=config.n_max)
parser.add_argument('--tqdm_opt',type=str2bool,default=config.tqdm_opt)
#parser.add_argument('--T',type=int,default=config.T)
parser.add_argument('--save_config',type=str2bool, default=config.save_config)
#parser.add_argument('--update_agg_res',type=str2bool,default=config.update_agg_res)
#parser.add_argument('--aggr_res_path',type=str, default=config.aggr_res_path)
#parser.add_argument('--rho',type=float,default=config.rho)
parser.add_argument('--allow_multi_gpu',type=str2bool)

parser.add_argument('--track_gpu',type=str2bool,default=config.track_gpu)
parser.add_argument('--track_cpu',type=str2bool,default=config.track_cpu)
parser.add_argument('--device',type=str, default=config.device)
parser.add_argument('--allow_zero_est',type=str2bool, default=config.allow_zero_est)
parser.add_argument('--torch_seed',type=int, default=config.torch_seed)
#parser.add_argument('--np_seed',type=int, default=config.np_seed)

#parser.add_argument('--noise_dist',type=str, default=config.noise_dist)
parser.add_argument('--sigma', type=float,default=config.sigma)
parser.add_argument('--p_t',type=float,default=config.p_t)
parser.add_argument('--p_range',type=str2floatList,default=config.p_range)
parser.add_argument('--g_target',type=float,default=config.g_target)
parser.add_argument('--g_range',type=str2floatList,default=config.g_range)
parser.add_argument('--N_range',type=str2intList,default=config.N_range)
parser.add_argument('--T_range',type=str2intList,default=config.T_range)
parser.add_argument('--rho_range', type=str2floatList,default=config.rho_range)
parser.add_argument('--alpha_range',type=str2floatList,default=config.alpha_range)
parser.add_argument('--v1_kernel',type=str2bool,default=config.v1_kernel)
parser.add_argument('--track_accept',type=str2bool,default=config.track_accept)
parser.add_argument('--track_calls',type=str2bool,default=config.track_calls)
parser.add_argument('--mh_opt',type=str2bool,default=config.mh_opt)
parser.add_argument('--adapt_d_t',type=str2bool,default=config.adapt_d_t)
parser.add_argument('--target_accept',type=float,default=config.target_accept)
parser.add_argument('--accept_spread',type=float,default=config.accept_spread)
parser.add_argument('--d_t_decay',type=float,default=config.d_t_decay)
parser.add_argument('--d_t_gain',type=float,default=config.d_t_gain)
args=parser.parse_args()

for k,v in vars(args).items():
    setattr(config, k, v)


if len(config.p_range)==0:
    config.p_range= [config.p_t]

if len(config.g_range)==0:
    config.g_range= [config.g_target]


if not config.allow_multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"



if config.torch_seed is None:
    config.torch_seed=int(time.time())
torch.manual_seed(seed=config.torch_seed)



if config.track_gpu:
    gpus=GPUtil.getGPUs()
    if len(gpus)>1:
        print("Multi gpus detected, only the first GPU will be tracked.")
    config.gpu_name=gpus[0].name

if config.track_cpu:
    config.cpu_name=cpuinfo.get_cpu_info()['brand_raw']
    config.cores_number=os.cpu_count()


if config.device is None:
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    if config.verbose>=5:
        print(config.device)
    device=config.device
else:
    device=config.device

d=config.d
#epsilon=config.epsilon


if not os.path.exists('./logs'):
    os.mkdir('./logs')
if not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir)

results_path='./logs/linear_gaussian_tests/results.csv'
if os.path.exists(results_path):
    results_g=pd.read_csv(results_path)
else:
    results_g=pd.DataFrame(columns=['p_t','mean_est','mean_time','mean_err','std_time','std_est','T','N','rho','alpha','n_rep','min_rate','method'])
    results_g.to_csv(results_path,index=False)
raw_logs_path=os.path.join(config.log_dir,'raw_logs')
if not os.path.exists(raw_logs_path):
    os.mkdir(raw_logs_path)




# if config.aggr_res_path is None:
#     aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
# else:
#     aggr_res_path=config.aggr_res_path



param_ranges = [config.N_range,config.T_range,config.rho_range,config.alpha_range,config.p_range]
param_lens=np.array([len(l) for l in param_ranges])
nb_runs= np.prod(param_lens)

mh_str="adjusted" if config.mh_opt else "unadjusted"
method=method_name+'_'+mh_str
save_every = 1
d=5

kernel_str='v1_kernel' if config.v1_kernel else 'v2_kernel'
kernel_function=t_u.langevin_kernel_pyt if config.v1_kernel else t_u.langevin_kernel_pyt2 
get_c_norm= lambda p:stat.norm.isf(p)
run_nb=0
iterator= tqdm(range(config.n_rep))
for p_t in config.p_range:
    c=get_c_norm(p_t)
    print(f'c:{c}')
    e_1= torch.Tensor([1]+[0]*(d-1)).to(device)
    V = lambda X: torch.clamp(input=c-X[:,0], min=0, max=torch.inf)
    
    gradV= lambda X: -torch.transpose(e_1[:,None]*(X[:,0]<c),dim0=1,dim1=0)
    
    norm_gen = lambda N: torch.randn(size=(N,d)).to(device)

    for g_t in config.g_range:
        for T in config.T_range:
            for alpha in config.alpha_range:       
                for N in config.N_range:
                    
                    run_nb+=1
                    print(f'Run {run_nb}/{nb_runs}')
                    times=[]
                    ests = []
                    
                    finished_flags=[]
                    iterator= tqdm(range(config.n_rep)) if config.tqdm_opt else range(config.n_rep)
                    print(f"Starting simulations with p_t:{p_t},g_t:{g_t},T:{T},alpha:{alpha},N:{N}")
                    for i in iterator:
                        t=time()
                        p_est,res_dict,=smc_pyt.LangevinSMCSimpAdaptPyt(gen=norm_gen,
                        l_kernel=kernel_function,N=N,min_rate=config.min_rate,
                        g_target=g_t,alpha=alpha,T=T,V= V,gradV=gradV,n_max=10000,return_log_p=False,
                        verbose=config.verbose, mh_opt=config.mh_opt,track_accept=config.track_accept,
                        allow_zero_est=config.allow_zero_est,gaussian =True,
                        target_accept=config.target_accept,accept_spread=config.accept_spread, 
                        adapt_d_t=config.adapt_d_t, d_t_decay=config.d_t_decay,
                        d_t_gain=config.d_t_gain
                        )
                        t1=time()-t
                        print(p_est)
                        finish_flag=res_dict['finished']
                        accept_rates=res_dict['accept_rates']
                        finished_flags.append(finish_flag)
                        times.append(t1)
                        ests.append(p_est)
                    times=np.array(times)  
                    ests=np.array(ests)
                    errs=np.abs(ests-p_t)
                    fin = np.array(finished_flags)
                    result_g={'p_t':p_t,'mean_est':ests.mean(),'mean_err':errs.mean(),
                    'mean_time':times.mean(),'std_time':times.std(),'std_est':ests.std(),'T':T,'N':N,
                    'g_target':g_t,'alpha':alpha,'n_rep':config.n_rep,'min_rate':config.min_rate,'d':d,
                    "method":method,"kernel":kernel_str,"finish_rate":fin.mean(),
                     'gpu_name':config.gpu_name ,'cpu_name':config.cpu_name}
                    if config.adapt_d_t:
                        result_g.update({'adapt_d_t':config.adapt_d_t,'target_accept':config.target_accept,
                        'accept_spread':config.accept_spread,'d_t_accept':config.d_t_decay,
                        'd_t_gain': config.d_t_gain})

                    results_g=results_g.append(result_g,ignore_index=True)
                    
                    if run_nb%save_every==0:
                        results_g.to_csv(results_path,index=False)
                    


                    {'p_t':p_t,'method':method_name,'gaussian_latent':str(config.gaussian_latent),
                    'N':config.N,'n_rep':config.n_rep,'T':config.T,'ratio':ratio,'K':K,'s':s
                    ,'min_rate':config.min_rate,'mean time':times.mean(),'std time':times.std()
                    ,'mean est':estimates.mean(),'bias':estimates.mean()-p_t,'mean abs error':abs_errors.mean(),
                    'mean rel error':rel_errors.mean(),'std est':estimates.std(),'freq underest':(estimates<p_t).mean()
                    ,'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,'cores_number':config.cores_number,
                    'batch_opt':config.batch_opt}