import dev.torch_utils as t_u 
import dev.langevin_base_pyt as smc_pyt
import scipy.stats as stat
import numpy as np
from tqdm import tqdm
from time import time
import os
import torch
import GPUtil
import cpuinfo
import argparse
from dev.utils import str2bool,str2floatList,str2intList

method_name="langevin_base_pyt"

#gaussian_linear
class config:
    N_range=[100]
    T_range=[1]
    min_rate=0.90
    rho_range=[1,10,100]
    alpha_range=[0.1]
    p_range=[1e-15]
    n_rep=10
    mh_opt=False
    



parser=argparse.ArgumentParser()
parser.add_argument('--log_dir',default=config.log_dir)
parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--N',type=int,default=config.N)
parser.add_argument('--verbose',type=float,default=config.verbose)
parser.add_argument('--d',type=int,default=config.d)
parser.add_argument('--min_rate',type=float,default=config.min_rate)
parser.add_argument('--alpha',type=float,default=config.alpha)
parser.add_argument('--n_max',type=int,default=config.n_max)
parser.add_argument('--tqdm_opt',type=str2bool,default=config.tqdm_opt)
parser.add_argument('--T',type=int,default=config.T)
parser.add_argument('--save_config',type=str2bool, default=config.save_config)
parser.add_argument('--update_agg_res',type=str2bool,default=config.update_agg_res)
parser.add_argument('--aggr_res_path',type=str, default=config.aggr_res_path)
parser.add_argument('--rho',type=float,default=config.rho)
parser.add_argument('--gaussian_latent',type=str2bool, default=config.gaussian_latent)
parser.add_argument('--allow_multi_gpu',type=str2bool)
parser.add_argument('--g_target',type=float,default=config.g_target)
parser.add_argument('--track_gpu',type=str2bool,default=config.track_gpu)
parser.add_argument('--track_cpu',type=str2bool,default=config.track_cpu)
parser.add_argument('--device',type=str, default=config.device)
parser.add_argument('--allow_zero_est',type=str2bool, default=config.allow_zero_est)
parser.add_argument('--torch_seed',type=int, default=config.torch_seed)
parser.add_argument('--np_seed',type=int, default=config.np_seed)
parser.add_argument('--export_to_onnx',type=str2bool, default=config.export_to_onnx)
parser.add_argument('--use_attack',type=str2bool,default=config.use_attack)
parser.add_argument('--attack',type=str,default=config.attack)
parser.add_argument('--epsilons',type=str2floatList,default=config.epsilons)
parser.add_argument('--input_index',type=int,default=config.input_index)
parser.add_argument('--lirpa_bounds',type=str2bool, default=config.lirpa_bounds)
parser.add_argument('--eps_min',type=float, default=config.eps_min)
parser.add_argument('--eps_max',type=float, default=config.eps_max)
parser.add_argument('--eps_num',type=int,default=config.eps_num)
parser.add_argument('--train_model',type=str2bool,default=config.train_model)
parser.add_argument('--noise_dist',type=str, default=config.noise_dist)
parser.add_argument('--a',type=float, default=config.a)
parser.add_argument('--N_range',type=str2intList,default=config.N_range)
parser.add_argument('--T_range',type=str2intList,default=config.T_range)
parser.add_argument('--rho_range', type=str2floatList,default=config.rho_range)
parser.add_argument('--alpha_range',type=str2floatList,default=config.alpha_range)
parser.add_argument('--download',type=str2bool, default=config.download)
args=parser.parse_args()

for k,v in vars(args).items():
    setattr(config, k, v)

if config.noise_dist is not None:
    config.noise_dist=config.noise_dist.lower()

if config.noise_dist not in ['uniform','gaussian']:
    raise NotImplementedError("Only uniform and Gaussian distributions are implemented.")

if not config.allow_multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

if config.np_seed is None:
    config.np_seed=int(time.time())
np.random.seed(seed=config.np_seed)

if config.torch_seed is None:
    config.torch_seed=int(time.time())
torch.manual_seed(seed=config.torch_seed)

if len(config.T_range)<1:
    config.T_range=[config.T]
if len(config.N_range)<1:
    config.N_range=[config.N]
if len(config.rho_range)<1:
    config.rho_range=[config.rho]
if len(config.alpha_range)<1:
    config.alpha_range=[config.alpha]

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

raw_logs_path=os.path.join(config.log_dir,'raw_logs')
if not os.path.exists(raw_logs_path):
    os.mkdir(raw_logs_path)


if config.epsilons is None:
    log_min,log_max=np.log(config.eps_min),np.log(config.eps_max)
    log_line=np.linspace(start=log_min,stop=log_max,num=config.eps_num)
    config.epsions=np.exp(log_line)

if config.aggr_res_path is None:
    aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
else:
    aggr_res_path=config.aggr_res_path



param_ranges = [config.N_range,config.T_range,config.rho_range,config.alpha_range,config.p_range]
param_lens=np.array([len(l) for l in param_ranges])
nb_runs= np.prod(param_lens)

mh_str="adjusted" if config.mh_opt else "unadjusted"
method="langevin_base_"+mh_str
save_every = 1
d=5
v1_kernel=False
kernel_str='v1_kernel' if v1_kernel else 'v2_kernel'
kernel_function=t_u.langevin_kernel_pyt if v1_kernel else t_u.langevin_kernel_pyt2 
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

    for rho in config.rho_range:
        for T in config.T_range:
            for alpha in config.alpha_range:       
                for N in config.N_range:
                    run_nb+=1
                    print(f'Run {run_nb}/{nb_runs}')
                    
                    times=[]
                    ests = []
                    finished_flags=[]
                    iterator= tqdm(range(config.n_rep))
                    for i in iterator:
                        t=time()
                        p_est,res_dict,=smc_pyt.LangevinSMCBasePyt2(gen=norm_gen,
                        l_kernel=kernel_function,N=N,min_rate=config.min_rate,
                        rho=rho,alpha=alpha,T=T,V= V,gradV=gradV,n_max=10000,return_log_p=False,
                        verbose=5, mh_opt=config.mh_opt,track_accept=True,allow_zero_est=True)
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
                    results_g=results_g.append({'p_t':p_t,'mean_est':ests.mean(),'mean_err':errs.mean(),
                    'mean_time':times.mean(),'std_time':times.std(),'std_est':ests.std(),'T':T,'N':N,
                    'rho':rho,'alpha':alpha,'n_rep':config.n_rep,'min_rate':config.min_rate,'d':d,
                    "method":method,"kernel":kernel_str,"finish_rate":fin.mean()},ignore_index=True)
                    
                    if run_nb%save_every==0:
                        results_g.to_csv('../logs/linear_gaussian_tests/results.csv',index=False)