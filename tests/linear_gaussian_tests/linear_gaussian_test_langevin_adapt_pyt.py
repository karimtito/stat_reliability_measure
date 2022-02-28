import stat_reliability_measure.dev.torch_utils as t_u 
import stat_reliability_measure.dev.langevin_adapt.langevin_adapt_pyt as smc_pyt
import scipy.stats as stat
import numpy as np
from tqdm import tqdm
from time import time
import os
import matplotlib.pyplot as plt
import torch
import GPUtil
import cpuinfo
import pandas as pd
import argparse
from stat_reliability_measure.dev.utils import str2bool,str2floatList,str2intList,float_to_file_float


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
    d=1024
    verbose=1
    log_dir='../../logs/linear_gaussian_tests'
    aggr_res_path = None
    update_agg_res=True
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
    adapt_d_t_mcmc=False
    target_accept=0.574
    accept_spread=0.1
    d_t_decay=0.999
    d_t_gain=1/d_t_decay
    v_min_opt=False
    ess_opt=False
    only_duplicated=False
    np_seed=0
    lambda_0=0.5
    test2=False

    s_opt=False
    s=1
    clip_s=True
    s_min=1e-3
    s_max=3
    s_decay=0.95
    s_gain=1.0001





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
parser.add_argument('--adapt_d_t_mcmc',type=str2bool,default=config.adapt_d_t_mcmc)
parser.add_argument('--update_agg_res',type=str2bool,default=config.update_agg_res)
parser.add_argument('--v_min_opt',type=str2bool,default=config.v_min_opt)
parser.add_argument('--ess_opt',type=str2bool,default=config.ess_opt)
parser.add_argument('--only_duplicated',type=str2bool,default=config.only_duplicated)
parser.add_argument('--lambda_0',type=float,default=config.lambda_0)
parser.add_argument('--test2',type=str2bool,default =config.test2)

parser.add_argument('--s_opt',type=str2bool,default=config.s_opt)
parser.add_argument('--s',type=float,default=config.s)
parser.add_argument('--clip_s',type=str2bool,default=config.clip_s)
parser.add_argument('--s_min',type=float,default=config.s_min) 
parser.add_argument('--s_max',type=float,default= config.s_max)
parser.add_argument('--s_decay',type=float,default=config.s_decay)
parser.add_argument('--s_gain',type =float,default= config.s_gain)

args=parser.parse_args()

for k,v in vars(args).items():
    setattr(config, k, v)


if len(config.p_range)==0:
    config.p_range= [config.p_t]

if len(config.g_range)==0:
    config.g_range= [config.g_target]


if len(config.N_range)==0:
    config.N_range= [config.N]


if len(config.T_range)==0:
    config.T_range= [config.T]


if not config.allow_multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"



if config.torch_seed is None:
    config.torch_seed=int(time.time())
torch.manual_seed(seed=config.torch_seed)

if config.np_seed is None:
    config.np_seed=int(time.time())
torch.manual_seed(seed=config.np_seed)



if config.track_gpu:
    gpus=GPUtil.getGPUs()
    if len(gpus)>1:
        print("Multi gpus detected, only the first GPU will be tracked.")
    config.gpu_name=gpus[0].name

if config.track_cpu:
    config.cpu_name=cpuinfo.get_cpu_info()[[key for key in cpuinfo.get_cpu_info().keys() if 'brand' in key][0]]
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


if not os.path.exists('../../logs'):
    os.mkdir('../../logs')
if not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir)

results_path='../../logs/linear_gaussian_tests/results.csv'
if os.path.exists(results_path):
    results_g=pd.read_csv(results_path)
else:
    results_g=pd.DataFrame(columns=['p_t','mean_est','mean_time','mean_err','std_time','std_est','T','N','rho','alpha','n_rep','min_rate','method'])
    results_g.to_csv(results_path,index=False)
raw_logs_path=os.path.join(config.log_dir,'raw_logs')
if not os.path.exists(raw_logs_path):
    os.mkdir(raw_logs_path)

loc_time= float_to_file_float(time())
log_name=method_name+'_'+loc_time
log_path=os.path.join(raw_logs_path,log_name)
os.mkdir(path=log_path)
config.json=vars(args)

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
adapt_func= smc_pyt.ESSAdaptBetaPyt if config.ess_opt else smc_pyt.SimpAdaptBetaPyt

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
                    loc_time= float_to_file_float(time())
                    log_name=method_name+f'_N_{N}_T_{T}_a_{float_to_file_float(alpha)}_g_{float_to_file_float(g_t)}'+loc_time.split('_')[0]
                    log_path=os.path.join(raw_logs_path,log_name)
                    os.mkdir(path=log_path)
                    run_nb+=1
                    print(f'Run {run_nb}/{nb_runs}')
                    times=[]
                    ests = []
                    calls=[]
                    finished_flags=[]
                    iterator= tqdm(range(config.n_rep)) if config.tqdm_opt else range(config.n_rep)
                    print(f"Starting simulations with p_t:{p_t},g_t:{g_t},T:{T},alpha:{alpha},N:{N}")
                    for i in iterator:
                        
                        if not config.test2:
                            t=time()
                            p_est,res_dict,=smc_pyt.LangevinSMCSimpAdaptPyt(gen=norm_gen,
                            l_kernel=kernel_function,N=N,min_rate=config.min_rate,
                            g_target=g_t,alpha=alpha,T=T,V= V,gradV=gradV,n_max=10000,return_log_p=False,
                            adapt_func=adapt_func,
                            verbose=config.verbose, mh_opt=config.mh_opt,track_accept=config.track_accept,
                            allow_zero_est=config.allow_zero_est,gaussian =True,
                            target_accept=config.target_accept,accept_spread=config.accept_spread, 
                            adapt_d_t=config.adapt_d_t, d_t_decay=config.d_t_decay,
                            d_t_gain=config.d_t_gain,
                            v_min_opt=config.v_min_opt,
                            v1_kernel=config.v1_kernel, lambda_0= config.lambda_0,
                            only_duplicated=config.only_duplicated,
                            )
                            t1=time()-t
                        else:
                            t=time()
                            p_est,res_dict,=smc_pyt.LangevinSMCSimpAdaptPyt2(gen=norm_gen,
                            l_kernel=kernel_function,N=N,min_rate=config.min_rate,
                            g_target=g_t,alpha=alpha,T=T,V= V,gradV=gradV,n_max=10000,return_log_p=False,
                            adapt_func=adapt_func,
                            verbose=config.verbose, mh_opt=config.mh_opt,track_accept=config.track_accept,
                            allow_zero_est=config.allow_zero_est,gaussian =True,
                            target_accept=config.target_accept,accept_spread=config.accept_spread, 
                            adapt_d_t=config.adapt_d_t, d_t_decay=config.d_t_decay,
                            d_t_gain=config.d_t_gain,
                            v_min_opt=config.v_min_opt,
                            v1_kernel=config.v1_kernel, lambda_0= config.lambda_0,
                            only_duplicated=config.only_duplicated,
                            s_opt=config.s_opt,
                            s=config.s,s_decay=config.s_decay,s_gain=config.s_gain,
                            s_min=config.s_min,s_max=config.s_max)
                            t1=time()-t

                        print(p_est)
                        finish_flag=res_dict['finished']
                        accept_rates=res_dict['accept_rates']
                        if config.track_accept:
                            accept_rates=res_dict['accept_rates']
                            np.savetxt(fname=os.path.join(log_path,f'accept_rates_{i}.txt')
                            ,X=accept_rates)
                            x_T=np.arange(len(accept_rates))
                            plt.plot(x_T,accept_rates)
                            plt.savefig(os.path.join(log_path,f'accept_rates_{i}.png'))
                            accept_rates_mcmc=res_dict['accept_rates_mcmc']
                            x_T=np.arange(len(accept_rates_mcmc))
                            plt.close()
                            plt.plot(x_T,accept_rates_mcmc)
                            plt.savefig(os.path.join(log_path,f'accept_rates_mcmc_{i}.png'))
                            plt.close()
                            np.savetxt(fname=os.path.join(log_path,f'accept_rates_mcmc_{i}.txt')
                            ,X=accept_rates_mcmc,)
                        finished_flags.append(finish_flag)
                        times.append(t1)
                        ests.append(p_est)
                        calls.append(res_dict['calls'])
                    times=np.array(times)
                    ests = np.array(ests)
                    calls=np.array(calls)
                
                    abs_errors=np.abs(ests-p_t)
                    rel_errors=abs_errors/p_t
                    bias=np.mean(ests)-p_t

                    times=np.array(times)  
                    ests=np.array(ests)
                    errs=np.abs(ests-p_t)
                    #fin = np.array(finished_flags)


                    np.savetxt(fname=os.path.join(log_path,'times.txt'),X=times)
                    np.savetxt(fname=os.path.join(log_path,'ests.txt'),X=ests)

                    plt.hist(times, bins=20)
                    plt.savefig(os.path.join(log_path,'times_hist.png'))
                    plt.hist(rel_errors,bins=20)
                    plt.savefig(os.path.join(log_path,'rel_errs_hist.png'))

                    #with open(os.path.join(log_path,'results.txt'),'w'):
                    results={"p_t":p_t,"method":method_name,'T':T,'N':N,
                    "g_target":g_t,'alpha':alpha,'n_rep':config.n_rep,'min_rate':config.min_rate,'d':d,
                    "method":method,"kernel":kernel_str,'adapt_t':config.adapt_d_t,
                    'mean_calls':calls.mean(),'std_calls':calls.std()
                    ,'mean time':times.mean(),'std time':times.std()
                    ,'mean est':ests.mean(),'bias':ests.mean()-p_t,'mean abs error':abs_errors.mean(),
                    'mean rel error':rel_errors.mean(),'std est':ests.std(),'freq underest':(ests<p_t).mean(), 
                    "v_min_opt":config.v_min_opt
                    ,'adapt_d_t_mcmc':config.adapt_d_t_mcmc,"adapt_d_t":config.adapt_d_t,""
                    "adapt_d_t_mcmc":config.adapt_d_t_mcmc,"d_t_decay":config.d_t_decay,"d_t_gain":config.d_t_gain,
                    "target_accept":config.target_accept,"accept_spread":config.accept_spread, 
                    "mh_opt":config.mh_opt,'only_duplicated':config.only_duplicated,
                    "np_seed":config.np_seed,"torch_seed":config.torch_seed
                    ,'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,'cores_number':config.cores_number,
                    "d":config.d
                    }

                    results_df=pd.DataFrame([results])
                    results_df.to_csv(os.path.join(log_path,'results.csv'),index=False)
                    if config.aggr_res_path is None:
                        aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                    else:
                        aggr_res_path=config.aggr_res_path
                    if config.update_agg_res:
                        if not os.path.exists(aggr_res_path):
                            cols=['p_t','method','N','rho','n_rep','T','alpha','min_rate','mean time','std time','mean est',
                            'bias','mean abs error','mean rel error','std est','freq underest','gpu_name','cpu_name']
                            aggr_res_df= pd.DataFrame(columns=cols)
                        else:
                            aggr_res_df=pd.read_csv(aggr_res_path)
                        aggr_res_df = pd.concat([aggr_res_df,results_df],ignore_index=True)
                        aggr_res_df.to_csv(aggr_res_path,index=False)
                    


                    