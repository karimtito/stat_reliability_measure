from __future__ import absolute_import
import numpy as np 
import numpy.linalg as LA
from time import time
from pydantic import conint
from scipy.special import betainc
import scipy.stats as stat
import matplotlib.pyplot as plt
import pandas as pd
import os
import GPUtil
import cpuinfo
import argparse
from tqdm import tqdm
import torch
from datetime import datetime

from stat_reliability_measure.dev.utils import  float_to_file_float,str2bool,str2intList,str2floatList, dichotomic_search
import stat_reliability_measure.dev.amls.amls_pyt as amls_pyt
method_name="amls_pyt"

class config:
    n_rep=10
    verbose=0
    min_rate=0.40
    clip_s=True
    s_min=8e-3
    s_max=3
    n_max=2000
    allow_zero_est=True
    
    N=40
    N_range=[]

    T=1
    T_range=[]

    ratio=0.6
    ratio_range=[]

    s=1
    s_range= []

    p_t=1e-10
    p_range=[]
    
    d = 1024
    epsilon = 1
    
    
    tqdm_opt=True
    save_config = True
    print_config=True
    update_agg_res=True
    aggr_res_path = None

    track_accept=False
    track_finish=True
    device = None

    log_dir="../../logs/gaussian_tests"
    batch_opt=True
    allow_multi_gpu=False
    track_gpu=True
    track_cpu=True
    core_numbers=None
    gpu_name=None 
    cpu_name=None
    cores_number=None

parser=argparse.ArgumentParser()

parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--verbose',type=float,default=config.verbose)
parser.add_argument('--min_rate',type=float,default=config.min_rate)
parser.add_argument('--clip_s',type=str2bool,default=config.clip_s)
parser.add_argument('--s_min',type=float,default=config.s_min)
parser.add_argument('--s_max',type=float,default=config.s_max)
parser.add_argument('--n_max',type=int,default=config.n_max)
parser.add_argument('--allow_zero_est',type=str2bool, default=config.allow_zero_est)

parser.add_argument('--N',type=int,default=config.N)
parser.add_argument('--N_range',type=str2intList,default=config.N_range)
parser.add_argument('--T',type=int,default=config.T)
parser.add_argument('--T_range',type=str2intList,default=config.T_range)
parser.add_argument('--ratio',type=float,default=config.ratio)
parser.add_argument('--ratio_range',type=float,default=config.ratio_range)
parser.add_argument('--s',type=float,default=config.s)
parser.add_argument('--s_range',type=str2floatList,default=config.s_range)
parser.add_argument('--p_t',type=float,default=config.p_t)
parser.add_argument('--p_range',type=str2floatList,default=config.p_range)

parser.add_argument('--d',type=int,default=config.d)
parser.add_argument('--epsilon',type=float, default=config.epsilon)
parser.add_argument('--tqdm_opt',type=bool,default=config.tqdm_opt)
parser.add_argument('--save_config', type=bool, default=config.save_config)
parser.add_argument('--print_config',type=bool , default=config.print_config)
parser.add_argument('--update_agg_res', type=bool,default=config.update_agg_res)

parser.add_argument('--aggr_res_path',type=str, default=config.aggr_res_path)
parser.add_argument('--track_gpu',type=str2bool,default=config.track_gpu)
parser.add_argument('--track_cpu',type=str2bool,default=config.track_cpu)
parser.add_argument('--log_dir',default=config.log_dir)
parser.add_argument('--allow_multi_gpu',type=str2bool,default=config.allow_multi_gpu)
parser.add_argument('--batch_opt',type=str2bool,default=config.batch_opt)
parser.add_argument('--track_accept',type=str2bool,default= config.track_accept)
parser.add_argument('--track_finish',type=str2bool,default=config.track_finish)
args=parser.parse_args()
for k,v in vars(args).items():
    setattr(config, k, v)

nb_runs=config.n_rep
if len(config.N_range)==0:
    config.N_range=[config.N]
nb_runs*=len(config.N_range)
if len(config.T_range)==0:
    config.T_range=[config.T]
nb_runs*=len(config.T_range)
if len(config.ratio_range)==0:
    config.ratio_range=[config.ratio]
nb_runs*=len(config.ratio_range)
if len(config.s_range)==0:
    config.s_range=[config.s]
nb_runs*=len(config.s_range)
if len(config.p_range)==0:
    config.p_range=[config.p_t]
nb_runs*=len(config.p_range)

if config.device is None:
    device= 'cuda:0' if torch.cuda.is_available() else 'cpu'


if not config.allow_multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

if config.track_gpu:
    gpus=GPUtil.getGPUs()
    if len(gpus)>1:
        print("Multi gpus detected, only the first GPU will be tracked.")
    config.gpu_name=gpus[0].name

if config.track_cpu:
    config.cpu_name=cpuinfo.get_cpu_info()[[key for key in cpuinfo.get_cpu_info().keys() if 'brand' in key][0]]
    config.cores_number=os.cpu_count()


epsilon=config.epsilon
d=config.d

if not os.path.exists('../../logs'):
    os.mkdir('../../logs')
    os.mkdir(config.log_dir)
elif not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir
    )
raw_logs_path=os.path.join(config.log_dir,'raw_logs/'+method_name)
if not os.path.exists(raw_logs_path):
    os.mkdir(raw_logs_path)


loc_time= datetime.today().isoformat().split('.')[0]
log_name=method_name+'_'+'_'+loc_time
exp_log_path=os.path.join(raw_logs_path,log_name)
os.mkdir(path=exp_log_path)
config.json=vars(args)
if config.print_config:
    print(config.json)
# if config.save_confi
# if config.save_config:
#     with open(file=os.path.join(),mode='w') as f:
#         f.write(config.json)

epsilon=config.epsilon
e_1 = torch.Tensor([1]+[0]*(d-1),device=config.device)
exp_res=[]
p_target_f=lambda h: 0.5*betainc(0.5*(d-1),0.5,(2*epsilon*h-h**2)/(epsilon**2))
i_run=0
for p_t in config.p_range:
    h,P_target = dichotomic_search(f=p_target_f,a=0,b=epsilon,thresh=p_t,n_max=100)
    c=epsilon-h
    
    if config.verbose>=5:
        print(f"P_target:{P_target}")
    arbitrary_thresh=40 #pretty useless a priori but should not hurt results
    def v_batch_pyt(X,c=c):
        return torch.clamp(input=torch.norm(X,p=2,dim=-1)*c-X[:,0],min=-arbitrary_thresh, max = None)
    amls_gen = lambda N: torch.randn(size=(N,d),device=config.device)
    batch_transform = lambda x: x
    normal_kernel =  lambda x,s : (x + s*torch.randn(size = x.shape,device=config.device))/np.sqrt(1+s**2) #normal law kernel, appliable to vectors 
    h_V_batch_pyt= lambda x: -v_batch_pyt(batch_transform(x)).reshape((x.shape[0],1))

    for T in config.T_range:
        for N in config.N_range: 
            for s in config.s_range:
                for ratio in config.ratio_range: 
                    loc_time= datetime.today().isoformat().split('.')[0]
                    log_name=method_name+f'_N_{N}_T_{T}_s_{float_to_file_float(s)}_r_{float_to_file_float(ratio)}_t_'+'_'+loc_time.split('_')[0]
                    log_path=os.path.join(exp_log_path,log_name)
                    os.mkdir(path=log_path)
                    i_run+=1
                    print(f"Starting run {i_run}/{nb_runs}")

                    K=int(N*ratio)
                    if config.verbose>3:
                        print(f"K/N:{K/N}")
                    times= []
                    rel_error= []
                    ests = [] 
                    calls=[]
                    if config.track_finish:
                        finish_flags=[]
                    for i in tqdm(range(config.n_rep)):
                        t=time()
                        if config.batch_opt:
                            amls_res=amls_pyt.ImportanceSplittingPytBatch(amls_gen, normal_kernel,K=K, N=N,s=s,  h=h_V_batch_pyt, 
                        tau=0 , n_max=config.n_max,clip_s=config.clip_s , 
                        s_min= config.s_min, s_max =config.s_max,verbose= config.verbose,
                        device=config.device,track_accept=config.track_accept)

                        else:
                            amls_res = amls_pyt.ImportanceSplittingPyt(amls_gen, normal_kernel,K=K, N=N,s=s,  h=h_V_batch_pyt, 
                        tau=0 , n_max=config.n_max,clip_s=config.clip_s , 
                        s_min= config.s_min, s_max =config.s_max,verbose= config.verbose,
                        device=config.device, )
                        t=time()-t
                        
                        est=amls_res[0]
                        
                        dict_out=amls_res[1]
                        if config.track_accept:
                            accept_logs=os.path.join(log_path,'accept_logs')
                            if not os.path.exists(accept_logs):
                                os.mkdir(path=accept_logs)
                            accept_rates=dict_out['accept_rates']
                            np.savetxt(fname=os.path.join(accept_logs,f'accept_rates_{i}.txt')
                            ,X=accept_rates)
                            x_T=np.arange(len(accept_rates))
                            plt.plot(x_T,accept_rates)
                            plt.savefig(os.path.join(accept_logs,f'accept_rates_{i}.png'))
                            plt.close()
                            accept_rates_mcmc=dict_out['accept_rates_mcmc']
                            x_T=np.arange(len(accept_rates_mcmc))
                            plt.plot(x_T,accept_rates_mcmc)
                            plt.savefig(os.path.join(accept_logs,f'accept_rates_mcmc_{i}.png'))
                            plt.close()
                            np.savetxt(fname=os.path.join(accept_logs,f'accept_rates_mcmc_{i}.txt')
                            ,X=accept_rates_mcmc)
                        if config.track_finish:
                            finish_flags.append(dict_out['finish_flag'])
                        times.append(t)
                        ests.append(est)
                        calls.append(dict_out['Count_h'])


                    times=np.array(times)
                    ests = np.array(ests)
                    abs_errors=np.abs(ests-p_t)
                    rel_errors=abs_errors/p_t
                    bias=np.mean(ests)-p_t

                    times=np.array(times)  
                    ests=np.array(ests)
                    calls=np.array(calls)
                    errs=np.abs(ests-p_t)

                    mean_calls=calls.mean()
                    std_calls=calls.std()
                    #fin = np.array(finished_flags)


                    np.savetxt(fname=os.path.join(log_path,'times.txt'),X=times)
                    np.savetxt(fname=os.path.join(log_path,'ests.txt'),X=ests)

                    plt.hist(times, bins=20)
                    plt.savefig(os.path.join(log_path,'times_hist.png'))
                    plt.hist(rel_errors,bins=20)
                    plt.savefig(os.path.join(log_path,'rel_errs_hist.png'))

                    #with open(os.path.join(log_path,'results.txt'),'w'):
                    results={'p_t':p_t,'method':method_name,
                    'N':N,'n_rep':config.n_rep,'T':T,'ratio':ratio,'K':K,'s':s
                    ,'min_rate':config.min_rate,'mean est':ests.mean()
                    ,'mean time':times.mean()
                    ,'std time':times.std(),
                    'mean_calls':mean_calls,
                    'std_calls':std_calls
                    ,'bias':ests.mean()-p_t,'mean abs error':abs_errors.mean(),
                    'mean rel error':rel_errors.mean(),'std est':ests.std(),'freq underest':(ests<p_t).mean()
                    ,'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,'cores_number':config.cores_number,
            'batch_opt':config.batch_opt,"d":d}
                    exp_res.append(results)
                    results_df=pd.DataFrame([results])
                    results_df.to_csv(os.path.join(log_path,'results.csv'),index=False)
                    if config.aggr_res_path is None:
                        aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                    else:
                        aggr_res_path=config.aggr_res_path
                    if config.update_agg_res:
                        if not os.path.exists(aggr_res_path):
                            cols=['p_t','method','N','rho','n_rep','T','alpha','min_rate','mean time','std time','mean est',
                            'mean_calls','std_calls',
                            'bias','mean abs error','mean rel error','std est','freq underest','gpu_name','cpu_name']
                            aggr_res_df= pd.DataFrame(columns=cols)
                        else:
                            aggr_res_df=pd.read_csv(aggr_res_path)
                        aggr_res_df = pd.concat([aggr_res_df,results_df],ignore_index=True)
                        aggr_res_df.to_csv(aggr_res_path,index=False)

exp_df=pd.DataFrame(exp_res)
exp_df.to_csv(os.path.join(exp_log_path,'exp_results.csv'),index=False)
