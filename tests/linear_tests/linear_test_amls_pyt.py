import numpy as np 
import numpy.linalg as LA
from time import time
from pydantic import conint
from scipy.special import betainc
import matplotlib.pyplot as plt
import pandas as pd
import os
import GPUtil
import cpuinfo
import argparse
from tqdm import tqdm
import torch

from stat_reliability_measure.dev.langevin_utils import langevin_kernel, project_ball, projected_langevin_kernel
from stat_reliability_measure.dev.utils import dichotomic_search, float_to_file_float,str2bool,str2intList,str2floatList
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
    gaussian_latent =False
    
    tqdm_opt=True
    save_config = True
    print_config=True
    update_agg_res=True
    aggr_res_path = None

    device = None

    log_dir="../../logs/linear_tests"
    batch_opt=False
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
parser.add_argument('--gaussian_latent',type=str2bool, default=config.gaussian_latent)

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

gaussian_latent= config.gaussian_latent
assert type(gaussian_latent)==bool, "The conversion of string to bool for 'gaussian_latent' failed"



if not os.path.exists('../../logs'):
    os.mkdir('../../logs')
    os.mkdir(config.log_dir)
elif not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir
    )
raw_logs_path=os.path.join(config.log_dir,'raw_logs')
if not os.path.exists(raw_logs_path):
    os.mkdir(raw_logs_path)

loc_time= datetime.today().isoformat().split('.')[0]
log_name=method_name+'_'+loc_time
log_path=os.path.join(raw_logs_path,log_name)
os.mkdir(path=log_path)
config.json=vars(args)
if config.print_config:
    print(config.json)
# if config.save_confi
# if config.save_config:
#     with open(file=os.path.join(),mode='w') as f:
#         f.write(config.json)

d=config.d
epsilon=config.epsilon
e_1 = torch.Tensor([1]+[0]*(d-1),device=config.device)
p_target_f=lambda h: 0.5*betainc(0.5*(d+1),0.5,(2*epsilon*h-h**2)/(epsilon**2))
i_run=0
for p_t in config.p_range:

    h,P_target = dichotomic_search(f=p_target_f,a=0,b=1,thresh=p_t,n_max=100)
    get_c= lambda p:1-dichotomic_search(f=p_target_f,a=0,b=1,thresh=p,n_max=100)[0]
    c=1-h
    if config.verbose>5:
        print(f"P_target:{P_target}")
    def V_batch_pyt(X,c=c):
        return torch.clamp(input=c-X[:,0],min=0, max = None)
    amls_gen = lambda N: torch.randn(size=(N,d+2),device=config.device)
    batch_transform = lambda x: (x/torch.norm(x,dim=-1)[:,None])[:d]
    normal_kernel =  lambda x,s : (x + s*torch.randn(size = x.shape,device=config.device))/np.sqrt(1+s**2) #normal law kernel, appliable to vectors 
    h_V_batch_pyt= lambda x: -V_batch_pyt(batch_transform(x)).reshape((x.shape[0],1))

    for T in config.T_range:
        for N in config.N_range: 
            for s in config.s_range:
                for ratio in config.ratio_range: 
                    i_run+=1
                    print(f"Starting run {i_run}/{nb_runs}")
                    K=int(N*ratio)
                    if config.verbose>3:
                        print(f"K/N:{K/N}")
                    times= []
                    rel_error= []
                    estimates = [] 
                    for i in tqdm(range(config.n_rep)):
                        t=time()
                        if config.batch_opt:
                            amls_res=amls_pyt.ImportanceSplittingPytBatch(amls_gen, normal_kernel,K=K, N=N,s=s,  h=h_V_batch_pyt, 
                        tau=0 , n_max=config.n_max,clip_s=config.clip_s , 
                        s_min= config.s_min, s_max =config.s_max,verbose= config.verbose,
                        device=config.device)

                        else:
                            amls_res = amls_pyt.ImportanceSplittingPyt(amls_gen, normal_kernel,K=K, N=N,s=s,  h=h_V_batch_pyt, 
                        tau=0 , n_max=config.n_max,clip_s=config.clip_s , 
                        s_min= config.s_min, s_max =config.s_max,verbose= config.verbose,
                        device=config.device)
                        t=time()-t
                        est=amls_res[0]
                        times.append(t)
                        estimates.append(est)

                    times=np.array(times)
                    estimates = np.array(estimates)
                    abs_errors=np.abs(estimates-config.p_t)
                    rel_errors=abs_errors/config.p_t
                    bias=np.mean(estimates)-config.p_t

                    np.savetxt(fname=os.path.join(log_path,'times.txt'),X=times)
                    np.savetxt(fname=os.path.join(log_path,'estimates.txt'),X=estimates)

                    plt.hist(times, bins=10)
                    plt.savefig(os.path.join(log_path,'times_hist.png'))
                    plt.hist(estimates,bins=10)
                    plt.savefig(os.path.join(log_path,'estimates_hist.png'))
                    #with open(os.path.join(log_path,'results.txt'),'w'):
                    results={'p_t':p_t,'method':method_name,'gaussian_latent':str(config.gaussian_latent),
                    'N':config.N,'n_rep':config.n_rep,'T':config.T,'ratio':ratio,'K':K,'s':s
                    ,'min_rate':config.min_rate,'mean time':times.mean(),'std time':times.std()
                    ,'mean est':estimates.mean(),'bias':estimates.mean()-p_t,'mean abs error':abs_errors.mean(),
                    'mean rel error':rel_errors.mean(),'std est':estimates.std(),'freq underest':(estimates<p_t).mean()
                    ,'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,'cores_number':config.cores_number,
                    'batch_opt':config.batch_opt}

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

        

