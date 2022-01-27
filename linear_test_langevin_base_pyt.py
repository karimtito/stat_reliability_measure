import numpy as np 
import torch
from time import time
from scipy.special import betainc
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from torch.random import seed
from tqdm import tqdm
#import psutil
import cpuinfo
import GPUtil
from dev.torch_utils import project_ball_pyt,projected_langevin_kernel_pyt,langevin_kernel_pyt
from dev.langevin_base_pyt import LangevinSMCBasePyt
from dev.utils import dichotomic_search, float_to_file_float, str2bool

method_name="langevin_base_pyt"

class config:
    log_dir="./logs/linear_tests"
    n_rep=10
    N=40
    verbose=0
    min_rate=0.40
    T=1
    rho=90
    alpha=0.025
    n_max=5000
    tqdm_opt=True
    p_t=1e-15
    d = 1024
    epsilon = 1
    allow_zero_est=False
    save_config=True
    print_config=True
    update_agg_res=True
    aggr_res_path=None
    gaussian_latent=False
    project_kernel=True
    allow_multi_gpu=False

    g_target=None
    track_gpu=True
    track_cpu=True
    gpu_name=None
    cpu_name=None
    cores_number=None
    device=None
    torch_seed=0
    np_seed=0
    tf_seed=None
    mh_opt=False
    

parser=argparse.ArgumentParser()
parser.add_argument('--log_dir',default=config.log_dir)
parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--N',type=int,default=config.N)
parser.add_argument('--verbose',type=float,default=config.verbose)
parser.add_argument('--d',type=int,default=config.d)
parser.add_argument('--p_t',type=float,default=config.p_t)
parser.add_argument('--min_rate',type=float,default=config.min_rate)
parser.add_argument('--alpha',type=float,default=config.alpha)
parser.add_argument('--n_max',type=int,default=config.n_max)
parser.add_argument('--epsilon',type=float, default=config.epsilon)
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
parser.add_argument('--mh_opt',type=str2bool,default=config.mh_opt)
args=parser.parse_args()




for k,v in vars(args).items():
    setattr(config, k, v)

if not config.allow_multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

if config.np_seed is None:
    config.np_seed=int(time.time())
np.random.seed(seed=config.np_seed)

if config.torch_seed is None:
    config.torch_seed=int(time.time())
torch.manual_seed(config.torch_seed)


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
epsilon=config.epsilon

if not os.path.exists('./logs'):
    os.mkdir('./logs')
    os.mkdir(config.log_dir)
raw_logs_path=os.path.join(config.log_dir,'raw_logs')
if not os.path.exists(raw_logs_path):
    os.mkdir(raw_logs_path)

if config.aggr_res_path is None:
    aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
else:
    aggr_res_path=config.aggr_res_path


loc_time= float_to_file_float(time())
log_name=method_name+'_'+loc_time
log_path=os.path.join(raw_logs_path,log_name)
os.mkdir(path=log_path)
config.json=vars(args)
if config.print_config:
    print(config.json)
# if config.save_config:

p_t=config.p_t
p_target_f=lambda h: 0.5*betainc(0.5*(d+1),0.5,(2*epsilon*h-h**2)/(epsilon**2))
h,P_target = dichotomic_search(f=p_target_f,a=0,b=1,thresh=p_t,n_max=100)
get_c= lambda p:1-dichotomic_search(f=p_target_f,a=0,b=1,thresh=p,n_max=100)[0]
c=1-h
e_1= torch.Tensor([1]+[0]*(d-1)).to(device)
if config.gaussian_latent:
    print("utilisation of gaussian latent space")
    e_1_d=torch.Tensor([1]+[0]*(d+1)).to(device)
    V_batch = lambda X: torch.clamp(c-X[:,0]/torch.norm(X,axis=1),min=0,max=torch.inf)
    gradV_batch = lambda X: (X[:,0]<c)[:,None]*(X/torch.norm(X,dim=1)[:,None]-e_1_d)
    mixing_kernel = langevin_kernel_pyt
    X_gen=lambda N: torch.randn(size=(N,d+2),device=device)
else:
    V_batch = lambda X: torch.clamp(c-X[:,0], min=0, max = torch.inf)
    gradV_batch = lambda X: -e_1[None]*(X[:,0]<c)[:,None]
    prjct_epsilon = lambda X: project_ball_pyt(X, R=epsilon)
    mixing_kernel = lambda X, gradV, delta_t,beta: projected_langevin_kernel_pyt(X,gradV,delta_t,beta, projection=prjct_epsilon)
    big_gen=lambda N: torch.randn(size= (N,d+2),device=device)
    norm_and_select= lambda X: (X/torch.norm(X,dim=1)[:,None])[:,:d] 
    X_gen = lambda N: epsilon*norm_and_select(big_gen(N))

if config.verbose>=5:
    print(f"P_target:{P_target}")





iterator=tqdm(range(config.n_rep)) if config.tqdm_opt else range(config.n_rep)
times,estimates=[],[]
for i in iterator:
    t=time()
    Langevin_est,_ = LangevinSMCBasePyt(gen=X_gen, l_kernel=mixing_kernel , V=V_batch, gradV= gradV_batch,min_rate=config.min_rate, N=config.N,
     beta_0 = 0, rho=config.rho, alpha = config.alpha, n_max=config.n_max, T=config.T, verbose=config.verbose)
    t=time()-t
    times.append(t)
    estimates.append(Langevin_est)

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
results={'p_t':config.p_t,'method':method_name,'gaussian_latent':str(config.gaussian_latent),
'N':config.N,'rho':config.rho,'n_rep':config.n_rep,'T':config.T,'alpha':config.alpha,'min_rate':config.min_rate,
'mean time':times.mean(),'std time':times.std(),'mean est':estimates.mean(),'bias':estimates.mean()-config.p_t,'mean abs error':abs_errors.mean(),
'mean rel error':rel_errors.mean(),'std est':estimates.std(),'freq underest':(estimates<config.p_t).mean()
,'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,'cores_number':config.cores_number,'torch_seed':config.torch_seed,
'np_seed':config.np_seed,"mh_opt":config.mh_opt}

results_df=pd.DataFrame([results])
results_df.to_csv(os.path.join(log_path,'results.csv'),index=False)

if config.update_agg_res:
    if not os.path.exists(aggr_res_path):
        cols=['p_t','method','N','rho','n_rep','T','alpha','min_rate','mean time','std time','mean est',
        'bias','mean abs error','mean rel error','std est','freq underest','gpu_name','cpu_name','allow']
        aggr_res_df= pd.DataFrame(columns=cols)
    else:
        aggr_res_df=pd.read_csv(aggr_res_path)
    aggr_res_df = pd.concat([aggr_res_df,results_df],ignore_index=True)
    aggr_res_df.to_csv(aggr_res_path,index=False)

