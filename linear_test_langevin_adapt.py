import numpy as np 
import numpy.linalg as LA
from time import time
from scipy.special import betainc
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from tqdm import tqdm
#import psutil
import cpuinfo
import GPUtil
from dev.langevin_utils import langevin_kernel,project_ball, projected_langevin_kernel
from dev.langevin_adapt import SimpAdaptLangevinSMC
from dev.utils import dichotomic_search, float_to_file_float, str2bool

method_name="langevin_adapt"

class config:
    log_dir="./logs/linear_tests"
    n_rep=10
    N=40
    verbose=0
    min_rate=0.40
    T=1
    rho=None
    alpha=0.1
    n_max=5000
    tqdm_opt=True
    p_t=1e-15
    d = 1024
    epsilon = 1
    save_config=True
    update_agg_res=True
    gaussian_latent='False'
    project_kernel=True
    allow_multi_gpu=False
    g_target=0.9
    track_gpu=True
    track_cpu=True
    gpu_name=None
    cpu_name=None
    cores_number=None

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
parser.add_argument('--rho',type=float,default=config.rho)
parser.add_argument('--gaussian_latent',type=str, default=config.gaussian_latent)
parser.add_argument('--allow_multi_gpu',type=str2bool)
parser.add_argument('--g_target',type=float,default=config.g_target)
parser.add_argument('--track_gpu',type=str2bool,default=config.track_gpu)
parser.add_argument('--track_cpu',type=str2bool,default=config.track_cpu)
args=parser.parse_args()

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
    config.cpu_name=cpuinfo.get_cpu_info()['brand_raw']
    config.cores_number=os.cpu_count()



gaussian_latent= True if config.gaussian_latent.lower() in ('true','yes','y') else False
assert type(gaussian_latent)==bool, "The conversion of string gaussian_latent failed"
config.json=vars(args)
print(config.json)

epsilon=config.epsilon
d=config.d

if not os.path.exists('./logs'):
    os.mkdir('./logs')
    os.mkdir(config.log_dir)

loc_time= float_to_file_float(time())
log_name=method_name+'_'+loc_time
log_path=os.path.join(config.log_dir,log_name)
os.mkdir(path=log_path)


e_1 = np.array([1]+[0]*(config.d-1))
if gaussian_latent:
    e_1_d = np.array([1]+[0]*(config.d+1))

big_gen= lambda N: np.random.normal(size=(N,d+2))
norm_and_select= lambda X: (X/LA.norm(X, axis=1)[:,None])[:,:d] 
alt_uniform_gen_epsilon = lambda N: epsilon*norm_and_select(big_gen(N))

#computing hyperspherical cap of height h
p_target=lambda h: 0.5*betainc(0.5*(d+1),0.5,(2*epsilon*h-h**2)/(epsilon**2))
h,P_target = dichotomic_search(f=p_target,a=0,b=1,thresh=config.p_t )
assert np.isclose(a=config.p_t, b=P_target), "The dichotomic search was not precise enough."
c=1-h
if gaussian_latent:
    print('using gaussian space')
    V_batch = lambda X: np.clip(c-X[:,0]/LA.norm(X,axis=1),a_min=0,a_max=np.inf)
    gradV_batch = lambda X: (X[:,0]<c)[:,None]*(c*X/LA.norm(X,axis=1)[:,None]-e_1_d[None])
    mixing_kernel = langevin_kernel
    X_gen=lambda N: np.random.normal(size=(N,d+2))
else:
    V_batch = lambda X: np.clip(c-X[:,0],a_min=0, a_max = np.inf)
    gradV_batch = lambda X: -e_1[None]*(X[:,0]<c)[:,None]
    prjct_epsilon = lambda X: project_ball(X, R=epsilon)
    mixing_kernel = lambda X, gradV, delta_t,beta: projected_langevin_kernel(X,gradV,delta_t,beta,
 projection=prjct_epsilon)
    big_gen=lambda N: np.random.normal(size= (N,d+2))
    norm_and_select= lambda X: (X/LA.norm(X, axis=1)[:,None])[:,:d] 
    X_gen = lambda N: epsilon*norm_and_select(big_gen(N))


if config.verbose>3:
    print(f"P_target:{P_target}")

iterator=tqdm(range(config.n_rep)) if config.tqdm_opt else range(config.n_rep)
times,estimates=[],[]
for i in iterator:
    t=time()
    Langevin_est = SimpAdaptLangevinSMC(gen=X_gen, V=V_batch, gradV= gradV_batch, N =config.N,
    l_kernel = mixing_kernel, alpha = config.alpha, n_max=config.n_max, T=config.T, verbose=config.verbose,
    g_target=config.g_target)
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
plt.hist(rel_errors,bins=10)
plt.savefig(os.path.join(log_path,'rel_errors.png'))


#with open(os.path.join(log_path,'results.txt'),'w'):
results={'p_t':config.p_t,'method':method_name,'gaussian_latent':str(config.gaussian_latent),
'N':config.N,'rho':config.rho,'n_rep':config.n_rep,'T':config.T,'alpha':config.alpha,'min_rate':config.min_rate,
'mean time':times.mean(),'std time':times.std(),'mean est':estimates.mean(),'bias':estimates.mean()-config.p_t,'mean abs error':abs_errors.mean(),
'mean rel error':rel_errors.mean(),'std est':estimates.std(),'freq underest':(estimates<config.p_t).mean()
,'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,'cores_number':config.cores_number,'g_target':config.g_target}
results_df=pd.DataFrame([results])
results_df.to_csv(os.path.join(log_path,'results.csv'),)
aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')

if config.update_agg_res:
    if not os.path.exists(aggr_res_path):
        print('aggregate results csv file not found')
        cols=['p_t','method','gaussian_latent','N','rho','n_rep','T','alpha','min_rate','mean time','std time','mean est',
        'bias','mean abs error','mean rel error','std est','freq underest','gpu_name','cpu_name','g_target']
        agg_res_df= pd.DataFrame(columns=cols)

    else:
        agg_res_df=pd.read_csv(aggr_res_path)
    agg_res_df = pd.concat([agg_res_df,results_df],ignore_index=True)
    agg_res_df.to_csv(aggr_res_path,index=False)
