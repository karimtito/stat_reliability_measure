import numpy as np 
import numpy.linalg as LA
from time import time
from scipy.special import betainc
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from tqdm import tqdm

from dev.langevin_utils import project_ball, projected_langevin_kernel
from dev.langevin_adapt import SimpAdaptLangevinSMC
from dev.utils import dichotomic_search, float_to_file_float

method_name="langevin_adapt"

class config:
    log_dir="./logs/linear_tests"
    n_rep=10
    N=40
    verbose=0
    min_rate=0.40
    T=1
    rho=90
    alpha=0.1
    n_max=2000
    tqdm_opt=True
    p_t=1e-15
    d = 1024
    epsilon = 1
    

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
parser.add_argument('--tqdm_opt',type=bool,default=config.tqdm_opt)


args=parser.parse_args()

for k,v in vars(args).items():
    setattr(config, k, v)

config.json=vars(args)
print(config.json)

epsilon=config.epsilon
d=config.d

loc_time= float_to_file_float(time())
log_name=method_name+'_'+loc_time
log_path=os.path.join(config.log_dir,log_name)
os.mkdir(path=log_path)


e_1 = np.array([1]+[0]*(config.d-1))

p_target=lambda h: 0.5*betainc(0.5*(d+1),0.5,(2*epsilon*h-h**2)/(epsilon**2))
h,P_target = dichotomic_search(f=p_target,a=0,b=1,thresh=config.p_t )
assert np.isclose(a=config.p_t, b=P_target), "The dichotomic search was not precise enough."

c=1-h
V_batch = lambda X: np.clip(c-X[:,0],a_min=0, a_max = np.inf)
gradV_batch = lambda X: -e_1[None]*(X[:,0]<c)[:,None] 
if config.verbose>5:
    print(f"P_target:{P_target}")

prjct_epsilon = lambda X: project_ball(X, R=epsilon)
prjct_epsilon_langevin_kernel = lambda X, gradV, delta_t,beta: projected_langevin_kernel(X,gradV,delta_t,beta, projection=prjct_epsilon)

big_gen= lambda N: np.random.normal(size=(N,d+2))
norm_and_select= lambda X: (X/LA.norm(X, axis=1)[:,None])[:,:d] 
alt_uniform_gen_epsilon = lambda N: epsilon*norm_and_select(big_gen(N))

iterator=tqdm(range(config.n_rep)) if config.tqdm_opt else range(config.n_rep)
times,estimates=[],[]
for i in iterator:
    t=time()
    Langevin_est = SimpAdaptLangevinSMC(alt_uniform_gen_epsilon , V=V_batch, gradV= gradV_batch, N =config.N,l_kernel = prjct_epsilon_langevin_kernel, alpha = config.alpha, n_max=config.n_max, T=config.T, verbose=config.verbose)
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
plt.savefig(os.path.join(log_path,'rel_errors_hist.png'))
#with open(os.path.join(log_path,'results.txt'),'w'):
results={'p_t':config.p_t,'method':method_name
,'N':config.N,'rho':config.rho,'n_rep':config.n_rep,'T':config.T,'alpha':config.alpha,'min_rate':pd.NA,
'mean time':times.mean(),'std time':times.std(),'mean abs error':abs_errors.mean(),
'mean rel error':rel_errors.mean(),'std est':estimates.std()}
results_df=pd.DataFrame([results])
results_df.to_csv(os.path.join(log_path,'results.csv'),)

