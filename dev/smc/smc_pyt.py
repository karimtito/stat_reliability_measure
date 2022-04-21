import torch 
import math
import numpy as np
from stat_reliability_measure.dev.torch_utils import TimeStepPyt, adapt_verlet_mcmc, apply_l_kernel,apply_simp_kernel, normal_kernel,verlet_mcmc



def dichotomic_search_d(f, a, b, thresh=0, n_max =50):
    """Implementation of dichotomic search of minimum solution for an decreasing function
        Args:
            -f: increasing function
            -a: lower bound of search space
            -b: upper bound of search space
            -thresh: threshold such that if f(x)>=thresh, x is considered to be a solution of the problem
    
    """
    low = a
    high = b
     
    i=0
    while i<n_max:
        i+=1
        if f(low)<=thresh:
            return low, f(low)
        mid = 0.5*(low+high)
        if f(mid)<thresh:
            high=mid
        else:
            low=mid

    return high, f(high)



def ESSAdaptBetaPyt(beta_old, v,lambda_0=1,max_beta=1e9,g_target=None,v_min_opt=None,multi_output=False):
    """Adaptive inverse temperature selection based on an approximation 
    of the ESS critirion 
    v_min_opt and g_target are unused dummy variables for compatibility
    """
    delta_beta=(lambda_0/v.std().item())
    if multi_output:
        new_beta=min(beta_old+delta_beta,max_beta)
        g=torch.exp(-(delta_beta)*v).mean()
        res=(new_beta,g)
    else:
        res=min(beta_old+delta_beta,max_beta)
    return res


def nextBetaSimpESS(beta_old, v,lambda_0=1,max_beta=1e9,multi_output=False):
    """Adaptive inverse temperature selection based on an approximation 
    of the ESS critirion 
    v_min_opt and g_target are unused dummy variables for compatibility
    """
    delta_beta=(lambda_0/v.std().item())
    res=min(beta_old+delta_beta,max_beta)
    return res,None

def bissection(f,a,b,target,thresh,n_max=100):
    """Implementation of binary search of method to find root of continuous function
           Args:
            -f: continuous function
            -a: lower bound of search space
            -b: upper bound of search space
            -thresh: threshold such that if f(x)>0, x is considered to be a solution of the problem
    
    """
    low = a
    assert f(a)>target-thresh
    assert f(b)<target+thresh
    high = b
     
    i=1
    while i<n_max:
        mid = 0.5*(low+high)
        if f(mid)>target+thresh:
            low=mid
        elif f(mid)<target-thresh:
            high=mid
        else:
            return mid,f(mid)
        i+=1
    mid = 0.5*(low+high)
    return mid, f(mid)



def nextBetaESS(beta_old,v,ess_alpha,max_beta=1e6,verbose=0,thresh=1e-3,debug=False,eps_ess=1e-10):
    """Adaptive selection of next inverse temperature based on the
     ESS critirion 
    v_min_opt and g_target are unused dummy variables for compatibility
    """
    N=v.shape[0]
    target_ess=int(ess_alpha*N)
    ESS = lambda beta: (torch.exp(-(beta-beta_old)*v).sum())**2/(torch.exp(-2*(beta-beta_old)*v).sum()+eps_ess)
    if debug:
        assert ESS(beta_old)>target_ess+thresh,"Target is chosen too close to N."
    if ESS(max_beta)>=target_ess-thresh:
        return max_beta,ESS(max_beta)
    else:
        new_beta,new_ess=bissection(f=ESS,a=beta_old,b=max_beta,target=target_ess,thresh=thresh)
        if verbose>=1:
            print(f"New beta:{new_beta}, new ESS:{new_ess}")
        return new_beta,new_ess
    
supported_beta_adapt={'ess':nextBetaESS,'simp_ess':nextBetaSimpESS}

def tuneFT(X,dt,L):
    pass




def SamplerSMC(gen,  V, gradV,adapt_func,ess_alpha=0.8,min_rate=0.8,alpha =0.1,N=300,T = 1,L=1,n_max=300, 
max_beta=1e6, verbose=False,device=None,
track_accept=False,track_beta=False,return_log_p=False,gaussian=False,
adapt_dt=False,
track_calls=True,track_dt=False,track_H=False,track_v_means=False,track_ratios=False,
target_accept=0.574,accept_spread=0.1,dt_decay=0.999,dt_gain=None,
dt_min=1e-5,dt_max=1e-2,v_min_opt=False,lambda_0=1,
debug=False,kappa_opt=False,
 track_ess=False,M_opt=False,adapt_step=False,alpha_p=0.1,FT=False,sig_dt=0.015):
    """
      Adaptive SMC estimator with transition kernels either based on:
      underdamped Langevin dynamics  (L=1) or Hamiltonian dynamics (L>1) kernels
      Args:
        gen: generator of iid samples X_i with respect to the reference measure  [fun]
        l_kernel: Langevin mixing kernel almost invariant to the Gibbs measure with 
                   a specific temperature. It can be overdamped or underdamped
                   and must take the form l_kernel(h,X)                             
        V: potential function                                            [fun]
        gradV: gradient of the potential function
         
        N: number of samples                                  [1x1] (2000)
       
        
        decay: decay rate of the strength of the kernel       [1x1] (0.9)
        T: number of repetitions of the mixing kernel (adaptative version ?)         
                                                                [1x1] (20)
        n_max: max number of iterations                       [1x1] (200)
        
        verbose: level of verbosity                           [1x1] (1)
      Returns:
         P_est: estimated probability
        
    """
    #cpu/gpu switch
    if device is None: 
        device= "cuda:0" if torch.cuda.is_available() else "cpu"

    # adaptative parameters
    if adapt_dt and dt_gain is None:
        dt_gain= 1/dt_decay

    mcmc_func=verlet_mcmc if not adapt_step else adapt_verlet_mcmc
    #Initializing failure probability
    g_prod=1

    # Trackers initialization
    if track_accept:
        accept_rate =[]
        accept_rates_mcmc=[]
    if track_ess:
        ess_=[]
    if track_dt:
        dt_s=[]
    if track_H:
        H_s=[]
    if track_beta:
        betas=[]
    if track_v_means: 
        v_means=[]
    if track_calls:
        Count_v=0
    if track_ratios:
        g_s=[g_prod]
    
    n = 0 # Number of iterations
    finished_flag=False
    beta=0
    reach_beta_inf=False
    scale_M=None
    ## For
    while not reach_beta_inf:
        n += 1
        if n >n_max:
            raise RuntimeError('The estimator failed. Increase n_max?')

        if n==1:
            X=gen(N)
            #p=torch.randn_like(X)
            v=V(X)

            d=X.shape[-1]
            dt_scalar =torch.clamp(alpha*TimeStepPyt(V,X,gradV),min=dt_min,max=dt_max)
            dt= torch.clamp(dt_scalar*torch.ones(size=(1,d),device=device)+sig_dt*torch.randn(size=(1,d),device=device),min=0,max=dt_max)
            if track_calls:
                Count_v = 3*N 
        else:
            if M_opt:
                scale_M=1/X.var(0)
                if verbose>=0.1:
                    print(f"avg. var:{X.var(0).mean()}")
            if adapt_step:
                X,v,nb_calls,dict_out=mcmc_func(q=X,beta=beta,gaussian=gaussian,
                    V=V,gradV=gradV,T=T, L=L,kappa_opt=kappa_opt,delta_t=dt,device=device,save_H=track_H,save_func=None,scale_M=scale_M,
                    alpha_p=alpha_p,dt_max=dt_max,sig_dt=sig_dt,FT=FT,verbose=verbose)
            else:
                X,v,nb_calls,dict_out=verlet_mcmc(q=X,beta=beta,gaussian=gaussian,
                                    V=V,gradV=gradV,T=T, L=L,kappa_opt=kappa_opt,delta_t=dt,device=device,save_H=track_H,save_func=None,
                                    scale_M=scale_M)
            if track_calls:
                Count_v+=nb_calls
            if track_accept:
                accept_rates_mcmc.append(dict_out['acc_rate'])
            if track_H:
                H_s.extend(list(dic_out['H']))
            if adapt_dt:
                accept_rate=dict_out['acc_rate'] 
                if verbose>=2.5:
                    print(f"Accept rate: {accept_rate}")
                if accept_rate>target_accept+accept_spread:
                    dt*=dt_gain
                elif accept_rate<target_accept-accept_spread: 
                    dt*=dt_decay
                if dt_min is not None:
                    dt=torch.clamp(dt,min=dt_min)
                if dt_max is not None:
                    dt=torch.clamp(dt,max=dt_max)
                if verbose>=2.5:
                    print(f"New mean dt:{dt.mean().item()}")
                if track_dt:
                    dt_s.append(dt.mean())
                


        #Printing progress
        if track_v_means or verbose:
            v_mean = v.mean()
            v_std = v.std()
            if track_v_means: 
                v_means.append(v_mean.item())
        if verbose:
            print('Iter = ',n, ' v_mean = ', v_mean.item(), " Calls = ", Count_v, "v_std = ", v_std.item())
        
        #Selecting next beta
        beta_old=beta
        beta,ess = adapt_func(beta_old,v,)
        if verbose>=0.5:
            print(f'Beta old {beta_old}, new beta {beta}, delta_beta={beta-beta_old}')
        
        if track_ess:
            ess_.append(ess)
        reach_beta_inf=beta==max_beta
        reach_rate=(v<=0).float().mean().item()
        if reach_beta_inf or reach_rate>=min_rate:
            beta=torch.inf
            G= v<=0
            g_iter=G.float().mean()
            g_prod*=g_iter
            break
        else:
            G = torch.exp(-(beta-beta_old)*v) #computes current value fonction
            
            U = torch.rand(size=(N,),device=device)
            
            
            if v_min_opt:
                G_v_min= torch.exp(-(beta-beta_old)*(v-v.min()))
                to_renew=(G_v_min<U) 
            else:
                to_renew = (G<U) 
        g_iter=G.mean()
        g_prod*=g_iter
        if verbose>=0.5:
            print(f"g_iter:{g_iter},g_prod:{g_prod}")
        if track_ratios:
            g_s.append(g_iter)
        
        nb_to_renew=to_renew.int().sum().item()
        if nb_to_renew>0:
            renew_idx=torch.multinomial(input=G/G.sum(), num_samples=nb_to_renew)
            #renew_idx = np.random.choice(a=np.arange(N),size=to_renew.sum(),p=G/G.sum())
            X[to_renew] = X[renew_idx]
   

    finished_flag=(v<=0).float().mean().item() >=min_rate
    if verbose>=1.:
        print(f"finished flag:{finished_flag}")
    P_est = g_prod.item()
    if verbose>=0.5:
            print(f"g_iter_final:{g_iter},g_final:{P_est}")
    dic_out = {'p_est':P_est,'X':X,'v':v,'finished':finished_flag}
    if track_accept:
        dic_out['accept_rates_mcmc']=np.array(accept_rates_mcmc)
    else:
        dic_out['accept_rates']=None
    if track_beta:
        dic_out['betas']=np.array(betas) 
    else:
        dic_out['betas']=None
    if return_log_p and P_est>0:
        dic_out['log_p_est']=math.log(P_est)
    else:
        dic_out['log_p_est']=None
    if track_calls:
        dic_out['calls']=Count_v
    if track_v_means: 
        dic_out['v_means']=np.array(v_means)
    if track_dt:
        dic_out['dts']=dt_s
    return P_est,dic_out
