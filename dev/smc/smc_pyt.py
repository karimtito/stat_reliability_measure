import torch 
import math
import numpy as np
from stat_reliability_measure.dev.torch_utils import TimeStepPyt, adapt_verlet_mcmc, adapt_verlet_mcmc2, apply_gaussian_kernel,verlet_mcmc
from stat_reliability_measure.dev.utils import valid_pars_type
from stat_reliability_measure.config import Config
from stat_reliability_measure.dev.torch_utils import V_pyt, gradV_pyt
from stat_reliability_measure.dev.smc.smc_utils import nextBetaESS,nextBetaSimpESS,SimpAdaptBetaPyt,ESSAdaptBetaPyt
    
supported_beta_adapt={'ess':nextBetaESS,'simp_ess':nextBetaSimpESS,'simp':ESSAdaptBetaPyt}

def SamplerSMC(gen, V, gradV,adapt_func='', min_rate=0.8, alpha =0.1, N=300, T = 1, L=1, n_max=5000, 
max_beta=1e6, verbose=False,device=None,ess_alpha=0.875,
track_accept=False,track_beta=False,return_log_p=False,gaussian=False,
adapt_dt=False, track_finish=False, save_X=False,save_v=False,
track_calls=True,track_dt=False,track_H=False,track_v_means=False,track_ratios=False,
target_accept=0.574,accept_spread=0.1,dt_decay=0.999,dt_gain=None,
dt_min=1e-5,dt_max=1e-2,v_min_opt=True,kappa_opt=False,
 track_ess=False,M_opt=False,adapt_step=False,alpha_p=0.1,FT=False,sig_dt=0.015,L_min=1,only_duplicated=False,
 GK_opt=False,GV_opt=False,dt_d=1,skip_mh=False,
 lambda_0=1,g_target=0.8,**kwargs):
    """
      Adaptive SMC estimator with transition kernels either based on:
      underdamped Langevin dynamics  (L=1) or Hamiltonian dynamics (L>1) kernels (GK_opt=False)
      or Gaussian-Verlet kernel (GV_opt=True,GK_opt=False)
      or simple Gaussian kernels (GK_opt=True)
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
    assert v_min_opt ,"v_min_opt must be True, otherwise all the samples can be killed at the same time"
    gradient_use= not (GK_opt or GV_opt)
    #assert method_config.adapt_func.lower() in smc_pyt.supported_beta_adapt.keys(),f"select adaptive function in {smc_pyt.supported_beta_adapt.keys}"
    #adapt_func=smc_pyt.supported_beta_adapt[method_config.adapt_func.lower()]
    #cpu/gpu switch
    if device is None: 
        device= "cuda:0" if torch.cuda.is_available() else "cpu"
    if adapt_func.lower()=='simp_ess':
        adapt_func = lambda beta,v : nextBetaSimpESS(beta_old=beta,v=v,lambda_0=lambda_0,max_beta=1e6)
    elif adapt_func.lower()=='simp':
        adapt_func = lambda beta,v: SimpAdaptBetaPyt(beta,v,g_target,v_min_opt=v_min_opt)
    elif adapt_func.lower()=='ess':
        adapt_func = lambda beta,v : nextBetaESS(beta_old=beta,v=v,ess_alpha=ess_alpha,max_beta=1e6)

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
        dt_means=[]
        dt_stds=[]

    if track_H:
        H_s=[]
    if track_beta:
        betas=[]
    if track_v_means: 
        v_means=[]
    
    Count_v=0
    if track_ratios:
        g_s=[g_prod]
    
    n = 0 # Number of iterations
    finish_flag=False
    beta=0
    reach_beta_inf=False
    scale_M=None
    ## For
    while not reach_beta_inf:
        n += 1
        if n >n_max:
            print('/!\ The estimator failed. Increase n_max?')
            break


        if n==1:
            X=gen(N)
            #p=torch.randn_like(X)
            v=V(X)
            Count_v+=N
            
            grad_v=gradV(X)
            Count_v+=2*N # each call to gradV costs 2 calls to V

            

            d=X.shape[-1]
            assert dt_d==1 or dt_d==d,"dt dimension can be 1 (isotropic diff.) or d (anisotropic diff.)"
            dt_scalar =alpha*TimeStepPyt(v_x=v,grad_v_x=grad_v)
            
            dt= torch.clamp(dt_scalar*torch.ones(size=(N,dt_d),device=device)+sig_dt*torch.randn(size=(N,dt_d),device=device),min=dt_min,max=dt_max)
            ind_L=torch.randint(low=L_min,high=L,size=(N,),).to(device).float() if L_min<L else L*torch.ones(size=(N,),device=device)
            
    
        else:
            if only_duplicated and nb_to_renew>0:
                Y=X[to_renew]
                v_y=v[to_renew]
                if gradient_use:
                    grad_v_y=grad_v[to_renew]
                else:
                    grad_v_y=None
                ind_L_y=ind_L[to_renew]
                dt_y=dt[to_renew]
            else:
                Y=X
                v_y=v
                if gradient_use:
                    grad_v_y=grad_v
                else:
                    grad_v_y=None
                ind_L_y=ind_L
                dt_y=dt
            if M_opt:
                scale_M=1/Y.var(0)
                if verbose:
                    print(f"Avg. var.:{Y.var(0).mean()}")
            if GK_opt:
                Y,v_y,nb_calls,dict_out=apply_gaussian_kernel(Y=Y,v_y=v_y,T=T,beta=beta,
                V=V,)
            else:
                
                if adapt_step:
                    Y,v_y,grad_v_y,nb_calls,dict_out=mcmc_func(q=Y,v_q=v_y,grad_V_q=grad_v_y,ind_L=ind_L_y,beta=beta,gaussian=gaussian,
                        V=V,gradV=gradV,T=T, L=L,kappa_opt=kappa_opt,delta_t=dt_y,device=device,save_H=track_H,save_func=None,scale_M=scale_M,
                        alpha_p=alpha_p,dt_max=dt_max,sig_dt=sig_dt,FT=FT,verbose=verbose,L_min=L_min,
                        gaussian_verlet=GV_opt,dt_min=dt_min,skip_mh=skip_mh)
                    if FT:
                        if only_duplicated and nb_to_renew>0:
                            dt[to_renew]=dict_out['dt']
                            ind_L[to_renew]=dict_out['ind_L']
                        else:
                            dt=dict_out['dt']
                            ind_L=dict_out['ind_L']
                        if verbose>=1.5:
                            print(f"New dt mean:{dt.mean().item()}, dt std:{dt.std().item()}")
                            print(f"New L mean: {ind_L.mean().item()}, L std:{ind_L.std().item()}")
                else:
                    Y,v_y,grad_v_y,nb_calls,dict_out=verlet_mcmc(q=Y,grad_V_q=grad_v_y,beta=beta,gaussian=gaussian,
                                        V=V,gradV=gradV,T=T, L=L,kappa_opt=kappa_opt,delta_t=dt,device=device,save_H=track_H,save_func=None,
                                        scale_M=scale_M)
                    if track_H:
                        H_s.extend(list(dict_out['H']))
            if track_calls:
                Count_v+=nb_calls
            if track_accept:
                accept_rates_mcmc.append(dict_out['acc_rate'])
                accept_rate=dict_out['acc_rate'] 
                if verbose>=2.5:
                    print(f"Accept rate: {accept_rate}")
            if adapt_dt:
                accept_rate=dict_out['acc_rate'] 
                if accept_rate>target_accept+accept_spread:
                    dt*=dt_gain
                elif accept_rate<target_accept-accept_spread: 
                    dt*=dt_decay
                dt=torch.clamp(dt,min=dt_min,max=dt_max)
                if verbose>=2.5:
                    print(f"New mean dt:{dt.mean().item()}")
            if track_dt:
                dt_means.append(dt.mean().item())
                dt_stds.append(dt.std().item())
            if only_duplicated and nb_to_renew>0:
                X[to_renew]=Y
                v[to_renew]=v_y
            else:
                X=Y
                v=v_y
                


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
        beta,ess = adapt_func(beta_old,v)
        if verbose>=0.5:
            print(f'Beta old {beta_old}, new beta {beta}, delta_beta={beta-beta_old}')
        
        
        if track_ess:
            ess_.append(ess)
            if verbose:
                print(f'next ESS:{ess}')
        reach_beta_inf=beta==max_beta
        reach_rate=(v<=0).float().mean().item()
        if reach_beta_inf or reach_rate>=min_rate:
            beta=math.inf
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
                surv_idx=torch.where(G_v_min>=U)[0]
            else:
                to_renew = (G<U)
                surv_idx=torch.where(G>=U)[0] 
        g_iter=G.mean()
        g_prod*=g_iter
        if verbose>=0.5:
            print(f"g_iter:{g_iter},g_prod:{g_prod}")
        if track_ratios:
            g_s.append(g_iter)

        nb_to_renew=to_renew.int().sum().item()
        if nb_to_renew>0:
            
            prenew_idx=torch.randint(low=0,high=len(surv_idx),size=(nb_to_renew,))
            renew_idx=surv_idx[prenew_idx]
        
            X[to_renew] = X[renew_idx]
    finish_flag=(v<=0).float().mean().item() >=min_rate
    if verbose>=1.:
        print(f"finished flag:{finish_flag}")
    P_est = max(g_prod.item(),1e-250)
    if verbose>=0.5:
            print(f"g_iter_final:{g_iter},g_final:{P_est}")
    dict_out = {'p_est':P_est,'finished':finish_flag}
    if save_X:
        dict_out['X']=X
    if save_v:
        dict_out['v']=v
    if track_finish:
        dict_out['finish_flag']=finish_flag
    if track_accept:
        
        dict_out['accept_rates_mcmc']=np.array(accept_rates_mcmc)
    else:
        dict_out['accept_rates']=None
    if track_beta:
        dict_out['betas']=np.array(betas) 
    else:
        dict_out['betas']=None
    if return_log_p and P_est>0:
        dict_out['log_p_est']=math.log(P_est)
    else:
        dict_out['log_p_est']=None
    if track_calls:
        dict_out['nb_calls']=Count_v
    if track_v_means: 
        dict_out['v_means']=np.array(v_means)
    if track_dt:
        dict_out['dt_means']=dt_means
        dict_out['dt_stds']=dt_stds
    return P_est,dict_out



def SamplerSmcMulti(gen,  V, gradV,adapt_func,min_rate=0.8,alpha =0.1,N=300,T = 1,L=1,n_max=5000, 
max_beta=1e6, verbose=False,device=None,
track_accept=False,track_beta=False,return_log_p=False,gaussian=False,
adapt_dt=False,
track_calls=True,track_dt=False,track_H=False,track_v_means=False,track_ratios=False,
target_accept=0.574,accept_spread=0.1,dt_decay=0.999,dt_gain=None,
dt_min=1e-5,dt_max=1e-2,v_min_opt=False,lambda_0=1,
debug=False,kappa_opt=False,
 track_ess=False,M_opt=False,adapt_step=False,alpha_p=0.1,FT=False,sig_dt=0.015,L_min=1,only_duplicated=False,
 GK_opt=False,GV_opt=False,dt_d=1,skip_mh=False):
    """
      Adaptive SMC estimator with transition kernels either based on:
      underdamped Langevin dynamics  (L=1) or Hamiltonian dynamics (L>1) kernels (GK_opt=False)
      or Gaussian-Verlet kernel (GV_opt=True,GK_opt=False)
      or simple Gaussian kernels (GK_opt=True)
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
        dt_means=[]
        dt_stds=[]
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
    finish_flag=False
    beta=0
    reach_beta_inf=False
    scale_M=None
    ## For
    while not reach_beta_inf:
        n += 1
        if n >n_max:
            print('/!\ The estimator failed. Increase n_max?')
            break


        if n==1:
            X=gen(N)
            #p=torch.randn_like(X)
            v=V(X)
            

            d=X.shape[-1]
            assert dt_d==1 or dt_d==d,"dt dimension can be 1 (isotropic diff.) or d (anisotropic diff.)"
            dt_scalar =alpha*TimeStepPyt(V,X,gradV)
            
            dt= torch.clamp(dt_scalar*torch.ones(size=(N,dt_d),device=device)+sig_dt*torch.randn(size=(N,dt_d),device=device),min=dt_min,max=dt_max)
            ind_L=torch.randint(low=L_min,high=L,size=(N,)).float() if L_min<L else L*torch.ones(size=(N,))
            if track_calls:
                Count_v = 3*N #N calls to V + 2*N calls to grad_V
    
        else:

            Y=X
            v_y=v
            ind_L_y=ind_L
            dt_y=dt
            if M_opt:
                scale_M=1/Y.var(0)
                if verbose:
                    print(f"Avg. var.:{Y.var(0).mean()}")
            if GK_opt:
                Y,v_y,nb_calls,dict_out=apply_gaussian_kernel(Y=Y,v_y=v_y,T=T,beta=beta,
                V=V,)
            else:
                
                if adapt_step:
                    Y,v_y,nb_calls,dict_out=mcmc_func(q=Y,v_q=v_y,ind_L=ind_L_y,beta=beta,gaussian=gaussian,
                        V=V,gradV=gradV,T=T, L=L,kappa_opt=kappa_opt,delta_t=dt_y,device=device,save_H=track_H,save_func=None,scale_M=scale_M,
                        alpha_p=alpha_p,dt_max=dt_max,sig_dt=sig_dt,FT=FT,verbose=verbose,L_min=L_min,
                        gaussian_verlet=GV_opt,dt_min=dt_min,skip_mh=skip_mh)
                    if FT:
            
                
                        dt=dict_out['dt']
                        ind_L=dict_out['ind_L']
                        if verbose>=1.5:
                            print(f"New dt mean:{dt.mean().item()}, dt std:{dt.std().item()}")
                            print(f"New L mean: {ind_L.mean().item()}, L std:{ind_L.std().item()}")
                else:
                    Y,v_y,nb_calls,dict_out=verlet_mcmc(q=Y,beta=beta,gaussian=gaussian,
                                        V=V,gradV=gradV,T=T, L=L,kappa_opt=kappa_opt,delta_t=dt,device=device,save_H=track_H,save_func=None,
                                        scale_M=scale_M)
                    if track_H:
                        H_s.extend(list(dict_out['H']))
            if track_calls:
                Count_v+=nb_calls
            if track_accept:
                accept_rates_mcmc.append(dict_out['acc_rate'])
                accept_rate=dict_out['acc_rate'] 
                if verbose>=2.5:
                    print(f"Accept rate: {accept_rate}")
            if adapt_dt:
                accept_rate=dict_out['acc_rate'] 
                if accept_rate>target_accept+accept_spread:
                    dt*=dt_gain
                elif accept_rate<target_accept-accept_spread: 
                    dt*=dt_decay
                dt=torch.clamp(dt,min=dt_min,max=dt_max)
                if verbose>=2.5:
                    print(f"New mean dt:{dt.mean().item()}")
            if track_dt:
                dt_means.append(dt.mean().item())
                dt_stds.append(dt.std().item())
    
            X=Y
            v=v_y
            


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
        beta,ess = adapt_func(beta_old,v)
        if verbose>=0.5:
            print(f'Beta old {beta_old}, new beta {beta}, delta_beta={beta-beta_old}')
        
        
        if track_ess:
            ess_.append(ess)
            if verbose:
                print(f'next ESS:{ess}')
        reach_beta_inf=beta==max_beta
        reach_rate=(v<=0).float().mean().item()
        if reach_beta_inf or reach_rate>=min_rate:
            beta=math.inf
            G= v<=0
            g_iter=G.float().mean()
            g_prod*=g_iter
            break
        else:
            G = torch.exp(-(beta-beta_old)*v) #computes current value fonction
        g_iter=G.mean()
        g_prod*=g_iter
        if verbose>=0.5:
            print(f"g_iter:{g_iter},g_prod:{g_prod}")
        if track_ratios:
            g_s.append(g_iter)

     
        new_idx=torch.multinomial(input=G/G.sum(), num_samples=N)
            #renew_idx = np.random.choice(a=np.arange(N),size=to_renew.sum(),p=G/G.sum())
        X = X[new_idx]
   

    finish_flag=(v<=0).float().mean().item() >=min_rate
    
    if verbose>=1.:
        print(f"finished flag:{finish_flag}")
    P_est = max(g_prod.item(),1e-250)
    if verbose>=0.5:
            print(f"g_iter_final:{g_iter},g_final:{P_est}")
    dict_out = {'p_est':P_est,'X':X,'v':v,'finished':finish_flag}
    if track_accept:
        dict_out['accept_rates_mcmc']=np.array(accept_rates_mcmc)
    else:
        dict_out['accept_rates']=None
    if track_beta:
        dict_out['betas']=np.array(betas) 
    else:
        dict_out['betas']=None
    if return_log_p and P_est>0:
        dict_out['log_p_est']=math.log(P_est)
    else:
        dict_out['log_p_est']=None
    if track_calls:
        dict_out['calls']=Count_v
    if track_v_means: 
        dict_out['v_means']=np.array(v_means)
    if track_dt:
        dict_out['dt_means']=dt_means
    return P_est,dict_out





