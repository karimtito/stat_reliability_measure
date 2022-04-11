import torch 
import math
import numpy as np
from stat_reliability_measure.dev.torch_utils import TimeStepPyt, apply_l_kernel,apply_simp_kernel, normal_kernel

def dichotomic_search_d(f, a, b, thresh=0, n_max =50):
    """Implementation of dichotomic search of minimum solution for an decreasing function
        Args:
            -f: increasing function
            -a: lower bound of search space
            -b: upper bound of search space
            -thresh: threshold such that if f(x)>0, x is considered to be a solution of the problem
    
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

def SimpAdaptBetaPyt(beta_old,v,g_target,search_method=dichotomic_search_d,max_beta=1e9,verbose=0,multi_output=False,v_min_opt=False,lambda_0=None):

    """ Simple adaptive mechanism to select next inverse temperature

    Returns:
        float: new_beta, next inverse temperature
        float: g, next mean weight
    """
    ess_beta = lambda beta : torch.exp(-(beta-beta_old)*(v)).mean() #decreasing funtion of the parameter beta
    if v_min_opt:
        v_min=v.min()
        ess_beta=lambda beta : torch.exp(-(beta-beta_old)*(v-v_min)).mean() 
    assert ((g_target>0) and (g_target<1)),"The target average weigh g_target must be a positive number in (0,1)"
    results= search_method(f=ess_beta,a=beta_old,b=max_beta,thresh = g_target) 
    new_beta, g = results[0],results[-1] 
    
    if verbose>0:
        print(f"g_target: {g_target}, actual g:{g}")

    if multi_output:
        if v_min_opt:
            g=torch.exp((-(new_beta-beta_old)*v)).mean() #in case we use v_min_opt we need to output original g
        return (new_beta,g)
    else:
        return new_beta

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


# """Implementation of Langevin Sequential Monte Carlo with adaptative tempering"""
# def LangevinSMCSimpAdaptPyt(gen, l_kernel,  V, gradV,g_target=0.9,min_rate=0.8,alpha =0.1,N=300,T = 1,n_max=300, 
# max_beta=1e6, verbose=False,adapt_func=SimpAdaptBetaPyt,allow_zero_est=False,device=None,mh_opt=False,mh_every=1,track_accept=False
# ,track_beta=False,return_log_p=False,gaussian=False, projection=None,track_calls=True,
# track_v_means=True,adapt_d_t=False,target_accept=0.574,accept_spread=0.1,d_t_decay=0.999,d_t_gain=None,
# v_min_opt=False,v1_kernel=True,lambda_0=1,
# debug=False,only_duplicated=False,track_delta_t=False):
#     """
#       Adaptive Langevin SMC estimator  
#       Args:
#         gen: generator of iid samples X_i                            [fun]
#         l_kernel: Langevin mixing kernel almost invariant to the Gibbs measure with 
#                    a specific temperature. It can be overdamped or underdamped
#                    and must take the form l_kernel(h,X)                             
#         V: potential function                                            [fun]
#         gradV: gradient of the potential function
         
#         N: number of samples                                  [1x1] (2000)
#         K: number of survivors                                [1x1] (1000)
        
#         decay: decay rate of the strength of the kernel       [1x1] (0.9)
#         T: number of repetitions of the mixing kernel (adaptative version ?)         
#                                                                 [1x1] (20)
#         n_max: max number of iterations                       [1x1] (200)
        
#         verbose: level of verbosity                           [1x1] (1)
#       Returns:
#          P_est: estimated probability
        
#     """
#     if adapt_d_t and d_t_gain is None:
#         d_t_gain= 1/d_t_decay
#     if device is None: 
#         device= "cuda:0" if torch.cuda.is_available() else "cpu"
#     # Internals
#     if track_accept:
#         accept_rates=[]
#         accept_rates_mcmc=[]
#     #d =gen(1).shape[-1] # dimension of the random vectors
#     n = 1 # Number of iterations
#     finished_flag=False
#     ## Init
#     # step A0: generate & compute potentials
#     X = gen(N) # generate N samples
#     if track_v_means: 
#         v_means=[]
    
#     v = V(X) # computes their potentials
#     Count_v = N # Number of calls to function V or it's  gradient
#     delta_t = alpha*TimeStepPyt(V,X,gradV)
#     Count_v+=2*N
#     beta_old = 0
#     if track_beta:
#         betas= [beta_old]
#     beta,g_0=adapt_func(beta_old=beta_old,v=v,g_target=g_target,multi_output=True,max_beta=max_beta,v_min_opt=v_min_opt,
#     lambda_0=lambda_0) 
#     g_prod=g_0
#     print(f"g_0:{g_0}")
#     ## For
#     while (v<=0).float().mean()<min_rate:
#         v_mean = v.mean()
#         v_std = v.std()
#         if verbose:
#             print('Iter = ',n, ' v_mean = ', v_mean.item(), " Calls = ", Count_v, "v_std = ", v_std.item())
#         if track_v_means: 
#             v_means.append(v_mean.item())
#         if verbose>=2:
#             print(f'Beta old {beta_old}, new beta {beta}, delta_beta={beta_old-beta}')
        
        
        
#         G = torch.exp(-(beta-beta_old)*v) #computes current value fonction
        
#         U = torch.rand(size=(N,),device=device)
#         if v_min_opt:
#             G_v_min= torch.exp(-(beta-beta_old)*(v-v.min()))
#             to_renew=(G_v_min<U) 
#         else:
#             to_renew = (G<U) 
#         nb_to_renew=to_renew.int().sum().item()
#         if nb_to_renew>0:
#             renew_idx=torch.multinomial(input=G/G.sum(), num_samples=nb_to_renew)
#             #renew_idx = np.random.choice(a=np.arange(N),size=to_renew.sum(),p=G/G.sum())
#             X[to_renew] = X[renew_idx]
        
#         n += 1 # increases iteration number
#         if n >=n_max:
#             if allow_zero_est:
#                 break
#             else:
#                 raise RuntimeError('The estimator failed. Increase n_max?')

        
#         if only_duplicated:
#             Y=X[to_renew]
#             v_y=v[to_renew]
#         else:
#             Y=X
#             v_y=v
#         local_accept_rates=[]
#         for _ in range(T):
#             if mh_every!=1:
#                 raise NotImplementedError("Metropolis-Hastings for more than one kernel step  is not implemented.")
#             if track_accept or mh_opt:
#                 cand_Y=l_kernel(Y,gradV, delta_t,beta)
#                 with torch.no_grad():
#                     #cand_v=V(cand_Y).detach()
#                     cand_v_y = V(cand_Y)
#                     Count_v+=N

                
#                 if not gaussian:
#                     high_diff= (Y-cand_Y-delta_t*gradV(cand_Y)).detach()
#                     low_diff=(cand_Y-Y-delta_t*gradV(Y)).detach()
#                 else:
#                     if v1_kernel:
#                         high_diff=(Y-(1-(delta_t/beta))*cand_Y-delta_t*gradV(cand_Y)).detach()
#                         low_diff=(cand_Y-(1-(delta_t/beta))*Y-delta_t*gradV(Y)).detach()
#                     else:
#                         #using Orstein-Uhlenbeck version 
#                         high_diff=(Y-np.sqrt(1-(2*delta_t/beta))*cand_Y-delta_t*gradV(cand_Y)).detach()
#                         low_diff=(cand_Y-np.sqrt(1-(2*delta_t/beta))*Y-delta_t*gradV(Y)).detach()
#                 Count_v+= 2*nb_to_renew if only_duplicated else 2*N
#                 log_a_high=-beta*(cand_v_y+(1/(4*delta_t))*torch.norm(high_diff,p=2 ,dim=1)**2)
                
                
                
#                 log_a_low= -beta*(v_y+(1/(4*delta_t))*torch.norm(low_diff,p=2,dim=1)**2)
#                 if gaussian: 
#                     log_a_high-= 0.5*torch.sum(cand_Y**2,dim=1)
#                     log_a_low-= 0.5*torch.sum(Y**2,dim=1)
#                 #alpha=torch.clamp(input=torch.eYp(log_a_high-log_a_low),max=1) /!\ clamping is not necessary
#                 alpha_=torch.exp(log_a_high-log_a_low)
#                 b_size= nb_to_renew if only_duplicated else N
#                 U=torch.rand(size=(b_size,),device=device) 
#                 accept=U<alpha_
#                 accept_rate=accept.float().mean().item()
#                 if track_accept:
#                     if verbose>=3:
#                         print(f"Local accept rate: {accept_rate}")
#                     accept_rates.append(accept_rate)
#                     local_accept_rates.append(accept_rate)
#                 if adapt_d_t:
#                     if accept_rate>target_accept+accept_spread:
#                         delta_t*=d_t_gain
#                     elif accept_rate<target_accept-accept_spread: 
#                         delta_t*=d_t_decay
#                 if mh_opt:
#                     with torch.no_grad():
#                         Y=torch.where(accept.unsqueeze(-1),input=cand_Y,other=Y)
                        
#                         v_y=torch.where(accept, input=cand_v_y,other=v_y)
#                         Count_v+= nb_to_renew if only_duplicated else N
#                         if debug:
                            
#                             v2 = V(Y)
#                             #Count_v+= N
#                             assert torch.equal(v,v2),"/!\ error in potential computation"
#                 else:
#                     Y=cand_Y
#                     v_y=cand_v_y
#             else:
#                 Y=l_kernel(Y, gradV, delta_t, beta)
                
#             Count_v= Count_v+ nb_to_renew if only_duplicated else Count_v+N
            
    

#         Y,v_y,nb_calls,dict_out=apply_l_kernel(Y=Y ,v_y=v_y,delta_t=delta_t,beta=beta,V=V,gradV=gradV,l_kernel=l_kernel,
#             T=T,mh_opt=mh_opt,device=device,v1_kernel=v1_kernel,adapt_d_t=adapt_d_t, track_accept=track_accept,
#             d_t_decay=d_t_decay,d_t_gain=d_t_gain,debug=False,target_accept=target_accept,accept_spread=accept_spread,
#             gaussian=gaussian, verbose=verbose,track_delta_t=track_delta_t)
#         Count_v+=nb_calls
#         if only_duplicated:
#             with torch.no_grad():
#                 X[to_renew]=Y
#                 v_y=V(Y)
#                 Count_v+=nb_to_renew
#                 v[to_renew]=V(Y)
#         else:
#             with torch.no_grad():
#                 X=Y
#                 v=V(X)
#         del Y
        
#         if track_accept:
#             local_accept_rates=dict_out['local_accept_rates']
#             accept_rates.extend(local_accept_rates)
#             accept_rates_mcmc.append(np.array(local_accept_rates).mean())
#         if adapt_d_t:
#             delta_t = dict_out['delta_t']
#         v = V(X)
#         Count_v+= N
#         beta_old = beta
#         beta,g_iter=adapt_func(beta_old=beta_old,v=v,g_target=g_target,multi_output=True,
#         max_beta=max_beta,v_min_opt=v_min_opt,lambda_0=lambda_0)
#         g_prod*=g_iter
#         if verbose>=1.5:
#             print(f"g_iter:{g_iter},g_prod:{g_prod}")
           
        
#     if (v<=0).float().mean()<min_rate:
#         finished_flag=True
#     P_est = (g_prod*(v<=0).float().mean()).item()
#     dic_out = {'p_est':P_est,'X':X,'v':v,'finished':finished_flag}
#     if track_accept:
#         dic_out['accept_rates']=np.array(accept_rates)
#         dic_out['accept_rates_mcmc']=np.array(accept_rates_mcmc)
#     else:
#         dic_out['accept_rates']=None
#     if track_beta:
#         dic_out['betas']=np.array(betas) 
#     else:
#         dic_out['betas']=None
#     if return_log_p and P_est>0:
#         dic_out['log_p_est']=math.log(P_est)
#     else:
#         dic_out['log_p_est']=None
#     if track_calls:
#         dic_out['calls']=Count_v
#     if track_v_means: 
#         dic_out['v_means']=np.array(v_means)

   
#     return P_est,dic_out




def LangevinSMCSimpAdaptPyt(gen,  V, gradV,l_kernel,apply_kernel=apply_l_kernel,g_target=0.9,min_rate=0.8,alpha =0.1,N=300,T = 1,n_max=300, 
max_beta=1e6, verbose=False,adapt_func=SimpAdaptBetaPyt,allow_zero_est=False,device=None,mh_opt=False,mh_every=1,track_accept=False
,track_beta=False,return_log_p=False,gaussian=False, projection=None,track_calls=True,
track_v_means=True,adapt_d_t=False,track_delta_t=False,target_accept=0.574,accept_spread=0.1,d_t_decay=0.999,d_t_gain=None,
d_t_min=1e-5,d_t_max=1e-2,v_min_opt=False,v1_kernel=True,lambda_0=1,s_opt=False,
debug=False,only_duplicated=False,s =1,s_decay=0.95,s_gain=1.0001
,clip_s=True,s_min=1e-3,s_max= 3,reject_ctrl=True,reject_thresh=0.1,prog_thresh=0.1,
mult_last=True):
    """
      Adaptive Langevin SMC estimator  
      Args:
        gen: generator of iid samples X_i                            [fun]
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
    if s_opt:
        kernel_pass =0
        rejection_rate=0
    if adapt_d_t and d_t_gain is None:
        d_t_gain= 1/d_t_decay
    if device is None: 
        device= "cuda:0" if torch.cuda.is_available() else "cpu"
    # Internals
    if track_accept:
        accept_rates=[]
        accept_rates_mcmc=[]
    
    #d =gen(1).shape[-1] # dimension of the random vectors
    n = 1 # Number of iterations
    finished_flag=False
    ## Init
    # step A0: generate & compute potentials
    X = gen(N) # generate N samples
    if track_v_means: 
        v_means=[]
    
    v = V(X) # computes their potentials
    Count_v = N # Number of calls to function V or it's  gradient
    delta_t = alpha*TimeStepPyt(V,X,gradV)
    if track_delta_t:
        delta_ts=[delta_t]
    Count_v+=2*N
    beta_old = 0
    if track_beta:
        betas= [beta_old]
    beta,g_0=adapt_func(beta_old=beta_old,v=v,g_target=g_target,multi_output=True,max_beta=max_beta,v_min_opt=v_min_opt,
    lambda_0=lambda_0) 
    g_prod=g_0
    print(f"g_0:{g_0}")
    ## For
    while (v<=0).float().mean()<min_rate:




        v_mean = v.mean()
        v_std = v.std()
        if verbose:
            print('Iter = ',n, ' v_mean = ', v_mean.item(), " Calls = ", Count_v, "v_std = ", v_std.item())
        if track_v_means: 
            v_means.append(v_mean.item())
        
        
        
        G = torch.exp(-(beta-beta_old)*v) #computes current value fonction
        
        U = torch.rand(size=(N,),device=device)
        if v_min_opt:
            G_v_min= torch.exp(-(beta-beta_old)*(v-v.min()))
            to_renew=(G_v_min<U) 
        else:
            to_renew = (G<U) 
        nb_to_renew=to_renew.int().sum().item()
        if nb_to_renew>0:
            renew_idx=torch.multinomial(input=G/G.sum(), num_samples=nb_to_renew)
            #renew_idx = np.random.choice(a=np.arange(N),size=to_renew.sum(),p=G/G.sum())
            X[to_renew] = X[renew_idx]
        
        n += 1 # increases iteration number
        if n >=n_max:
            if allow_zero_est:
                break
            else:
                raise RuntimeError('The estimator failed. Increase n_max?')

        
        if only_duplicated:
            Y=X[to_renew]
            v_y=v[to_renew]
        else:
            Y=X
            v_y=v
        if not s_opt:   
            Y,v_y,nb_calls,dict_out=apply_l_kernel(Y=Y ,v_y=v_y,delta_t=delta_t,beta=beta,V=V,gradV=gradV,l_kernel=l_kernel,
            T=T,mh_opt=mh_opt,device=device,v1_kernel=v1_kernel,adapt_d_t=adapt_d_t, track_accept=track_accept,
            d_t_decay=d_t_decay,d_t_gain=d_t_gain,debug=False,target_accept=target_accept,accept_spread=accept_spread,
            gaussian=gaussian, verbose=verbose,track_delta_t=track_delta_t,
            d_t_min=d_t_min,d_t_max=d_t_max)
            if adapt_d_t:
                delta_t = dict_out['delta_t']
                if track_delta_t:
                    delta_ts.extend(dict_out['delta_ts'])
        else:
            Y,v_y,nb_calls,dict_out=apply_simp_kernel(Y,v_y=v_y,simp_kernel=normal_kernel,T=T,s=s,
            V=V,decay=s_decay,clip_s=clip_s,s_min =s_min,s_max=s_max,debug=debug,reject_thresh=reject_thresh,
            verbose=verbose, kernel_pass=kernel_pass,track_accept=track_accept,rejection_rate=rejection_rate,
            reject_ctrl=reject_ctrl,beta=beta,device=device,gaussian=gaussian)
            kernel_pass+=dict_out['l_kernel_pass']
            if reject_ctrl:
                s=dict_out['s']
                rejection_rate=dict_out['rejection_rate']

            

            
        
        Count_v+=nb_calls


        if only_duplicated:
            with torch.no_grad():
                X[to_renew]=Y
                # v_y=V(Y)
                # Count_v+=nb_to_renew
                v[to_renew]=v_y
        else:
            with torch.no_grad():
                X=Y
                v=v_y
        del Y
        
        if track_accept:
            local_accept_rates=dict_out['local_accept_rates']
            accept_rates.extend(local_accept_rates)
            accept_rates_mcmc.append(np.array(local_accept_rates).mean())
        
        #v = V(X)
        #Count_v+= N
        beta_old = beta
        beta,g_iter=adapt_func(beta_old=beta_old,v=v,g_target=g_target,multi_output=True,
        max_beta=max_beta,v_min_opt=v_min_opt,lambda_0=lambda_0)
        if s_opt:
            if rejection_rate<=reject_thresh and (beta-beta_old)/beta_old<prog_thresh:
                s = s*s_gain if not clip_s else np.clip(s*s_gain,s_min,s_max)
                if verbose>1:
                    print('Strength of kernel increased!')
                    print(f's={s}')
        if verbose>=0.5:
            print(f'Beta old {beta_old}, new beta {beta}, delta_beta={beta_old-beta}')
        
        g_prod*=g_iter
        if verbose>=0.5:
            print(f"g_iter:{g_iter},g_prod:{g_prod}")
           
    reach_rate=  (v<=0).float().mean().item() 

    finished_flag=reach_rate<min_rate
    
    P_est = (g_prod*reach_rate).item() if mult_last else g_prod.item()
    if verbose>=0.5:
            print(f"g_iter_final:{reach_rate},g_final:{P_est}")
    dic_out = {'p_est':P_est,'X':X,'v':v,'finished':finished_flag}
    if track_accept:
        dic_out['accept_rates']=np.array(accept_rates)
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
    if track_delta_t:
        dic_out['delta_ts']=delta_ts
    return P_est,dic_out
