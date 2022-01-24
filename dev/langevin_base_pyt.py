import torch
from dev.torch_utils import TimeStepPyt
import math
import numpy as np 
# def TimeStepPyt(V,X,gradV,p=1,p_p=2):
#     V_mean= V(X).mean()
#     V_grad_norm_mean = ((torch.norm(gradV(X),dim = 1,p=p_p)**p).mean())**(1/p)
#     with torch.no_grad():
#         result=V_mean/V_grad_norm_mean
#     return result

# """ Basic implementation of Langevin Sequential Monte Carlo """
# def LangevinSMCBasePyt(gen, l_kernel,   V, gradV,rho=10,beta_0=0, min_rate=0.8,alpha =0.1,N=300,T = 1,n_max=300, 
# verbose=False,adapt_func=None,allow_zero_est=False,device=None,mh_opt=False,mh_every=1,track_accept=False,return_log_p=True):
#     """
#       Basic version of a Langevin-based SMC estimator 
#       it can be Metropolis-adjusted (mh_opt=True) or not (mh_opt=False)
#       Args:
#          gen: generator of iid samples X_i                            [fun]
#          l_kernel: Langevin mixing kernel invariant to the Gibbs measure with 
#                    a specific temperature. It can be overdamped or underdamped
#                    and must take the form l_kernel(h,X)                             
#          V: potential function                                            [fun]
#          gradV: gradient of the potential function
         
#          N: number of samples                                  [1x1] (2000)
#          K: number of survivors                                [1x1] (1000)
        
#          decay: decay rate of the strength of the kernel       [1x1] (0.9)
#          T: number of repetitions of the mixing kernel (adaptative version ?)         
#                                                                 [1x1] (20)
#          n_max: max number of iterations                       [1x1] (200)
        
#         verbose: level of verbosity                           [1x1] (1)
#       Returns:
#          P_est: estimated probability
        
#     """

#     Internals
 
#     d =gen(1).shape[-1] # dimension of the random vectors
#     n = 1 # Number of iterations
#     finished_flag=True

#     if mh_opt and track_accept:
#         accept_rates=[]
        
#     # Init
#     step A0: generate & compute potentials
#     X = gen(N) # generate N samples
#     if device is None:
#         device=X.device

#     w= (1/N)*torch.ones(N,device=device)
#     log_w= torch.log(w)
#     with torch.no_grad():
#         v = V(X) # computes their potentials
#     Count_v = N # Number of calls to function V or it's  gradient
#     delta_t = (alpha*TimeStepPyt(V,X,gradV)).detach()
#     if verbose>=5:
#         print(f'delta_t: {delta_t}')
#     Count_v+=2*N
#     beta_old = beta_0
#     # For
#     while n<n_max and (v<=0).float().mean()<min_rate:
#         v_mean = v.mean()
#         v_std = v.std()
#         if verbose:
#             print('Iter = ',n, ' v_mean = ', v_mean.item(), " Calls = ", Count_v, "v_std = ", v_std.item())
#         if adapt_func is None:
#             delta_beta = rho*delta_t
#             beta = beta_old+delta_beta
#         else:
#             beta = adapt_func(beta,)
#         beta=beta.item()
        
#         G = torch.exp(-(beta-beta_old)*v).to('cpu') #computes current value fonction
#         log_G=(-(beta-beta_old)*v)
        
#         log_w+= log_G #update log weights
#         w = w * G #updates weights
#         if verbose>=5:
#             print(f'log_G: {log_G}') 
#             print(f'log weights: {log_w}')
#         n += 1 # increases iteration number
#         if n >=n_max:
#             w=torch.exp(log_w)
#             P_est = (w*(v.detach()<=0).float()).sum()
#             if allow_zero_est:
#                 if return_log_p:
#                     min_v= v.min()
#                     result=(torch.log(P_est).item(), min_v.item(),X,v.item())
#                 else:
#                     result=(P_est.item(),True)
#                 if track_accept:
#                     result+=(accept_rates,)
#             else:
#                 raise RuntimeError('The estimator failed. Increase n_max?')
        
#         for t in range(T):
#             if mh_opt and ((n*T+t)%mh_every==0):
#                 cand_X=l_kernel(X,gradV, delta_t,beta)
#                 with torch.no_grad():
#                     cand_v=V(cand_X).detach()
#                     cand_v = V(cand_X)
#                     Count_v+=N
#                 high_diff= (X-cand_X-delta_t*gradV(cand_X)).detach()
#                 log_a_high=-beta*(cand_v+(1/(4*delta_t))*torch.norm(high_diff,p=2 ,dim=1)**2)
#                 low_diff=(cand_X-X-delta_t*gradV(X)).detach()
#                 log_a_low= -beta*(v+(1/(4*delta_t))*torch.norm(low_diff,p=2,dim=1)**2)
#                 alpha=torch.clamp(input=torch.exp(log_a_high-log_a_low),max=1) /!\ clamping is not necessary
#                 alpha=torch.exp(log_a_high-log_a_low)
#                 U=torch.rand(size=(N,),device=device)
#                 accept=U<alpha
#                 if track_accept:
#                     accept_rates.append(accept.float().mean().item())
#                 X=torch.where(accept.unsqueeze(-1),input=cand_X,other=X)
#                 v=torch.where(accept, input=cand_v,other=v)
#             else:
#                 X=l_kernel(X, gradV, delta_t, beta)
            
#             Count_v+= N

#         X.require_grad=True
#         with torch.no_grad():
#             v = V(X)
#             Count_v+= N
#         beta_old = beta
#     if verbose>=5:
#         print(v<=0)
#     w=torch.exp(log_w)
#     P_est = (w*(v.detach()<=0).float()).sum()
    
#     if return_log_p:
#         min_v= v.min()
#         result=(torch.log(P_est).item(), min_v.item(),X,v.item())
#     else:
#         result=(P_est.item(),True)
    
#     TODO: return (P_est,dict_out)
#     dic_out = {'v':v.item(),'X':X.detach().item(),'w':w.item(),'finished':finished_flag}
#     if mh_opt and track_accept:
#         result+=(accept_rates,)
#     return result,dic_out





""" Basic implementation of Langevin Sequential Monte Carlo """
def LangevinSMCBasePyt(gen, l_kernel,   V, gradV,rho=10,beta_0=0, min_rate=0.8,alpha =0.1,N=300,T = 1,n_max=300, 
verbose=False,adapt_func=None,allow_zero_est=False,device=None,mh_opt=False,mh_every=1,track_accept=False,track_beta=False,return_log_p=True,
gaussian=True):
    """
      Basic version of a Langevin-based SMC estimator 
      it can be Metropolis-adjusted (mh_opt=True) or not (mh_opt=False)
      Args:
         gen: generator of iid samples X_i                            [fun]
         l_kernel: Langevin mixing kernel invariant to the Gibbs measure with 
                   a specific temperature. It can be overdamped or underdamped
                   and must take the form l_kernel(h,X)                             
         V: potential function                                            [fun]
         gradV: gradient of the potential function
         
         N: number of samples                                  [1x1] (2000)
         K: number of survivors                                [1x1] (1000)
        
         decay: decay rate of the strength of the kernel       [1x1] (0.9)
         T: number of repetitions of the mixing kernel (adaptative version ?)         
                                                                [1x1] (20)
         n_max: max number of iterations                       [1x1] (200)
        
        verbose: level of verbosity                           [1x1] (1)
      Returns:
         P_est: estimated probability
        
    """

    # Internals
 
    #d =gen(1).shape[-1] # dimension of the random vectors
    n = 1 # Number of iterations
    finish_flag=True #we assume the algorithm will finish in time

    if mh_opt and track_accept:
        accept_rates=[]
    if track_beta:
        betas=[]
    ## Init
    # step A0: generate & compute potentials
    X = gen(N) # generate N samples
    if device is None:
        device=X.device

    w= (1/N)*torch.ones(N,device=device)
    log_w= torch.log(w)
    with torch.no_grad():
        v = V(X) # computes their potentials
    Count_v = N # Number of calls to function V or it's  gradient
    delta_t = (alpha*TimeStepPyt(V,X,gradV)).detach()
    if verbose>=5:
        print(f'delta_t: {delta_t}')
    Count_v+=2*N
    beta_old = beta_0
    ## For
    while n<n_max and (v<=0).float().mean()<min_rate:
        v_mean = v.mean()
        v_std = v.std()
        if verbose:
            print('Iter = ',n, ' v_mean = ', v_mean.item(), " Calls = ", Count_v, "v_std = ", v_std.item())
        if adapt_func is None:
            delta_beta = rho*delta_t
            beta = beta_old+delta_beta
        else:
            beta = adapt_func(beta,)
        beta=beta.item()
        if track_beta:
            betas.append(beta)
        #G = torch.exp(-(beta-beta_old)*v).to('cpu') #computes current value fonction
        log_G=(-(beta-beta_old)*v)
        
        log_w+= log_G #update log weights
        #w = w * G #updates weights
        if verbose>=5:
            print(f'log_G: {log_G}') 
            print(f'log weights: {log_w}')
        n += 1 # increases iteration number
        if n >=n_max:
            
            if allow_zero_est:
                finish_flag=False
                
            else:
                raise RuntimeError('The estimator failed. Increase n_max?')
            break
        
        for t in range(T):
            assert mh_every==1,"Metropolis-Hastings for more than one kernel step  is not implemented."
            if mh_opt and ((n*T+t)%mh_every==0):
                cand_X=l_kernel(X,gradV, delta_t,beta)
                with torch.no_grad():
                    #cand_v=V(cand_X).detach()
                    cand_v = V(cand_X)
                    Count_v+=N
                high_diff= (X-cand_X-delta_t*gradV(cand_X)).detach()
                log_a_high=-beta*(cand_v+(1/(4*delta_t))*torch.norm(high_diff,p=2 ,dim=1)**2)
                
                low_diff=(cand_X-X-delta_t*gradV(X)).detach()
                log_a_low= -beta*(v+(1/(4*delta_t))*torch.norm(low_diff,p=2,dim=1)**2)
                if gaussian: 
                    log_a_high-= 0.5*torch.sum(cand_X**2,dim=1)
                    log_a_low-= 0.5*torch.sum(X**2,dim=1)
                #alpha=torch.clamp(input=torch.exp(log_a_high-log_a_low),max=1) /!\ clamping is not necessary
                alpha=torch.exp(log_a_high-log_a_low)
                U=torch.rand(size=(N,),device=device)
                accept=U<alpha
                if track_accept:
                    accept_rates.append(accept.float().mean().item())
                X=torch.where(accept.unsqueeze(-1),input=cand_X,other=X)
                # with torch.no_grad():
                #     v2 = V(X)
                #     Count_v+= N
                v=torch.where(accept, input=cand_v,other=v)
                # assert torch.equal(v,v2),"/!\ error in potential computation"
            else:
                X=l_kernel(X, gradV, delta_t, beta)
            
            Count_v+= N

        X.require_grad=True
        with torch.no_grad():
            v = V(X)
            Count_v+= N
        beta_old = beta
   
   
   
    if verbose>=5:
        print(v<=0)
    w=torch.exp(log_w)
    P_est = ((w*(v.detach()<=0).float()).sum()).item()
    
    dic_out = {'p_est':P_est,'X':X,'v':v,'finished':finish_flag}
    if mh_opt and track_accept:
        dic_out['accept_rates']=np.array(accept_rates)
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
    return P_est,dic_out