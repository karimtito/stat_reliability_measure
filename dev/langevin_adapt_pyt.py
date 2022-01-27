import torch 
import math
import numpy as np
from dev.torch_utils import TimeStepPyt

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

def SimpAdaptBetaPyt(beta_old,v,g_target,search_method=dichotomic_search_d,max_beta=1e6,verbose=0,multi_output=False):

    ess_beta = lambda beta : (-(beta-beta_old)*v).mean() #increasing funtion of the parameter beta
    assert ((g_target>0) and (g_target<1)),"The target average weigh g_target must be a positive number in (0,1)"
    log_thresh= math.log(g_target)
    results= search_method(f=ess_beta,a=beta_old,b=max_beta,  thresh = log_thresh)
    new_beta, g = results[0],math.exp(results[-1])
    if verbose>0:
        print(f"g_target: {g_target}, actual g: {g}")

    if multi_output:
        return (new_beta,g)
    else:
        return new_beta


"""Implementation of Langevin Sequential Monte Carlo with adaptative tempering"""
def LangevinSMCSimpAdaptPyt(gen, l_kernel,   V, gradV,g_target=0.9,min_rate=0.8,alpha =0.1,N=300,T = 1,n_max=300, 
max_beta=1e6, verbose=False,adapt_func=SimpAdaptBetaPyt,allow_zero_est=False,device=None,mh_opt=False,mh_every=1,track_accept=False
,track_beta=False,return_log_p=False,gaussian=False, projection=None,track_calls=True,
track_v_means=True,adapt_d_t=False,target_accept=0.574,accept_spread=0.1,d_t_decay=0.999,d_t_gain=None,
debug=False):
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
        K: number of survivors                                [1x1] (1000)
        
        decay: decay rate of the strength of the kernel       [1x1] (0.9)
        T: number of repetitions of the mixing kernel (adaptative version ?)         
                                                                [1x1] (20)
        n_max: max number of iterations                       [1x1] (200)
        
        verbose: level of verbosity                           [1x1] (1)
      Returns:
         P_est: estimated probability
        
    """
    if adapt_d_t and d_t_gain is None:
        d_t_gain= 1/d_t_decay
    if device is None: 
        device= "cuda:0" if torch.cuda.is_available() else "cpu"
    # Internals
    if mh_opt and track_accept:
        accept_rates=[]
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
    Count_v+=2*N
    beta_old = 0
    if track_beta:
        betas= [beta_old]
    beta,g_0=adapt_func(beta_old=beta_old,v=v,g_target=g_target,multi_output=True,max_beta=max_beta) 
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
        if verbose>=2:
            print(f'Beta old {beta_old}, new beta {beta}, delta_beta={beta_old-beta}')
        
        
        
        G = torch.exp(-(beta-beta_old)*v) #computes current value fonction
        
        U = torch.rand(size=(N,),device=device)
        to_renew = (G<U)
        nb_to_renew=to_renew.int().sum()
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
        
        for t in range(T):
            if mh_every!=1:
                raise NotImplementedError("Metropolis-Hastings for more than one kernel step  is not implemented.")
            if mh_opt and ((n*T+t)%1==0):
                cand_X=l_kernel(X,gradV, delta_t,beta)
                with torch.no_grad():
                    #cand_v=V(cand_X).detach()
                    cand_v = V(cand_X)
                    Count_v+=N

                
                if projection is None:
                    high_diff= (X-cand_X-delta_t*gradV(cand_X)).detach()
                    low_diff=(cand_X-X-delta_t*gradV(X)).detach()
                

                log_a_high=-beta*(cand_v+(1/(4*delta_t))*torch.norm(high_diff,p=2 ,dim=1)**2)
                
                
                
                log_a_low= -beta*(v+(1/(4*delta_t))*torch.norm(low_diff,p=2,dim=1)**2)
                if gaussian: 
                    log_a_high-= 0.5*torch.sum(cand_X**2,dim=1)
                    log_a_low-= 0.5*torch.sum(X**2,dim=1)
                #alpha=torch.clamp(input=torch.exp(log_a_high-log_a_low),max=1) /!\ clamping is not necessary
                alpha=torch.exp(log_a_high-log_a_low)
                U=torch.rand(size=(N,),device=device)
                accept=U<alpha
                accept_rate=accept.float().mean().item()
                if track_accept:
                    accept_rates.append(accept_rate)
                if adapt_d_t:
                    if accept_rate>target_accept+accept_spread:
                        delta_t*=d_t_gain
                    elif accept_rate<target_accept-accept_spread: 
                        accept_rate*=d_t_decay
                X=torch.where(accept.unsqueeze(-1),input=cand_X,other=X)
                
                v=torch.where(accept, input=cand_v,other=v)
                if debug:
                    with torch.no_grad():
                        v2 = V(X)
                        Count_v+= N
                    assert torch.equal(v,v2),"/!\ error in potential computation"
            else:
                X=l_kernel(X, gradV, delta_t, beta)
            
            Count_v+= N

        v = V(X)
        Count_v+= N
        beta_old = beta
        beta,g_iter=adapt_func(beta_old=beta_old,v=v,g_target=g_target,multi_output=True,max_beta=max_beta)
        g_prod*=g_iter
        if verbose>=1.5:
            print(f"g_iter:{g_iter},g_prod:{g_prod}")
           
        
    if (v<=0).float().mean()<min_rate:
        finished_flag=True
    P_est = (g_prod*(v<=0).float().mean()).item()
    dic_out = {'p_est':P_est,'X':X,'v':v,'finished':finished_flag}
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
    if track_calls:
        dic_out['calls']=Count_v
    if track_v_means: 
        dic_out['v_means']=np.array(v_means)
    return P_est,dic_out
