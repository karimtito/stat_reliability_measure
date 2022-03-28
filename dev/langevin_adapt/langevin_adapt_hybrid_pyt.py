import torch 
import math
import numpy as np
from stat_reliability_measure.dev.torch_utils import TimeStepPyt,apply_l_kernel,apply_simp_kernel

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

norm_kernel=lambda X,s: 1/math.sqrt(1+s**2)*(X+s*torch.randn_like(X))


"""Implementation of Langevin Sequential Monte Carlo with adaptative tempering"""
def LangevinSMCAdaptHybridPyt(gen, l_kernel,score_func,  V, gradV,simp_kernel=norm_kernel,g_target=0.9,min_rate=0.8,alpha =0.1,N=300,K=250,T = 1,n_max=300, 
max_beta=1e6, verbose=False,adapt_func=SimpAdaptBetaPyt,allow_zero_est=False,device=None,mh_opt=False,mh_every=1,track_accept=False
,track_beta=False,return_log_p=False,gaussian=False, projection=None,track_calls=True,
track_v_means=True,adapt_d_t=False,target_accept=0.574,accept_spread=0.1,d_t_decay=0.999,d_t_gain=None,d_t_max=None, d_t_min=None,
v_min_opt=False,v1_kernel=True,lambda_0=1, s=1,
debug=False,only_duplicated=False, L_target=0,
rejection_ctrl = True, reject_thresh=0.9, gain_rate = 1.0001, prog_thresh=0.01,clip_s=False
,s_min=8e-3,s_max=3,decay=0.95,g_t_0=0.65,s_opt=True,track_delta_t=False):
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
        if track_delta_t:
            delta_ts=[]
    if device is None: 
        device= "cuda:0" if torch.cuda.is_available() else "cpu"
    # Internals
    track_reject=track_accept
    if track_accept:
        accept_rates=[]
        accept_rates_mcmc=[]
    if track_reject:
        reject_rates=[]
        reject_rates_mcmc=[]
    #d =gen(1).shape[-1] # dimension of the random vectors
    n = 1 # Number of iterations
    finished_flag=False
    ## Init
    # step A0: generate & compute potentials
    X = gen(N) # generate N samples
    if track_v_means: 
        v_means=[]
    with torch.no_grad():
        SX=score_func(X)
    ind= torch.argsort(input=SX,dim=0,descending=True).squeeze(-1)
    L_j= SX[ind][K] # set the threshold to (K+1)-th score level
    V_= lambda X: torch.clamp(L_j-score_func(X),min=0)
    
    v=torch.clamp(L_j-SX,min=0)
    #v = V(X) # computes their potentials
    Count_v = N # Number of calls to function V or it's  gradient
    if verbose>=1:
        print('Iter = ',n, ' L_j = ', L_j.item(), "h_mean",SX.mean().item(),  " Calls = ", Count_v)
    
    delta_t = alpha*TimeStepPyt(V,X,gradV)
    Count_v+=2*N
    beta_old = 0
    if track_beta:
        betas= [beta_old]
    beta,g_0=adapt_func(beta_old=beta_old,v=v,g_target=g_t_0,multi_output=True,max_beta=max_beta,v_min_opt=v_min_opt,
    lambda_0=lambda_0) 
    
    if verbose>=1:
            print(f'Beta old {beta_old}, new beta {beta}, delta_beta={beta_old-beta}')
    
    g_prod=g_0
    if verbose>=1:
        print(f"g_0:{g_0}")
    thresh=1e-3
    kernel_pass=0
    rejection_rate=0

    while L_j<L_target-thresh:
        
        n += 1 # increases iteration number
        if n >=n_max:
            if allow_zero_est:
                break
            else:
                raise RuntimeError('The estimator failed. Increase n_max?')
            
        with torch.no_grad():

            y = X[ind[0:K],:]
            Sy = SX[ind[0:K]] # Keep their scores in SY
            if verbose>=1.5:
                worst_score=Sy[-1]
                print(f"worst score:{worst_score}")
            
            #VY = torch.clamp(L_j-SY, min=0)
            Vy=v[ind[0:K]]
            
           
            #Z = torch.zeros((N-K,d),device=device)
            #SZ = torch.zeros((N-K,1),device=device)
            #resample randomly from K best particles
            ind_=torch.multinomial(input=torch.ones(size=(K,)),num_samples=N-K,replacement=True).squeeze(-1)
             # step D: refresh samples
            Z=y[ind_,:] 
            SZ=Sy[ind_]
            VZ=Vy[ind_]
        # if not kernel_test:
        #     l_reject_rates=[]
        #     for _ in range(T):
        #         with torch.no_grad():
        #             W = simp_kernel(Z,s) # propose a refreshed samples
        #             kernel_pass+=(N-K)
        #             #SW = score_func(W) # compute their scores
        #             #VW_= torch.clamp(L_j-SW,min=0)
        #             VW=V_(W)
        #             #assert torch.equal(input=VW,other=VW_)
        #             Count_v+= (N-K)
        #             #accept_flag= SW>L_j
                    
                        
                    
                    

                                    
        #             Count_v+= 2*nb_to_renew if only_duplicated else 2*N
        #             high_diff=(W-(1/math.sqrt(1+s**2))*Z)
        #             low_diff=(Z-(1/math.sqrt(1+s**2))*W)
                    
        #             log_a_high=-beta*VW-(s**2/(1+s**2))*torch.sum(high_diff**2,dim=1)
        #             log_a_low= -beta*VZ-(s**2/(1+s**2))*torch.sum(low_diff**2,dim=1)


        #             if gaussian: 
        #                 log_a_high-= 0.5*torch.sum(W**2,dim=1)
        #                 log_a_low-= 0.5*torch.sum(Z**2,dim=1)
        #             #alpha=torch.clamp(input=torch.eYp(log_a_high-log_a_low),max=1) /!\ clamping is not necessary
        #             alpha_=torch.exp(log_a_high-log_a_low)
        #             b_size= N-K
        #             U=torch.rand(size=(b_size,),device=device) 
        #             accept_flag=U<alpha_
        #             if verbose>=2:
        #                 print(f"Local accept ratio :{accept_flag.float().mean().item()}")
        #             Z=torch.where(accept_flag.unsqueeze(-1),input=W,other=Z)
        #             VZ=torch.where(accept_flag,input=VW,other=VZ)
                
                    
        #             #assert torch.equal(input=VZ,other=VZ_)
        #             reject_flag=1-accept_flag.float()
        #             rejection_rate  = (kernel_pass-(N-K))/kernel_pass*rejection_rate+(1./kernel_pass)*((1-accept_flag.float()).sum().item())
                    
        #             if track_reject:
        #                 l_reject_rates.append(reject_flag.float().mean().item())
        #                 reject_rates.append(reject_flag.float().mean().item())


        #             if rejection_ctrl and rejection_rate>=reject_thresh:
                        
        #                 s = s*decay if not clip_s else np.clip(s*decay,a_min=s_min,a_max=s_max)
        #                 if verbose>1:
        #                     print('Strength of kernel diminished!')
        #                     print(f's={s}')
        if s_opt:
            Z,VZ,nb_calls,dict_out=apply_simp_kernel(Y=Z,v_y=VZ,simp_kernel=simp_kernel,T=T,beta=beta,
            s=s,V=V_,gaussian=gaussian,device=device,decay=decay,clip_s=clip_s,s_min=s_min,s_max=s_max,
            debug=debug,verbose=verbose,rejection_rate=rejection_rate,kernel_pass=kernel_pass,track_accept=track_accept,
            reject_ctrl=rejection_ctrl,reject_thresh=reject_thresh,d_t_max=d_t_max ,d_t_min=d_t_min)
            kernel_pass=dict_out['l_kernel_pass']
            if rejection_ctrl:
                s=dict_out['s']
                rejection_rate=dict_out['rejection_rate']
        else:
            Z,VZ,nb_calls,dict_out=apply_l_kernel(Y=Z ,v_y=VZ,delta_t=delta_t,beta=beta,V=V,gradV=gradV,l_kernel=l_kernel,
            T=T,mh_opt=mh_opt,device=device,v1_kernel=v1_kernel,adapt_d_t=adapt_d_t, track_accept=track_accept,
            d_t_decay=d_t_decay,d_t_gain=d_t_gain,debug=False,target_accept=target_accept,accept_spread=accept_spread,
            gaussian=gaussian, verbose=verbose,track_delta_t=track_delta_t)
            if adapt_d_t:
                delta_t = dict_out['delta_t']
                if track_delta_t:
                    delta_ts.extend(dict_out['delta_ts'])
            #raise NotImplementedError("Langevin option for level-progression is not implemented yet.")

        Count_v+=nb_calls

        if track_accept:
            local_accept_rates=dict_out['local_accept_rates']
            accept_rates.extend(local_accept_rates)
            accept_rate_mcmc=np.array(local_accept_rates).mean()
            if verbose>=2:
                print(accept_rate_mcmc)
            accept_rates_mcmc.append(accept_rate_mcmc)

        
        
             
        SZ=score_func(Z)
        VZ=torch.clamp(L_j-SZ,min=0)
        #SY=score_func(Y)
        with torch.no_grad():
        # step A: update set X and the scores
            X[:K,:] = y # copy paste the old samples of Y into X
            SX[:K] = Sy
            v[:K] = Vy
            X[K:N,:] = Z # copy paste the new samples of Z into X
            SX[K:N] = SZ
            v[K:N] = VZ
            # step B: Find new threshold
            
            ind = torch.argsort(SX,dim=0,descending=True).squeeze(-1) # sort in descending order
            S_sort= SX[ind]
            new_tau = S_sort[K]
        if verbose>=1.5:
            print(f"new_tau={new_tau}")
        assert new_tau>L_j,"L_j sequence should be increasing!"
        

        if s_opt:
            if rejection_rate<=reject_thresh and (new_tau-L_j)/L_j<prog_thresh:
                s = s*gain_rate if not clip_s else np.clip(s*gain_rate,s_min,s_max)
                if verbose>1:
                    print('Strength of kernel increased!')
                    print(f's={s}')
        L_j = torch.clamp(S_sort[K], max=0) # set the threshold to (K+1)-th
        old_v=v
        with torch.no_grad():
            v=torch.clamp(L_j-SX,min=0)
        G=torch.exp(-beta*(v-old_v))
        g_iter=G.mean().item()
        g_prod*=g_iter
        if verbose>=1.5:
            print(f"g_iter:{g_iter},g_prod:{g_prod}")
        h_mean = SX.mean()
        if verbose>=1:
            print('Iter = ',n, ' L_j = ', L_j.item(), "h_mean",h_mean.item(),  " Calls = ", Count_v," beta = ",beta)

        if track_reject:
            if verbose>1:
                print(f'Rejection rate: {rejection_rate}')
            #rejection_rates+=[rejection_rate]

    old_v=v        
    L_j=0
    v=V_(X)
    G=torch.exp(-beta*(v-old_v))
    g_iter=G.mean().item()
    g_prod*=g_iter
    if verbose>=1.5:
        print(f"g_iter:{g_iter},g_prod:{g_prod}")

    #g_prod=g_iter*(v)
    #L_j=0 -> we finish by taking beta to +infty
    while (v<=0).float().mean()<min_rate:
        n += 1 # increases iteration number
        if n >=n_max:
            if allow_zero_est:
                break
            else:
                raise RuntimeError('The estimator failed. Increase n_max?')
        v_mean = v.mean()
        v_std = v.std()
        if verbose:
            print('Iter = ',n, ' v_mean = ', v_mean.item(), " Calls = ", Count_v, "v_std = ", v_std.item(),"beta =",beta)
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
        
        

        local_accept_rates=[]
        if only_duplicated:
            Y=X[to_renew]
            v_y=v[to_renew]
        else:
            Y=X
            v_y=v
        for _ in range(T):
    
            if track_accept or mh_opt:
                cand_Y=l_kernel(Y,gradV, delta_t,beta)
                with torch.no_grad():
                    #cand_v=V(cand_Y).detach()
                    cand_v_y = V(cand_Y)
                    Count_v+=N

                
                if not gaussian:
                    high_diff=(Y-cand_Y-delta_t*gradV(cand_Y)).detach()
                    low_diff=(cand_Y-Y-delta_t*gradV(Y)).detach()
                else:
                    if v1_kernel:
                        high_diff=(Y-(1-(delta_t/beta))*cand_Y-delta_t*gradV(cand_Y)).detach()
                        low_diff=(cand_Y-(1-(delta_t/beta))*Y-delta_t*gradV(Y)).detach()
                    else:
                        #using Orstein-Uhlenbeck version 
                        high_diff=(Y-np.sqrt(1-(2*delta_t/beta))*cand_Y-delta_t*gradV(cand_Y)).detach()
                        low_diff=(cand_Y-np.sqrt(1-(2*delta_t/beta))*Y-delta_t*gradV(Y)).detach()
                Count_v+= 2*nb_to_renew if only_duplicated else 2*N
                log_a_high=-beta*(cand_v_y+(1/(4*delta_t))*torch.norm(high_diff,p=2 ,dim=1)**2)
                log_a_low= -beta*(v_y+(1/(4*delta_t))*torch.norm(low_diff,p=2,dim=1)**2)
                if gaussian: 
                    log_a_high-= 0.5*torch.sum(cand_Y**2,dim=1)
                    log_a_low-= 0.5*torch.sum(Y**2,dim=1)
                #alpha=torch.clamp(input=torch.eYp(log_a_high-log_a_low),max=1) /!\ clamping is not necessary
                alpha_=torch.exp(log_a_high-log_a_low)
                b_size= nb_to_renew if only_duplicated else N
                U=torch.rand(size=(b_size,),device=device) 
                accept=U<alpha_
                accept_rate=accept.float().mean().item()
                if track_accept:
                    if verbose>=3:
                        print(f"Local accept rate: {accept_rate}")
                    accept_rates.append(accept_rate)
                    local_accept_rates.append(accept_rate)
                if adapt_d_t:
                    if accept_rate>target_accept+accept_spread:
                        delta_t*=d_t_gain
                    elif accept_rate<target_accept-accept_spread: 
                        delta_t*=d_t_decay
                    if d_t_min is not None:
                        delta_t=max(delta_t,d_t_min)
                    if d_t_max is not None:
                        delta_t=min(delta_t,d_t_max)
                    
                if mh_opt:
                    with torch.no_grad():
                        Y=torch.where(accept.unsqueeze(-1),input=cand_Y,other=Y)
                        
                        v_y=torch.where(accept, input=cand_v_y,other=v_y)
                        Count_v+= nb_to_renew if only_duplicated else N
                        if debug:
                            
                            v2 = V(Y)
                            #Count_v+= N
                            assert torch.equal(v,v2),"/!\ error in potential computation"
                else:
                    Y=cand_Y
                    v_y=cand_v_y
            else:
                Y=l_kernel(Y, gradV, delta_t, beta)


            Count_v= Count_v+ nb_to_renew if only_duplicated else Count_v+N
        
        Y,v_y,nb_calls,dict_out=apply_l_kernel(Y=Y ,v_y=v_y,delta_t=delta_t,beta=beta,V=V,gradV=gradV,
            l_kernel=l_kernel,T=T,mh_opt=mh_opt,device=device,v1_kernel=v1_kernel,adapt_d_t=adapt_d_t, track_accept=track_accept,
            d_t_decay=d_t_decay,d_t_gain=d_t_gain,debug=False,target_accept=target_accept,
            accept_spread=accept_spread,gaussian=gaussian, verbose=verbose,track_delta_t=track_delta_t,
            d_t_min=d_t_min,d_t_max=d_t_max)
        if adapt_d_t:
            delta_t = dict_out['delta_t']
            if track_delta_t:
                delta_ts.extend(dict_out['delta_ts'])
        
        if track_accept:
            accept_rates_mcmc.append(np.array(local_accept_rates).mean())    
        if only_duplicated:
            with torch.no_grad():
                X[to_renew]=Y
                v_y=V(Y)
                Count_v+=nb_to_renew
                v[to_renew]=V(Y)
        else:
            with torch.no_grad():
                X=Y
                v=V(X)
                Count_v+=N
        del Y
        
        
        
        
        beta_old = beta
        beta,g_iter=adapt_func(beta_old=beta_old,v=v,g_target=g_target,multi_output=True,max_beta=max_beta,v_min_opt=v_min_opt)
        g_prod*=g_iter
        if verbose>=1.5:
            print(f"g_iter:{g_iter},g_prod:{g_prod}")
        
        if verbose>=2:
            print(f'Beta old {beta_old}, new beta {beta}, delta_beta={beta_old-beta}')
    if (v<=0).float().mean()<min_rate:
        finished_flag=True
    g_iter_final=(v<=0).float().mean()
    P_est = (g_prod*g_iter_final).item()
    dic_out = {'p_est':P_est,'X':X,'v':v,'finished':finished_flag}
    if track_accept:
        dic_out['accept_rates']=np.array(accept_rates)
        dic_out['accept_rates_mcmc']=np.array(accept_rates_mcmc)
    if track_reject:
        dic_out['reject_rates']=np.array(reject_rates)
        dic_out['reject_rates_mcmc']=np.array(reject_rates_mcmc)
    else:
        dic_out['reject_rates']=None
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


