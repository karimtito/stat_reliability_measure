import scipy.stats as stat 
import numpy as np 
import torch
from stat_reliability_measure.dev.torch_utils import TimeStepPyt, adapt_verlet_mcmc,verlet_mcmc



def score(v,Lambda):
    s = -v/Lambda.reshape(v.shape)
    return s
    
def HybridMLS(gen,V,gradV,tau,score=score,gibbs_kernel=None,
N=2000,K=1000,s=1,decay=0.95,T = 30,L=1,n_max = 300, alpha = 0.2,alpha_q=0.95,
verbose=1,device=None,track_accept=False,
adapt_step=True,kappa_opt=False, alpha_p:float=0.1,FT:bool=True,
only_duplicated:bool=True,adapt_dt=False,
target_accept=0.574,accept_spread=0.1,dt_decay=0.999,dt_gain=None,
dt_min=1e-5,dt_max=1e-2,L_min=1,
track_v=False,track_dt=False,track_H=False
, GV_opt=False,dt_d=1,skip_mh=False,scale_M=torch.tensor([1.]),gaussian=True, sig_dt=0.015,
exp_rate=1.):
    """
      Hybrid Importance splitting estimator using gradient information 
      via exponential auxiliary variables
      Args:
         gen: generator of iid samples X_i                            [fun]
         gibbs_kernel: mixing gibbs_kernel invariant to f_X           [fun]
         score: score function                                        [fun]
         V: potential function                                        [fun]
         tau: threshold. The rare event is defined as score(X)>tau    [1x1]
         
         N: number of samples                                  [1x1] (2000)
         K: number of survivors                                [1x1] (1000)
         s: strength of the the mixing gibbs_kernel                  [1x1] (1)
         decay: decay rate of the strength of the gibbs_kernel       [1x1] (0.9)
         T: number of repetitions of the mixing gibbs_kernel         [1x1] (20)
         n_max: max number of iterations                       [1x1] (200)
         alpha: level of confidence interval                   [1x1] (0.95)
         verbose: level of verbosity                           [1x1] (1)
      Returns:
         P_est: estimated probability
         dic_out: a dictionary containing additional data
           -dic_out['Var_est']: estimated variance
           -dic_out.['CI_est']: estimated confidence of interval
           -dic_out.['Xrare']: Examples of the rare event 
    """
    exp_dist = torch.distributions.Exponential(rate=torch.tensor([exp_rate]))
    if device is None:
        device= "cuda:0" if torch.cuda.is_available() else "cpu"
    exp_gen = lambda N: exp_dist.sample((N,)).to(device)
    scale_M= scale_M.to(device)
    if adapt_dt and dt_gain is None:
        dt_gain= 1/dt_decay
    if track_accept:
        accept_rate =[]
        accept_rates_mcmc=[]
    if track_dt:
        dt_means=[]
        dt_stds = []
    if track_H:
        H_s=[]
    if track_v: 
        v_means=[]
        v_stds=[]
    
    
    
    # Internals 
    q = -stat.norm.ppf((1-alpha_q)/2) # gaussian quantile
    d =gen(1).shape[-1] # dimension of the random vectors
    n = 1 # Number of iterations
    finish_flag=False
    ## Init
    # step A0: generate & compute scores
    X = gen(N) # generate N samples
    Lambda = exp_gen(N)
    VX = V(X)
    Count_V = N # Number of calls to function score
    SX = score(VX,Lambda) # compute their scores
    gradient_use=not (GV_opt)
    if gradient_use:
        grad_VX = gradV(X)
        Count_V += 2*N # Each call to the gradient costs 2 calls to V
    
    
    #step B: find new threshold
    ind_= torch.argsort(input=SX,dim=0,descending=True).squeeze(-1)
    

    S_sort= SX[ind_]
    tau_j = S_sort[K] # set the threshold to (K+1)-th highest score
    beta_j = -1/tau_j.item() if tau_j.item()<0 else np.inf
    #s_mean = SX.mean()
    V_mean = VX.mean()
    if verbose>=1:
        print('Iter = ',n, ' tau_j = ', tau_j.item(), "beta_j",beta_j, "V_mean",V_mean.item(),  " Calls = ", Count_V)
    assert dt_d==1 or dt_d==d,"dt dimension can be 1 (isotropic diff.) or d (anisotropic diff.)"
    dt_scalar =alpha*TimeStepPyt(v_x=VX,grad_v_x=grad_VX)
    dt= torch.clamp(dt_scalar*torch.ones(size=(N,dt_d),device=device)+sig_dt*torch.randn(size=(N,dt_d),device=device),min=dt_min,max=dt_max)
    ind_L=torch.randint(low=L_min,high=L,size=(N,)).float() if L_min<L else L*torch.ones(size=(N,))
  
    ## While
    
    while (tau_j<tau).item():              #strict inequality
        n += 1 # increase iteration number
        if n >=n_max:
            print('/!\ The estimator failed. Increase n_max?')
            break
        
        # step C: Keep K highest scores samples in Y
        Y = X[ind_[0:K],:]
        SY = SX[ind_[0:K]] # Keep their scores in SY
        
        VY = VX[ind_[0:K]]
        ind_L_Y = ind_L[(ind_[0:K]).to(ind_L.device)]
        dt_Y = dt[ind_[0:K]]
        Lambda_Y = Lambda[ind_[0:K]]
        # step D: refresh samples
        #Z = torch.zeros((N-K,d),device=device)
        #SZ = torch.zeros((N-K,1),device=device)
        
        ind=torch.multinomial(input=torch.ones(size=(K,)),num_samples=N-K,replacement=True).squeeze(-1)
        Z=Y[ind,:] 
        Lambda_Z = Lambda_Y[ind]
        ind_L_Z = ind_L_Y[ind]
        dt_Z = dt_Y[ind]
        if N-K==1:
            Z=Z.unsqueeze(0)
        SZ=SY[ind]
        VZ=VY[ind]
        if GV_opt:
            grad_VZ=None
        else:
            grad_VZ=grad_VX[ind]
        if gibbs_kernel is None:
            gibbs_kernel = verlet_mcmc if not adapt_step else adapt_verlet_mcmc
        if adapt_step:
                Z,VZ,grad_VZ,nb_calls,dict_out=gibbs_kernel(q=Z,v_q=VZ,grad_V_q=grad_VZ,ind_L=ind_L_Z,beta=beta_j,gaussian=gaussian,
                    V=V,gradV=gradV,T=T, L=L,kappa_opt=kappa_opt,delta_t=dt_Z,device=device,
                    save_H=track_H,save_func=None,scale_M=scale_M,
                    alpha_p=alpha_p,dt_max=dt_max,sig_dt=sig_dt,FT=FT,verbose=verbose,L_min=L_min,
                    gaussian_verlet=GV_opt,dt_min=dt_min,skip_mh=skip_mh)
                if FT:
                    if only_duplicated:
                        dt_Z=dict_out['dt']
                        ind_L_Z=dict_out['ind_L']
                    else:
                        dt=dict_out['dt']
                        ind_L=dict_out['ind_L']
                    if verbose>=1.5:
                        print(f"New dt mean:{dt.mean().item()}, dt std:{dt.std().item()}")
                        print(f"New L mean: {ind_L.mean().item()}, L std:{ind_L.std().item()}")
        else:
            Z,VZ,grad_VZ,nb_calls,dict_out=gibbs_kernel(q=Z,grad_V_q=grad_VZ,beta=beta_j,gaussian=gaussian,
                                V=V,gradV=gradV,T=T, L=L,kappa_opt=kappa_opt,delta_t=dt,device=device,save_H=track_H,save_func=None,
                                scale_M=scale_M,GV_opt=GV_opt,verbose=verbose)
            if track_H:
                H_s.extend(list(dic_out['H']))
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
        
        Count_V+=nb_calls
            
           
        Lambda_Z = beta_j*VZ+exp_gen(N-K).squeeze(1)
        
        SZ = score(VZ,Lambda_Z)

       
        # step A: update set X and the scores
        X[:K,:] = Y # copy paste the old samples of Y into X
       
        SX[:K] = SY
        VX[:K] = VY
        dt[:K] = dt_Y
        ind_L[:K] = ind_L_Y

        X[K:N,:] = Z # copy paste the new samples of Z into X
        SX[K:N] = SZ
        VX[K:N] = VZ
        grad_VX[K:N] = grad_VZ
        dt[K:N] = dt_Z
        ind_L[K:N] = ind_L_Z
        # step B: Find new threshold
        ind_ = torch.argsort(SX,dim=0,descending=True).squeeze(-1) # sort in descending order
        S_sort= SX[ind_]
        # new_tau = S_sort[K]
        # if (new_tau-tau_j)/tau_j<prog_thresh:
        #     s = s*gain_rate if not clip_s else np.clip(s*gain_rate,s_min,s_max)
        #     if verbose>1:
        #         print('Strength of gibbs_kernel increased!')
        #         print(f's={s}')

        tau_j = S_sort[K] # set the threshold to (K+1)-th
        beta_j = -1/tau_j.item() if tau_j.item()<0 else np.inf
        V_mean = VX.mean()
        #h_mean = SX.mean()
        if verbose>=1:
            print('Iter = ',n, ' tau_j = ', tau_j.item(), " beta_j = ", beta_j, " V_mean =",V_mean.item(),  " Calls = ", Count_V)
        if verbose>=2.5:
            print(f'Current prob. estim:{(K/N)**(n-1)}')
        
    # step E: Last round
    if (tau_j>=tau).item():
        finish_flag=True
    K_last = (SX>=tau).sum().item() # count the nb of score above the target threshold

    #Estimation

    p = K/N
    p_last = K_last/N
    P_est = (p**(n-1))*p_last
    Var_est = (P_est**2)*((n-1)*(1-p)/p + (1-p_last)/p_last)/N if p_last>0 else 0
    P_bias = P_est*n*(1-p)/p/N
    CI_est = P_est*np.array([1,1]) + q*np.sqrt(Var_est)*np.array([-1,1])
    Xrare = X[(SX>=tau).reshape(-1),:] if p_last>0 else None

    dic_out = {"Var_est":Var_est,"CI_est": CI_est,"N":N,"K":K,"s":s,"decay":decay,"T":T,"Count_V":Count_V,
    "P_bias":P_bias,"n":n,"Xrare":Xrare}
    if track_accept:
        # dic_out['accept_rates']=np.array(accept_rate)
        dic_out['accept_rates_mcmc']=np.array(accept_rates_mcmc)
    if track_dt:
        dic_out['dt_means']=dt_means
        dic_out['dt_stds']=dt_stds
    dic_out['finish_flag']=finish_flag


       
    return P_est,dic_out