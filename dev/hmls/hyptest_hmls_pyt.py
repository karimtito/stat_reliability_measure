import scipy.stats as stat 
import numpy as np 
import torch
from stat_reliability_measure.dev.torch_utils import TimeStepPyt, adapt_verlet_mcmc,verlet_mcmc

def score(v,Lambda):
    s = -v/Lambda.reshape(v.shape)
    return s
    
def HypTestHybridMLS(gen,V,gradV,tau=0.,score=score,
gibbs_kernel=None,p_c=10**(-5), alpha_test=0.99,
N=2000,s=1,decay=0.95,T = 30,L=1,n_max = 300, alpha = 0.2,
alpha_est=0.95,track_calls=True,
verbose=1,device=None,track_accept=False,
adapt_step=True,kappa_opt=False, alpha_p:float=0.1,FT:bool=True,
only_duplicated:bool=True,adapt_dt=False,
target_accept=0.574,accept_spread=0.1,dt_decay=0.999,dt_gain=None,
dt_min=1e-5,dt_max=1e-2,L_min=1,track_X=False,
track_dt=False,track_H=False, track_levels=False,
GV_opt=False,dt_d=1,skip_mh=False,scale_M=torch.tensor([1.]),
gaussian=True, sig_dt=0.015,exp_rate=1.):
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
         s: strength of the the mixing gibbs_kernel                  [1x1] (1)
         decay: decay rate of the strength of the gibbs_kernel       [1x1] (0.9)
         T: number of repetitions of the mixing gibbs_kernel         [1x1] (20)
         n_max: max number of iterations                       [1x1] (200)
         alpha: level of confidence interval                   [1x1] (0.95)
         verbose: level of verbosity                           [1x1] (1)
      Returns:
         P_est: estimated probability
         dic_out: a dictionary containing additional data
           -dic_out['var_est']: estimated variance
           -dic_out.['CI_est']: estimated confidence of interval
           -dic_out.['Xrare']: Examples of the rare event 
    """
    assert only_duplicated,"Only duplicated is not implemented yet"
    K = 1
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
    if track_levels: 
        levels=[]
    # Internals 
    p = (N-1)/N  # probability of survival of a sample at each iteration
    q = -stat.norm.ppf((1-alpha_est)/2) # gaussian quantile
    d =gen(1).shape[-1] # dimension of the random vectors
    confidence_level_m = lambda y :stat.gamma.sf(-np.log(p_c),a=y, scale =1/N)
    m, _ = dichotomic_search(f = confidence_level_m, a=1, b=n_max, thresh=alpha_test)
    m = int(m)+1
    if verbose:
        print(f"Starting Hybrid Last Particle algorithm with {m}, to certify p<p_c={p_c}, with confidence level alpha ={1-alpha_test}.")
    if m>=n_max:
        raise AssertionError(f"Confidence level requires more than n_max={n_max} iterations... increase n_max ?")
    # Init
    P_est = 0
    var_est = 0
    check = 0
    CI_est = np.zeros((2))
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
    if track_levels:
        levels.append(tau_j.item())
        
    assert dt_d==1 or dt_d==d,"dt dimension can be 1 (isotropic diff.) or d (anisotropic diff.)"
    dt_scalar =alpha*TimeStepPyt(v_x=VX,grad_v_x=grad_VX)
    dt= torch.clamp(dt_scalar*torch.ones(size=(N,dt_d),device=device)+sig_dt*torch.randn(size=(N,dt_d),device=device),
                    min=dt_min,max=dt_max)
    ind_L=torch.randint(low=L_min,high=L,size=(N,)).float() if L_min<L else L*torch.ones(size=(N,))
  
    ## While
    
    while (n<=m) and (tau_j<tau).item():   # strict inequality
        n += 1 # increase iteration number
        i_dead = torch.argmin(SX,dim = 0) # find the index of the worst sample

        if n >=n_max:
            print('/!\ The estimator failed. Increase n_max?')
            break
        if verbose>=2.5:
            print(f"Current prob. estim:{(K/N)**(n-1)}")
        
        i_new = np.random.choice(list(set(range(N))-set([i_dead])))
        check+=1
        # step C: Keep K highest scores samples in Y
        # step D: refresh samples
        #Z = torch.zeros((N-K,d),device=device)
        #SZ = torch.zeros((N-K,1),device=device)
        z0=X[i_new,:] # init 
        z0.unsqueeze_(0)
        Lambda_Z = Lambda[i_new]
        ind_L_Z = ind_L[i_new]
        dt_Z = dt[i_new]

        if N-K==1:
            Z=Z.unsqueeze(0)
        SZ=SX[i_new]
        VZ=VX[i_new]
        if GV_opt:
            grad_VZ=None
        else:
            grad_VZ=grad_VX[i_new]
        if gibbs_kernel is None:
            gibbs_kernel = verlet_mcmc if not adapt_step else adapt_verlet_mcmc
        if adapt_step:
                Z,VZ,grad_VZ,nb_calls,dict_out=gibbs_kernel(q=Z,v_q=VZ,grad_v_q=grad_VZ,
                ind_L=ind_L_Z,beta=beta_j,gaussian=gaussian,
                    V=V,gradV=gradV,T=T, L=L,kappa_opt=kappa_opt,delta_t=dt_Z,
                    device=device,
                    save_H=track_H,save_func=None,scale_M=scale_M,
                    alpha_p=alpha_p,dt_max=dt_max,sig_dt=sig_dt,FT=FT,verbose=verbose,
                    L_min=L_min,
                    gaussian_verlet=GV_opt,dt_min=dt_min,skip_mh=skip_mh)
                if FT:
                    dt_Z=dict_out['dt']
                    ind_L_Z=dict_out['ind_L']
                    if verbose>=1.5:
                        print(f"New dt mean:{dt.mean().item()}, dt std:{dt.std().item()}")
                        print(f"New L mean: {ind_L.mean().item()}, L std:{ind_L.std().item()}")
        else:
            Z,VZ,grad_VZ,nb_calls,dict_out=gibbs_kernel(q=Z,grad_v_q=grad_VZ,beta=beta_j,
                        gaussian=gaussian,V=V,gradV=gradV,T=T, L=L,kappa_opt=kappa_opt,
                        delta_t=dt,device=device,save_H=track_H,save_func=None,
                                scale_M=scale_M,GV_opt=GV_opt,verbose=verbose)
            if track_H:
                H_s.extend(list(dic_out['H']))
        if track_accept:
                accept_rate=dict_out['acc_rate'] 
                accept_rates_mcmc.append(accept_rate)
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
        i_dead = torch.argmin(SX,dim = 0) # find the index of the worst sample
        X[i_dead,:] = z0
        SX[i_dead] = sz0
        VX[i_dead] = VZ
        dt[i_dead] = dt_Z
        ind_L[i_dead] = ind_L_Z
        grad_VX[i_dead] = grad_VZ
        Lambda[i_dead] = Lambda_Z
        beta_j = -1/tau_j.item() if tau_j.item()<0 else np.inf
        V_mean = VX.mean()
        #h_mean = SX.mean()
        if verbose>=1:
            print('Iter = ',n, ' tau_j = ', tau_j.item(), " beta_j = ", beta_j, " V_mean =",V_mean.item(),  " Calls = ", Count_V)
        if track_levels:
            levels.append(tau_j.item())
        if verbose>=2.5:
            print(f'Current prob. estim: {p**(n-1)}')
    # step E: Last round
    if (tau_j>=tau).item():
        finish_flag=True
    K_last = (SX>=tau).sum().item() # count the nb of score above the target threshold
    #Estimation
    P_est = (K_last/N)*((N-1)/N)**n # estimate the probability of failure
    if tau_j>tau:
        var_est = P_est**2*(P_est**(-1/N)-1)    
        CI_est[0] = P_est*np.exp(-q/np.sqrt(N)*np.sqrt(-np.log(P_est)+(q**2)/4/N) - (q**2)/2/N)
        CI_est[1] = P_est*np.exp(q/np.sqrt(N)*np.sqrt(-np.log(P_est)+(q**2)/4/N) - (q**2)/2/N)
        s_out = {'var_est':var_est,'CI_est':CI_est,'Iter':n,'Calls':Count_h,'Sample size':N}
        s_out['Cert']=False    
        s_out['Xrare'] = X
    else:
        s_out = {'var_est':None, 'CI_est':[0,p_c],'Iter':n,'Calls':Count_h,'Sample size':N}
        P_est = p_c
        s_out['Cert']=True 
        s_out['Xrare']= None
    if track_accept:
        s_out['accept_rates']=accept_rates_mcmc
    if track_dt:
        s_out['dt_means']=dt_means
        s_out['dt_stds']=dt_stds
    if track_H:
        s_out['H']=H_s
    if track_levels:
        s_out['levels']=levels
    return P_est, s_out
