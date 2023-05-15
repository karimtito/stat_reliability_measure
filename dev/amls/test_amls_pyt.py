import scipy.stats as stat 
import numpy as np 
import torch
import math
from stat_reliability_measure.dev.utils import dichotomic_search

def TestImportanceSplittingPyt(gen,kernel,h,tau,N=2000,s=1,decay=0.95
            ,p_c=10**(-5),T = 30,n_max = 5000, alpha_est = 0.95, alpha_test=0.99,
verbose=1, track_rejection=False, rejection_ctrl = True,  gain_rate = 1.0001, 
prog_thresh=0.01,clip_s=False,s_min=1e-3,s_max=5,device=None,track_accept=False,
reject_forget_rate =1, gain_forget_rate=1,check_every=10,accept_ratio=0.9,
gain_thresh=0.01):
    """
      PyTorch implementationf of Importance splitting estimator 
      Args:
         gen: generator of iid samples X_i                            [fun]
         kernel: mixing kernel invariant to f_X                       [fun]
         h: score function                                            [fun]
         tau: threshold. The rare event is defined as h(X)>tau        [1x1]
         
         N: number of samples                                  [1x1] (2000)
         K: number of survivors                                [1x1] (1000)
         s: strength of the the mixing kernel                  [1x1] (1)
         decay: decay rate of the strength of the kernel       [1x1] (0.9)
         T: number of repetitions of the mixing kernel         [1x1] (20)
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

    if device is None:
        device= "cuda:0" if torch.cuda.is_available() else "cpu"
    # Internals 
    q = -stat.norm.ppf((1-alpha_est)/2) # gaussian quantile
    d =gen(1).shape[-1] # dimension of the random vectors
    n = 1 # Number of iterations
    p = (N-1)/N  # probability of survival of a sample at each iteration
    if track_accept: 
        accept_rates=[]
    confidence_level_m = lambda y :stat.gamma.sf(-np.log(p_c),a=y, scale =1/N) 
    m, _ = dichotomic_search(f = confidence_level_m, a=1, b=n_max, thresh=alpha_test)
    m = int(m)+1
    if verbose:
        print(f"Starting Last Particle algorithm with {m}, to certify p<p_c={p_c}, with confidence level alpha ={1-alpha_test}.")
    if m>=n_max:
        raise AssertionError(f"Confidence level requires more than n_max={n_max} iterations... increase n_max ?")
    ## Init
    P_est = 0
    Var_est = 0
    check = 0    
    CI_est = np.zeros((2))
    # step A0: generate & compute scores 
    X = gen(N) # generate N samples
    SX = h(X) # compute their scores
    Count_h = N # Number of calls to function h
    rejection_rate=0
    kernel_pass=0
    rejection_rates=[0]
    ## While
    while (n<=m) and (tau_j<tau).item():
        
        n += 1 # increase iteration number

        i_dead = torch.argmin(SX,dim = 0) # find the index of the worst sample
        #print(SX[i_dead], tau_j )
        if tau_j!=-np.inf and tau_j!=0:
            gain = np.abs((SX[i_dead]-tau_j)/tau_j)
        else:
            gain=0
        gamma = 1+gain_forget_rate*(n-1)
        avg_gain = (1-gamma/n)*avg_gain + (gamma/n)*gain
        if n>1 and avg_gain<gain_thresh and reject_rate<(1-accept_ratio):
                s = s/decay
                if verbose>=1 and check%check_every==0:
                    print('Strength of kernel increased!')
                    print(f's={s}')

        tau_j = SX[i_dead] # set the new threshold
        h_mean = SX.mean() 
        if verbose>=1:
            print('Iter = ',n, ' tau_j = ', tau_j.item(), "h_mean",h_mean.item(),  " Calls = ", Count_h)


        
        #ind=torch.multinomial(input=torch.ones(size=(K,)),num_samples=N-K,replacement=True)

        #Z=Y[ind,:] 
        #SZ=SY[ind]
        i_new = np.random.choice(list(set(range(N))-set([i_dead])))
        check+=1
        #u = np.random.choice(range(K),size=1,replace=False) # pick a sample at random in Y
        #TODO implement tracking of instant accept rate and rejection rate
        
        #accept=0 
        Count_accept = 0
        z0=X[i_new,:] # init
        for _ in range(T):
            w = kernel(z0,s) # propose a refreshed sample
            kernel_pass+=1
            
                
            sw = h(w) # compute its score
            
            Count_h = Count_h + 1
            if sw.item()>tau_j.item(): # accept if true
                z0 = w
                sz0 = sw
                Count_accept+=1
                
        X[i_dead,:] = z0
        SX[i_dead] = sz0
        
        Count_h+=T
        gamma = T+reject_forget_rate*kernel_pass
        reject_rate = (1-gamma/(kernel_pass+T))*reject_rate + gamma*(1-Count_accept/T)/(kernel_pass+T) 
        if check%check_every==0 and verbose>=1:
            print(f'Accept ratio:{Count_accept/T}')
            print(f'Reject rate:{reject_rate}')
        kernel_pass+=T
        
        if rejection_ctrl & reject_rate > (1-accept_ratio):
            s = s*decay
            if verbose>=1 and check%check_every==0:
                print('Strength of kernel diminished!')
                print(f's={s}')
        Count_accept = 0

        
        
       

       
        i_dead = torch.argmin(SX,dim = 0) # find the index of the worst sample
        new_tau = SX[i_dead] # set the new threshold
        
        if (new_tau-tau_j)/tau_j<prog_thresh:
            s = s*gain_rate if not clip_s else np.clip(s*decay,s_min,s_max)
            if verbose>1:
                print('Strength of kernel increased!')
                print(f's={s}')

        tau_j = new_tau # set the threshold to (K+1)-th

        
        h_mean = SX.mean()
        if verbose>=1:
            print('Iter = ',n, ' tau_j = ', tau_j.item(), "h_mean",h_mean.item(),  " Calls = ", Count_h)
        if track_rejection:
            if verbose>1:
                print(f'Rejection rate: {rejection_rate}')
            rejection_rates+=[rejection_rate]

    # step E: Last round
    K_last = (SX>=tau).sum().item() # count the nb of score above the target threshold

    #Estimation

    
    if tau_j>tau:
        Var_est = P_est**2*(P_est**(-1/N)-1)    
        CI_est[0] = P_est*np.exp(-q/np.sqrt(N)*np.sqrt(-np.log(P_est)+(q**2)/4/N) - (q**2)/2/N)
        CI_est[1] = P_est*np.exp(q/np.sqrt(N)*np.sqrt(-np.log(P_est)+(q**2)/4/N) - (q**2)/2/N)
        s_out = {'Var_est':Var_est,'CI_est':CI_est,'Iter':n,'Calls':Count_h,'Sample size':N}
        s_out['Cert']=False    
        s_out['Xrare'] = X
    else:
        s_out = {'Var_est':None, 'CI_est':[0,p_c],'Iter':n,'Calls':Count_h,'Sample size':N}
        P_est = p_c
        s_out['Cert']=True 
        s_out['Xrare']= None
    return P_est, s_out


def s_to_dt(s):
    return s/math.sqrt(1+s**2)
