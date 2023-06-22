import numpy as np
import eagerpy as ep
from stat_reliability_measure.dev.utils import dichotomic_search
import scipy.stats as stat
import math
""" Implementation of the last particle algorithm for statistical reliability measure
"""

def ImportanceSplittingLp(gen,kernel,h,tau=0,N=100,s=0.1,decay=0.9,T = 20, accept_ratio = 0.9, 
alpha_est = 0.95, alpha_test=0.99,verbose=1, gain_thresh=0.01, check_every=3, p_c = 10**(-20),n_max = int(10**6), 
  reject_forget_rate =0, gain_forget_rate=0, reject_thresh=0.005):
    """
      Importance splitting last particle estimator, i.e. the importance splitting algorithm with K=N-1

      Args:
        gen: generator of iid samples X_i                            [fun]
        kernel: mixing kernel invariant to f_X                       [fun]
        h: score function from gaussian vector                       [fun]
        tau: threshold. The rare events are defined as h(X)>tau_j    [tx1]
         
        
        N: number of samples                                  [1x1] (100)
        s: strength of the the kernel                         [1x1] (0.1)
        T: number of repetitions of the mixing kernel         [1x1] (20)
        n_max: max number of iterations                       [1x1] (200)
        n_mod: check each n_mod iteration                     [1x1] (100)
        decay: decay rate of the strength                      [1x1] (0.9)
        accept_ratio: lower bound of accept ratio             [1x1] (0.5)
        alpha: level of confidence interval                   [1x1] (0.95)
        verbose: level of verbosity                           [1x1] (0)
       
       Returns:
         P_est: estimated probability
         s_out: a dictionary containing additional data
           -s_out['Var_est']: estimated variance
           -s_out['CI_est']: estimated confidence of interval
           -s_out['Xrare']: Examples of the rare event 
           -s_out['result']: Result of the estimation/hypothesis testing process

    """

    # Internals
    q = -stat.norm.ppf((1-alpha_est)/2) # gaussian quantile
    #d =gen(1).shape[-1] # dimension of the random vectors
    k = 1 # Number of iterations
    p = (N-1)/N       
    confidence_level_m = lambda y :stat.gamma.sf(-np.log(p_c),a=y, scale =1/N) 
    m, _ = dichotomic_search(f = confidence_level_m, a=100, b=n_max, thresh=alpha_test)
    m = int(m)+1
    if verbose:
        print(f"Starting Last Particle algorithm with {m}, to certify p<p_c={p_c}, with confidence level alpha ={1-alpha_test}.")
    if m>=n_max:
        raise AssertionError(f"Confidence level requires more than n_max={n_max} iterations... increase n_max ?")
    tau_j = -np.inf
    P_est = 0
    Var_est = 0
    CI_est = np.zeros((2))
    kernel_pass=0
    Count_accept = 0
    check=0
    ## Init
    # step A0: generate & compute scores
    X = gen(N) # generate N samples
    SX = h(X) # compute their scores
    Count_h = N # Number of calls to function h
    reject_rate = 0
    avg_gain=0
    #step B: find new threshold
    ## While
    while (k<=m):
        #find new threshold
        i_dead = np.argmin(SX,axis = None) # sort in descending order
        #print(SX[i_dead], tau_j )
        if tau_j!=-np.inf and tau_j!=0:
            gain = np.abs((SX[i_dead]-tau_j)/tau_j)
        else:
            gain=0

        gamma = 1+gain_forget_rate*(k-1)
        avg_gain = (1-gamma/k)*avg_gain + (gamma/k)*gain
        if k>1 and avg_gain<gain_thresh and reject_rate<reject_thresh:
                s = s/decay
                if verbose>=1 and check%check_every==0:
                    print('Strength of kernel increased!')
                    print(f's={s}')

        tau_j = SX[i_dead] # set the threshold to the last particule's score
        if tau_j>tau:
            P_est= p**(k-1)
            break #it is useless to compute new minimum if desired level has already been reached
        if verbose>=1 and check%check_every==0:
            print('Iter = ',k, ' tau_j = ', tau_j, " Calls = ", Count_h)
        check+=1
        
        # Refresh samples
        i_new = np.random.choice(list(set(range(N))-set([i_dead])))
        z0 = X[i_new,:]
        sz0 = SX[i_new]
        for t in range(T):
            w = kernel(z0,s)
            sw = h(w)
            if sw>=tau_j:
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
        
        if reject_rate > (1-accept_ratio):
            s = s*decay
            if verbose>=1 and check%check_every==0:
                print('Strength of kernel diminished!')
                print(f's={s}')
        Count_accept = 0
        k += 1 # increase iteration number
    
    
    if tau_j>tau:
        Var_est = P_est**2*(P_est**(-1/N)-1)    
        CI_est[0] = P_est*np.exp(-q/np.sqrt(N)*np.sqrt(-np.log(P_est)+(q**2)/4/N) - (q**2)/2/N)
        CI_est[1] = P_est*np.exp(q/np.sqrt(N)*np.sqrt(-np.log(P_est)+(q**2)/4/N) - (q**2)/2/N)
        s_out = {'Var_est':Var_est,'CI_est':CI_est,'Iter':k,'Calls':Count_h,'Sample size':N}
        s_out['Cert']=False    
        s_out['Xrare'] = X
    else:
        s_out = {'Var_est':None, 'CI_est':[0,p_c],'Iter':k,'Calls':Count_h,'Sample size':N}
        P_est = p_c
        s_out['Cert']=True 
        s_out['Xrare']= None
    return P_est, s_out

def s_to_dt(s):
    return s/math.sqrt(1+s**2)