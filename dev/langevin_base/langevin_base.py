
import numpy as np
import numpy.linalg as LA

from stat_reliability_measure.dev.langevin_utils import TimeStep




""" Basic implementation of Langevin Sequential Monte Carlo """
def LangevinSMCBase(gen, l_kernel,   V, gradV,rho=1,beta_0=0, min_rate=0.8,alpha =0.1,N=300,T = 1,n_max=300, 
verbose=False,adapt_func=None, allow_zero_est=False,track_calls=False):
    """
      Basic version of a Langevin-based SMC estimator
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
    finished_flag=False
    ## Init
    # step A0: generate & compute potentials
    X = gen(N) # generate N samples
    
    w= (1/N)*np.ones(N)
    v = V(X) # computes their potentials
    Count_v = N # Number of calls to function V or it's  gradient
    delta_t = alpha*TimeStep(V,X,gradV)
    Count_v+=2*N
    beta_old = beta_0
    ## For
    while n<n_max and (v<=0).mean()<min_rate:
        v_mean = v.mean()
        v_std = v.std()
        if verbose:
            print('Iter = ',n, ' v_mean = ', v_mean, " Calls = ", Count_v, "v_std = ", v_std)
        if adapt_func is None:
            delta_beta = rho*delta_t
            beta = beta_old+delta_beta
        else:
            beta = adapt_func(beta,)
        
        G = np.exp(-(beta-beta_old)*v) #computes current value fonction
        
        
        w = w * G #updates weights
        n += 1 # increases iteration number
        if n >=n_max:
            if allow_zero_est:
                break 
            else:
                raise RuntimeError('The estimator failed. Increase n_max?')
        
        for t in range(T):
            X=l_kernel(X, gradV, delta_t, beta)
            Count_v+= N
        v = V(X)
        Count_v+= N
        beta_old = beta
    if n<=n_max:
        finished_flag=True
    P_est = (w*(v<=0).astype(int)).sum()        
    dic_out={"finished":finished_flag} 
    if track_calls: 
        dic_out['calls']=Count_v
    return P_est,dic_out