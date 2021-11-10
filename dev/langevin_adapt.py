import numpy as np 
from dev.langevin_utils import TimeStep

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


def SimpAdaptBeta(beta_old,v,g_target,search_method=dichotomic_search_d,max_beta=1e6,verbose=0,multi_output=False):

    ess_beta = lambda beta : (np.exp(-(beta-beta_old)*v)).mean() #increasing funtion of the parameter beta
    results= search_method(f=ess_beta,a=beta_old,b=max_beta,  thresh = g_target)
    new_beta, g = results[0],results[-1]
    if verbose>0:
        print(f"g_target: {g_target}, actual g: {g}")

    if multi_output:
        return (new_beta,g)
    else:
        return new_beta

"""Implementation of Langevin Sequential Monte Carlo with adaptative tempering"""
#TODO def AdaptLangevinSMC: 
def SimpAdaptLangevinSMC(gen, l_kernel,   V, gradV,g_target=0.1,min_rate=0.8,alpha =0.1,N=300,T = 1,n_max=300, max_beta=1e6, verbose=False,adapt_func=SimpAdaptBeta,rho=1):
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
    

    # Internals
 
    #d =gen(1).shape[-1] # dimension of the random vectors
    n = 1 # Number of iterations

    ## Init
    # step A0: generate & compute potentials
    X = gen(N) # generate N samples
    
    
    v = V(X) # computes their potentials
    Count_v = N # Number of calls to function V or it's  gradient
    delta_t = alpha*TimeStep(V,X,gradV)
    Count_v+=2*N
    beta_old = 0
    
    beta,g_0=SimpAdaptBeta(beta_old=beta_old,v=v,g_target=g_target,multi_output=True,max_beta=max_beta) 
    g_ =[g_0]
    ## For
    while (v<=0).sum()<N:
        
        v_mean = v.mean()
        v_std = v.std()
        if verbose:
            print('Iter = ',n, ' v_mean = ', v_mean, " Calls = ", Count_v, "v_std = ", v_std)
            print(f'Beta old {beta_old}, new beta {beta}, delta_beta={beta_old-beta}')
        
        
        
        G = np.exp(-(beta-beta_old)*v) #computes current value fonction
        
        U = np.random.uniform(low=0,high=1,size=N)
        to_renew = G<U
        renew_idx = np.random.choice(a=np.arange(N),size=to_renew.sum(),p=G/G.sum())
        X[to_renew] = X[renew_idx]
        n += 1 # increases iteration number
        if n >=n_max:
            raise RuntimeError('The estimator failed. Increase n_max?')
        
        for _ in range(T):
            X=l_kernel(X, gradV, delta_t, beta)
            Count_v+= 2*N
        v = V(X)
        Count_v+= N
        beta_old = beta
        beta,g_iter=SimpAdaptBeta(beta_old=beta_old,v=v,g_target=g_target,multi_output=True,max_beta=max_beta)
        g_.append(g_iter)
        
    P_est = np.prod(g_)
   

    return P_est