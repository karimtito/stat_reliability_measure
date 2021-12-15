import torch
from torch_utils import TimeStepPyt

# def TimeStepPyt(V,X,gradV,p=1,p_p=2):
#     V_mean= V(X).mean()
#     V_grad_norm_mean = ((torch.norm(gradV(X),dim = 1,p=p_p)**p).mean())**(1/p)
#     with torch.no_grad():
#         result=V_mean/V_grad_norm_mean
#     return result

""" Basic implementation of Langevin Sequential Monte Carlo """
def LangevinSMCBasePyt(gen, l_kernel,   V, gradV,rho=10,beta_0=0, min_rate=0.8,alpha =0.1,N=300,T = 1,n_max=300, 
verbose=False,adapt_func=None,accept_zero_est=False,device=None):
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

    ## Init
    # step A0: generate & compute potentials
    X = gen(N) # generate N samples
    if device is None:
        device=X.device

    w= (1/N)*torch.ones(N)
    v = V(X) # computes their potentials
    Count_v = N # Number of calls to function V or it's  gradient
    delta_t = alpha*TimeStepPyt(V,X,gradV)
    print(delta_t)
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
        
        G = torch.exp(-(beta-beta_old)*v).to('cpu') #computes current value fonction
       
        print(G) 
        
        w = w * G #updates weights
        print(w)
        n += 1 # increases iteration number
        if n >=n_max:
            if accept_zero_est:
                return  0
            else:
                raise RuntimeError('The estimator failed. Increase n_max?')
        
        for t in range(T):
            X=l_kernel(X, gradV, delta_t, beta)
            Count_v+= N

        X.require_grad=True
        v = V(X).detach().cpu()
        Count_v+= N
        beta_old = beta
    print(v<=0)
    
    P_est = (w.cpu()*(v.detach().cpu()<=0).float()).sum()        
    return P_est