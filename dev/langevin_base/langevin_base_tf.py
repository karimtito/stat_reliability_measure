import tensorflow as tf
from dev.tf_utils import TimeStepTF


def LangevinSMCBaseTF(gen, l_kernel,   V, gradV,rho=1,beta_0=0, min_rate=0.8,alpha =0.1,N=300,T = 1,n_max=300,
step_decay=0., verbose=False,adapt_func=None, allow_zero_est=False):
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
    
    w= (1/N)*tf.ones(N)
    v = V(X) # computes their potentials
    Count_v = N # Number of calls to function V or it's  gradient
    
    delta_t = alpha*TimeStepTF(V,X,gradV)
    if verbose>1.5:
        print(f"Initial time step: dt={delta_t}")
    Count_v+=2*N
    beta_old = beta_0
    ## For
    while n<n_max and tf.reduce_mean(tf.cast(v<=0,tf.float32))<min_rate:
        v_mean = tf.reduce_mean(v)
        v_std = tf.math.reduce_std(v)
    
        if verbose:
            print('Iter = ',n, ' v_mean = ', v_mean.numpy(), " Calls = ", Count_v, "v_std = ", v_std.numpy())
        if adapt_func is None:
            delta_beta = rho*delta_t
            beta = beta_old+delta_beta
        else:
            beta = adapt_func(beta,)
        
        G = tf.math.exp(-(beta-beta_old)*v) #computes current value fonction
        
        
        w = w * G #updates weights
        n += 1 # increases iteration number
        if n >=n_max:
            if allow_zero_est: 
                return tf.math.reduce_sum(w*tf.cast(v<=0,dtype=tf.float32)),False
            else:
                raise RuntimeError('The estimator failed. Increase n_max?')
        
        for t in range(T):
            X=l_kernel(X, gradV, delta_t, beta)
            Count_v+= N
        v = V(X)
        Count_v+= N
        beta_old = beta
        delta_t=delta_t*(1-step_decay)
        if verbose>1.5:
            print(f"New time step: dt={delta_t}")
    P_est = tf.math.reduce_sum(w*tf.cast(v<=0,dtype=tf.float32))
    return P_est,True