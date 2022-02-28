import scipy.stats as stat 
import numpy as np 


def ImportanceSplitting(gen,kernel,h,tau,N=2000,K=1000,s=1,decay=0.95,T = 30,n_max = 300, alpha = 0.95,
verbose=1, track_rejection=False, rejection_ctrl = False, reject_thresh=0.9, gain_rate = 1.0001, 
prog_thresh=0.01,clip_s=False,s_min=1e-3,s_max=5):
    """
      Importance splitting estimator
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

    # Internals
    q = -stat.norm.ppf((1-alpha)/2) # gaussian quantile
    d =gen(1).shape[-1] # dimension of the random vectors
    n = 1 # Number of iterations

    ## Init
    # step A0: generate & compute scores
    X = gen(N) # generate N samples
    SX = h(X) # compute their scores
    Count_h = N # Number of calls to function h
    
    #step B: find new threshold
    ind = np.argsort(SX,axis=None)[::-1] # sort in descending order
    S_sort= SX[ind]
    tau_j = S_sort[K] # set the threshold to (K+1)-th
    h_mean = SX.mean()
    if verbose>=1:
        print('Iter = ',n, ' tau_j = ', tau_j, "h_mean",h_mean,  " Calls = ", Count_h)

    rejection_rate=0
    kernel_pass=0
    rejection_rates=[0]
    ## While
    while (n<n_max) and (tau_j<tau):
        n += 1 # increase iteration number
        if n >=n_max:
            raise RuntimeError('The estimator failed. Increase n_max?')
        # step C: Keep K highest scores samples in Y
        Y = X[ind[0:K],:]
        SY = SX[ind[0:K]] # Keep their scores in SY
        # step D: refresh samples
        Z = np.zeros((N-K,d))
        SZ = np.zeros((N-K,1))
        for k in range(N-K):
            u = np.random.choice(range(K),size=1,replace=False) # pick a sample at random in Y
            z0 = Y[u,:]
            accept_flag = False
            for t in range(T):
                w = kernel(z0,s) # propose a refreshed sample
                kernel_pass+=1
                
                    
                sw = h(w) # compute its score
                Count_h = Count_h + 1
                if sw>tau_j: # accept if true
                    z0 = w
                    sz0 = sw
                    accept_flag = True # monitor if accepted
                elif track_rejection:
                    rejection_rate=((kernel_pass-1.)/kernel_pass)*rejection_rate+(1/kernel_pass)
            Z[k,:] = z0 # a fresh sample
            SZ[k] = sz0 # its score
            if rejection_ctrl and rejection_rate>=reject_thresh:
                
                s = s*decay if not clip_s else np.clip(s*decay,a_min=s_min,a_max=s_max)
                if verbose>1:
                    print('Strength of kernel diminished!')
                    print(f's={s}')
            if not accept_flag:
                s = s * decay if not clip_s else np.clip(s*decay,a_min=s_min,a_max=s_max)# decrease the strength of the mixing kernel


       
        # step A: update set X and the scores
        X[:K,:] = Y # copy paste the old samples of Y into X
        SX[:K] = SY
        X[K:N,:] = Z # copy paste the new samples of Z into X
        SX[K:N] = SZ
        # step B: Find new threshold
        ind = np.argsort(SX,axis=None)[::-1] # sort in descending order
        S_sort= SX[ind]
        new_tau = S_sort[K]
        if (new_tau-tau_j)/tau_j<prog_thresh:
            s = s*gain_rate if not clip_s else np.clip(s*decay,s_min,s_max)

            if verbose>1:
                    print('Strength of kernel diminished!')
                    print(f's={s}')

        tau_j = S_sort[K] # set the threshold to (K+1)-th

        
        h_mean = SX.mean()
        if verbose>=1:
            print('Iter = ',n, ' tau_j = ', tau_j, "h_mean",h_mean,  " Calls = ", Count_h)
        if track_rejection:
            if verbose>1:
                print(f'Rejection rate: {rejection_rate}')
            rejection_rates+=[rejection_rate]

    # step E: Last round
    K_last = (SX>=tau).sum() # count the nb of score above the target threshold

    #Estimation
    p = K/N
    p_last = K_last/N
    P_est = (p**(n-1))*p_last
    Var_est = (P_est**2)*((n-1)*(1-p)/p + (1-p_last)/p_last)/N
    P_bias = P_est*n*(1-p)/p/N
    CI_est = P_est*np.array([1,1]) + q*np.sqrt(Var_est)*np.array([-1,1])
    Xrare = X[(SX>=tau).reshape(-1),:]
    dic_out = {"Var_est":Var_est,"CI_est": CI_est,"N":N,"K":K,"s":s,"decay":decay,"T":T,"Count_h":Count_h,
    "P_bias":P_bias,"n":n,"Xrare":Xrare}
    if track_rejection:
        dic_out["rejection_rates"]=np.array(rejection_rates)
        dic_out["Avg. rejection rate"]=rejection_rate
    

       
    return P_est,dic_out