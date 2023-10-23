import numpy as np 
import scipy.stats as stat
from utils import dichotomic_search

""" Implementation of last particle variant """



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
           -s_out['var_est']: estimated variance
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
    var_est = 0
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
        if tau_j!=-np.inf:
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
        var_est = P_est**2*(P_est**(-1/N)-1)    
        CI_est[0] = P_est*np.exp(-q/np.sqrt(N)*np.sqrt(-np.log(P_est)+(q**2)/4/N) - (q**2)/2/N)
        CI_est[1] = P_est*np.exp(q/np.sqrt(N)*np.sqrt(-np.log(P_est)+(q**2)/4/N) - (q**2)/2/N)
        s_out = {'var_est':var_est,'CI_est':CI_est,'Iter':k,'Calls':Count_h,'Sample size':N}
        s_out['Cert']=False    
        s_out['Xrare'] = X
    else:
        s_out = {'var_est':None, 'CI_est':[0,p_c],'Iter':k,'Calls':Count_h,'Sample size':N}
        P_est = p_c
        s_out['Cert']=True 
        s_out['Xrare']= None
    return P_est, s_out


def ImportanceSplittingLpBatch(gen,kernel_b,h,h_big,nb_system=5,d=784,tau=0,N=100,s=0.1,decay=0.92,T = 20, accept_ratio = 0.9, 
alpha_est = 0.95, alpha_test=0.99,verbose=1, gain_thresh=0.01, check_every=3, p_c = 10**(-20),n_max = int(10**6), 
  reject_forget_rate =0, gain_forget_rate=0, reject_thresh=0.005,fast_decay=True, fast_d=1):
    """
      Importance splitting last particle estimator, i.e. the importance splitting algorithm with K=N-1
      with several particle systems.

      Args:
        gen: generator of iid samples X_i                            [fun]
        kernel_batch: mixing kernel invariant to f_X                 [fun]
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
           -s_out['var_est']: estimated variance
           -s_out['CI_est']: estimated confidence of interval
           -s_out['Xrare']: Examples of the rare event 
           -s_out['result']: Result of the estimation/hypothesis testing process

    """
    q = -stat.norm.ppf((1-alpha_est)/2) # gaussian quantile
    s_b = s*np.ones(nb_system)
    
    k = 1 # Number of iterations
    p = (N-1)/N       
    confidence_level_m = lambda y :stat.gamma.sf(-np.log(p_c),a=y, scale =1/N) 
    m, _ = dichotomic_search(f = confidence_level_m, a=100, b=n_max, thresh=alpha_test)
    m = int(m)+1
    if verbose:
        print(f"Starting Last Particle algorithm with {m}, to certify p<p_c={p_c}, with confidence level alpha ={1-alpha_test}.")
    if m>=n_max:
        raise AssertionError(f"Confidence level requires more than n_max={n_max} iterations... increase n_max ?")
    tau_j = np.array(nb_system*[-np.inf])
    
    is_done = np.zeros(nb_system)
    done_k = -np.ones(nb_system)
    
    kernel_pass= 0
    Count_accept = np.zeros(nb_system)
    check=0

    X = gen(nb_system*N).reshape((nb_system,N,d)) # generate N*nb_system samples
    SX = h_big(X.reshape((nb_system*N,d))).reshape((nb_system,N)) # compute their scores
    Count_h = nb_system*N # Number of calls to function h
    reject_rate = np.zeros(nb_system)
    avg_gain= np.zeros(nb_system)
    Xrare = -np.ones((nb_system,N,d))
    nb_system_c = nb_system #current number, as systems can get deleted as algorithm goes 
    real_indices = np.arange(nb_system)  #keeping track of initial systems indices as systems gets deleted
    local_indices = np.arange(nb_system_c)
    while (k<=m):
        #find new threshold
        
        i_deads = np.argmin(SX,axis = 1) # sort in descending order
        #we switch the 'last' particle in terms of score and the first particle as indices go, for simplicity
        tempXs, tempSs = np.array(X[:,0],copy=True), np.array(SX[:,0],copy=True)
        X[:,0], SX[:,0] = X[local_indices,i_deads],SX[local_indices,i_deads]
        X[local_indices,i_deads],SX[local_indices,i_deads] = tempXs, tempSs
        del tempSs, tempXs
        #print(SX[i_dead], tau_j )
        if k>1:
            gain = np.abs((SX[local_indices, i_deads]-tau_j[None])/tau_j[None])
        else:
            gain=np.zeros(nb_system_c)

        gamma = 1+gain_forget_rate*(k-1)
        avg_gain = (1-gamma/k)*avg_gain + (gamma/k)*gain
        if k>1: 
            is_too_low =  (avg_gain<gain_thresh) * (reject_rate<reject_thresh)
            if is_too_low.sum()>0:
                s_b = s_b/decay*is_too_low+s_b*(1-is_too_low)
                s_b = s_b.reshape(-1)
            if verbose>=1 and check%check_every==0:
                print('Strengths of kernels updated!')
                print(f's_b={s_b}')

        tau_j = SX[:,0] # set the threshold to the last particules's scores
        if (tau_j>tau).sum()>0:
            is_over = np.where(tau_j>tau)[0]
            if verbose:
                print(f"System(s):{is_over} reached required level.")
            #we need to kill systems that have reached required level, while taking this into account for the real systems indices
            is_done[real_indices[is_over]],done_k[real_indices[is_over]]=1,k
            if is_done.sum()==nb_system:
                break   #if all the systems have reached the final level we can stop the itertions there
            nb_system_c-=len(is_over)
            local_indices = np.arange(nb_system_c)
            Xrare[is_over] = X[is_over]
            X,SX = np.delete(X,is_over, axis=0),np.delete(SX,is_over, axis=0)
            gain, avg_gain,tau_j =  np.delete(gain,is_over), np.delete(avg_gain,is_over), np.delete(tau_j,is_over)  
            reject_rate, Count_accept = np.delete(reject_rate,is_over), np.delete(Count_accept,is_over)
            real_indices = np.delete(real_indices,is_over)
            s_b = np.delete(s_b ,is_over)



            
        if verbose>=1 and check%check_every==0:
            print('Iter = ',k, ' tau_j = ', tau_j, " Calls = ", Count_h)
        check+=1

        # Refresh samples
        i_news = np.random.choice(range(1,N),size=nb_system_c)        
        z0s = X[local_indices,i_news]
        sz0s = SX[local_indices,i_news]
        for _ in range(T):
            w = kernel_b(z0s,s_b) #kernel_b must take into account the number of systems and different strengths
            sw = h(w, real_indices)
            is_good_move = sw>=tau_j
            z0s,sz0s  = z0s*(1-is_good_move)[:,None] + is_good_move[:,None]*w, sz0s  *(1-is_good_move) + is_good_move*sw
            
            Count_accept = Count_accept + is_good_move
        
        X[:,0] = z0s
        SX[:,0] = sz0s
        
        del z0s, sz0s

        Count_h+=T*nb_system_c
        gamma = T+reject_forget_rate*kernel_pass
        reject_rate = (1-gamma/(kernel_pass+T))*reject_rate + gamma*(1-Count_accept/T)/(kernel_pass+T) 
        if check%check_every==0 and verbose>=1:
            print(f'Accept ratios (local averages):{Count_accept/T}')
            print(f'Reject rates (moving averages):{reject_rate}')
        kernel_pass+=T
        is_zero_accept = Count_accept==0
        is_too_high = reject_rate > (1-accept_ratio)
        if is_too_high.sum()>0:
            s_b = s_b*decay*is_too_high+s_b*(1-is_too_high)
            s_b = s_b.reshape(-1)
            if fast_decay:
                s_b = s_b*decay**fast_d*is_zero_accept+(1-is_zero_accept)*s_b
        if verbose>=1 and check%check_every==0:
            print('Strengths of kernel updated!')
            print(f's_b={s_b}')
        Count_accept = np.zeros(nb_system_c)
        k += 1 # increase iteration number
    
    
    
    if is_done.sum()>0:
        P_est = p**(done_k-1)*is_done+(1-is_done)*p_c
        var_est = is_done*P_est**2*(P_est**(-1/N)-1)-(1-is_done)
        CI_est = np.zeros((nb_system,2))
        CI_est[:,0] = is_done*(P_est*np.exp(-q/np.sqrt(N)*np.sqrt(-np.log(P_est)+(q**2)/4/N) - (q**2)/2/N))
        CI_est[:,1] = is_done*(P_est*np.exp(q/np.sqrt(N)*np.sqrt(-np.log(P_est)+(q**2)/4/N) - (q**2)/2/N)) + (1-is_done)*p_c
        cert_ = 1-is_done 
        s_out ={'var_est':var_est,'CI_est':CI_est,'Iter':k,'Calls':Count_h,'Sample size':N,'Cert':cert_}
        s_out['Xrare'] = Xrare
    else:
        s_out = {'var_est': -np.ones(nb_system), 'CI_est':np.array(nb_system*[0,p_c]),'Iter':k,'Calls':Count_h,'Sample size':N} 
        s_out['Cert']= np.array([True]*nb_system)
        s_out['Xrare']= None
        P_est = np.array(nb_system*[p_c])
    return P_est, s_out


def ImportanceSplitting(gen,kernel,h,tau,N=2000,K=1000,s=1,decay=0.99,T = 30,n_max = 300, alpha = 0.95,
verbose=1, track_rejection=False, rejection_ctrl = False, rej_threshold=0.9, gain_rate = 1.0001, 
prog_thresh=0.01):
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
         s_out: a dictionary containing additional data
           -s_out['var_est']: estimated variance
           -s_out.['CI_est']: estimated confidence of interval
           -s_out.['Xrare']: Examples of the rare event 
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
            if rejection_ctrl and rejection_rate>=rej_threshold:
                print('Strength of kernel diminished!')
                s = s*decay
                print(f's={s}')
            if not accept_flag:
                s = s * decay # decrease the strength of the mixing kernel


       
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
            s = s*gain_rate
            print('Strength of kernel increased!')
            print(f's={s}')

        tau_j = S_sort[K] # set the threshold to (K+1)-th

        
        h_mean = SX.mean()
        if verbose>=1:
            print('Iter = ',n, ' tau_j = ', tau_j, "h_mean",h_mean,  " Calls = ", Count_h)
            if track_rejection:
                print(f'Rejection rate: {rejection_rate}')
                rejection_rates+=[rejection_rate]

    # step E: Last round
    K_last = (SX>=tau).sum() # count the nb of score above the target threshold

    #Estimation
    p = K/N
    p_last = K_last/N
    P_est = (p**(n-1))*p_last
    var_est = (P_est**2)*((n-1)*(1-p)/p + (1-p_last)/p_last)/N
    P_bias = P_est*n*(1-p)/p/N
    CI_est = P_est*np.array([1,1]) + q*np.sqrt(var_est)*np.array([-1,1])
    Xrare = X[(SX>=tau).reshape(-1),:]
    s_out = {"var_est":var_est,"CI_est": CI_est,"N":N,"K":K,"s":s,"decay":decay,"T":T,"Count_h":Count_h,
    "P_bias":P_bias,"n":n,"Xrare":Xrare}
    if track_rejection:
        s_out["rejection_rates"]=np.array(rejection_rates)
        s_out["Avg. rejection rate"]=rejection_rate
    

       
    return P_est,s_out