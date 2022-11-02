import eagerpy as ep
import numpy as np



def apply_native_to_ep(x: ep.Tensor)-> ep.Tensor:
    return ep.astensor(f(x.raw))

def ep_normal_like(x: ep.Tensor)-> ep.Tensor:
    return x.normal(shape=x.shape)

def ep_std(x, axis=0):
    x = ep.astensor(x)
    result = (x.square().mean(axis=axis)-x.mean(axis=axis).square).sqrt()
    return result

def gaussian_kernel_ep(x,dt,scale_M=1):
    kappa= 1. / (1 + ep.sqrt(1 - dt**2*(1/scale_M)))
    return x-dt**2*kappa*(1/scale_M)*x+dt*((scale_M)**(-1/2))*x.normal(shape=x.shape)

def time_step_ep(V,X:ep.Tensor,gradV,p:int=1): 
    """ 
    Args:
        V (_type_): potential function in the native DL framework
        X (ep.Tensor): perturbation (latent) vector
        gradV (_type_): gradient of potential function in the native DL framework
        p (int, optional): norm order. Defaults to 1.

    Returns:
        ep.Tensor: _description_
    """
    V_mean= ep.astensor(V(X.raw)).mean()
    grads_ep = ep.astensor(gradV(X.raw))
    V_grad_norm_mean = ((ep.norms.l2(grads_ep,axis=1)**p).mean())**(1/p)
    
    time_step=V_mean/V_grad_norm_mean
    return time_step

def apply_gaussian_kernel_ep(Y:ep.Tensor, v_y,T:int,beta:float,dt, V,
 gaussian:bool, adapt_dt:bool=False,
 debug:bool=False,verbose=1
,kernel_pass=0, track_accept:bool=False,
save_Ys=False,scale_M=1):
    nb=Y.shape[0]
    Ys=[Y]
    VY=v_y 
    nb_calls=0
    l_accept_rates=[]
    l_kernel_pass=0
    for _ in range(T):
        Z = gaussian_kernel_ep(Y,dt,scale_M=scale_M) # propose refreshed samples
        kernel_pass+=nb
        l_kernel_pass+=nb
        # compute their scores
        VZ= apply_native_to_ep(f=V,x=Z)
        nb_calls+=nb
        log_a_high=-beta*VZ
        log_a_low= -beta*VY
        if gaussian: 
            log_a_high-= 0.5*Z.square().sum(axis=1)
            log_a_low-= 0.5*Y.square().sum(axis=1)
        alpha_= ep.exp(log_a_high-log_a_low)
        b_size= nb
        U= Y.uniform(size=(b_size,))
        accept_flag=U<alpha_
        if verbose>=2.5:
            print(accept_flag.float32().mean().item())
        Y=Y.where(accept_flag.reshape((-1,1)),Z,Y)
        Ys.append(Y)
        VY=Y.where(accept_flag,VZ,VY)
        if track_accept:
            l_accept_rate=accept_flag.float32().mean().item()
            if verbose>=3:
                print(f"local accept_rate:{l_accept_rate}")
            l_accept_rates.append(l_accept_rate)
    dict_out={'l_kernel_pass':l_kernel_pass}
    if adapt_dt:
        dict_out['dt']=dt
    if track_accept:
        dict_out['acc_rate']=np.array(l_accept_rates).mean()
    if save_Ys:
        dict_out['Ys']=Ys
    return Y,VY,nb_calls,dict_out


def score_function_ep(X,y_0,model):
    y = apply_native_to_ep(f=model,x=X)
    y_diff = ep.concatenate([y[:,:y_0], y[:,(y_0+1):]],axis=1) - y[:,y_0].reshape((-1,1))
    s, _ = y_diff.max(axis=1)
    return s


def hamiltonian_ep(X,p,V,beta,scale_M=1,gaussian=True):
    U = beta*apply_native_to_ep(f=V,x=X) +0.5*X.square().sum(axis=1) if gaussian else beta*apply_native_to_ep(f=V,x=X)
    H = U + 0.5*((1/scale_M)*p**2).sum(axis=1)
    return H

def verlet_mcmc(q,beta:float,gaussian:bool,V,gradV,T:int,delta_t,L:int=1,
kappa_opt:bool=True,save_H=True,save_func=None,device='cpu',scale_M=None,gaussian_verlet=False,ind_L=None):
    """ Simple EagerPy implementation of Hamiltonian MCMC 

    Args:
        q (_type_): _description_
        p (_type_): _description_
        beta (_type_): _description_
        gaussian (_type_): _description_
        V (_type_): _description_
        gradV (_type_): _description_
        T (_type_): _description_
        delta_t (_type_): _description_
        kappa_opt (bool, optional): _description_. Defaults to True.
        save_H (bool, optional): _description_. Defaults to True.
        save_Q (bool, optional): _description_. Defaults to True.
        save_func (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    acc  = 0
    p=ep_normal_like(q)
    if scale_M is not None:
        sqrt_M = ep.sqrt(scale_M)
        p=sqrt_M*ep_normal_like(q)
    (N,d)=q.shape
    H_old = hamiltonian_ep(X=q,p=p,beta=beta, gaussian=gaussian,V=V)
    nb_calls=N
    if save_H:
        H_ = np.zeros(T+1)
        H_[0]=H_old.mean()

    if save_func is not None:
        saved=[save_func(q,p)]
    for i  in range(T):
        q_trial,p_trial=verlet_kernel_ep(X=q,gradV=gradV, p_0=p,delta_t=delta_t,beta=0,L=L,kappa_opt=kappa_opt,
        scale_M=scale_M,ind_L=ind_L,GV=gaussian_verlet)
        nb_calls+=ind_L.sum().item()
        H_trial= hamiltonian_ep(X=q_trial, p=p_trial,V=V,beta=beta,gaussian=gaussian)
        nb_calls+=N
        
        alpha= q.uniform(size=(N,))
        delta_H=ep.clip(-(H_trial-H_old),max=0)
        accept=ep.exp(delta_H)>alpha
        acc+=accept.sum()
        q=p.where(accept.unsqueeze(-1),q_trial,q)
        p = ep_normal_like(q)
        if scale_M is not None:
            p = sqrt_M*p
        H_old= hamiltonian_ep(X=q, p=p,V=V,beta=beta,gaussian=gaussian)
        if save_H:
            H_[i+1] = H_old.mean()
        nb_calls+=N
        

        if save_func is not None:
            saved.append(save_func(q,p))
    
    dict_out={'acc_rate':acc/(T*N)}

    if save_H:
        dict_out['H']=H_
    if save_func is not None:
        dict_out['saved']=saved
    v_q=V(q)
    nb_calls+=N
    return q,v_q,nb_calls,dict_out


def verlet_kernel_ep(X, gradV, delta_t, beta,L,ind_L=None,p_0=None,lambda_=0, gaussian=True,kappa_opt=False,scale_M=None,GV=False):
    """ HMC (L>1) / Underdamped-Langevin (L=1) kernel with Verlet integration (a.k.a. Leapfrog scheme)

    """
    if ind_L is None:
        ind_L=L*np.ones(size=(X.shape[0],))
    q_t = X
    # if no initial momentum is given we draw it randomly from gaussian distribution
    if scale_M is None:
        scale_M=1
    if p_0 is None:                        
        p_t=ep.sqrt(scale_M)*ep_normal_like(X)

    else:
        p_t = p_0
    grad_q=lambda p,dt:dt*(p/scale_M)
    if kappa_opt:
        kappa= 2. / (1 + (1 - delta_t**2)**(1/2)) #if scale_M is 1 else 2. / (1 + torch.sqrt(1 - delta_t**2*(1/scale_M)))
       
    else:
        kappa=q_t.ones_like()
    k=1
    i_k=ind_L>=k
    while (i_k).sum()>0:
        #I. Verlet scheme
        #computing half-point momentum
        #p_t = p_t-0.5*dt*gradV(X) / norms(gradV(X))
        
        if not GV:
            p_t[i_k] = p_t[i_k]-0.5*beta*delta_t[i_k]*gradV(q_t[i_k]) 
        
        
        p_t[i_k] = p_t[i_k]- 0.5*delta_t[i_k]*kappa[i_k]*q_t[i_k]
        if p_t.isnan().any():
            print("p_t",p_t)
        #updating position
        q_t[i_k] = (q_t[i_k] + grad_q(p_t[i_k],delta_t[i_k]))
        assert not q_t.isnan().any(),"Nan values detected in q"
        #updating momentum again
        if not GV:
            p_t[i_k] = p_t[i_k] -0.5*beta*delta_t[i_k]*gradV(q_t[i_k])
        p_t[i_k] =p_t[i_k] -0.5*kappa[i_k]*delta_t[i_k]*q_t[i_k]
        #II. Optional smoothing of momentum memory
        p_t[i_k] = ep.exp(-lambda_*delta_t[i_k])*p_t[i_k]
        k+=1
        i_k=ind_L>=k
    return q_t,p_t