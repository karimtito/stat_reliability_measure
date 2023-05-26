import eagerpy as ep
import numpy as np
from stat_reliability_measure.dev.tf_utils import compute_V_grad_tf2, compute_V_tf2, compute_h_tf, norm_dist_tf
from stat_reliability_measure.dev.torch_utils import compute_V_pyt, compute_V_grad_pyt, compute_h_pyt, norm_dist_pyt


def apply_native_to_ep(f,x: ep.Tensor)-> ep.Tensor:
    return ep.astensor(f(x.raw))

def ep_normal_like(x: ep.Tensor)-> ep.Tensor:
    return x.normal(shape=x.shape)

def ep_std(x, axis=0):
    #x = ep.astensor(x)
    result = (x.square().mean(axis=axis)-x.mean(axis=axis).square()).sqrt()
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
    V_mean=V(X).mean()
    grads_ep = gradV(X)
    V_grad_norm_mean = ((ep.norms.l2(grads_ep,axis=1)**p).mean())**(1/p)
    
    time_step=V_mean/V_grad_norm_mean
    return time_step

def apply_gaussian_kernel_ep(Y:ep.Tensor, v_y,T:int,beta:float,dt, V,
 gaussian:bool, adapt_dt:bool=False,
 verbose=1
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
        VZ= V(Z)
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


def hamiltonian_nat_ep(X,p,V,beta,scale_M=1,gaussian=True):
    U = beta*apply_native_to_ep(f=V,x=X) +0.5*X.square().sum(axis=1) if gaussian else beta*apply_native_to_ep(f=V,x=X)
    H = U + 0.5*((1/scale_M)*p**2).sum(axis=1)
    return H

def hamiltonian_ep(X,p,V,beta,scale_M=1,gaussian=True):
    U = beta*V(X) +0.5*X.square().sum(axis=1) if gaussian else beta*V(X)
    H = U + 0.5*((1/scale_M)*p**2).sum(axis=1)
    return H

def verlet_mcmc_ep(q,beta:float,gaussian:bool,V,gradV,T:int,delta_t, grad_V_q=None,L:int=1,
kappa_opt:bool=True,save_H=True,save_func=None,device='cpu',scale_M=None,gaussian_verlet=False,ind_L=None):
    """ Simple EagerPy implementation of hamiltonian_ep MCMC 

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
        q_trial,p_trial,grad_calls,grad_V_q=verlet_kernel_ep(X=q,gradV=gradV, p_0=p,delta_t=delta_t,beta=0,L=L,kappa_opt=kappa_opt,
        scale_M=scale_M,ind_L=ind_L,GV=gaussian_verlet)
        nb_calls+=grad_calls
        H_trial= hamiltonian_ep(X=q_trial, p=p_trial,V=V,beta=beta,gaussian=gaussian)
        nb_calls+=N
        
        alpha= q.uniform(size=(N,))
        delta_H=ep.clip(-(H_trial-H_old),min_=None,max_=0)
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
    return q,v_q,grad_V_q,nb_calls,dict_out

def adapt_verlet_mcmc_ep(q,v_q,ind_L, beta:float,gaussian:bool,V,gradV,T:int,delta_t:float,
        grad_V_q=None,L:int=1,kappa_opt:bool=True,save_H=True,save_func=None,device='cpu',scale_M=None,alpha_p:float=0.1,
prop_d=0.1,FT=False,dt_max=None,dt_min=None,sig_dt=0.015,
verbose=0,L_min=1,gaussian_verlet=False,skip_mh=False):
    """ Simple implementation of hamiltonian_ep dynanimcs MCMC 

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
        alpha_p 

    Returns:
        _type_: _description_
    """
   
    acc  = 0
    T_max=T
    if scale_M is None:
        scale_M=1
        sqrt_M=1
    else:
        sqrt_M = scale_M.sqrt()
    p=sqrt_M*ep_normal_like(q)
    if FT:
        
        maha_dist = lambda x,y: ((1/scale_M)*(x-y)**2).sum(1)
        ones_L= np.ones_like(ind_L)
    (N,d)=q.shape
    o_old=q+q**2
    mu_old,sig_old=(o_old).mean(0),ep_std(o_old)
    H_old= hamiltonian_ep(X=q, p=p,V=V,beta=beta,gaussian=gaussian,scale_M=scale_M)
    nb_calls=0 #we can reuse the potential value v_q of previous iteration: no new potential computation
    if save_H:
        H_ = np.zeros(T+1)
        H_[0]=H_old.mean().item()

    if save_func is not None:
        saved=[save_func(q,p)]
    prod_correl = q.ones(shape=(d,))
    i=0
 
    while (prod_correl>alpha_p).sum().item()>=prop_d*d and i<T_max:
        q_trial,p_trial,grad_calls,grad_V_q=verlet_kernel_ep(X=q,gradV=gradV,grad_V_q=grad_V_q, p_0=p,delta_t=delta_t,beta=beta,L=L,kappa_opt=kappa_opt,
        scale_M=scale_M, ind_L=ind_L,GV=gaussian_verlet)
        
        nb_calls+=2*grad_calls # counting gradient calls as 2 potential calls
        H_trial= hamiltonian_ep(X=q_trial, p=p_trial,V=V,beta=beta,gaussian=gaussian,scale_M=scale_M)
        nb_calls+=N # N new potentials are computed 
        delta_H= ep.clip(-(H_trial-H_old), min_=-1e5,max_=0)
        if FT:
            exp_weight = ep.exp(ep.clip(delta_H,min_=-1e5,max_=0))
            m_distances= maha_dist(x=q,y=q_trial)/q.from_numpy(ind_L.astype(np.float32))
            lambda_i=m_distances*exp_weight+1e-5*m_distances.ones_like()
            if lambda_i.isnan().any():
                print(f"NaN values in lambda_i:")
                print(lambda_i)
            elif (lambda_i==0).sum()>0:
                print(f"zero values in lambda_i")
                print(f"lambda_i")
                print(f"exp_weight:{exp_weight}")
                print(f"m_distances:{m_distances}")
                if (m_distances==0).sum()>0:
                    print(f"q_trial:{q_trial}")
                    print(f"q:{q}")
                print(f"ind_L:{ind_L}")
            norm_lambda=(lambda_i/lambda_i.sum()).numpy()
            sel_ind =np.where(np.random.multinomial(n=1,pvals=norm_lambda,size = (N,)))[1]
            
            delta_t = ep.clip(delta_t[sel_ind]+sig_dt*ep_normal_like(delta_t),min_=dt_min,max_=dt_max,)

            noise_L=np.random.rand(N)
            
            ind_L= np.clip(ind_L[sel_ind]+((noise_L>=(2/3)).astype(float)-(noise_L<=1/3).astype(float))*ones_L,a_min=L_min,a_max=L+1e-3)
            
        if skip_mh:
            q=q_trial
            nb_accept=N
        else:
            alpha=delta_H.uniform(shape=(N,))
            accept=ep.exp(delta_H)>alpha
            nb_accept=accept.sum().item()
            acc+=nb_accept
            q=accept.reshape((-1,1)).where(q_trial,q)
        if nb_accept>0:
            o_new=q+q**2
            mu_new,sig_new=o_new.mean(0),ep_std(o_new)
            correl=((o_new-mu_new)*(o_old-mu_old)).mean(0)/(sig_old*sig_new)
            prod_correl*=correl
            mu_old, sig_old = mu_new, sig_new
            
    
        p = ep_normal_like(q)
        if scale_M is not None:
            p = sqrt_M*p
        H_old= hamiltonian_ep(X=q, p=p,V=V,beta=beta,gaussian=gaussian,scale_M=scale_M)
        #no new potential is computed 
        if save_H:
            H_[i+1] = H_old.mean()
        
        

        if save_func is not None:
            saved.append(save_func(q,p))
        i+=1
    if verbose:
        print(f"T_final={i}")
    dict_out={'acc_rate':acc/(i*N)}

    if save_H:
        dict_out['H']=H_
    if save_func is not None:
        dict_out['saved']=saved
    if FT:
        dict_out['dt']=delta_t
        dict_out['ind_L']=ind_L
    v_q= V(q)
    #no new potential computation (already done in the computation of H_new)
    
    return q,v_q,grad_V_q,nb_calls,dict_out

def verlet_kernel_ep(X, gradV, delta_t, beta,L,grad_V_q=None,
    ind_L=None,p_0=None,lambda_=0, gaussian=True,kappa_opt=False,scale_M=None,GV=False):
    """ HMC (L>1) / Underdamped-Langevin (L=1) kernel with Verlet integration (a.k.a. Leapfrog scheme)

    """
    nb_particles = X.shape[0]
    grad_calls=0
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
        kappa= 2. / (1 + (1 - delta_t**2)**(1/2))        
    else:
        kappa=q_t.ones_like()
    k=1
    i_k=(ind_L>=k)
    if not GV and grad_V_q is None:
        grad_V_q = gradV(q_t)
        grad_calls+=2*nb_particles
    while (i_k).sum()>0:
        #I. Verlet scheme
        #computing half-point momentum
        #p_t = p_t-0.5*dt*gradV(X) / norms(gradV(X))
        i_k_ep = (p_t.from_numpy(i_k)).reshape((-1,1))
        if not GV:
            #TODO /!\ find a way to compute gradient only for p_t[i_k] as in PyTorch implementation
            p_t = i_k_ep.where(p_t-0.5*beta*delta_t*grad_V_q,p_t)
        
        p_t =  i_k_ep.where(p_t- 0.5*delta_t*kappa*q_t,p_t)
        if p_t.isnan().any():
            print("p_t",p_t)
        #updating position
        q_t = i_k_ep.where(q_t + grad_q(p_t,delta_t),q_t)
        assert not q_t.isnan().any(),"Nan values detected in q"
        #updating momentum again
        if not GV:
            grad_V_q = gradV(q_t)
            grad_calls+=2*nb_particles
            p_t = i_k_ep.where(p_t-0.5*beta*delta_t*grad_V_q,p_t)
        p_t = i_k_ep.where(p_t- 0.5*delta_t*kappa*q_t,p_t)
        #II. Optional smoothing of momentum memory
        p_t = i_k_ep.where(ep.exp(-lambda_*delta_t)*p_t,p_t)
        del i_k_ep
        k+=1
        i_k=ind_L>=k
    return q_t,p_t, grad_calls, grad_V_q

norm_dist_dic = {ep.TensorFlowTensor: norm_dist_tf,ep.PyTorchTensor: norm_dist_pyt}
compute_grad_dic = {ep.TensorFlowTensor: compute_V_grad_tf2, ep.PyTorchTensor: compute_V_grad_pyt }
compute_V_dic = {ep.TensorFlowTensor: compute_V_tf2, ep.PyTorchTensor: compute_V_pyt }
compute_h_dic = {ep.TensorFlowTensor: compute_h_tf, ep.PyTorchTensor: compute_h_pyt }

def gradV_ep(x_:ep.Tensor,model,target_class,input_shape,low:ep.Tensor,high:ep.Tensor
,gaussian_latent=True,reshape=True,gaussian_prior=False) -> ep.Tensor:
    if gaussian_latent and not gaussian_prior:
        norm_dist=norm_dist_dic[type(x_)](loc=0.,scale=1.)
        u=ep.astensor(norm_dist.cdf(x_.raw))
    else:
        u=x_
    if reshape:
        u=ep.reshape(u,(u.shape[0],)+input_shape)
    
    x_p = u if gaussian_prior else low+(high-low)*u 
    if gaussian_prior:
        x_p=ep.maximum(x_p,low.reshape(low.shape+(1,1)))
        x_p=ep.minimum(x_p,high.reshape(high.shape+(1,1)))
    _,grad_x_p = compute_grad_dic[type(x_)](model=model,input_=x_p.raw,target_class=target_class)
    grad_x_p = ep.astensor(grad_x_p)
    grad_u=ep.reshape(grad_x_p,x_.shape) if gaussian_prior else ep.reshape((high-low)*grad_x_p,x_.shape)
    if gaussian_latent and not gaussian_prior:
        grad_x=ep.exp(ep.astensor(norm_dist.log_prob(x_.raw)))*grad_u
    else:
        grad_x=grad_u
    return grad_x

def V_ep(x_,model,target_class,input_shape,low,high,gaussian_latent=True,reshape=True,gaussian_prior=False) -> ep.Tensor:
    if gaussian_latent:
        norm_dist=norm_dist_dic[type(x_)](loc=0.,scale=1.)
        u=apply_native_to_ep(f=norm_dist.cdf, x=x_)
    else:
        u=x_
    if reshape:
        u=ep.reshape(u,(u.shape[0],)+input_shape)   
    x_p = u if gaussian_prior else low+(high-low)*u 
    if gaussian_prior:
        x_p=ep.maximum(x_p,low.reshape(low.shape+(1,1)))
        x_p=ep.minimum(x_p,high.reshape(high.shape+(1,1)))
    v = ep.astensor(compute_V_dic[type(x_)](model=model,input_=x_p.raw,target_class=target_class))
    return v