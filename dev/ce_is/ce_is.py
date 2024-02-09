import torch
import scipy.stats as stats
import numpy as np
from math import sqrt
from stat_reliability_measure.dev.torch_utils import NormalCDFLayer, NormalToUnifLayer
from stat_reliability_measure.dev.mpp_utils import gradient_binary_search, gaussian_space_attack, mpp_search_newton, binary_search_to_zero
from stat_reliability_measure.dev.imp_sampling.is_pyt import GaussianImportanceWeight2, GaussianImportanceWeight3

def gen_batched(N_ce,batch_size,gen):
    xs = []
    for _ in range(N_ce//batch_size):
        xs.append(gen(batch_size))
    if N_ce%batch_size!=0:
        xs.append(gen(N_ce%batch_size))
    return torch.cat(xs,dim=0)

def CrossEntropyIS(gen,h,rho=0.5,N:int=int(1e4), t_max=100, estimate_covar=False,
                   estimate_var=False, N_ce=None, theta_0=None,ce_masri=False,
                   batch_size:int=int(1e3),save_rare:bool=False,verbose=0.,track_X:bool=False,
               nb_calls_mpp=0,sigma_bias=1.,G=None,gradG=None,  t_transform=None,y_clean=None,
               save_mpp=False,save_theta=False,save_thetas=False,save_sigma=False,save_sigmas=False,
               model=None,search_method='mpp_search',u_mpp=None,save_weights=True, epsilon=0.1,
               eps_real_mpp=1e-2,real_mpp=True, steps=100,stepsize=1e-2,gamma=0.05,
               alpha_CI=0.05, num_iter=32,stop_eps=1E-2,stop_cond_type='beta',
               random_init=False, sigma_init=0.1,**kwargs):
    """ Cross-Entropy Importance Sampling for Rare Event Simulation

    Args:
        gen (_type_): random generator
        h (_type_): score function
        N_ce (int, optional): number of samples to use. Defaults to int(1e4).
        batch_size (int, optional): batch size. Defaults to int(1e2).
        save_rare (bool, optional): Option to track adversarial examples. Defaults to False.
        verbose (float, optional): Level of verbosity. Defaults to False.


    Returns:
        float: probability of failure
    """
    u_rare = []
    if N_ce is None:
        N_ce = N
    K = int(N_ce*rho)
    test = gen(1)
    d= test.numel()
    
    
    sigma_0 = torch.ones((1,d),device=test.device)
    device = test.device
    del test
    zero_latent = torch.zeros((1,d),device=device)
    if theta_0 is None:
        theta_t = zero_latent
    else:
        theta_t = theta_0
    if estimate_covar:
        Sigma_t = torch.eye(d)
    if save_thetas:
        thetas = [theta_t]
    sigma_t=sigma_0
    if save_sigmas:
        sigmas = [sigma_t]
    U = gen_batched(N_ce,batch_size=batch_size,gen=gen)+theta_t
    
    with torch.no_grad():
        SU = h(U)
    Count_h = N_ce
    ind= torch.argsort(input=SU,dim=0,descending=True).squeeze(-1)
    S_sort = SU[ind]
    
    gamma_t = S_sort[K]
    h_mean = SU.mean()
    t=0
    
    if verbose>=1:
        print('Iter = ',t, ' tau_j = ', gamma_t.item(), "h_mean",h_mean.item(),  " Calls = ", Count_h)
    U_surv = U[ind[0:K]]
    gauss_weights = GaussianImportanceWeight2( x=U_surv,mu_1 = zero_latent, sigma_1=sigma_0,mu_2=theta_t,sigma_2=sigma_t )
    if gauss_weights.sum()==0:
        print('gauss_weights.sum()==0 at first iteration')
        
    if gauss_weights.sum()<1e-10:
        print('gauss_weights.sum()<1e-10 at first iteration')
        
    normed_weights = (gauss_weights/gauss_weights.sum()).unsqueeze(1)
    while t<t_max and gamma_t<0:
        t+=1
        theta_t = (normed_weights*U_surv).sum(0).unsqueeze(0)


        if ce_masri:
            print("using CE-Masri")
            theta_norm = torch.norm(theta_t)
            u_t = theta_t/theta_norm
            y = (u_t[None,:]*U_surv).sum(-1)
            v_hat = (normed_weights.squeeze(1)*(y-theta_norm).square()).sum()
            
        
            seed_U = gen_batched(N_ce,batch_size=batch_size,gen=gen)
            U= theta_t[:] + (seed_U +  ((v_hat-1)/(1+sqrt(v_hat)))*(u_t.unsqueeze(0)*(u_t[None,:]*seed_U).sum(-1)))
        else:   
        
            if estimate_covar:
                Sigma_t = torch.eye(d)
            elif estimate_var:
                sigma_t = (normed_weights*(U_surv-theta_t)**2 ).sum(0).unsqueeze(0).sqrt()
            
            U = theta_t[:] + gen_batched(N_ce,batch_size=batch_size,gen=gen)*sigma_t
        with torch.no_grad():
            SU = h(U)
        Count_h += N_ce
        ind= torch.argsort(input=SU,dim=0,descending=True).squeeze(-1)
        S_sort = SU[ind]
        gamma_t = S_sort[K]
        h_mean = SU.mean()
        if verbose>=1:
            print('Iter = ',t, ' tau_j = ', gamma_t.item(), "h_mean",h_mean.item(),  " Calls = ", Count_h)
        U_surv = U[ind[0:K]]
        gauss_weights = GaussianImportanceWeight2( x=U_surv,mu_1 = zero_latent, sigma_1=sigma_0,mu_2=theta_t,sigma_2=sigma_t )
        if gauss_weights.sum()==0:
            print('gauss_weights.sum()==0')
            break
        if gauss_weights.sum()<1e-10:
            print('gauss_weights.sum()<1e-10')
            break
        normed_weights = (gauss_weights/gauss_weights.sum()).unsqueeze(1)
        if save_thetas:
            thetas.append(theta_t)
        if save_sigmas: 
            sigmas.append(sigma_t)
            
    
    n= 0
    pre_var = 0.
    p_f = 0.
    gen_bias = lambda n: theta_t + gen(n) * sigma_t
    if save_weights:
        weights = []
    if save_rare:
        u_rare = []
    for _ in range(N//batch_size):
        x_mc = gen_bias(batch_size)
        with torch.no_grad():
            rare_event= h(x_mc)>=0
        if save_rare:
            u_rare.append(x_mc[rare_event])
        
        n+= batch_size
        gauss_weights=  GaussianImportanceWeight2(x=x_mc,mu_1=zero_latent,sigma_1=sigma_0,mu_2=theta_t,sigma_2=sigma_t)
        p_local = (rare_event)*gauss_weights
        pre_var_local = (rare_event)*gauss_weights**2
        del x_mc,rare_event
        if save_weights:
            weights.append(p_local)
        p_f = ((n-batch_size)/n)*p_f+(batch_size/n)*p_local.float().mean()
        pre_var = ((n-batch_size)/n)*pre_var+(batch_size/n)*pre_var_local.float().mean()
           
    if N%batch_size!=0:
        rest = N%batch_size
        x_mc = gen_bias(rest)
        with torch.no_grad():
            rare_event= h(x_mc)>=0
        if save_rare:
            u_rare.append(x_mc[rare_event])
        gauss_weights=  GaussianImportanceWeight2(x=x_mc,mu_1=zero_latent,sigma_1=sigma_0,mu_2=theta_t,sigma_2=sigma_t)
        p_local = (rare_event)*gauss_weights
        pre_var_local = (rare_event)*gauss_weights**2
        if save_weights:
            weights.append(p_local)
        del x_mc,rare_event
        n+= rest
        
        p_f = ((N-rest)/N)*p_f+(rest/N)*p_local.float().mean()
        pre_var = ((N-rest)/N)*pre_var+(rest/N)*pre_var_local.float().mean()
      
    dict_out = {'nb_calls':Count_h+N,'p_f':p_f,}
    var_est = (1/N)*(pre_var-p_f**2).item()
    dict_out['var_est'] = var_est
    dict_out['std_est'] = np.sqrt(var_est)
    if save_weights:
        dict_out['weights'] = torch.cat(weights,dim=0).to('cpu').numpy()
    if save_rare:
        dict_out['u_rare'] = torch.cat(u_rare,dim=0).to('cpu').numpy()
    if save_thetas:
        dict_out['thetas'] = torch.cat(thetas,dim=0).to('cpu').numpy()
    if save_theta:
        dict_out['theta'] = theta_t.to('cpu').numpy()
    if save_sigmas:
        dict_out['sigmas'] = torch.cat(sigmas,dim=0).to('cpu').numpy()
    if save_sigma:
        dict_out['sigma'] = sigma_t.to('cpu').numpy()
        
    
    CI = stats.norm.interval(1-alpha_CI,loc=p_f.item(),scale=np.sqrt(var_est))
    dict_out['CI']=CI
    return p_f.cpu().item(),dict_out
     
        
            
        
            
        
    
    
    
        