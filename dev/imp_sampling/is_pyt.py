import torch
import scipy.stats as stats
import numpy as np
from math import sqrt
from stat_reliability_measure.dev.torch_utils import NormalCDFLayer, NormalToUnifLayer
from stat_reliability_measure.dev.mpp_utils import gradient_binary_search, gaussian_space_attack, mpp_search_newton, binary_search_to_zero

def GaussianImportanceWeight(x,mu_1,mu_2,sigma_1=1.,sigma_2=1.,d=1):
    """ Computes importance weights for Gaussian distributions for Importance Sampling

    Args:

        x (torch.tensor): input
        mu_1 (torch.tensor): mean of first Gaussian
        mu_2 (torch.tensor): mean of second Gaussian
        sigma_1 (torch.tensor): standard deviation of first Gaussian
        sigma_2 (torch.tensor): standard deviation of second Gaussian

    Returns:
        torch.tensor: importance weights i.e. ratio of densities of the two Gaussians at x
    """
    density_ratio = torch.exp(-0.5*((x-mu_1)/sigma_1).square().sum(-1)+0.5*((x-mu_2)/sigma_2).square().sum(-1)) * (sigma_2**d/sigma_1**d)
    return density_ratio


def GaussianImportanceWeight2(x,mu_1,mu_2,sigma_1=1.,sigma_2=1.,d=1):
    """ Computes importance weights for Gaussian distributions for Importance Sampling

    Args:

        x (torch.tensor): input
        mu_1 (torch.tensor): mean of first Gaussian
        mu_2 (torch.tensor): mean of second Gaussian
        sigma_1 (torch.tensor): standard deviation of first Gaussian
        sigma_2 (torch.tensor): standard deviation of second Gaussian

    Returns:
        torch.tensor: importance weights i.e. ratio of densities of the two Gaussians at x
    """
    density_ratio = torch.exp(-0.5*((x-mu_1)/sigma_1).square().sum(-1)+0.5*((x-mu_2)/sigma_2).square().sum(-1)) * (sigma_2.prod()/sigma_1.prod())
    return density_ratio



def mpp_search(grad_f, zero_latent,num_iter=100,stop_cond_type='grad_norm',
               stop_eps=1e-3,
            mult_grad_calls=2.,debug=False,print_every=10,):
    """ Search algorithm for the Most Probable Point (u_mpp) with a Newton method.  """
    x= zero_latent
    grad_fx,f_x = grad_f(x)
    grad_calls=1
    beta=torch.norm(x,dim=-1)
    k= 0
    stop_cond=False
    print_count=0
    debug_ = debug
    
    while k<num_iter and  (~stop_cond or f_x>0): 
        k+=1
        a = grad_fx/torch.norm(grad_fx,dim=-1)
        beta_new = beta + f_x/torch.norm(grad_fx,dim=-1)
        u_new=-a*beta_new
        grad_f_xnew,f_x = grad_f(u_new)
        grad_calls+=1 
        if debug_:
            debug = print_count%print_every==0
        if stop_cond_type not in ['grad_norm','norm','beta']:
            raise NotImplementedError(f"Method {stop_cond_type} is not implemented.")
        if stop_cond_type=='grad_norm':
            diff = torch.norm(grad_fx-grad_f_xnew,dim=-1)
        elif stop_cond_type=='norm':
            diff = torch.norm(x-u_new,dim=-1)
        elif stop_cond_type=='beta':
            diff = torch.abs(beta-beta_new)
           
            
        if debug:
                print(f"{stop_cond_type}_diff: {diff}")
        stop_cond = (diff<stop_eps).item()
        beta=beta_new
        if debug:
            print(f'beta: {beta}')
        if debug_:
            print_count+=1
        x=u_new
        grad_fx=grad_f_xnew
    if k==num_iter and debug:
        print("Warning: maximum number of iteration has been reached for MPP search")
    nb_calls= mult_grad_calls*grad_calls
    dict_out = {'nb_calls':nb_calls}
    return x, dict_out

def hlrf(grad_f, zero_latent,num_iter=10,stop_cond_type='grad_norm',
               stop_eps=1e-3,step_size=1.,save_every=1,save_history=False,
            mult_grad_calls=2.,debug=False,print_every=1,):
    """ Search algorithm for the Most Probable Point (u_mpp) with a Newton method.  """
    
    

    k = 0
    stop_cond = False
    print_count = 0
    save_count = 0
    debug_ = debug
    u = zero_latent
    if save_history:
        u_history = [u]
    grad_f_x,f_x = grad_f(u)
    norm_grad = torch.norm(grad_f_x,dim=-1)
    beta = f_x / norm_grad
    
    a = -grad_f_x / norm_grad
    grad_calls = 1
    while k < num_iter and (not stop_cond): 

        
       
        u_new = u + step_size * (beta * a - u)
        grad_f_xnew,f_xnew = grad_f(u_new)
        norm_grad = torch.norm(grad_f_xnew,dim=-1)
        a_new = -grad_f_xnew / norm_grad
        beta_new = torch.sum(u_new*a_new,dim=1) + f_xnew / norm_grad
        if debug_:
            debug = print_count % print_every == 0
        if stop_cond_type not in ['grad_norm','norm','beta']:
            raise NotImplementedError(f"Method {stop_cond_type} is not implemented.")
        if stop_cond_type=='grad_norm':
            diff = torch.norm(grad_f_xnew - grad_f_x)
        elif stop_cond_type=='norm':
            diff = torch.norm(u_new - u)
        elif stop_cond_type=='beta':
            diff = torch.abs(beta - beta_new)
           
            
        if debug:
                print(f"{stop_cond_type}_diff: {diff}")
                print(f"beta: {beta}")
                print(f"beta_new: {beta_new}")
                print(f"u: {u}")
                print(f"u_new: {u_new}")
                
        if k % save_every == 0:
            save_count += 1
            if debug:
                print(f"Saving u at iteration {save_count}")
            if save_history:
                u_history.append(u_new)
        stop_cond = (diff<stop_eps).item()
        beta=beta_new
        grad_f_x=grad_f_xnew
        f_x = f_xnew
        beta = beta_new
        u = u_new
        if debug:
            print(f'beta: {beta}')
        if debug_:
            print_count+=1
        k+=1
    if k==num_iter and debug:
        print("Warning: maximum number of iteration has been reached for MPP search")
    nb_calls = mult_grad_calls*grad_calls
    dict_out = {'nb_calls':nb_calls}
    if save_history:
        u_history = torch.stack(u_history)
        dict_out['u_history'] = u_history
    return u, dict_out

search_methods_list=['mpp_search','gradient_binary_search',''
                     'carlini','carlini-wagner','cw','carlini_wagner','carlini_wagner_l2','brendel','brendel-bethge','brendel_bethge',
                     'adv','adv_attack','hlrf','fmna','fast_minimum_norm_attack','fmna_l2']

def GaussianIS(x_clean,gen,h,N:int=int(1e4),batch_size:int=int(1e3),save_rare:bool=False,verbose=0.,track_X:bool=False,
               nb_calls_mpp=0,sigma_bias=1.,G=None,gradG=None,  t_transform=None,y_clean=None,save_mpp=False,
               model=None,search_method='mpp_search',u_mpp=None,save_weights=True, epsilon=0.1,
               eps_real_mpp=1e-2,real_mpp=True, steps=100,stepsize=1e-2,gamma=0.05,
               alpha_CI=0.05, num_iter=32,stop_eps=1E-2,stop_cond_type='beta',
               random_init=False, sigma_init=0.1,**kwargs):
    """ Gaussian importance sampling algorithm to compute probability of failure

    Args:
        gen (_type_): random generator
        h (_type_): score function
        N (int, optional): number of samples to use. Defaults to int(1e4).
        batch_size (int, optional): batch size. Defaults to int(1e2).
        save_rare (bool, optional): Option to track adversarial examples. Defaults to False.
        verbose (float, optional): Level of verbosity. Defaults to False.


    Returns:
        float: probability of failure
    """
    d=x_clean.numel()
    zero_latent = torch.zeros((1,d),device=x_clean.device)
   
    if u_mpp is None:
        
        if search_method not in search_methods_list:
            raise NotImplementedError(f"Method {search_method} is not implemented.")
        if search_method=='mpp_search':
            assert gradG is not None, "gradG must be provided for mpp_search"
            debug=verbose>=1
            u_mpp,dict_out=mpp_search(zero_latent=zero_latent, 
                        grad_f= gradG,stop_cond_type=stop_cond_type,
                        num_iter=num_iter,stop_eps=stop_eps,debug=debug,) 
            nb_calls_mpp = dict_out['nb_calls']
        elif search_method=='hlrf':
            assert gradG is not None, "gradG must be provided for mpp_search"
            debug=verbose>=1
            u_mpp,dict_out=hlrf(zero_latent=zero_latent, 
                        grad_f= gradG,stop_cond_type=stop_cond_type,step_size=stepsize,
                        num_iter=num_iter,stop_eps=stop_eps,debug=debug,)
            nb_calls_mpp = dict_out['nb_calls']
        elif search_method=='gradient_binary_search':
            assert gradG is not None, "gradG must be provided for gradient_binary_search"
            assert G is not None, "G must be provided for gradient_binary_search"
            u_mpp=gradient_binary_search(zero_latent=zero_latent,gradG=gradG, G=G)
        elif search_method.lower() in ['carlini','carlini-wagner','cw','carlini_wagner','carlini_wagner_l2','adv','adv_attack']:
            assert (model is not None), "model must be provided for Carlini-Wagner attack"
            assert t_transform is not None, "t_transform must be provided for Carlini-Wagner attack"
            u_mpp= gaussian_space_attack(x_clean=x_clean,y_clean=y_clean,model=model,noise_dist='uniform',
                                        t_transform=t_transform,num_iter=num_iter,
                      attack = search_method,steps=steps, stepsize=stepsize, max_dist=None, epsilon=epsilon,
                        sigma=1.,random_init=random_init , sigma_init=sigma_init,**kwargs)
        elif search_method.lower() in ['brendel','brendel-bethge','brendel_bethge']:
            assert t_transform is not None, "t_transform must be provided for Brendel-Bethge attack"
            assert (model is not None), "model must be provided for Brendel-Bethge attack"
            u_mpp= gaussian_space_attack(x_clean=x_clean,y_clean=y_clean,model=model,noise_dist='uniform',
                                         t_transform=t_transform, num_iter=num_iter,
                        attack = search_method,steps=steps, stepsize=stepsize, max_dist=None, epsilon=epsilon,
                            sigma=1., random_init=random_init , sigma_init=sigma_init,**kwargs)
        
        elif search_method.lower() in ['fmna','fast_minimum_norm_attack','fmna_l2']:
            assert t_transform is not None, "t_transform must be provided for FMNA attack"
            assert (model is not None), "model must be provided for FMNA attack"
            u_mpp= gaussian_space_attack(x_clean=x_clean,y_clean=y_clean,model=model,noise_dist='uniform',
                                         t_transform=t_transform, 
                        attack = search_method,num_iter=num_iter,steps=steps, stepsize=gamma, max_dist=None, epsilon=epsilon,
                            sigma=1., random_init=random_init , sigma_init=sigma_init,**kwargs)
            
        
    else:
        assert nb_calls_mpp>0, "nb_calls_mpp must be provided if u_mpp is provided" 
    if batch_size>N:
        batch_size=N
    if real_mpp:    
        lambda_,calls = binary_search_to_zero(G=G,x=u_mpp,eps=eps_real_mpp)   
        nb_calls_mpp+=calls
        u_mpp = lambda_*u_mpp
    
    
    u_mpp = u_mpp.reshape((1,d))
    beta_HL = torch.norm(u_mpp,dim=-1)
    if verbose>=1:
        print(f"beta_HL: {beta_HL}")
        
        
    gen_bias = lambda n: u_mpp+sigma_bias*gen(n)
    p_f = 0
    pre_var = 0
    n=0
    u_rares=[]
    if save_weights:
        weights=[]
    for _ in range(N//batch_size):
        u_mc = gen_bias(batch_size)
        with torch.no_grad():
            rare_event= h(u_mc)>=0
        if save_rare:
            u_rares.append(u_mc[rare_event])
        
        n+= batch_size
        gauss_weights=  GaussianImportanceWeight(x=u_mc,mu_1=zero_latent,mu_2=u_mpp,sigma_2=sigma_bias,d=d)
        p_local = (rare_event)*gauss_weights
        pre_var_local = (rare_event)*gauss_weights**2
        del u_mc,rare_event
        if save_weights:
            weights.append(p_local)
        p_f = ((n-batch_size)/n)*p_f+(batch_size/n)*p_local.float().mean()
        pre_var = ((n-batch_size)/n)*pre_var+(batch_size/n)*pre_var_local.float().mean()
        
        
    if N%batch_size!=0:
        rest = N%batch_size
        u_mc = gen_bias(rest)
        with torch.no_grad():
            rare_event= h(u_mc)>=0
        if save_rare:
            u_rares.append(u_mc[rare_event])
        gauss_weights=  GaussianImportanceWeight(x=u_mc,mu_1=zero_latent,mu_2=u_mpp,sigma_2=sigma_bias,d=d)
        p_local = (rare_event)*gauss_weights
        pre_var_local = (rare_event)*gauss_weights**2
        if save_weights:
            weights.append(p_local)
        del u_mc,rare_event
        n+= rest
        
        p_f = ((N-rest)/N)*p_f+(rest/N)*p_local.float().mean()
        pre_var = ((N-rest)/N)*pre_var+(rest/N)*pre_var_local.float().mean()
    
    dict_out = {'nb_calls':N+nb_calls_mpp}
    var_est = (1/N)*(pre_var-p_f**2).item()
    dict_out['var_est']=var_est
    dict_out['std_est']=np.sqrt(var_est)
    if save_weights:
        dict_out['weights']=torch.cat(weights,dim=0).to('cpu').numpy()
    if save_mpp:
        dict_out['mpp']=u_mpp.to('cpu').numpy()
    CI = stats.norm.interval(1-alpha_CI,loc=p_f.item(),scale=np.sqrt(var_est))
    dict_out['CI']=CI
    if save_rare:
        dict_out['u_rare']=torch.cat(u_rares,dim=0).to('cpu').numpy()
    return p_f.cpu().item(),dict_out


def gaussian_space_attack(x_clean,y_clean,model,noise_dist='uniform',
                      attack = 'Carlini',num_iter=50,steps=100, stepsize=1e-2, max_dist=None, epsilon=0.1, t_transform=None,
                        sigma=1.,x_min=-int(1e2),x_max=int(1e2), random_init=False , sigma_init=0.5,real_uniform=False,**kwargs):
    """ Performs an attack on the latent space of the model."""
    device= x_clean.device
    if max_dist is None:
        max_dist = sqrt(x_clean.numel())*sigma
    if attack.lower() in ('carlini','cw','carlini-wagner','carliniwagner','carlini_wagner','carlini_wagner_l2'):
        attack = 'Carlini'
        assert (x_clean is not None) and (y_clean is not None), "x_clean and y_clean must be provided for Carlini-Wagner attack"
        assert (model is not None), "model must be provided for Carlini-Wagner attack"
        import foolbox as fb
        attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=num_iter, 
                                          stepsize=stepsize,
                                        
                                          steps=steps,)
    elif attack.lower() in ('brendel','brendel-bethge','brendel_bethge'):
        attack = 'Brendel'
        import foolbox as fb
        attack = fb.attacks.L2BrendelBethgeAttack(binary_search_steps=num_iter,
        lr=stepsize, steps=steps,)
    elif attack.lower() in ('fmna','fast_minimum_norm_attack','fmna_l2','fmna2'):
        attack='FMNA'
        import foolbox as fb
        attack = fb.attacks.L2FMNAttack(binary_search_steps=num_iter,steps=steps)
    else:
        raise NotImplementedError(f"Search method '{attack}' is not implemented.")
    
    if noise_dist.lower() in ('gaussian','normal'):
        
        fmodel=fb.models.PyTorchModel(model, bounds=(x_min, x_max),device=device)
        if not random_init:
            x_0 = x_clean
        else:
            print(f"Random init with sigma_init={sigma_init}")
            x_0 = x_clean + sigma_init*torch.randn_like(x_clean)
        
        _,advs,success= attack(fmodel, x_0.unsqueeze(), y_clean.unsqueeze(0), epsilons=[max_dist])
        assert success.item(), "The attack failed. Try to increase the number of iterations or steps."
        design_point= advs[0]-x_clean
        del advs
        
    elif noise_dist.lower() in ('uniform','unif'):
        if t_transform is None:
            if not real_uniform:
                t_transform = NormalCDFLayer(device=device, offset=x_clean,epsilon =epsilon)
            else:
                t_transform = NormalToUnifLayer(device=device, x_clean=x_clean,epsilon =epsilon)
        fake_bounds=(x_min,x_max)
        total_model = torch.nn.Sequential(t_transform,
                                          model)
        if not random_init:
            x_0 = torch.zeros_like(x_clean)
        else:
            print(f"Random init with sigma_init={sigma_init}")
            x_0 = sigma_init*torch.randn_like(x_clean)
        total_model.eval()
        fmodel=fb.models.PyTorchModel(total_model, bounds=fake_bounds,
                device=device, )
        _,advs,success= attack(fmodel,x_0.unsqueeze(0) , y_clean.unsqueeze(0), 
                               epsilons=[max_dist])
        design_point= advs[0]
        del advs 
    dict_out={'nb_calls':1}
    return design_point  


def gradient_binary_search(zero_latent,gradG, G,alpha=0.5,num_iter=20):
    """ Binary search algorithm to find the failure point in the direction of the gradient of the limit state function.

    Args:
        zero_latent (torch.tensor): zero latent point
        gradG (callable): gradient of the limit state function
        num_iter (int, optional): number of iterations. Defaults to 20.

    Returns:
        torch.tensor: failure point
    """
    x= zero_latent
    gradG_x,G_x = gradG(x)
    while G_x>0:
        x= x+alpha*gradG_x
        gradG_x,G_x = gradG(x)
    a=zero_latent
    b = x 
    for _ in range(num_iter):
        c = (a+b)/2
        _,G_c = gradG(c)
        if G_c>0:
            b=c
        else:
            a=c
    
    return b