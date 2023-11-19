import torch
import scipy.stats as stats
import numpy as np
from math import sqrt
from stat_reliability_measure.dev.torch_utils import NormalCDFLayer, NormalToUnifLayer

def GaussianImportanceWeight(x,mu_1,mu_2,sigma_1=1.,sigma_2=1.):
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
    density_ratio = torch.exp(-0.5*((x-mu_1)/sigma_1).square().sum(-1)+0.5*((x-mu_2)/sigma_2).square().sum(-1)) * (sigma_2/sigma_1)
    return density_ratio




def mpp_search(grad_f, zero_latent,max_iter=100,stop_cond_type='grad_norm',
               stop_eps=1e-3,
            mult_grad_calls=2.,debug=False,print_every=10):
    """ Search algorithm for the Most Probable Point (X_mpp) with a Newton method.  """
    x= zero_latent
    grad_fx,f_x = grad_f(x)
    grad_calls=1
    beta=torch.norm(x,dim=-1)
    k= 0
    stop_cond=False
    print_count=0
    debug_ = debug
    
    while k<max_iter and  (~stop_cond or f_x>0): 
        k+=1
        a = grad_fx/torch.norm(grad_fx,dim=-1)
        beta_new = beta + f_x/torch.norm(grad_fx,dim=-1)
        x_new=-a*beta_new
        grad_f_xnew,f_x = grad_f(x_new)
        grad_calls+=1 
        if debug_:
            debug = print_count%print_every==0
        if stop_cond_type not in ['grad_norm','norm','beta']:
            raise NotImplementedError(f"Method {stop_cond_type} is not implemented.")
        if stop_cond_type=='grad_norm':
            grad_norm_diff = torch.norm(grad_fx-grad_f_xnew,dim=-1)
            if debug:
                print(f"grad_norm_diff: {grad_norm_diff}")
            stop_cond = (grad_norm_diff<stop_eps).item()
        elif stop_cond_type=='norm':
            norm_diff = torch.norm(x-x_new,dim=-1)
            if debug:
                print(f"norm_diff: {norm_diff}")
            stop_cond = (norm_diff<stop_eps).item()
        elif stop_cond_type=='beta':
            beta_diff = torch.abs(beta-beta_new)
            if debug:
                print(f"beta_diff: {beta_diff}")
            stop_cond = (beta_diff<stop_eps).item()
        
        beta=beta_new
        if debug:
            print(f'beta: {beta}')
        if debug_:
            print_count+=1
        x=x_new
        grad_fx=grad_f_xnew
    if k==max_iter and debug_:
        print("Warning: maximum number of iteration has been reached")
    nb_calls= mult_grad_calls*grad_calls
    return x, nb_calls




search_methods= {'mpp_search':mpp_search ,'gradient_binary_search':'gradient_binary_search',}
search_methods_list=['mpp_search','gradient_binary_search',
                     'carlini','carlini-wagner','cw','carlini_wagner','carlini_wagner_l2','brendel','brendel-bethge','brendel_bethge',
                     'adv','adv_attack','hlrf']
def GaussianIS(x_clean,gen,h,N:int=int(1e4),batch_size:int=int(1e3),track_advs:bool=False,verbose=0.,track_X:bool=False,
               nb_calls_mpp=0,sigma_bias=1.,G=None,gradG=None, real_uniform=False, normal_cdf_layer=None,
               model=None,search_method='mpp_search',X_mpp=None,save_weights=False, epsilon=0.1,
               alpha_CI=0.05, max_iter=15,random_init=False,**kwargs):
    """ Gaussian importance sampling algorithm to compute probability of failure

    Args:
        gen (_type_): random generator
        h (_type_): score function
        N (int, optional): number of samples to use. Defaults to int(1e4).
        batch_size (int, optional): batch size. Defaults to int(1e2).
        track_advs (bool, optional): Option to track adversarial examples. Defaults to False.
        verbose (float, optional): Level of verbosity. Defaults to False.


    Returns:
        float: probability of failure
    """
    d=x_clean.numel()
    zero_latent = torch.zeros((1,d),device=x_clean.device)
    if X_mpp is None:
        if search_method not in search_methods.keys():
            raise NotImplementedError(f"Method {search_method} is not implemented.")
        if search_method=='mpp_search':
            assert gradG is not None, "gradG must be provided for mpp_search"
            debug=verbose>=1
            X_mpp,nb_calls_mpp=mpp_search(zero_latent=zero_latent, 
                        grad_f= gradG,stop_cond_type='beta',
                        max_iter=max_iter,stop_eps=1E-2,debug=debug,**kwargs) 
        elif search_method=='gradient_binary_search':
            assert gradG is not None, "gradG must be provided for gradient_binary_search"
            assert G is not None, "G must be provided for gradient_binary_search"
            X_mpp=gradient_binary_search(zero_latent=zero_latent,gradG=gradG, G=G)
        elif search_method.lower() in ['carlini','carlini-wagner','cw','carlini_wagner','carlini_wagner_l2','adv','adv_attack']:
            assert (model is not None), "model must be provided for Carlini-Wagner attack"
            assert normal_cdf_layer is not None, "normal_cdf_layer must be provided for Carlini-Wagner attack"
            X_mpp= gaussian_space_attack(x_clean=x_clean,y_clean=None,model=model,noise_dist='uniform',
                      attack = search_method,num_iter=10,steps=100, stepsize=1e-2, max_dist=None, epsilon=epsilon,
                        sigma=1.,random_init=False , sigma_init=0.5,**kwargs)
        elif search_method.lower() in ['brendel','brendel-bethge','brendel_bethge']:
            assert normal_cdf_layer is not None, "normal_cdf_layer must be provided for Brendel-Bethge attack"
            assert (model is not None), "model must be provided for Brendel-Bethge attack"
            X_mpp= gaussian_space_attack(x_clean=x_clean,y_clean=None,model=model,noise_dist='uniform',
                        attack = search_method,num_iter=10,steps=100, stepsize=1e-2, max_dist=None, epsilon=epsilon,
                            sigma=1., random_init=False , sigma_init=0.5,**kwargs)
            
        
    else:
        assert nb_calls_mpp>0, "nb_calls_mpp must be provided if X_mpp is provided" 
    if batch_size>N:
        batch_size=N
    d = x_clean.shape[-1]
    gen_bias = lambda n: X_mpp+sigma_bias*gen(n)
    p_f = 0
    pre_var = 0
    n=0
    x_advs=[]
    if save_weights:
        weights=[]
    for _ in range(N//batch_size):
        x_mc = gen_bias(batch_size)
        with torch.no_grad():
            rare_event= h(x_mc)>=0
        if track_advs:
            x_advs.append(x_mc[rare_event])
        
        n+= batch_size
        gauss_weights=  GaussianImportanceWeight(x=x_mc,mu_1=zero_latent,mu_2=X_mpp,sigma_2=sigma_bias)
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
        if track_advs:
            x_advs.append(x_mc[rare_event])
        gauss_weights=  GaussianImportanceWeight(x=x_mc,mu_1=zero_latent,mu_2=X_mpp,sigma_2=sigma_bias)
        p_local = (rare_event)*gauss_weights
        pre_var_local = (rare_event)*gauss_weights**2
        if save_weights:
            weights.append(p_local)
        del x_mc,rare_event
        n+= rest
        
        p_f = ((N-rest)/N)*p_f+(rest/N)*p_local.float().mean()
        pre_var = ((N-rest)/N)*pre_var+(rest/N)*pre_var_local.float().mean()
    dict_out = {'nb_calls':N+nb_calls_mpp}
    var_est = (1/N)*(pre_var-p_f**2).item()
    dict_out['var_est']=var_est
    dict_out['std_est']=np.sqrt(var_est)
    dict_out['weights']=torch.cat(weights,dim=0).to('cpu').numpy()
    CI = stats.norm.interval(1-alpha_CI,loc=p_f.item(),scale=np.sqrt(var_est))
    dict_out['CI']=CI
    if track_advs:
        dict_out['advs']=torch.cat(x_advs,dim=0).to('cpu').numpy()
    return p_f.cpu(),dict_out


def gaussian_space_attack(x_clean,y_clean,model,noise_dist='uniform',
                      attack = 'Carlini',num_iter=10,steps=100, stepsize=1e-2, max_dist=None, epsilon=0.1, normal_cdf_layer=None,
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
    else:
        raise NotImplementedError(f"Search method '{attack}' is not implemented.")
    
    if noise_dist.lower() in ('gaussian','normal'):
        
        fmodel=fb.models.PyTorchModel(model, bounds=(x_min, x_max),device=device)
        if not random_init:
            x_0 = x_clean
        else:
            print(f"Random init with sigma_init={sigma_init}")
            x_0 = x_clean + sigma_init*torch.randn_like(x_clean)
        
        _,advs,success= attack(fmodel, x_0[:1], y_clean.unsqueeze(0), epsilons=[max_dist])
        assert success.item(), "The attack failed. Try to increase the number of iterations or steps."
        design_point= advs[0]-x_clean
        del advs
        
    elif noise_dist.lower() in ('uniform','unif'):
        if normal_cdf_layer is None:
            if not real_uniform:
                normal_cdf_layer = NormalCDFLayer(device=device, offset=x_clean,epsilon =epsilon)
            else:
                normal_cdf_layer = NormalToUnifLayer(device=device, offset=x_clean,epsilon =epsilon)
        fake_bounds=(x_min,x_max)
        total_model = torch.nn.Sequential(normal_cdf_layer,
                                          model)
        if not random_init:
            x_0 = torch.zeros_like(x_clean)
        else:
            print(f"Random init with sigma_init={sigma_init}")
            x_0 = sigma_init*torch.randn_like(x_clean)
        total_model.eval()
        fmodel=fb.models.PyTorchModel(total_model, bounds=fake_bounds,
                device=device, )
        _,advs,success= attack(fmodel,x_0[:1] , y_clean.unsqueeze(0), 
                               epsilons=[max_dist])
        design_point= advs[0]
        del advs 

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