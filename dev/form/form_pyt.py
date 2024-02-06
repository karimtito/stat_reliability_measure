import scipy.stats as stats 
from stat_reliability_measure.dev.torch_utils import NormalCDFLayer
from math import sqrt
import torch
from stat_reliability_measure.dev.mpp_utils import gradient_binary_search,gaussian_space_attack,mpp_search_newton, search_methods_list, binary_search_to_zero

          

def FORM_pyt(x_mpp=None,x_clean=None, gradG=None, G=None,y_clean=None,model=None,noise_dist='uniform',
             search_method='Carlini',t_transform=None,nb_calls=-1.,verbose=0,
             num_iter=10,steps=100, stepsize=1e-2, max_dist=None, epsilon=0.1,eps_real_mpp=1e-3,
              sigma=1.,x_min=0.,x_max=1., random_init=False , sigma_init=0.5, real_mpp=True,**kwargs):

    """Computes the probability of failure using the First Order Method (FORM)
    Args:
        x_clean (torch.Tensor): clean input
        y_clean (torch.Tensor): clean label
        model (torch.nn.Module): model to attack
        noise_dist (str): distribution of the noise. Can be 'uniform' or 'gaussian'
        search_method (str): search method for the most probable point. Can be 'Carlini' or 'gradient_descent'
        num_iter (int): number of iterations for the search method
        steps (int): number of steps for the search method
        stepsize (float): step size for the search method
        max_dist (float): maximum distance between the clean input and the adversarial example
        epsilon (float): maximum perturbation for the search method (if uniform)
        sigma (float): standard deviation of the noise (if Gaussian)
        x_min (float): minimum value of the input
        x_max (float): maximum value of the input
    
    Returns:
        p_fail (float): probability of failure
        dict_out (dict): dictionary containing the parameters of the attack

    """
    assert x_mpp is not None or (x_clean is not None and model is not None and y_clean is not None) 
    device= x_clean.device if (x_mpp is None) else x_mpp.device
    search_method=search_method.lower()
    if x_mpp is None:
        zero_latent=torch.zeros_like(x_clean)
        if search_method not in search_methods_list:
            raise NotImplementedError(f"Method {search_method} is not implemented.")
        if search_method=='mpp_search':
            assert gradG is not None, "gradG must be provided for mpp_search"
            debug=verbose>=1
            x_mpp,nb_calls=mpp_search_newton(zero_latent=zero_latent, 
                        grad_f= gradG,stop_cond_type='beta',
                        max_iter=num_iter,stop_eps=1E-2,debug=debug,) 
        elif search_method=='gradient_binary_search':
            assert gradG is not None, "gradG must be provided for gradient_binary_search"
            assert G is not None, "G must be provided for gradient_binary_search"
            x_mpp=gradient_binary_search(zero_latent=zero_latent,gradG=gradG, G=G)
        elif search_method.lower() in ['carlini','carlini-wagner','cw','carlini_wagner','carlini_wagner_l2','adv','adv_attack']:
            assert (model is not None), "model must be provided for Carlini-Wagner attack"
            assert t_transform is not None, "t_transform must be provided for Carlini-Wagner attack"
            x_mpp= gaussian_space_attack(x_clean=x_clean,y_clean=y_clean,model=model,noise_dist='uniform',
                                
                      attack = search_method,num_iter=10,steps=100, stepsize=1e-2, max_dist=None, epsilon=epsilon,
                        sigma=1.,random_init=False , sigma_init=0.5,**kwargs)
        elif search_method.lower() in ['brendel','brendel-bethge','brendel_bethge']:
            assert t_transform is not None, "t_transform must be provided for Brendel-Bethge attack"
            assert (model is not None), "model must be provided for Brendel-Bethge attack"
            x_mpp= gaussian_space_attack(x_clean=x_clean,y_clean=y_clean,model=model,noise_dist='uniform',
                        attack = search_method,num_iter=10,steps=100, stepsize=1e-2, max_dist=None, epsilon=epsilon,
                            sigma=1., random_init=False , sigma_init=0.5,**kwargs)
        elif search_method.lower() in ['fmna','fast_mininum_norm_attack','fmna_l2']: 
            assert (model is not None), "model must be provided for FMNA attack"
            assert t_transform is not None, "t_transform must be provided for FMNA attack"
            x_mpp= gaussian_space_attack(x_clean=x_clean,y_clean=y_clean,model=model,noise_dist='uniform',
                        attack = search_method,num_iter=10,steps=100, stepsize=1e-2, max_dist=None, epsilon=epsilon,
                            sigma=1., random_init=False , sigma_init=0.5,**kwargs)
            

    if real_mpp:
        assert G is not None, "G must be provided for real_mpp"
        x_mpp = binary_search_to_zero(G=G,x=x_mpp,eps=eps_real_mpp)
    design_point=x_mpp
    nb_calls=nb_calls
    l2dist= design_point.norm(p=2).detach().cpu().item()
    p_fail= stats.norm.cdf(-l2dist/sigma)
    
    dict_out={'nb_calls':nb_calls,'l2dist':l2dist}
    return p_fail,dict_out



            