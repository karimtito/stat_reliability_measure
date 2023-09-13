import scipy.stats as stats 
from stat_reliability_measure.dev.torch_utils import NormalCDFLayer
from math import sqrt
import torch

def FORM_pyt(x_clean,y_clean,model,noise_dist='uniform',
             search_method='Carlini',
             num_iter=10,steps=100, stepsize=1e-2, max_dist=None, epsilon=0.1,
              sigma=1.,x_min=0.,x_max=1., random_init=False , sigma_init=0.5,**kwargs):

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
    device= x_clean.device
    if max_dist is None:
        max_dist = sqrt(x_clean.numel())*sigma # maximum distance between the clean input and the adversarial example
    if search_method.lower() in ('carlini','cw','carlini-wagner','carliniwagner','carlini_wagner','carlini_wagner_l2'):
        assert (x_clean is not None) and (y_clean is not None), "x_clean and y_clean must be provided for Carlini-Wagner attack"
        assert (model is not None), "model must be provided for Carlini-Wagner attack"
        import foolbox as fb
        attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=num_iter, 
                                          stepsize=stepsize,
                                        
                                          steps=steps,)
    elif search_method.lower() in ('brendel','brendel-bethge','brendel_bethge'):
        import foolbox as fb
        attack = fb.attacks.L2BrendelBethgeAttack(binary_search_steps=num_iter,
        lr=stepsize, steps=steps,)
    else:
        raise NotImplementedError(f"Search method '{search_method}' is not implemented.")
        
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
        normal_cdf_layer = NormalCDFLayer(device=device, offset=x_clean,epsilon =epsilon)
        fake_bounds=(-10.,10.)
        total_model = torch.nn.Sequential(normal_cdf_layer, model)
        if not random_init:
            x_0 = torch.zeros_like(x_clean)
        else:
            print(f"Random init with sigma_init={sigma_init}")
            x_0 = sigma_init*torch.randn_like(x_clean)
        total_model.eval()
        fmodel=fb.models.PyTorchModel(total_model, bounds=fake_bounds,
                device=device, )
        _,advs,success= attack(fmodel,x_0[:1] , y_clean.unsqueeze(0), epsilons=[max_dist])
        design_point= advs[0]
        del advs
    else:
        raise NotImplementedError(f"Noise distribution '{noise_dist}' is not implemented.")
    
            



    l2dist= design_point.norm(p=2)
    p_fail= stats.norm.cdf(-l2dist.cpu()/sigma)
    nb_calls= num_iter*steps
    dict_out={'nb_calls':nb_calls}
    return p_fail,dict_out


#gradient descent in 1 dimension to find zero of a function f
def find_zero_gd_pyt(f, grad_f, x0, obj='min', stepsize=1e-2, max_iter=100, tol=1e-3,random_init=False):
    x = x0
    
    if random_init:
        x = x0 + stepsize*torch.randn_like(x0)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    f_calls = 0
    sign= 1 if obj=='max' else -1
    for i in range(max_iter):
        x = x + sign*stepsize * grad_f(x)
        f_calls += 2 # we count 2 calls for each gradient evaluation (forward and backward)
        if sign*f(x) > -tol:
            f_calls += 1 # we count one more call for the last function evaluation
            break
    if i == max_iter-1:
        print('Warning: max_iter reached in find_zero_gd')
    return x, f_calls

def mpp_search(f, grad_f, x_clean,max_iter=100,stop_cond_type='grad_norm',stop_eps=1e-3):
    """ Search algorithm for the Most Probable Point (MPP) 
        according to the course 'Probabilistc Engineering Design' from University of Missouri  """
    x=x_clean 
    grad_fx = grad_f(x)
    f_calls+=2
    beta=torch.norm(x)
    k= 0
    stop_cond=False
    while k<max_iter & ~stop_cond: 
        k+=1
        a = grad_fx/torch.norm(grad_fx)
        beta_new = beta + f(x)/torch.norm(grad_fx)
        f_calls+=1
        x_new=-a*beta
        grad_f_xnew = grad_f(x_new)
        f_calls+=2
        if stop_cond_type not in ['grad_norm']:
            raise NotImplementedError(f"Method {stop_cond_type} is not implemented.")
        if stop_cond_type=='grad_norm':
            stop_cond = torch.norm(grad_fx-grad_f_xnew)<stop_eps
        beta=beta_new
        x=x_new
        grad_fx=grad_f_xnew
    if k==max_iter:
        print("Warning: maximum number of iteration has been reached")
    return x, f_calls

        
            