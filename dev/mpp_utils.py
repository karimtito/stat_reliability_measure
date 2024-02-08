import torch
from math import sqrt
from .torch_utils import NormalCDFLayer,NormalToUnifLayer

search_methods_list=['mpp_search','gradient_binary_search','carliniwagner'
                     'carlini','carlini-wagner','cw','carlini_wagner','carlini_wagner_l2','brendel','brendel-bethge','brendel_bethge','brendelbethge', 
                     'fmna','fmna_attack','fmna_attack_l2',
                     'adv','adv_attack','hlrf']


def binary_search_to_zero(G,x,lambda_min=0.,lambda_max=4.,eps=1e-3, max_iter=32,verbose=False):
    """binary search to find the zero of a function"""
    
    i=1
    if G(x)>0:
        
        a = 1
        b = lambda_max

        c = (a+b)/2
        G_c = G(c*x)
        while G_c.abs()>eps and i<max_iter:
            i+=1
            G_c = G(c*x)
            if verbose:
                print(f"c={c},G_c={G_c}")
            if G(c*x)>0:
                a = c
            else:
                b = c
            c = (a+b)/2
            G_c = G(c*x)
    else:
        a = lambda_min
        b = 1
        c = (a+b)/2
        G_c = G(c*x)
        while G_c.abs()>eps and i<max_iter:
            i+=1
            G_c = G(c*x)
            if verbose:
                print(f"c={c},G_c={G_c}")
            if G(c*x)>0:
                a = c
            else:
                b = c
            c = (a+b)/2
            G_c = G(c*x)
    if i==max_iter:
        print("Warning: maximum number of iteration has been reached")
    return (b, i)

def mpp_search_newton(grad_f, zero_latent,max_iter=100,stop_cond_type='grad_norm',
               stop_eps=1e-3,
            mult_grad_calls=2.,debug=False,print_every=10):
    """ Search algorithm for the Most Probable Point (u_mpp) with a Newton method.  """
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





def gaussian_space_attack(x_clean,y_clean,model,noise_dist='uniform',
                      attack = 'Carlini',num_iter=10,steps=100, stepsize=1e-2, max_dist=None, epsilon=0.1, t_transform=None,
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
    elif attack.lower() in ('fmna','fast_mininum_norm_attack','fmna_l2'):
        attack = 'FMNA'
        import foolbox as fb
        attack = fb.attacks.L2FMNAttack(binary_search_steps=num_iter,
        gamma=stepsize, steps=steps,)
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