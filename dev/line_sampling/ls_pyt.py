import torch
import scipy.stats as stats
import numpy as np
from math import sqrt
from stat_reliability_measure.dev.torch_utils import NormalCDFLayer
from stat_reliability_measure.dev.imp_sampling.is_pyt import mpp_search, search_methods_list,hlrf, gaussian_space_attack,  binary_search_to_zero,gradient_binary_search
from stat_reliability_measure.dev.torch_utils import normal_dist


def batch_dichotomic_search(batch_G,beta_min=0,beta_max=4,eps=1e-5,nb_iter=20,batch_size=100,device='cpu'):
    """ Batch dichotomic search algorithm to find the failure point in the direction of the gradient of the limit state function.

    Args:
        g (callable): limit state function
        a_min (torch.tensor): lower bound
        a_max (torch.tensor): upper bound
        eps (float, optional): tolerance. Defaults to 1e-3.

    Returns:
        torch.tensor: failure point
    """
  
    a = torch.ones((batch_size,)).to(device)*beta_min
    b = torch.ones((batch_size,)).to(device)*beta_max
    G_max = batch_G(b) 
    count_calls =batch_size
    while G_max.max().item()>0 and b.median().item()<1e2:
        b = torch.where(batch_G(b)>0,2*b,b)
        G_max = batch_G(b)
    k=0
    while torch.max(b-a)>eps and k < nb_iter:
        k+=1
        c = (a+b)/2
        
        a = torch.where(batch_G(c)>0,c,a)
        b = torch.where(batch_G(c)>0,b,c)
        count_calls+=batch_size
    return b,count_calls

def gen_hyperplane(n,alpha_mpp,gen):
        x_n = gen(n)
        return x_n-torch.sum(alpha_mpp*x_n,dim=1)[:,None]*alpha_mpp[0][None,:]



def LineSampling(x_clean,gen,G,N:int=int(1e4),epsilon=None,y_clean=None,batch_size:int=int(1e3),save_rare:bool=False,verbose=0.,track_X:bool=False,
               nb_calls_mpp=0,h=None,gradG=None,t_transform=None,num_iter=20,stepsize=0.01,sigma_init=0.5,steps=100,random_init=False,
               model=None,search_method='mpp_search',u_mpp=None,stop_cond_type='beta',stop_eps=1e-3,save_mpp=False,default_params=True,
            num_classes=10,gamma=0.3,check_mpp=False,
               alpha_CI=0.05, real_mpp=False, eps_real_mpp=1e-3,**kwargs):
    """ Line sampling algorithm to compute probability of failure, using an importance direction alpha,
    and a line search in this direction to find the failure point for gaussian samples on the orthogonal space.
    """
    d = x_clean.numel()
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
            nb_calls=dict_out['nb_calls']
        if search_method=='hlrf':
            assert gradG is not None, "gradG must be provided for mpp_search"
            debug=verbose>=1
            u_mpp,dict_out=hlrf(zero_latent=zero_latent, 
                        grad_f= gradG,stop_cond_type=stop_cond_type,step_size=0.95,
                        num_iter=num_iter,stop_eps=stop_eps,debug=debug,)
            nb_calls=dict_out['nb_calls']
        elif search_method=='gradient_binary_search':
            assert gradG is not None, "gradG must be provided for gradient_binary_search"
            assert G is not None, "G must be provided for gradient_binary_search"
            u_mpp=gradient_binary_search(zero_latent=zero_latent,gradG=gradG, G=G)
        elif search_method.lower() in ['carlini','carlini-wagner','cw','carlini_wagner','carlini_wagner_l2','adv','adv_attack']:
            assert (model is not None), "model must be provided for Carlini-Wagner attack"
            assert t_transform is not None, "t_transform must be provided for Carlini-Wagner attack"
            u_mpp= gaussian_space_attack(x_clean=x_clean,y_clean=y_clean,model=model,noise_dist='uniform',
                                        t_transform=t_transform,num_iter=num_iter,default_params=default_params,
                      attack = search_method,steps=steps, stepsize=stepsize, max_dist=None, epsilon=epsilon,
                        sigma=1.,random_init=random_init , sigma_init=sigma_init,**kwargs)
        elif search_method.lower() in ['brendel','brendel-bethge','brendel_bethge']:
            assert t_transform is not None, "t_transform must be provided for Brendel-Bethge attack"
            assert (model is not None), "model must be provided for Brendel-Bethge attack"
            u_mpp= gaussian_space_attack(x_clean=x_clean,y_clean=y_clean,model=model,noise_dist='uniform',
                                         t_transform=t_transform, num_iter=num_iter,default_params=default_params,
                        attack = search_method,steps=steps, stepsize=stepsize, max_dist=None, epsilon=epsilon,
                            sigma=1., random_init=random_init , sigma_init=sigma_init,**kwargs)
        
        elif search_method.lower() in ['fmna','fast_minimum_norm_attack','fmna_l2']:
            assert t_transform is not None, "t_transform must be provided for FMNA attack"
            assert (model is not None), "model must be provided for FMNA attack"
            u_mpp= gaussian_space_attack(x_clean=x_clean,y_clean=y_clean,model=model,noise_dist='uniform',
                                         t_transform=t_transform, default_params=default_params,
                        attack = search_method,num_iter=num_iter,steps=steps, stepsize=gamma, max_dist=None, epsilon=epsilon,
                            sigma=1., random_init=random_init , sigma_init=sigma_init,**kwargs)
        elif search_method.lower() in ('bp','boundary','boundary_projection'):
            assert (model is not None), "model must be provided for BP attack"
            u_mpp= gaussian_space_attack(x_clean=x_clean,y_clean=y_clean,model=model,noise_dist='uniform',t_transform=t_transform,
                                         default_params=default_params,
                        attack = search_method,steps=steps, stepsize=gamma, max_dist=None, epsilon=epsilon,num_classes=num_classes,
                            sigma=1., random_init=random_init , sigma_init=sigma_init,**kwargs)
        elif search_method.lower() in ('ihl_rf','ihlrf','hlrf'):
            assert (model is not None), "model must be provided for iHLRF attack"
            u_mpp= gaussian_space_attack(x_clean=x_clean,y_clean=y_clean,model=model,noise_dist='uniform',t_transform=t_transform,
                        attack = search_method,steps=steps, stepsize=gamma, max_dist=None, epsilon=epsilon,default_params=default_params,
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
    alpha_mpp = u_mpp/torch.norm(u_mpp,dim=-1)
    beta_HL = torch.norm(u_mpp,dim=-1).item()
  
    
    if verbose>=1:
        print(f"beta_HL: {beta_HL}")
    def gen_hyperplane(n,alpha_mpp,gen):
        x_n = gen(n)
   
        return x_n-torch.sum(alpha_mpp*x_n,dim=1)[:,None]*alpha_mpp[0][None,:]
    p_f = 0
    pre_var = 0
    n=0
    u_rares=[]
    nb_calls = 0
    for _ in range(N//batch_size):
        n+=batch_size
        u_ls = gen_hyperplane(batch_size,alpha_mpp,gen)
        batch_G = lambda betas: G(u_ls+betas[:,None]*alpha_mpp)
        betas_mc,count_calls = batch_dichotomic_search(batch_G,beta_min=0,beta_max=10 * beta_HL,eps=1e-5,nb_iter=25,device=x_clean.device,
                                           batch_size=batch_size)
        nb_calls+=count_calls
        pfs = normal_dist.cdf(-betas_mc)
        p_f_local = pfs.mean()
        pre_var_local = pfs.var()
        
        p_f = ((n-batch_size)/n)*p_f+(batch_size/n)*p_f_local
        pre_var = ((n-batch_size)/n)*pre_var+(batch_size/n)*pre_var_local
        
        
    if N%batch_size!=0:
        rest = N%batch_size
        n+=rest
        u_ls = gen_hyperplane(rest)
        batch_G = lambda betas: G(u_ls+betas*alpha_mpp)
        betas_mc,count_calls = batch_dichotomic_search(batch_G,beta_min=0,beta_max=10 * beta_HL,eps=1e-5,nb_iter=25)
        pfs = normal_dist.cdf(-betas_mc)
        p_f_local = pfs.mean()
        pre_var_local = pfs.var()
        nb_calls+=count_calls
        p_f = ((n-rest)/n)*p_f+(rest/n)*p_f_local
        pre_var = ((n-rest)/n)*pre_var+(rest/n)*pre_var_local
        
    dict_out = {'nb_calls': (nb_calls+nb_calls_mpp)}
    var_est = (1/N)*(pre_var-p_f**2).item()
    dict_out['var_est']=var_est
    dict_out['std_est']=np.sqrt(var_est)
    
   
    dict_out['mpp']=u_mpp.to('cpu').numpy()
    CI = stats.norm.interval(1-alpha_CI,loc=p_f.item(),scale=np.sqrt(var_est))
    dict_out['CI']=CI
    if save_rare:
        dict_out['u_rare']=None
    return p_f.cpu().item(),dict_out


# def gaussian_space_attack(x_clean,y_clean,model,noise_dist='uniform',
#                       attack = 'Carlini',num_iter=50,steps=100, stepsize=1e-2, max_dist=None, epsilon=0.1, t_transform=None,
#                         sigma=1.,x_min=-int(1e2),x_max=int(1e2), random_init=False , sigma_init=0.5,real_uniform=False,**kwargs):
#     """ Performs an attack on the latent space of the model."""
#     device= x_clean.device
#     if max_dist is None:
#         max_dist = sqrt(x_clean.numel())*sigma
#     if attack.lower() in ('carlini','cw','carlini-wagner','carliniwagner','carlini_wagner','carlini_wagner_l2'):
#         attack = 'Carlini'
#         assert (x_clean is not None) and (y_clean is not None), "x_clean and y_clean must be provided for Carlini-Wagner attack"
#         assert (model is not None), "model must be provided for Carlini-Wagner attack"
#         import foolbox as fb
#         attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=num_iter, 
#                                           stepsize=stepsize,
                                        
#                                           steps=steps,)
#     elif attack.lower() in ('brendel','brendel-bethge','brendel_bethge'):
#         attack = 'Brendel'
#         import foolbox as fb
#         attack = fb.attacks.L2BrendelBethgeAttack(binary_search_steps=num_iter,
#         lr=stepsize, steps=steps,)
#     elif attack.lower() in ('fmna','fast_minimum_norm_attack','fmna_l2','fmna2'):
#         attack='FMNA'
#         import foolbox as fb
#         attack = fb.attacks.L2FMNAttack(binary_search_steps=num_iter,steps=steps)
#     else:
#         raise NotImplementedError(f"Search method '{attack}' is not implemented.")
    
#     if noise_dist.lower() in ('gaussian','normal'):
        
#         fmodel=fb.models.PyTorchModel(model, bounds=(x_min, x_max),device=device)
#         if not random_init:
#             x_0 = x_clean
#         else:
#             print(f"Random init with sigma_init={sigma_init}")
#             x_0 = x_clean + sigma_init*torch.randn_like(x_clean)
        
#         _,advs,success= attack(fmodel, x_0.unsqueeze(), y_clean.unsqueeze(0), epsilons=[max_dist])
#         assert success.item(), "The attack failed. Try to increase the number of iterations or steps."
#         design_point= advs[0]-x_clean
#         del advs
        
#     elif noise_dist.lower() in ('uniform','unif'):
#         if t_transform is None:
#             if not real_uniform:
#                 t_transform = NormalCDFLayer(device=device, offset=x_clean,epsilon =epsilon)
#             else:
#                 t_transform = NormalToUnifLayer(device=device, x_clean=x_clean,epsilon =epsilon)
#         fake_bounds=(x_min,x_max)
#         total_model = torch.nn.Sequential(t_transform,
#                                           model)
#         if not random_init:
#             x_0 = torch.zeros_like(x_clean)
#         else:
#             print(f"Random init with sigma_init={sigma_init}")
#             x_0 = sigma_init*torch.randn_like(x_clean)
#         total_model.eval()
#         fmodel=fb.models.PyTorchModel(total_model, bounds=fake_bounds,
#                 device=device, )
#         _,advs,success= attack(fmodel,x_0.unsqueeze(0) , y_clean.unsqueeze(0), 
#                                epsilons=[max_dist])
#         design_point= advs[0]
#         del advs 

#     return design_point   


# def gradient_binary_search(zero_latent,gradG, G,alpha=0.5,num_iter=20):
#     """ Binary search algorithm to find the failure point in the direction of the gradient of the limit state function.

#     Args:
#         zero_latent (torch.tensor): zero latent point
#         gradG (callable): gradient of the limit state function
#         num_iter (int, optional): number of iterations. Defaults to 20.

#     Returns:
#         torch.tensor: failure point
#     """
#     x= zero_latent
#     gradG_x,G_x = gradG(x)
#     while G_x>0:
#         x= x+alpha*gradG_x
#         gradG_x,G_x = gradG(x)
#     a=zero_latent
#     b = x 
#     for _ in range(num_iter):
#         c = (a+b)/2
#         _,G_c = gradG(c)
#         if G_c>0:
#             b=c
#         else:
#             a=c
    
#     return b