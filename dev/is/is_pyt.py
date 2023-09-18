import torch

def GaussianImportanceWeight(x,mu_1,mu_2,sigma_1,sigma_2):
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
    density_ratio = torch.exp(-0.5*((x-mu_1)/sigma_1)**2+0.5*((x-mu_2)/sigma_2)**2) * (sigma_2/sigma_1)
    return density_ratio
    
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

search_methods= {'mpp_search':mpp_search }

def GaussianIS(x_clean,h,N:int=int(1e4),batch_size:int=int(1e2),track_advs:bool=False,verbose=0.,track_X:bool=False,
               sigma_bias=1.,search_method='mpp_search',**kwargs):
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
    MPP = search_methods[search_method] 
    gen_bias = torch.distributions.normal.Normal(loc=MPP, scale=sigma_bias)
    p_f = 0
    n=0
    x_advs=[]
    for _ in range(N//batch_size):
        x = gen_bias(batch_size)
        h_MC = h(x)
        if track_advs:
            x_advs.append(x_mc[h_MC>=0])
        del x_mc
        n+= batch_size
        p_f = ((n-batch_size)/n)*p_f+(batch_size/n)*(h_MC>=0).float().mean()
        del h_MC
        
    if N%batch_size!=0:
        rest = N%batch_size
        x_mc = gen_bias(rest)
        h_MC = h(x_mc)
        if track_advs:
            x_advs.append(x_mc[h_MC>=0])
        p_local = (h_MC>=0)*GaussianImportanceWeight(x=x_mc,mu_1=x_clean,mu_2=MPP)
        del x_mc
        N+= rest
        
        p_f = ((N-rest)/N)*p_f+(rest/N)*p_local.float().mean()
        del h_MC
    assert N==N
    dict_out = {'nb_calls':N}
    if track_advs:
        dict_out['advs']=torch.cat(x_advs,dim=0).to('cpu').numpy()
    
    
    return p_f.cpu(),dict_out