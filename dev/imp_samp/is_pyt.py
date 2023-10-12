import torch

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
    """ Search algorithm for the Most Probable Point (X_mpp) 
        according to the course 'Probabilistc Engineering Design' from University of Missouri  """
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
    if k==max_iter:
        print("Warning: maximum number of iteration has been reached")
    nb_calls= mult_grad_calls*grad_calls
    return x, nb_calls




search_methods= {'mpp_search':mpp_search ,}

def GaussianIS(x_clean,gen,h,N:int=int(1e4),batch_size:int=int(1e2),track_advs:bool=False,verbose=0.,track_X:bool=False,
               nb_calls_mpp=0,sigma_bias=1.,G=None,gradG=None,
               model=None,search_method='mpp_search',**kwargs):
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
    if search_method not in search_methods.keys():
        raise NotImplementedError(f"Method {search_method} is not implemented.")
    if search_method=='mpp_search':
        assert gradG is not None, "gradG must be provided for mpp_search"
        d=x_clean.shape[-1]
        zero_latent = torch.zeros((1,d),device=x_clean.device)
        X_mpp,nb_calls_mpp=mpp_search(zero_latent=zero_latent, 
                    grad_f= gradG,stop_cond_type='beta',
                    max_iter=10,stop_eps=1E-2,debug=True) 
    
    if batch_size>N:
        batch_size=N
    d = x_clean.shape[-1]
    gen_bias = lambda n: X_mpp+sigma_bias*gen(n)
    p_f = 0
    pre_var = 0
    n=0
    x_advs=[]
    for _ in range(N//batch_size):
        x_mc = gen_bias(batch_size)
        h_MC = h(x_mc)
        if track_advs:
            x_advs.append(x_mc[h_MC>=0])
        
        n+= batch_size
        p_local = (h_MC>=0)*GaussianImportanceWeight(x=x_mc,mu_1=x_clean,mu_2=X_mpp)
        pre_var_local = (h_MC>=0)*GaussianImportanceWeight(x=x_mc,mu_1=x_clean,mu_2=X_mpp)**2
        del x_mc
        p_f = ((n-batch_size)/n)*p_f+(batch_size/n)*p_local.float().mean()
        pre_var = ((n-batch_size)/n)*pre_var+(batch_size/n)*pre_var_local.float().mean()
        del h_MC
        
    if N%batch_size!=0:
        rest = N%batch_size
        x_mc = gen_bias(rest)
        h_MC = h(x_mc)
        if track_advs:
            x_advs.append(x_mc[h_MC>=0])
        p_local = (h_MC>=0)*GaussianImportanceWeight(x=x_mc,mu_1=x_clean,mu_2=X_mpp)
        del x_mc
        N+= rest
        
        p_f = ((N-rest)/N)*p_f+(rest/N)*p_local.float().mean()
        del h_MC
    
    dict_out = {'nb_calls':N+nb_calls_mpp}
    var_est = (1/N)*(pre_var-p_f**2).item()
    dict_out['var_est']=var_est
    if track_advs:
        dict_out['advs']=torch.cat(x_advs,dim=0).to('cpu').numpy()
    
    
    return p_f.cpu(),dict_out