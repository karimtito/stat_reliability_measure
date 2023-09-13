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
    

def GaussianIS(x_clean,h,N:int=int(1e4),batch_size:int=int(1e2),track_advs:bool=False,verbose=0.,track_X:bool=False,
               sigma_bias=1.,):
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
        x_mc = gen(rest)
        h_MC = h(x_mc)
        if track_advs:
            x_advs.append(x_mc[h_MC>=0])
        del x_mc
        N+= rest
        p_f = ((N-rest)/N)*p_f+(rest/N)*(h_MC>=0).float().mean()
        del h_MC
    assert N==N
    dict_out = {'nb_calls':N}
    if track_advs:
        dict_out['advs']=torch.cat(x_advs,dim=0).to('cpu').numpy()
    
    
    return p_f.cpu(),dict_out