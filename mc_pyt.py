import torch
def MC_pf(gen,h,N:int=int(1e4),batch_size:int=int(1e2),track_advs:bool=False,verbose=0.,track_X:bool=False):
    """ Computes probability of failure 

    Args:
        X (_type_): vector of clean inputs 
        y (_type_): vector of labels
        gen (_type_): random generator
        h (_type_): score function
        N (int, optional): number of samples to use. Defaults to int(1e4).
        batch_size (int, optional): batch size. Defaults to int(1e2).
        track_advs (bool, optional): Option to track adversarial examples. Defaults to False.
        verbose (float, optional): Level of verbosity. Defaults to False.

    Returns:
        float: probability of failure
    """
    p_f = 0
    n=0
    run_est=[]
    for _ in range(N//batch_size):
        x_mc = gen(batch_size)
        h_MC = h(x_mc)
        if track_advs:
            run_est.append(x_mc[h_MC>=0])
        del x_mc
        n+= batch_size
        p_f = ((n-batch_size)/n)*p_f+(batch_size/n)*(h_MC>=0).float().mean()
        del h_MC
        
    if N%batch_size!=0:
        rest = N%batch_size
        x_mc = gen(rest)
        h_MC = h(x_mc)
        if track_advs:
            run_est.append(x_mc[h_MC>=0])
        del x_mc
        N+= rest
        p_f = ((N-rest)/N)*p_f+(rest/N)*(h_MC>=0).float().mean()
        del h_MC
    assert N==N
    dict_out = {'nb_calls':N}
    if track_advs:
        dict_out['u_rare']=torch.cat(run_est,dim=0).to('cpu').numpy()
    p_f = p_f.cpu().item()
    dict_out['CI_low']=p_f-1.96*(p_f*(1-p_f)/N)**.5
    dict_out['CI_up']=p_f+1.96*(p_f*(1-p_f)/N)**.5
    dict_out['var_est']=p_f*(1-p_f)/N
    return p_f,dict_out