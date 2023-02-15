def MC_pf(gen,score,N_mc:int=int(1e4),batch_size:int=int(1e2),track_advs:bool=False):
    """ Computes probability of failure 

    Args:
        X (_type_): _description_
        y (_type_): _description_
        gen (_type_): _description_
        score (_type_): _description_
        N_mc (int, optional): _description_. Defaults to int(1e4).
        batch_size (int, optional): _description_. Defaults to int(1e2).
        track_advs (bool, optional): _description_. Defaults to False.

    Returns:
        float: probability of failure
    """
    p_f = 0
    N=0
    x_advs=[]
    for _ in range(N_mc//batch_size):
        x_mc = gen(batch_size)
        score_MC = score(x_mc)
        if track_advs:
            x_advs.append(x_mc[score_MC>=0])
        del x_mc
        N+= batch_size
        p_f = ((N-batch_size)/N)*p_f+(batch_size/N)*(score_MC>=0).float().mean()
        del score_MC
        
    if N_mc%batch_size!=0:
        rest = N_mc%batch_size
        x_mc = gen(rest)
        score_MC = score(x_mc)
        if track_advs:
            x_advs.append(x_mc[score_MC>=0])
        del x_mc
        N+= rest
        p_f = ((N-rest)/N)*p_f+(rest/N)*(score_MC>=0).float().mean()
        del score_MC
    assert N==N_mc
    if track_advs:
        return p_f, x_advs
    return p_f