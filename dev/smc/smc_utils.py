import torch
def dichotomic_search_d(f, a, b, thresh=0, n_max =50):
    """Implementation of dichotomic search of minimum solution for an decreasing function
        Args:
            -f: increasing function
            -a: lower bound of search space
            -b: upper bound of search space
            -thresh: threshold such that if f(x)>=thresh, x is considered to be a solution of the problem
    """
    low = a
    high = b
     
    i=0
    while i<n_max:
        i+=1
        if f(low)<=thresh:
            return low, f(low)
        mid = 0.5*(low+high)
        if f(mid)<thresh:
            high=mid
        else:
            low=mid

    return high, f(high)

def SimpAdaptBetaPyt(beta_old,v,g_target,search_method=dichotomic_search_d,max_beta=1e6,verbose=0,multi_output=True,v_min_opt=False):

    """ Simple adaptive mechanism to select next inverse temperature

    Returns:
        float: new_beta, next inverse temperature
        float: g, next mean weight
    """
    ess_beta = lambda beta : torch.exp(-(beta-beta_old)*(v)).mean() #decreasing funtion of the parameter beta
    if v_min_opt:
        v_min=v.min()
        ess_beta=lambda beta : torch.exp(-(beta-beta_old)*(v-v_min)).mean() 
    assert ((g_target>0) and (g_target<1)),"The target average weigh g_target must be a positive number in (0,1)"
    results= search_method(f=ess_beta,a=beta_old,b=max_beta,thresh = g_target) 
    new_beta, g = results[0],results[-1] 
    
    if verbose>0:
        print(f"g_target: {g_target}, actual g:{g}")

    if multi_output:
        if v_min_opt:
            g=torch.exp((-(new_beta-beta_old)*v)).mean() #in case we use v_min_opt we need to output original g
        return (new_beta,g)
    else:
        return new_beta


def ESSAdaptBetaPyt(beta_old, v,lambda_0=1,max_beta=1e9,g_target=None,v_min_opt=None,multi_output=False):
    """Adaptive inverse temperature selection based on an approximation 
    of the ESS critirion 
    v_min_opt and g_target are unused dummy variables for compatibility
    """
    delta_beta=(lambda_0/v.std().item())
    if multi_output:
        new_beta=min(beta_old+delta_beta,max_beta)
        g=torch.exp(-(delta_beta)*v).mean()
        res=(new_beta,g)
    else:
        res=min(beta_old+delta_beta,max_beta)
    return res


def nextBetaSimpESS(beta_old, v,lambda_0=1,max_beta=1e9,multi_output=False):
    """Adaptive inverse temperature selection based on an approximation 
    of the ESS critirion 
    v_min_opt and g_target are unused dummy variables for compatibility
    """
    v_std=v.std().item()
    if v_std==0:
        v_std=1e-9
    delta_beta=(lambda_0/v.std().item())
    res=min(beta_old+delta_beta,max_beta)
    return res,None

def bissection(f,a,b,target,thresh,n_max=100):
    """Implementation of binary search of method to find root of continuous function
           Args:
            -f: continuous function
            -a: lower bound of search space
            -b: upper bound of search space
            -thresh: threshold such that if f(x)>0, x is considered to be a solution of the problem
    
    """
    low = a
    assert f(a)>target-thresh
    assert f(b)<target+thresh
    high = b
     
    i=1
    while i<n_max:
        mid = 0.5*(low+high)
        if f(mid)>target+thresh:
            low=mid
        elif f(mid)<target-thresh:
            high=mid
        else:
            return mid,f(mid)
        i+=1
    mid = 0.5*(low+high)
    return mid, f(mid)



def nextBetaESS(beta_old,v,ess_alpha,max_beta=1e6,verbose=0,thresh=1e-3,debug=False,eps_ess=1e-10):
    """Adaptive selection of next inverse temperature based on the
     ESS critirion 
    v_min_opt and g_target are unused dummy variables for compatibility
    """
    N=v.shape[0]
    target_ess=int(ess_alpha*N)
    ESS = lambda beta: (torch.exp(-(beta-beta_old)*v).sum())**2/(torch.exp(-2*(beta-beta_old)*v).sum()+eps_ess)
    if debug:
        assert ESS(beta_old)>target_ess+thresh,"Target is chosen too close to N."
    if ESS(max_beta)>=target_ess-thresh:
        return max_beta,ESS(max_beta)
    else:
        new_beta,new_ess=bissection(f=ESS,a=beta_old,b=max_beta,target=target_ess,thresh=thresh)
        if verbose>=1:
            print(f"New beta:{new_beta}, new ESS:{new_ess}")
        return new_beta,new_ess