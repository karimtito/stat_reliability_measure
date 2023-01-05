import scipy.stats as stats 
from stat_reliability_measure.dev.torch_utils import norm_batch_tensor
from math import sqrt

def FORM_pyt(dp_search_method,X,y,score,d,max_dist=None,**kwargs):
    """Computes the probability of failure using the First Order Method

    Args:
        dp_search_method (_type_): design point search method (eg. HLRF, iHLRF...)
        X (_type_): _description_
        y (_type_): _description_
        model (_type_): _description_
        d (_type_): _description_
        max_dist (_type_, optional): _description_. Defaults to None.
    """
    if max_dist is None:
        max_dist = sqrt(d)
    design_points=dp_search_method(score,X,y,max_dist=max_dist,**kwargs)
    l2dist= norm_batch_tensor(design_points,d=d)
    p_fail= stats.norm.cdf(-l2dist.cpu())
    return p_fail
