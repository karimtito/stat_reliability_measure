import scipy.stats as stats 
from stat_reliability_measure.dev.torch_utils import norm_batch_tensor
from math import sqrt
import torch

def FORM_pyt(dp_search_method,X,score,d,**kwargs):
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
    design_points=dp_search_method(score,X,**kwargs)
    l2dist= norm_batch_tensor(design_points,d=d)
    p_fail= stats.norm.cdf(-l2dist)
    return p_fail

class method_config:
    optim_steps=10
    opt_steps_list = []
    tol=1e-3
    max_iter=100
    max_iter_range=[]
    random_init=False

#gradient descent in 1 dimension to find zero of a function f
def find_zero_gd_pyt(f, grad_f, x0, obj='min', step_size=1e-2, max_iter=100, tol=1e-3,random_init=False):
    x = x0
    
    if random_init:
        x = x0 + step_size*torch.randn_like(x0)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    f_calls = 0
    sign= 1 if obj=='max' else -1
    for i in range(max_iter):
        x = x + sign*step_size * grad_f(x)
        f_calls += 2 # we count 2 calls for each gradient evaluation (forward and backward)
        if sign*f(x) > -tol:
            f_calls += 1 # we count one more call for the last function evaluation
            break
    if i == max_iter-1:
        print('Warning: max_iter reached in find_zero_gd')
    return x, f_calls

def design_point_search(f, grad_f, x_0, step_size=1e-2,max_iter=100):
    """ Finds the design point using HL-RF algorithm """