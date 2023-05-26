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

def mpp_search(f, grad_f, x_0,max_iter=100,stop_cond_type='grad_norm',stop_eps=1e-3):
    """ Search algorithm for the Most Probable Point (MPP) 
        according to 'Probabilistc Engineering Design' source from University of Missouri  """
    x=x_0 
    grad_fx = grad_f(x)
    f_calls+=2
    beta=torch.norm(x)
    k= 0
    stop_cond=False
    while k<max_iter &  ~stop_cond: 
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

        
            