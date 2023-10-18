import torch
import scipy.stats as stats
import numpy as np
from math import sqrt
from stat_reliability_measure.dev.torch_utils import NormalCDFLayer
from stat_reliability_measure.dev.imp_sampling.is_pyt import mpp_search, search_methods


def LineSampling(x_clean,gen,h,N:int=int(1e4),batch_size:int=int(1e3),track_advs:bool=False,verbose=0.,track_X:bool=False,
               nb_calls_mpp=0,G=None,gradG=None,
               model=None,search_method='mpp_search',X_mpp=None,
               alpha_CI=0.05,**kwargs):
    """ Line sampling algorithm to compute probability of failure, using an importance direction alpha,
    and a line search in this direction to find the failure point for gaussian samples on the orthogonal space.
    """
    d = x_clean.numel()
    zero_latent = torch.zeros((1,d),device=x_clean.device)
    if X_mpp is None:
        if search_method not in search_methods.keys():
            raise NotImplementedError(f"Method {search_method} is not implemented.")
        if search_method=='mpp_search':
            assert gradG is not None, "gradG must be provided for mpp_search"
            debug=verbose>=1
            X_mpp,nb_calls_mpp=mpp_search(zero_latent=zero_latent, 
                        grad_f= gradG,stop_cond_type='beta',
                        max_iter=5,stop_eps=1E-2,debug=debug)
        