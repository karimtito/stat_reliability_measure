
import argparse

def dichotomic_search(f, a, b, thresh=0, n_max =50):
    """Implementation of dichotomic search of minimum solution for an increasing function
        Args:
            -f: increasing function
            -a: lower bound of search space
            -b: upper bound of search space
            -thresh: threshold such that if f(x)>0, x is considered to be a solution of the problem
    
    """
    low = a
    high = b
     
    i=0
    while i<n_max:
        i+=1
        if f(low)>=thresh:
            return low, f(low)
        mid = 0.5*(low+high)
        if f(mid)>thresh:
            high=mid
        else:
            low=mid

    return high, f(high)

def float_to_file_float(x):
    x=str(x).replace('.','_').replace(',','_')
    return x

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(in_str,split_chr=',',type_out=None):
    l = in_str.strip('[').strip(']').split(split_chr)
    if type_out is not None:
        l=[type_out(e) for e in l]
    return l