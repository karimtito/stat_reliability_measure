import argparse
import numpy as np

datasets_dims={'mnist':784,'cifar10':3*1024,'cifar100':3*1024,'imagenet':3*224**2}
datasets_num_c={'mnist':10,'cifar10':10,'imagenet':1000}
datasets_means={'mnist':0,'cifar10':(0.4914, 0.4822, 0.4465),'cifar100':[125.3/255.0, 123.0/255.0, 113.9/255.0]}
datasets_stds={'mnist':1,'cifar10':(0.2023, 0.1994, 0.2010),'cifar100':[63.0/255.0, 62.1/255.0, 66.7/255.0]}

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

def float_to_file_float(x):
    x=str(x).replace('.','_').replace(',','_')
    return x

def path_to_meanlog10(path):
    ests=np.loadtxt(path)
    return np.log10(ests).mean()

def path_to_stdlog10(path):
    ests=np.loadtxt(path)
    return np.log10(ests).std()

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

def get_sel_df(df,cols=None,vals=None,conds=None,triplets=None):
    cvc_flag=(cols is not None and vals is not None and conds is not None)
    triplet_flag=triplets is not None
    assert triplet_flag or cvc_flag,"triplets or (cols,vals,conds) should be not empty"
    mask=np.ones(len(df))
    if cvc_flag:
        iterator=zip(cols,vals,conds)
        if triplet_flag:
            iterator=list(iterator)
            iterator.extend(triplets)
    else:
        iterator = triplets
    for (col,val,cond) in iterator:
        if cond.lower() in ['eq','equal','same','==','=']:
            mask*=(df[col]==val).apply(float)
        elif cond.lower() in ['supeq','superequal','>=']:
            mask*=(df[col]>=val).apply(float)
        elif cond.lower() in ['infeq','inferequal','<=']:
            mask*=(df[col]<=val).apply(float)
        elif cond.lower() in ['inf','inferior','<']:
            mask*=(df[col]<val).apply(float)
        elif cond.lower() in ['sup','superior','>']:
            mask*=(df[col]>val).apply(float)
        if cond.lower() in ['neq','nonequal','!=']:
            mask*=(df[col]!=val).apply(float)
        elif cond.lower() in ['contains','cont']:
            mask*=(df[col].apply(lambda x: val in x)).apply(float)
    mask=mask.apply(bool)
    return df[mask]

str2floatList=lambda x: str2list(in_str=x, type_out=float)
str2intList=lambda x: str2list(in_str=x, type_out=int)
low_str=lambda x: str(x).lower()