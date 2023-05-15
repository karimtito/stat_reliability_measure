import os
import torch
import numpy as np
import time
from datetime import datetime
from stat_reliability_measure.home import ROOT_DIR


def update_config(config):
    if config.input_stop is None:
        config.input_stop=config.input_start+1

    if config.noise_dist is not None:
        config.noise_dist=config.noise_dist.lower()

    if config.noise_dist not in ['uniform','gaussian']:
        raise NotImplementedError("Only uniform and Gaussian distributions are implemented.")

    

    if config.np_seed is None:
        config.np_seed=int(time.time())
    

    if config.torch_seed is None:
        config.torch_seed=int(time.time())

    if len(config.T_range)<1:
        config.T_range=[config.T]
    if len(config.N_range)<1:
        config.N_range=[config.N]

    if len(config.N_range_alt)<1:
        config.N_range_alt=config.N_range
    if len(config.s_range)<1:
        config.s_range=[config.s]
    if len(config.ratio_range)<1:
        config.ratio_range=[config.ratio]


    if config.track_gpu:
        import GPUtil
        gpus=GPUtil.getGPUs()
        if len(gpus)>1:
            print("Multi gpus detected, only the first GPU will be tracked.")
        config.gpu_name=gpus[0].name

    if config.track_cpu:
        import cpuinfo
        config.cpu_name=cpuinfo.get_cpu_info()[[key for key in cpuinfo.get_cpu_info().keys() if 'brand' in key][0]]
        config.cores_number=os.cpu_count()
    #epsilon=config.epsilon


    

    

    if config.epsilons is None:
        log_min,log_max=np.log(config.eps_min),np.log(config.eps_max)
        log_line=np.linspace(start=log_min,stop=log_max,num=config.eps_num)
        config.epsilons=np.exp(log_line)

    