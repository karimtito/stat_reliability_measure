"""script to launch experiments as described in the experimental design
     
"""
from pathlib import Path
import numpy as np
import os
class config:
    p_range=[1e-2,1e-4,1e-6,1e-8,1e-10]
    n_rep=300
    methods_list=['MC','MLS','H_SMC','MALA_SMC','RW_SMC']
    params_dic={'MC': {'N_range':[int(1e5),int(1e6)],'batch_size':[128]},

                'MLS': {'N_range':[int(1e2,1e3)]}
                }

    log_dir=Path('./low_prob_logs/exp_1')



def main():
    completed_path= Path(config.log_dir,'completed.csv')
    if not config.log_dir.exists():
        os.mkdir(config.log_dir)
    for method in config.methods_list:
        if method=='MC':
            import exp_1.exp_1_MC as exp_1_MC
            for key,value in config.params_dic['MC'].items:
                setattr(exp_1_MC.config, key, value)
            exp_1_MC.main()
            del exp_1_MC 
        if method=='MLS':
            import exp_1.exp_1_MLS as exp_1_MLS
            for key,value in config.params_dic['MLS'].items:
                setattr(exp_1_MLS.config, key, value)
            exp_1_MLS.main()
            del exp_1_MLS

if __name__ == '__main__':
    main()
