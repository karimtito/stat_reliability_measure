""" Script to launch low probability level threshold
    experiments as described in the experimental design 
"""
from pathlib import Path
import numpy as np
import pandas as pd
import os
from itertools import product
import argparse
from stat_reliability_measure.dev.utils import str2list,str2floatList,get_sel_df



class config:
    p_range=[1e-2,1e-4,1e-6,1e-8,1e-10]
    n_rep=300
    np_seed=0
    torch_seed=0
    methods_list=['MC','MLS_SMC','H_SMC','MALA_SMC','RW_SMC']
    params_dic={'MC': {'N_range':[int(1e5),int(1e6)],'b_range':[int(1e5)]},

                'MLS_SMC': {'N_range':[int(1e2),int(1e3)],'ratio_range':[0.1,0.5,0.9],
                        'T_range':[1,10,100],},
                'H_SMC':{'N_range':[int(1e2),int(1e3)],'e_range':[0.1,0.5,0.9],
                        'T_range':[1,10,100], 'L_range':[10]},
                'MALA_SMC':{'N_range':[int(1e2),int(1e3)],'e_range':[0.1,0.5,0.9],
                        'T_range':[1,10,100],},
                'RW_SMC':{'N_range':[int(1e2),int(1e3)],'e_range':[0.1,0.5,0.9],
                        'T_range':[1,10,100],},
                }
    log_dir='./logs/exp_1_low_probs'
    verbose=0


parser = argparse.ArgumentParser()
parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--p_range',type=str2floatList,default=config.p_range)
parser.add_argument('--verbose',type=float,default=config.verbose)
parser.add_argument('--methods_list',type=str2list,default=config.methods_list)
args=parser.parse_args()
for k,v in vars(args).items():
    setattr(config, k, v)

def get_completion_rate(n_rep:int=config.n_rep, p_range:list=config.p_range,
    params_dic:dict=config.params_dic, log_dir:str=config.log_dir):
    """compute experiments completion rate and prints to stdout

    Args:
        p_range: list of failure probility levels
        n_rep: number of repetitions considered
        params_dic (dict): list of parameters dictionnaries
        log_dir (str): string of log directory
    """
    count_tot =0 
    count_compl= 0
    aggr_res_path=os.path.join(log_dir,'aggr_res.csv')
    if not os.path.exists(aggr_res_path):
        return 0
    aggr_res_df = pd.read_csv(aggr_res_path)
    for method,params in params_dic.items():
        params_keys=list(params.keys())
        
        params_values=list(params.values())
        params_keys.append('p_t')
        params_values.append(p_range)
        range_dict={'N_range':'N','T_range':'T','p_range':'p_t','ratio_range':'ratio',
                    'b_range':'batch_size','e_range':'ess_alpha','L_range':'L'}
        # removing the '_range' from parameters keys
        params_keys = [range_dict[key] if 'range' in key else key for key in params_keys]
        product_params=product(*params_values)
        for params_vals in product_params:
            count_tot+=1
            same_exp_df = get_sel_df(df=aggr_res_df,cols=params_keys, vals=params_vals, 
                triplets=[('method',method,'=='),('n_rep',n_rep,'==')])  
            # if a similar experiment has been done in the current log directory we skip it
            if len(same_exp_df)>0:
                count_compl+=1

    print(f"Total exp: {count_tot}")
    print(f"Completed exp: {count_compl}")
    return count_compl/count_tot
                


def main():
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
    if not os.path.exists(aggr_res_path):
        print(f"Completion rate: 0%")
        cols=['p_t','method','N','rho','n_rep','T','alpha','min_rate','mean_time','std_time','mean_est','s',
            'bias','mean abs error','batch_size','mean_rel_error','std_est','freq underest','gpu_name','cpu_name','ratio',
            'ess_alpha','L']
        aggr_res_df= pd.DataFrame(columns=cols)
        aggr_res_df.to_csv(aggr_res_path,index=False)
    else:
        compl_rate=get_completion_rate()
        print(f"Completion rate: {compl_rate*100:.1f}%")
    for method in config.methods_list:
        
        if method=='MC':
            import exp_1.exp_1_MC as exp
        elif method=='MLS_SMC':
            import exp_1.exp_1_MLS as exp
        elif method=='H_SMC':
            import exp_1.exp_1_H_SMC as exp
        elif method.lower() in ('rw_smc','rw'):
            import exp_1.exp_1_RW as exp
        elif method.lower() in ('mala_smc','mala'):
            import exp_1.exp_1_MALA as exp
        # elif method=='FORM':
        #     import exp_1.exp_1_FORM as exp_1_FORM
        else:
            raise NotImplementedError(f"Method {method} is not implemented yet.")
        for key,value in config.params_dic[method].items():
            setattr(exp.config, key, value)
        exp.config.n_rep=config.n_rep
        
        exp.config.p_range=config.p_range
        
        exp.config.verbose=config.verbose
        exp.config.log_dir = config.log_dir
        exp.config.repeat_exp=False
        exp.config.update_aggr_res=True
        exp.config.track_cpu=True
        exp.config.track_gpu=True
        exp.main()
        del exp
        #     for key,value in config.params_dic['FORM'].items:
        #         setattr(exp_1_FORM.config, key, value)
        #     exp_1_MC.main()
        #     del exp_1_MC 
        print(f"Experiments for method {method} completed.")
        compl_rate=get_completion_rate()
        print(f"Overall completion rate: {compl_rate*100:.1f}%")

if __name__ == '__main__':
    main()
