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
from stat_reliability_measure.dev.utils import float_to_file_float



class config:
    dataset_models_epsilons= {'mnist':{'dnn2':[0.15]},'cifar10':{'convnet':[0.1]}}
    n_rep=200
    image_idx=0
    np_seed=0
    torch_seed=0
    robust_model=False
    robust_eps=0.1
    repeat_exp=True
    methods_list=['MC','H_SMC','MLS_SMC','MALA_SMC','RW_SMC']
    params_dic={'MC': {'mnist':
                            {'N_range':[int(1e5),int(1e6)],'b_range':[int(1e5)]},
                       'cifar10': 
                            {'N_range':[int(1e4),int(1e5)],'b_range':[int(4e3)]}
                            },

                'MLS_SMC': {'mnist':
                            {'N_range':[int(1e2),int(5e2)],'ratio_range':[0.2,0.5,0.8],
                        'T_range':[1,10,100]},
                       'cifar10': 
                            {'N_range':[int(1e2),int(2e2)],'ratio_range':[0.2,0.5,0.8],
                        'T_range':[1,10]}
                            },
                
                
    
                'H_SMC':{'mnist':
                            {'N_range':[int(1e2),int(5e2)],'e_range':[0.2,0.5,0.8],
                        'T_range':[1,10,100],'L_range':[10]},
                       'cifar10': 
                            {'N_range':[int(1e2),int(2e2)],'e_range':[0.2,0.5,0.8],
                        'T_range':[1,10],'L_range':[5]}
                            },
                'MALA_SMC':{'mnist':
                            {'N_range':[int(1e2),int(5e2)],'e_range':[0.2,0.5,0.8],
                        'T_range':[1,10,100]},
                       'cifar10': 
                            {'N_range':[int(1e2),int(2e2)],'e_range':[0.2,0.5,0.8],
                        'T_range':[1,10]}
                            },
                'RW_SMC':{'mnist':
                            {'N_range':[int(1e2),int(5e2)],'e_range':[0.2,0.5,0.8],
                        'T_range':[1,10,100]},
                       'cifar10': 
                            {'N_range':[int(1e2),int(2e2)],'e_range':[0.2,0.5,0.8],
                        'T_range':[1,10]}
                            },
                }
    log_dir='./logs/exp_2_low_probs'
    verbose=0
    


parser = argparse.ArgumentParser()
parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--torch_seed',type=int,default=config.torch_seed)
parser.add_argument('--np_seed',type=int,default=config.np_seed)
parser.add_argument('--verbose',type=float,default=config.verbose)
parser.add_argument('--methods_list',type=str2list,default=config.methods_list)
args=parser.parse_args()
for k,v in vars(args).items():
    setattr(config, k, v)

def get_completion_rate(config):
    """compute experiments completion rate and prints to stdout

    Args:
        p_range: list of failure probility levels
        n_rep: number of repetitions considered
        params_dic (dict): list of parameters dictionnaries
        log_dir (str): string of log directory
    """
    n_rep:int=config.n_rep
    data_m_eps:dict=config.dataset_models_epsilons
    params_dic:dict=config.params_dic
    log_dir:str=config.log_dir
    count_tot =0 
    count_compl= 0
    aggr_res_path=os.path.join(log_dir,'aggr_res.csv')
    if not os.path.exists(aggr_res_path):
        return 0
    aggr_res_df = pd.read_csv(aggr_res_path)
    for dataset,model_eps_dic in data_m_eps.items():
        for model_arch,epsilons in model_eps_dic.items():
            for method,params in params_dic.items():
                params=params[dataset]
                params_keys=list(params.keys())
                
                params_values=list(params.values())
                
                
                range_dict={'N_range':'N','T_range':'T','epsilons':'epsilon','ratio_range':'ratio',
                            'b_range':'batch_size','e_range':'ess_alpha','L_range':'L'}
                # removing the '_range' from parameters keys
                params_keys = [range_dict[key] if ('range' in key) else key for key in params_keys]
                params_keys.append('epsilon')
                
                params_values.append(epsilons)
                product_params=product(*params_values)
                if config.robust_model:
                    c_robust_eps = float_to_file_float(config)
                model_name=model_arch +'_' + dataset if not config.robust_model else f"model_{model_arch}_{dataset}_robust_{c_robust_eps}"
                for params_vals in product_params:
                    count_tot+=1
                    
                    same_exp_df = get_sel_df(df=aggr_res_df,cols=params_keys, vals=params_vals, 
                        triplets=[('method',method,'=='),('n_rep',n_rep,'=='),
                        ('model_name',model_name,'=='),('image_idx',config.image_idx,'=='),
                        ('dataset',dataset,'==')])  
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
        cols=['epsilon','method','dataset','N','model_name','image_idx','rho','ratio','n_rep','T','alpha','min_rate','mean_time','std_time','mean_est','s',
            'bias','mean abs error','batch_size','mean_rel_error','std_est','freq underest','gpu_name','cpu_name','ratio',
            'ess_alpha','L','last_particle']
        aggr_res_df= pd.DataFrame(columns=cols)
        aggr_res_df.to_csv(aggr_res_path,index=False)
    else:
        compl_rate=get_completion_rate(config)
        print(f"Completion rate: {compl_rate*100:.1f}%")
    for dataset,model_eps_dic in config.dataset_models_epsilons.items():
        for model_arch,epsilons in model_eps_dic.items():
            for method in config.methods_list:
                
                if method=='MC':
                    import exp_2.exp_2_MC as exp
                elif method=='MLS_SMC':
                    import exp_2.exp_2_MLS as exp
                elif method=='H_SMC':
                    import exp_2.exp_2_HSMC as exp
                elif method.lower() in ('rw_smc','rw'):
                    import exp_2.exp_2_RW as exp
                elif method.lower() in ('mala_smc','mala'):
                    import exp_2.exp_2_MALA as exp
                # elif method=='FORM':
                #     import exp_2.exp_2_FORM as exp_2_FORM
                else:
                    raise NotImplementedError(f"Method {method} is not implemented yet.")
                for key,value in config.params_dic[method][dataset].items():
                    print(f"key:{key},value:{value}")
                    setattr(exp.config, key, value)
                exp.config.n_rep=config.n_rep
                exp.config.model_arch = model_arch
                exp.config.dataset = dataset
                exp.config.epsilons=epsilons
                exp.config.verbose=config.verbose
                exp.config.log_dir = config.log_dir
                exp.config.repeat_exp=False
                exp.config.update_aggr_res=True
                exp.config.track_cpu=True
                exp.config.track_gpu=True
                exp.config.torch_seed=config.torch_seed
                exp.config.np_seed=config.np_seed
                exp.main()
                del exp
                #     for key,value in config.params_dic['FORM'].items:
                #         setattr(exp_2_FORM.config, key, value)
                #     exp_2_MC.main()
                #     del exp_2_MC 
                print(f"Experiments for method {method} completed.")
                compl_rate=get_completion_rate(config)
                print(f"Overall completion rate: {compl_rate*100:.1f}%")

if __name__ == '__main__':
    main()
