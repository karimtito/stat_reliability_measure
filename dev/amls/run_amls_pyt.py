import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from stat_reliability_measure.home import ROOT_DIR
from time import time
from datetime import datetime
import stat_reliability_measure.dev.torch_utils as t_u
import stat_reliability_measure.dev.utils as utils
from stat_reliability_measure.dev.utils import  float_to_file_float,str2bool,str2intList,str2floatList, str2list
from stat_reliability_measure.dev.utils import get_sel_df, simple_vars, range_vars, range_dict_to_lists
import stat_reliability_measure.dev.mls.amls_uniform as amls_mls
from stat_reliability_measure.dev.amls.amls_utils import MLS_SMC_Config
import stat_reliability_measure.dev.amls.amls_pyt as amls_pyt
from config import Exp2Config
from itertools import product as cartesian_product


def run_amls_exp(model, X, y, epsilon_range=None, noise_dist='',dataset_name = 'dataset',
                 model_name='model',
                 verbose=0.1, x_min=0,x_max=None, mask_opt=False,mask_val=0.,mask_cond =
                 log_hist_=False, aggr_res_path=None,
                 log_txt_=False,exp_config=None,method_config=None,**kwargs):
    """ Running reliability experiments on neural network model with supervised data (X,y)

    
    """
    if method_config is None:
        method_config = MLS_SMC_Config()
        method_config.verbose=verbose
    if exp_config is None:
        exp_config=Exp2Config(model=model,X=X,y=y,
        dataset_name=dataset_name,model_name=model_name,epsilon_range=epsilon_range,
        aggr_res_path=aggr_res_path,x_min=x_min,x_max=x_max,mask_opt=mask_opt,
        mask_val=mask_val,verbose=verbose)
        exp_config.aggr_res_path=aggr_res_path
    for k,v in kwargs.items():
        if hasattr(method_config,k):
            if hasattr(exp_config,k) and  not ('config' in k):
                raise ValueError(f"parameter {k} is in both exp_config and method_config")
            setattr(method_config, k, v)
        elif hasattr(exp_config,k):
            setattr(exp_config, k, v)
        else:
            raise ValueError(f"unknown configuration parameter {k}")
    method_config.update()
    exp_config.update()
    

    method_config.exp_config=exp_config
    param_ranges = [v for k,v in range_vars(method_config)]

    
    param_lens=np.array([len(l) for l in param_ranges])
    param_cart = cartesian_product(param_ranges)
    nb_runs= np.prod(param_lens)*len(exp_config.epsilon_range)*exp_config.nb_inputs
    if verbose>0:
        print(f"Running reliability experiments on architecture {exp_config.model_arch} trained on {exp_config.dataset}.")
        print(f"Testing {exp_config.noise_dist} noise pertubation with epsilon in {exp_config.epsilon_range}")

    if exp_config.verbose>0:
        exp_config.print_config()
        method_config.print_config()

    exp_config.config_path=os.path.join(exp_config.exp_log_path,'exp_config.json')
    exp_config.to_json()
    method_config.config_path=os.path.join(exp_config.exp_log_path,'method_config.json')
    method_config.to_json()
    method_range_dict = range_vars(method_config)
    method_param_lists = range_dict_to_lists(range_dict=method_range_dict)
    i_exp=0
    param_ranges= list(range_vars(method_config).keys()) + list(range_vars(exp_config).keys())
    if verbose>0:
        print(param_ranges)
    lenghts=np.array([len(L) for L in param_ranges])
    nb_exps= np.prod(lenghts)*exp_config.nb_inputs
    method_name="MLS_SMC"
    reuslts_df=pd.DataFrame({})
    use_cuda= "cuda" in exp_config.device
    for l in range(len(exp_config.X)):
        with torch.no_grad():
            x_0,y_0 = exp_config.X[l], exp_config.y[l]
        input_shape=x_0.shape
        #x_0.requires_grad=True
        for idx in range(len(exp_config.epsilon_range)):
            
            
            exp_config.epsilon = exp_config.epsilon_range[idx]
           
            pgd_success= (exp_config.attack_success[idx][l]).item() if exp_config.use_attack else None 
            p_l,p_u=None,None
            if exp_config.lirpa_bounds:
                from stat_reliability_measure.dev.lirpa_utils import get_lirpa_bounds
                # Step 2: define perturbation. Here we use a Linf perturbation on input image.
                p_l,p_u=get_lirpa_bounds(x_0=x_0,y_0=y_0,model=model,epsilon=exp_config.epsilon,
                num_classes=exp_config.num_classes,noise_dist=exp_config.noise_dist,a=exp_config.a,device=exp_config.device)
                p_l,p_u=p_l.item(),p_u.item()
            def prop(x):
                y = model(x)
                y_diff = torch.cat((y[:,:y_0], y[:,(y_0+1):]),dim=1) - y[:,y_0].unsqueeze(-1)
                y_diff, _ = y_diff.max(dim=1)
                return y_diff #.max(dim=1)
                
            lists_cart= cartesian_product(*method_param_lists)
            for method_params in lists_cart:
                i_exp+=1
                aggr_res_path=os.path.join(exp_config.log_dir,'aggr_res.csv')
                if (not exp_config.repeat_exp) and exp_config.update_aggr_res and os.path.exists(aggr_res_path):
                    aggr_res_df = pd.read_csv(aggr_res_path)
                    same_exp_df = get_sel_df(df=aggr_res_df,triplets=[('method_name',method_name,'='),
                    ('model_name',exp_config.model_name,'='),('epsilon',exp_config.epsilon,'='),('image_idx',l,'='),('n_rep',config.n_rep,'='),
        ('N',method_config.N,'='),('T',method_config.T,'='),
        ('s',method_config.s,'='),('last_particle',method_config.last_particle,'=='),
        ('ratio',method_config.ratio,'=')] )  
                    # if a similar experiment has been done in the current log directory we skip it
                    if len(same_exp_df)>0 and exp_config.verbose>0:
                        print(f"Skipping {method_name} run {i_exp}/{nb_exps}")
                        print(f"with model: {model_name}, img_idx:{l},eps:{exp_config.epsilon},T:{T},N:{N},s:{s},K:{K}")
                        continue
                loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
                log_name=method_name+'_e_'+float_to_file_float(exp_config.epsilon_range[idx])+'_N_'+str(N)+'_T_'+str(T)+'_s_'+float_to_file_float(s)
                log_name=log_name+'_r_'+float_to_file_float(method_config.ratio)+'_'+'_'+loc_time
                log_path=os.path.join(exp_log_path,log_name)
                
                print(f"Starting {method_name} run {i_exp}/{nb_exps}")
                                
                
                K=int(method_config.N*method_config.ratio) if not method_config.last_particle else method_config.N-1
                if exp_config.verbose>0:
                    print(f"with model: {model_name}, img_idx:{l},eps:{exp_config.epsilon},T:{method_config.T},N:{method_config.N},s:{method_config.s},K:{K}")
                if exp_config.verbose>3:
                    print(f"K/N:{K/method_config.N}")
                times= []
                rel_error= []
                ests = [] 
                log_ests=[]
                calls=[]
                if method_config.track_finish:
                    finish_flags=[]
                iterator = tqdm(range(exp_config.n_rep)) if exp_config.tqdm_opt else range(exp_config.n_rep)
                for i in tqdm(range(exp_config.n_rep)):
                    t=time()
                    lg_p,nb_calls,max_val,x,levels,dic=amls_mls.multilevel_uniform(prop=prop,
                    count_particles=method_config.N,count_mh_steps=method_config.T,x_min=exp_config.x_min,
                    x_max=exp_config.x_max,
                    x_sample=x_0,sigma=exp_config.epsilon,rho=method_config.ratio,CUDA=use_cuda,debug=(exp_config.verbose>=1))


                    t=time()-t
                    # we don't need adversarial examples and highest score
                    del x
                    del max_val
                    log_ests.append(lg_p)
                    est=np.exp(lg_p)
                    if exp_config.verbose:
                        print(f"Est:{est}")
                    # dict_out=amls_res[1]
                    # if config.track_accept:
                    #     accept_logs=os.path.join(log_path,'accept_logs')
                    #     if not os.path.exists(accept_logs):
                    #         os.mkdir(path=accept_logs)
                    #     accept_rates=dict_out['accept_rates']
                    #     np.savetxt(fname=os.path.join(accept_logs,f'accept_rates_{i}.txt')
                    #     ,X=accept_rates)
                    #     x_T=np.arange(len(accept_rates))
                    #     plt.plot(x_T,accept_rates)
                    #     plt.savefig(os.path.join(accept_logs,f'accept_rates_{i}.png'))
                    #     plt.close()
                    #     accept_rates_mcmc=dict_out['accept_rates_mcmc']
                    #     x_T=np.arange(len(accept_rates_mcmc))
                    #     plt.plot(x_T,accept_rates_mcmc)
                    #     plt.savefig(os.path.join(accept_logs,f'accept_rates_mcmc_{i}.png'))
                    #     plt.close()
                    #     np.savetxt(fname=os.path.join(accept_logs,f'accept_rates_mcmc_{i}.txt')
                    #     ,X=accept_rates_mcmc)
                    if method_config.track_finish:
                        finish_flags.append(levels[-1]>=0)
                    times.append(t)
                    ests.append(est)
                    calls.append(nb_calls)
    
                times=np.array(times)
                ests = np.array(ests)
                log_ests=np.array(log_ests)
                mean_est=ests.mean()
                calls=np.array(calls)
                mean_calls=calls.mean()
                std_est=ests.std()
                
                q_1,med_est,q_3=np.quantile(a=ests,q=[0.25,0.5,0.75])
                std_rel=std_est/mean_est**2 if mean_est >0 else 0
                std_rel_adj=std_rel*mean_calls
                print(f"mean est:{ests.mean()}, std est:{ests.std()}")
                print(f"mean calls:{calls.mean()}")
                print(f"std. re.:{std_rel}")
                print(f"std. rel. adj.:{std_rel_adj}")
                
                if method_config.track_finish:
                    finish_flags=np.array(finish_flags)
                    freq_finished=finish_flags.mean()
                    freq_zero_est=(ests==0).mean()
                else:
                    freq_zero_est,freq_finished=None,None
                #finished=np.array(finish_flag)
                if method_config.track_finish and freq_finished<1:
                    unfinish_est=ests[~finish_flags]
                    unfinish_times=times[~finish_flags]
                    unfinished_mean_est=unfinish_est.mean()
                    unfinished_mean_time=unfinish_times.mean()
                else:
                    unfinished_mean_est,unfinished_mean_time=None,None
                if os.path.exists(log_path):
                    log_path=log_path+'_rand_'+str(np.random.randint(low=0,high=9))
                os.mkdir(log_path)
                
                times_path=os.path.join(log_path,'times.txt')
                
                est_path=os.path.join(log_path,'ests.txt')
                

                std_log_est=log_ests.std()
                mean_log_est=log_ests.mean()
                lg_q_1,lg_med_est,lg_q_3=np.quantile(a=ests,q=[0.25,0.5,0.75])
                lg_est_path=os.path.join(log_path,'lg_ests.txt')
                

                if log_txt_:
                    np.savetxt(fname=times_path,X=times)
                    np.savetxt(fname=est_path,X=ests)
                    np.savetxt(fname=lg_est_path,X=ests)

                
                if log_hist_:
                    plt.hist(times, bins=10)
                    plt.savefig(os.path.join(log_path,'times_hist.png'))
                    plt.hist(ests,bins=10)
                    plt.savefig(os.path.join(log_path,'ests_hist.png'))
                
                

                #with open(os.path.join(log_path,'results.txt'),'w'):
                results={"image_idx":l,'mean_calls':calls.mean(),'std_calls':calls.std()
                ,'mean_time':times.mean(),'std_time':times.std()
                ,'mean_est':ests.mean(),'std_est':ests.std(), 'est_path':est_path,'times_path':times_path,
                "std_rel":std_rel,"std_rel_adj":std_rel*mean_calls,
                "lirpa_safe":exp_config.lirpa_safe,
                'q_1':q_1,'q_3':q_3,'med_est':med_est,"lg_est_path":lg_est_path,
                "mean_log_est":mean_log_est,"std_log_est":std_log_est,
                "lg_q_1":lg_q_1,"lg_q_3":lg_q_3,"lg_med_est":lg_med_est,
                "log_path":log_path,"log_name":log_name,}
                results.update(simple_vars(method_config))
                results.update(simple_vars(exp_config))
                exp_res.append(results)
                results_df=pd.DataFrame([results])
                results_df.to_csv(os.path.join(log_path,'results.csv'),)
                if exp_config.aggr_res_path is None:
                    aggr_res_path=os.path.join(exp_config.log_dir,'agg_res.csv')
                else: 
                    aggr_res_path=exp_config.aggr_res_path

                if exp_config.update_aggr_res:
                    if not os.path.exists(aggr_res_path):
                        print(f'aggregate results csv file not found \n it will be build at {aggr_res_path}')
                        cols=['method_name','gaussian_latent','N','rho','n_rep','T','epsilon','alpha','min_rate','mean_time','std_time','mean_est',
                        'std_est','freq underest','g_target']
                        cols+=['freq_finished','freq_zero_est','unfinished_mean_est','unfinished_mean_time']
                        cols+=['pgd_success','p_l','p_u','gpu_name','cpu_name','np_seed','torch_seed','noise_dist','datetime']
                        agg_res_df= pd.DataFrame(columns=cols)

                    else:
                        agg_res_df=pd.read_csv(aggr_res_path)
                    agg_res_df = pd.concat([agg_res_df,results_df],ignore_index=True)
                    agg_res_df.to_csv(aggr_res_path,index=False)
    return config,results_df,agg_res_df
