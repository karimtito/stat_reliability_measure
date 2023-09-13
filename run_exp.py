import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from datetime import datetime
import stat_reliability_measure.dev.torch_utils as t_u
import stat_reliability_measure.dev.utils as utils
from stat_reliability_measure.dev.utils import  float_to_file_float,str2bool,str2intList,str2floatList, str2list
from stat_reliability_measure.dev.utils import get_sel_df, simple_vars, range_vars, range_dict_to_lists
import stat_reliability_measure.dev.mls.amls_uniform as amls_mls
from stat_reliability_measure.dev.amls.amls_config import MLS_SMC_Config
import stat_reliability_measure.dev.amls.amls_pyt as amls_pyt
import stat_reliability_measure.dev.mls.amls_uniform as amls_webb
from stat_reliability_measure.dev.mls.webb_config import MLS_Webb_Config
from stat_reliability_measure.dev.form.form_pyt import FORM_pyt as FORM_pyt
from stat_reliability_measure.dev.form.form_config import FORM_config
from stat_reliability_measure.dev.smc.smc_pyt import SamplerSMC, SamplerSmcMulti
from stat_reliability_measure.dev.smc.smc_config import SMCSamplerConfig
from stat_reliability_measure.dev.mc.mc_config import CrudeMC_Config
from stat_reliability_measure.dev.mc.mc_pyt import MC_pf
from stat_reliability_measure.dev.hmls.hmls_config import HMLS_Config
from stat_reliability_measure.dev.hmls.hmls_pyt import HybridMLS
from stat_reliability_measure.config import Exp2Config
from itertools import product as cartesian_product



method_config_dict={'amls':MLS_SMC_Config,'amls_webb':MLS_Webb_Config,
                    'mls_webb':MLS_Webb_Config,
                    'hmls':HMLS_Config,
                    'mala':SMCSamplerConfig,'amls_batch':MLS_SMC_Config,
                    'mc':CrudeMC_Config,'crudemc':CrudeMC_Config,'crude_mc':CrudeMC_Config,
                    'form':FORM_config,'rw_smc':SMCSamplerConfig,
                    'smc_multi':SMCSamplerConfig,
                    
                    'hmc':SMCSamplerConfig,'smc':SMCSamplerConfig,}
method_func_dict={'amls':amls_pyt.ImportanceSplittingPyt,'mala':SamplerSMC,
                  'rw_smc':SamplerSMC,
                  'hmls':HybridMLS,
                  'mc':MC_pf,'crudemc':MC_pf,'crude_mc':MC_pf, 
                  'amls_webb': amls_webb.multilevel_uniform,'form':FORM_pyt,
                  'mls_webb':amls_webb.multilevel_uniform,
                    'hmc':SamplerSMC,'smc':SamplerSMC,'smc_multi':SamplerSmcMulti, 
                    'amls_batch':amls_pyt.ImportanceSplittingPytBatch,
                    'smc_multi':SamplerSMC}

def run_est(model, X, y, method='amls_webb', epsilon_range=[], 
                     noise_dist='uniform',dataset_name = 'dataset',
                 model_name='model', 
                 verbose=0, x_min=0,x_max=1., mask_opt=False,mask_idx=None,
                 mask_cond=None,p_ref=None,
                 log_hist_=True, aggr_res_path='',batch_opt=False,
                 log_txt_=True,exp_config=None,method_config=None,
                 smc_multi=False,**kwargs):
    """ Running reliability experiments on neural network model with supervised data (X,y)
        values 
    """
    method=method.lower()
    if method=='amls' and batch_opt:
        method='amls_batch'
        if verbose>0:
            print("batch option is only available for amls_batch method")
    if mask_opt: 
        assert (mask_idx is not None) or (mask_cond is not None), "if using masking option, either 'mask_idx' or 'mask_cond' should be given"
    if method_config is None:
        method_config = method_config_dict[method]()
    if exp_config is None:
        exp_config=Exp2Config(model=model,X=X,y=y,
        dataset_name=dataset_name,model_name=model_name,epsilon_range=epsilon_range,
        aggr_res_path=aggr_res_path,x_min=x_min,x_max=x_max,mask_opt=mask_opt,
        mask_idx=mask_idx,mask_cond=mask_cond,verbose=verbose)
    else:
        exp_config.model,exp_config.X,exp_config.y,=model,X,y
        exp_config.dataset, exp_config.model_name, exp_config.epsilon_range=dataset_name,model_name,epsilon_range
        exp_config.aggr_res_path, exp_config.verbose=aggr_res_path ,float(verbose)
        exp_config.noise_dist,exp_config.x_min,exp_config.x_max, = noise_dist,x_min,x_max,
        if mask_opt:
            exp_config.mask_opt=mask_opt
            exp_config.mask_idx=mask_idx
            exp_config.mask_cond=mask_cond
        else:
            exp_config.mask_opt=False
    for k,v in kwargs.items():
        if hasattr(method_config,k):
            if hasattr(exp_config,k) and  not ('config' in k):
                raise ValueError(f"parameter {k} is in both exp_config and method_config")
            setattr(method_config, k, v)
        elif hasattr(exp_config,k):
            setattr(exp_config, k, v)
        else:
            raise ValueError(f"unknown configuration parameter {k}")
    if method=='rw_smc':
        method_config.GV_opt=True
    if method=='hmc':
        method.L=5
    method_config.update()
    exp_config.update(method_name=method_config.method_name)
    method_config.exp_config=exp_config
    param_ranges = [v for k,v in range_vars(method_config).items()]
    
    param_lens=np.array([len(l) for l in param_ranges])  
    nb_exps= np.prod(param_lens)*len(exp_config.epsilon_range)*exp_config.nb_inputs
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
    print(f"with parameters in {method_range_dict}")
    method_param_lists = range_dict_to_lists(range_dict=method_range_dict)
    i_exp=0
    if verbose>1.0:
        print(f"Experiment range parameters: {method_param_lists}")
    
    method_name=method_config.method_name
    results_df=pd.DataFrame({})
    if exp_config.aggr_res_path is None:
        aggr_res_path=os.path.join(exp_config.log_dir,'agg_res.csv')
    else: 
        aggr_res_path=exp_config.aggr_res_path
   
    if not os.path.exists(aggr_res_path):
        print(f'aggregate results csv file not found \n it will be build at {aggr_res_path}')
        cols=['method_name','gaussian_latent','N','rho','n_rep','T','epsilon','alpha','min_rate','mean_time','std_time','mean_est',
        'std_est','freq underest','g_target']
        cols+=['freq_finished','freq_zero_est','unfinished_mean_est','unfinished_mean_time']
        cols+=['pgd_success','p_l','p_u','gpu_name','cpu_name','np_seed','torch_seed','noise_dist','datetime']
        agg_res_df= pd.DataFrame(columns=cols)
    else:
        agg_res_df=pd.read_csv(aggr_res_path)
    estimation_func = method_func_dict[method]
    track_X = hasattr(method_config,'track_X') and method_config.track_X
    for l in range(len(exp_config.X)):
        with torch.no_grad():
            x_clean,y_clean = exp_config.X[l], exp_config.y[l]
        for idx in range(len(exp_config.epsilon_range)):
            exp_config.epsilon = exp_config.epsilon_range[idx]
            pgd_success= (exp_config.attack_success[idx][l]).item() if exp_config.use_attack else None 
            if exp_config.lirpa_bounds:
                from stat_reliability_measure.dev.lirpa_utils import get_lirpa_bounds
                # Step 2: define perturbation. Here we use a Linf perturbation on input image.
                p_l,p_u=get_lirpa_bounds(x_clean=x_clean,y_clean=y_clean,model=model,epsilon=exp_config.epsilon,
                num_classes=exp_config.num_classes,noise_dist=exp_config.noise_dist,a=exp_config.a,device=exp_config.device)
                exp_config.p_l,exp_config.p_u=p_l.item(),p_u.item()
            if exp_config.lirpa_cert:
                from stat_reliability_measure.dev.lirpa_utils import get_lirpa_cert
                exp_config.lirpa_safe,exp_config.time_lirpa_safe=get_lirpa_cert(x_clean=x_clean,y_clean=y_clean,
                                epsilon = exp_config.epsilon, num_classes=exp_config.num_classes                                                
                                ,model=model, device=exp_config.device)
        
            lists_cart= cartesian_product(*method_param_lists)
            for method_params in lists_cart:
                i_exp+=1
                
                vars(method_config).update(method_params)
                method_keys= list(simple_vars(method_config).keys())
                method_vals= list(simple_vars(method_config).values())
                clean_method_args_str = str(method_params).replace('{','').replace('}','').replace("'",'').replace('(',"").replace(')',"").replace(',',':')
                aggr_res_path=os.path.join(exp_config.log_dir,'aggr_res.csv')
                if exp_config.update_aggr_res and os.path.exists(exp_config.aggr_res_path):
                    aggr_res_df = pd.read_csv(exp_config.aggr_res_path)
                if exp_config.update_aggr_res:
                    if os.path.exists(exp_config.aggr_res_path):
                        aggr_res_df = pd.read_csv(exp_config.aggr_res_path)
                        if 'exp_id' in aggr_res_df.columns:
                            
                            last_exp_id = int(aggr_res_df['exp_id'].max())
                        else:
                            last_exp_id = -1
                    else:
                        last_exp_id = -1
                    
                    if not exp_config.repeat_exp and os.path.exists(exp_config.aggr_res_path):
                        aggr_res_df = pd.read_csv(exp_config.aggr_res_path)
                        if len(aggr_res_df)>0:
                            same_method_df = get_sel_df(df=aggr_res_df,triplets=[('method_name',method_config.method_name,'=')])
                            if len(same_method_df)>0:
                                print("Experiment already done for method: "+method_config.method_name)
                                try:
                                    triplets=[('model_name',exp_config.model_name,'='),('epsilon',exp_config.epsilon,'='),
                                    ('image_idx',l,'='),('n_rep',exp_config.n_rep,'='),('noise_dist',exp_config.noise_dist,'=')]

                                    same_exp_df = get_sel_df(df=same_method_df, cols=method_keys, vals=method_vals, 
                                    triplets =triplets)
                                    # if a similar experiment has been done in the current log directory we skip it

                                    if len(same_exp_df)>0:
                                        print(f"Skipping {method_config.method_name} run {i_exp}/{nb_exps}, with model: {exp_config.model_name}, img_idx:{l},eps:{exp_config.epsilon},"+clean_method_args_str)
                                        continue
                                except KeyError:
                                    if exp_config.verbose>=1:
                                        print(f"No similar experiment found for {method_config.method_name} run {i_exp}/{nb_exps}, with model: {exp_config.model_name}, img_idx:{l},eps:{exp_config.epsilon},"+clean_method_args_str)
                                    exp_config.exp_id = last_exp_id+1
                            else:
                                exp_config.exp_id = last_exp_id+1
                        else:
                            exp_config.exp_id = last_exp_id+1
                    else:
                        exp_config.exp_id = last_exp_id+1
                    if not hasattr(exp_config,'exp_id'):
                        exp_config.exp_id=last_exp_id+1
                loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
                log_name=method_name+'_e_'+float_to_file_float(exp_config.epsilon_range[idx])+f'_n_{exp_config.n_rep}'+loc_time
               
                log_path=os.path.join(exp_config.exp_log_path,log_name)
                
                print(f"Starting {method_config.method_name} simulation {i_exp}/{nb_exps}, with model: {exp_config.model_name}, img_idx:{l},eps:{exp_config.epsilon},"+clean_method_args_str)
                                
             
                if exp_config.verbose>0:
                    print(f"with model: {model_name}, img_idx:{l},eps:{exp_config.epsilon},")
              
                times= []
                rel_error= []
                ests = [] 
                log_ests=[]
                calls=[]
                
                if hasattr(method_config,'track_finish') and method_config.track_finish:
                    finish_flags=[]
                    track_finish=True
                else:
                    track_finish = False
                if hasattr(method_config,'track_levels') and method_config.track_levels:
                    track_levels=True
                    levels_list=[]
                    max_iter=0
                else:
                    track_levels=False
                if track_X:
                    X_list=[]
                if hasattr(method_config,'track_advs') and method_config.track_advs:
                    track_advs=True
                    advs_list = []
                else:
                    track_advs=False
                    advs_list=None
                if exp_config.tqdm_opt:
                    if exp_config.notebook:
                        from tqdm.notebook import tqdm
                    else:
                        from tqdm import tqdm
                    iterator = tqdm(range(exp_config.n_rep))
                else: 
                    iterator = range(exp_config.n_rep)

                exp_required = {'verbose':exp_config.verbose}
                requires_keys = [k for k in simple_vars(method_config).keys() if 'requires_' in k]
                for k in requires_keys:
                    if getattr(method_config,k):
                        required_key = k.replace('requires_','')
                        exp_required[required_key]=getattr(exp_config,required_key)
                
                #selecting only method configuration variables relevent to the estimation function
                func_args_vars = {k:simple_vars(method_config)[k] for k in simple_vars(method_config).keys() if ('require' not in k) and ('name' not in k) and ('calls' not in k)}
                args_dict = {**func_args_vars,**exp_required}
                method_config.update()
                for i in iterator:
                    t=time()
                    p_est,dict_out=estimation_func(**args_dict,)
                    t=time()-t
                    
                    
                    nb_calls=dict_out['nb_calls']
                    ests.append(p_est)
                    log_est= np.log(p_est) if p_est>0 else -250.
                    log_ests.append(log_est)
                    times.append(t)
                    calls.append(nb_calls)
                    if exp_config.verbose>1:
                        print(f"Est:{p_est}")
                    if hasattr(method_config,'track_accept'):
                        if method_config.track_accept:
                            accept_logs=os.path.join(log_path,'accept_logs')
                            if not os.path.exists(accept_logs):
                                os.mkdir(path=accept_logs)
                            accept_rates=dict_out['acc_ratios']
                            np.savetxt(fname=os.path.join(accept_logs,f'accept_rates_{i}.txt')
                            ,X=accept_rates)
                            x_T=np.arange(len(accept_rates))
                            plt.plot(x_T,accept_rates)
                            plt.savefig(os.path.join(accept_logs,f'accept_rates_{i}.png'))
                            plt.close()
                            exp_config.mean_acc_ratio = np.array(dict_out['acc_ratios']).mean()
                    if track_finish:
                        finish_flags.append(dict_out['finish_flag'])
                    if track_levels:
                        new_levels= dict_out['levels']
                        if len(new_levels)>max_iter:
                            for i in range(max_iter):
                                levels_list[i].append(new_levels[i])
                            for i in range(max_iter,len(new_levels)):
                                levels_list.append([new_levels[i]])
                            max_iter=len(new_levels)
                    if track_X:
                        X_list.append(dict_out['X'])
                    if track_advs:
                        advs_list.append(dict_out['advs'])

                    del dict_out
    
                times=np.array(times)
                ests = np.array(ests)
                log_ests=np.array(log_ests)
                mean_est=ests.mean()
                calls=np.array(calls)
                mean_calls=calls.mean()
                std_est=ests.std()
                if track_levels:
                    mean_levels = [np.array(l).mean() for l in levels_list]
                    std_levels = [np.array(l).std() for l in levels_list]  

                
                
                q_1,med_est,q_3=np.quantile(a=ests,q=[0.25,0.5,0.75])
                std_rel=std_est/mean_est**2 if mean_est >0 else 0
                std_rel_adj=std_rel*mean_calls
                print(f"mean est:{ests.mean()}, std est:{ests.std()}")
                print(f"mean calls:{calls.mean()}")
                print(f"std. re.:{std_rel}")
                print(f"std. rel. adj.:{std_rel_adj}")
                if p_ref is not None:
                    rel_error = np.abs(ests-p_ref)/p_ref
                    print(f"mean rel. error:{rel_error.mean()}")
                    print(f"std rel. error:{rel_error.std()}")
                    print(f"stat performance (per 1k calls):{rel_error.std()*(mean_calls/1000)}")
                
                if track_finish:
                    finish_flags=np.array(finish_flags)
                    freq_finished=finish_flags.mean()
                    freq_zero_est=(ests==0).mean()
                else:
                    freq_zero_est,freq_finished=None,None
                #finished=np.array(finish_flag)
                if track_finish and freq_finished<1:
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
                    if track_levels:
                        mean_levels_path=os.path.join(log_path,'mean_levels.txt')
                        std_levels_path=os.path.join(log_path,'std_levels.txt')
                        exp_config.levels_path=mean_levels_path
                        np.savetxt(fname=mean_levels_path,X=mean_levels)
                        np.savetxt(fname=std_levels_path,X=std_levels)

                if log_hist_:
                    plt.hist(times, bins=10)
                    plt.savefig(os.path.join(log_path,'times_hist.png'))
                    plt.close()
                    plt.hist(ests,bins=10)
                    plt.savefig(os.path.join(log_path,'ests_hist.png'))
                    plt.close()
                    plt.hist(log_ests,bins=10)
                    plt.savefig(os.path.join(log_path,'log_ests_hist.png'))
                    plt.close()
                    if track_levels:
                        T = len(mean_levels)
                        plt.errorbar(x=np.arange(T),y=mean_levels,yerr=std_levels)
                        plt.savefig(os.path.join(log_path,'levels.png'))
                        exp_config.levels_png=os.path.join(log_path,'levels.png')
                        plt.close()
                #with open(os.path.join(log_path,'results.txt'),'w'):
                result={"image_idx":l,'mean_calls':calls.mean(),'std_calls':calls.std()
                ,'mean_time':times.mean(),'std_time':times.std()
                ,'mean_est':ests.mean(),'std_est':ests.std(), 'est_path':est_path,'times_path':times_path,
                "std_rel":std_rel,"std_rel_adj":std_rel*mean_calls,
               
                'q_1':q_1,'q_3':q_3,'med_est':med_est,"lg_est_path":lg_est_path,
                "mean_log_est":mean_log_est,"std_log_est":std_log_est,
                "lg_q_1":lg_q_1,"lg_q_3":lg_q_3,"lg_med_est":lg_med_est,
                "log_path":log_path,"log_name":log_name}
                if track_finish:
                    result.update({"freq_finished":freq_finished,"freq_zero_est":freq_zero_est,
                    "unfinished_mean_est":unfinished_mean_est,"unfinished_mean_time":unfinished_mean_time})
           
        
                if p_ref is not None:
                    result.update({"rel_error":rel_error.mean(),"std_rel_error":rel_error.std(),
                                   "p_ref":p_ref})
                result.update(simple_vars(method_config))
                result.update(simple_vars(exp_config))
                
                result_df=pd.DataFrame([result])
                results_df = pd.concat([results_df,result_df],ignore_index=True)
                results_df.to_csv(os.path.join(exp_config.exp_log_path,'results.csv'),)
                
                if exp_config.update_aggr_res:
                    agg_res_df = pd.concat([agg_res_df,result_df],ignore_index=True)
                    agg_res_df.to_csv(aggr_res_path,index=False)

                

    if len(results_df)>0:
        p_est = results_df['mean_est'].mean()
    else:
        p_est = None
    dict_out = {'results_df':results_df,'agg_res_df':agg_res_df,
                'method_config':method_config, 'exp_config':exp_config}
    if track_X: 
        dict_out['X_list']=X_list
    if track_advs:
        dict_out['advs_list']=advs_list
    return p_est, dict_out