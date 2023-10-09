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
from stat_reliability_measure.config import ExpModelConfig
from itertools import product as cartesian_product

def run_amls_exp(model, X, y, epsilon_range=[], noise_dist='uniform',dataset_name = 'dataset',
                 model_name='model',
                 verbose=0.1, x_min=0,x_max=1., mask_opt=False,mask_idx=None,mask_cond=None,
                 log_hist_=True, aggr_res_path='',
                 log_txt_=True,exp_config=None,method_config=None,**kwargs):
    """ Running reliability experiments on neural network model with supervised data (X,y)
        values 
    """
    if mask_opt: 
        assert (mask_idx is not None) or (mask_cond is not None), "if using masking option, either 'mask_idx' or 'mask_cond' should be given"
    if method_config is None:
        method_config = MLS_SMC_Config()
        method_config.verbose=verbose
    if exp_config is None:
        exp_config=ExpModelConfig(model=model,X=X,y=y,
        dataset_name=dataset_name,model_name=model_name,epsilon_range=epsilon_range,
        aggr_res_path=aggr_res_path,x_min=x_min,x_max=x_max,mask_opt=mask_opt,
        mask_idx=mask_idx,mask_cond=mask_cond,verbose=verbose)
    else:
        exp_config.model,exp_config.X,exp_config.y,=model,X,y
        exp_config.dataset, exp_config.model_name, exp_config.epsilon_range=dataset_name,model_name,epsilon_range
        exp_config.aggr_res_path, exp_config.verbose=aggr_res_path ,verbose
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
    method_config.update()
    exp_config.update(method_name=method_config.method_name)
    

    method_config.exp_config=exp_config
    param_ranges = [v for k,v in range_vars(method_config).items()]

    
    param_lens=np.array([len(l) for l in param_ranges])

    param_cart = cartesian_product(param_ranges)
  
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
    method_param_lists = range_dict_to_lists(range_dict=method_range_dict)
    i_exp=0
    
    if verbose>1.0:
        print(f"Experiment range parameters: {method_param_lists}")
    
    method_name="MLS_SMC"
    results_df=pd.DataFrame({})
    if exp_config.aggr_res_path is None:
        aggr_res_path=os.path.join(exp_config.log_dir,'agg_res.csv')
    else: 
        aggr_res_path=exp_config.aggr_res_path
   
    if not os.path.exists(aggr_res_path):
        print(f'aggregate results csv file not found \n it will be build at {aggr_res_path}')
        cols=['method_name','from_gaussian','N','ratio','n_rep','T','epsilon','alpha','min_rate','mean_time','std_time','mean_est',
        'std_est','freq underest','g_target']
        cols+=['freq_finished','freq_zero_est','unfinished_mean_est','unfinished_mean_time']
        cols+=['pgd_success','p_l','p_u','gpu_name','cpu_name','np_seed','torch_seed','noise_dist','datetime']
        agg_res_df= pd.DataFrame(columns=cols)

    else:
        agg_res_df=pd.read_csv(aggr_res_path)
    use_cuda= "cuda" in exp_config.device
    
    
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
            def prop(x):
                y = model(x)
                y_diff = torch.cat((y[:,:y_clean], y[:,(y_clean+1):]),dim=1) - y[:,y_clean].unsqueeze(-1)
                y_diff, _ = y_diff.max(dim=1)
                return y_diff #.max(dim=1)
                
            lists_cart= cartesian_product(*method_param_lists)
            for method_params in lists_cart:
                i_exp+=1
                
                vars(method_config).update(method_params)
                method_keys= list(simple_vars(method_config).keys())
                method_vals = list(simple_vars(method_config).values())
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
                                try:
                                    same_exp_df = get_sel_df(df=same_method_df, cols=method_keys, vals=method_vals, 
                                    triplets =[('model_name',exp_config.model_name,'='),('epsilon',exp_config.epsilon,'='),
                                    ('image_idx',l,'='),('n_rep',exp_config.n_rep,'='),('noise_dist',exp_config.noise_dist,'=')])
                                    # if a similar experiment has been done in the current log directory we skip it

                                    if len(same_exp_df)>0:
                                        print(f"Skipping {method_config.method_name} run {i_exp}/{nb_exps}, with model: {exp_config.model_name}, img_idx:{l},eps:{exp_config.epsilon},"+str(method_params).replace('{','').replace('}','').replace("'",''))
                                        continue
                                except KeyError:
                                    if exp_config.verbose>=5:
                                        print(f"No similar experiment found for {method_config.method_name} run {i_exp}/{nb_exps}, with model: {exp_config.model_name}, img_idx:{l},eps:{exp_config.epsilon},"+str(method_params).replace('{','').replace('}','').replace("'",''))
                    exp_config.exp_id = last_exp_id+1
                loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
                log_name=method_name+'_e_'+float_to_file_float(exp_config.epsilon_range[idx])+f'_n_{exp_config.n_rep}'+'_N_'+str(method_config.N)+'_T_'+str(method_config.T)+'_s_'+str(method_config.s)+'_K_'+str(method_config.K)+'_r_'+float_to_file_float(method_config.ratio)+'_'+loc_time
               
                log_path=os.path.join(exp_config.exp_log_path,log_name)
                
                print(f"Starting {method_name} run {i_exp}/{nb_exps}")
                                
                
                K=int(method_config.N*method_config.ratio) if not method_config.last_particle else method_config.N-1
                if exp_config.verbose>0:
                    print(f"with model: {model_name}, img_idx:{l},eps:{exp_config.epsilon},T:{method_config.T},N:{method_config.N},s:{method_config.s},K:{method_config.K}")
                if exp_config.verbose>3:
                    print(f"K/N:{method_config.K/float(method_config.N)})")
                times= []
                rel_error= []
                ests = [] 
                log_ests=[]
                calls=[]
                if method_config.track_finish:
                    finish_flags=[]
                if exp_config.tqdm_opt:
                    if exp_config.notebook:
                        from tqdm.notebook import tqdm
                    else:
                        from tqdm import tqdm
                    iterator = tqdm(range(exp_config.n_rep))
                else: 
                    iterator = range(exp_config.n_rep)

                for i in tqdm(range(exp_config.n_rep)):
                    t=time()
                    p_est,dict_out=amls_mls.multilevel_uniform(
                        score=exp_config.score,
                  N=method_config.N,T=method_config.T,x_min=exp_config.x_min,
                    x_max=exp_config.x_max,
                    x_clean=x_clean,epsilon=exp_config.epsilon,ratio=method_config.ratio,CUDA=use_cuda,
                    verbose=exp_config.verbose
                    )

                    t=time()-t
                    # we don't need adversarial examples and highest score
                    log_est=np.log(p_est) if p_est>0 else -250.0
                    log_ests.append(log_est) 
                    if exp_config.verbose>1:
                        print(f"Est:{est}")
                   
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
                    if method_config.track_finish:
                        finish_flags.append(levels[-1]>=0)
                    times.append(t)
                    ests.append(p_est)
                    calls.append(dict_out['nb_calls'])
    
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
                log10_ests=log_ests/np.log(10)
                std_log10_est=log10_ests.std()
                mean_log10_est=log10_ests.mean()
                mean_log_est=log_ests.mean()
                lg_q_1,lg_med_est,lg_q_3=np.quantile(a=ests,q=[0.25,0.5,0.75])
                lg10_q_1,lg10_med_est,lg10_q_3=np.quantile(a=log10_ests,q=[0.25,0.5,0.75])
                lg_est_path=os.path.join(log_path,'log10_ests.txt')
                
                if log_txt_:
                    np.savetxt(fname=times_path,X=times)
                    np.savetxt(fname=est_path,X=ests)
                    np.savetxt(fname=lg_est_path,X=log_ests/np.log(10))
                if log_hist_:
                    plt.hist(times, bins=exp_config.n_rep//5)
                    plt.savefig(os.path.join(log_path,'times_hist.png'))
                    plt.close()
                    plt.hist(ests,bins=exp_config.n_rep//5)
                    plt.savefig(os.path.join(log_path,'ests_hist.png'))
                    plt.close()
                    plt.hist(log_ests/np.log(10),bins=exp_config.n_rep//5)
                    plt.savefig(os.path.join(log_path,'log10_ests_hist.png'))
                    plt.close()
                #with open(os.path.join(log_path,'results.txt'),'w'):
                result={"image_idx":l,'mean_calls':calls.mean(),'std_calls':calls.std()
                ,'mean_time':times.mean(),'std_time':times.std()
                ,'mean_est':ests.mean(),'std_est':ests.std(), 'est_path':est_path,'times_path':times_path,
                "std_rel":std_rel,"std_rel_adj":std_rel*mean_calls,
               
                'q_1':q_1,'q_3':q_3,'med_est':med_est,"lg_est_path":lg_est_path,
                "mean_log_est":mean_log_est,"std_log_est":std_log_est,
                "lg_q_1":lg_q_1,"lg_q_3":lg_q_3,"lg_med_est":lg_med_est,
                "mean_log10_est":mean_log10_est,"std_log10_est":std_log10_est,
                "lg10_q_1":lg10_q_1,"lg10_q_3":lg10_q_3,"lg10_med_est":lg10_med_est,
                "log_path":log_path,"log_name":log_name}
                result.update(simple_vars(method_config))
                result.update(simple_vars(exp_config))
                
                result_df=pd.DataFrame([result])
                results_df = pd.concat([results_df,result_df],ignore_index=True)
                results_df.to_csv(os.path.join(exp_config.exp_log_path,'results.csv'),)
                
                if exp_config.update_aggr_res:
                    agg_res_df = pd.concat([agg_res_df,result_df],ignore_index=True)
                    agg_res_df.to_csv(aggr_res_path,index=False)

                


    
    return results_df, exp_config, method_config, agg_res_df
