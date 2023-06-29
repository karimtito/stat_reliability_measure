import stat_reliability_measure.dev.torch_utils as t_u
import stat_reliability_measure.dev.smc.smc_pyt as smc_pyt
import numpy as np
from tqdm import tqdm
from time import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
from stat_reliability_measure.dev.utils import float_to_file_float
from stat_reliability_measure.dev.utils import get_sel_df, simple_vars, range_vars, range_dict_to_lists
from stat_reliability_measure.config import Exp2Config
from itertools import product as cartesian_product
def main():
    method_config=smc_pyt.SMCSamplerConfig()
    exp_config=Exp2Config()
    parser= exp_config.get_parser()
    parser = method_config.add_parsargs(parser)
    args=parser.parse_args()
    for k,v in vars(args).items():
        if hasattr(method_config,k):
            if hasattr(exp_config,k) and  not ('config' in k):
                raise ValueError(f"parameter {k} is in both exp_config and method_config")
            setattr(method_config, k, v)
        elif hasattr(exp_config,k):
            setattr(exp_config, k, v)
        else:
            raise ValueError(f"unknown method_config parameter {k}")
    method_config.update()
    exp_config.update(method_name = method_config.get_method_name())

    method_config.exp_config=exp_config
    param_ranges = [method_config.N_range,method_config.T_range,method_config.L_range,method_config.ess_alpha_range,method_config.alpha_range,
                    exp_config.epsilon_range]

    
    param_lens=np.array([len(l) for l in param_ranges])
    param_cart = cartesian_product(param_ranges)
    nb_runs= np.prod(param_lens)*len(exp_config.epsilon_range)*exp_config.nb_inputs
    method=method_config.method_name
    print(f"Running reliability experiments on architecture {exp_config.model_arch} trained on {exp_config.dataset}.")
    print(f"Testing {exp_config.noise_dist} noise pertubation with epsilon in {exp_config.epsilon_range}")

    #exp_config.print_config()
    #method_config.print_config()
    exp_config.config_path=os.path.join(exp_config.exp_log_path,'exp_config.json')
    exp_config.to_json()
    method_config.config_path=os.path.join(exp_config.exp_log_path,'method_config.json')
    method_config.to_json()
    method_range_dict = range_vars(method_config)
    method_param_lists = range_dict_to_lists(range_dict=method_range_dict)


    run_nb=0
    iterator= tqdm(range(exp_config.n_rep))
    exp_res=[]
    for l in range(exp_config.input_start,exp_config.input_stop):
        print(exp_config.nb_inputs)
        with torch.no_grad():
            x_0,y_0 = exp_config.X[l], exp_config.y[l]
        x_0.requires_grad=True
        exp_config.x_0=x_0
        exp_config.y_0=y_0
        for idx in range(len(exp_config.epsilon_range)):
            
            
            exp_config.epsilon = exp_config.epsilon_range[idx]

            if exp_config.use_attack:
                pgd_success= (exp_config.attack_success[idx][l]).item()
            p_l,p_u=None,None
            if exp_config.lirpa_bounds:
                from stat_reliability_measure.dev.lirpa_utils import get_lirpa_bounds
                # Step 2: define perturbation. Here we use a Linf perturbation on input image.
                p_l,p_u=get_lirpa_bounds(x_0=x_0,y_0=y_0,model=exp_config.model,epsilon=exp_config.epsilon,
                num_classes=exp_config.num_classes,noise_dist=exp_config.noise_dist,a=method_config.a,device=exp_config.device)
                p_l,p_u=p_l.item(),p_u.item()
            lirpa_safe=None
            if exp_config.lirpa_cert:
                assert exp_config.noise_dist.lower()=='uniform',"Formal certification only makes sense for uniform distributions"
                from stat_reliability_measure.dev.lirpa_utils import get_lirpa_cert
                lirpa_safe=get_lirpa_cert(x_0=x_0,y_0=y_0,model=exp_config.model,epsilon=exp_config.epsilon,
                num_classes=exp_config.num_classes,device=exp_config.device)
            lists_cart= cartesian_product(*method_param_lists)
            for method_params in lists_cart:
                exp_config.method_name=method_config.get_method_name()
                if exp_config.update_aggr_res and os.path.exists(exp_config.aggr_res_path):
                    aggr_res_df = pd.read_csv(exp_config.aggr_res_path)
                vars(method_config).update(method_params)
                method_keys= list(simple_vars(method_config).keys())
                method_vals = list(simple_vars(method_config).values())
                run_nb+=1
                if exp_config.update_aggr_res:
                    if os.path.exists(exp_config.aggr_res_path):
                        aggr_res_df = pd.read_csv(exp_config.aggr_res_path)
                        if 'exp_id' in aggr_res_df.columns:
                            last_exp_id = aggr_res_df['exp_id'].max()
                        else:
                            last_exp_id = -1
                    else:
                        last_exp_id = -1
                    if not exp_config.repeat_exp and os.path.exists(exp_config.aggr_res_path):
                        aggr_res_df = pd.read_csv(exp_config.aggr_res_path)
                        if len(aggr_res_df)>0:
                            same_method_df = get_sel_df(df=aggr_res_df,triplets=[('method_name',method,'=')])
                            if len(same_method_df)>0:
                                try:
                                    same_exp_df = get_sel_df(df=same_method_df, cols=method_keys, vals=method_vals, 
                                    triplets =[('model_name',exp_config.model_name,'='),('epsilon',exp_config.epsilon,'='),
                                    ('image_idx',l,'='),('n_rep',exp_config.n_rep,'='),('noise_dist',exp_config.noise_dist,'=')])
                                    # if a similar experiment has been done in the current log directory we skip it

                                    if len(same_exp_df)>0:
                                        print(f"Skipping {method_config.method_name} run {run_nb}/{nb_runs}, with model: {exp_config.model_name}, img_idx:{l},eps:{exp_config.epsilon},"+str(method_params).replace('{','').replace('}','').replace("'",''))
                                        continue
                                    else:
                                        print(f"No similar experiment found")
                                except KeyError:
                                    if exp_config.verbose>=5:
                                        print(f"No similar experiment found for {method_config.method_name} run {run_nb}/{nb_runs}, with model: {exp_config.model_name}, img_idx:{l},eps:{exp_config.epsilon},"+str(method_params).replace('{','').replace('}','').replace("'",''))
                                    exp_config.exp_id = last_exp_id+1
                    else:
                        exp_config.exp_id = last_exp_id+1
                
                #loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
            
                log_name=method_config.method_name+f'_N_{method_config.N}_T_{method_config.T}_L_{method_config.L}_a_{float_to_file_float(method_config.alpha)}_ess_{float_to_file_float(method_config.ess_alpha)}'
                log_path=os.path.join(exp_config.exp_log_path,log_name)
                log_path_prop=log_path
                k=1 
                index_level=0
                while os.path.exists(log_path_prop) and index_level<5:
                    if k==10: 
                        log_path= log_path_prop
                        k=1
                        index_level+=1
                    log_path_prop = log_path+'_'+str(k) 
                    k+=1 
                log_path = log_path_prop
                os.makedirs(log_path,exist_ok=True)
                times=[]
                ests = []
                calls = []
                finished_flags=[]
                iterator= tqdm(range(exp_config.n_rep)) if exp_config.tqdm_opt else range(exp_config.n_rep)
                print(f"Starting {method} simulations {run_nb}/{nb_runs} with model:{exp_config.model_name} img_idx:{l},eps={exp_config.epsilon}, ")
                print(f"Saving results in {log_path}")
                for i in iterator:
                    t=time()
                    p_est,res_dict,=smc_pyt.SamplerSMC(gen=exp_config.gen,V= method_config.V_classif,
                                    adapt_func = method_config.adapt_func,
                                    gradV=method_config.gradV_classif,**simple_vars(method_config))
                    
                    t1=time()-t
                    if exp_config.verbose>2:
                        print(f"est:{p_est}")
                    #finish_flag=res_dict['finished']
                    
                    if method_config.track_accept:
                        accept_logs=os.path.join(log_path,'accept_logs')
                        if not os.path.exists(accept_logs):
                            os.mkdir(path=accept_logs)
                        accept_rates_mcmc=res_dict['accept_rates_mcmc']
                        np.savetxt(fname=os.path.join(accept_logs,f'accept_rates_mcmc_{i}.txt')
                        ,X=accept_rates_mcmc,)
                        x_T=np.arange(len(accept_rates_mcmc))
                        plt.plot(x_T,accept_rates_mcmc)
                        plt.savefig(os.path.join(accept_logs,f'accept_rates_mcmc_{i}.png'))
                        plt.close()
                        

                    if (method_config.adapt_dt or method_config.adapt_step) and method_config.track_dt:
                        dt_means=res_dict['dt_means']
                        dt_stds=res_dict['dt_stds']
                        dts_path=os.path.join(log_path,'dts')
                        if not os.path.exists(dts_path):
                            os.mkdir(path=dts_path)
                        np.savetxt(fname=os.path.join(dts_path,f'dt_means_{i}.txt'),X=dt_means)
                        np.savetxt(fname=os.path.join(dts_path,f'dt_stds_{i}.txt'),X=dt_stds)
                        x_T=np.arange(len(dt_means))
                        plt.errorbar(x_T,dt_means,yerr=dt_stds,label='dt')
                        plt.savefig(os.path.join(dts_path,f'dt_{i}.png'))
                        plt.close()
                    
                    
                    times.append(t1)
                    ests.append(p_est)
                    calls.append(res_dict['calls'])
                times=np.array(times)
                ests = np.array(ests)
                calls=np.array(calls)
            
                mean_calls=calls.mean()
                std_est=ests.std()
                mean_est=ests.mean()
                std_rel=std_est/mean_est
                std_rel_adj=std_rel*mean_calls
                print(f"mean est:{ests.mean()}, std est:{ests.std()}")
                print(f"mean calls:{calls.mean()}")
                print(f"std. rel.:{std_rel}")
                print(f"std. rel. adj.:{std_rel_adj}")
                q_1,med_est,q_3=np.quantile(a=ests,q=[0.25,0.5,0.75])
                times=np.array(times)  
                ests=np.array(ests)
                log_ests=np.log(np.clip(ests,a_min=1e-250,a_max=1))
                std_log_est=log_ests.std()
                mean_log_est=log_ests.mean()
                lg_q_1,lg_med_est,lg_q_3=np.quantile(a=ests,q=[0.25,0.5,0.75])
                lg_est_path=os.path.join(log_path,'lg_ests.txt')
                np.savetxt(fname=lg_est_path,X=ests)
                    #fin = np.array(finished_flags)

                times_path=os.path.join(log_path,'times.txt')
                np.savetxt(fname=times_path,X=times)
                est_path=os.path.join(log_path,'ests.txt')
                np.savetxt(fname=est_path,X=ests)

                plt.hist(times, bins=20)
                plt.savefig(os.path.join(log_path,'times_hist.png'))
                plt.close()
                
                plt.hist(times, bins=20)
                plt.savefig(os.path.join(log_path,'times_hist.png'))
                plt.close()

                #with open(os.path.join(log_path,'results.txt'),'w'):
            
                results={"image_idx":l,'mean_calls':calls.mean(),'std_calls':calls.std()
                ,'mean_time':times.mean(),'std_time':times.std()
                ,'mean_est':ests.mean(),'std_est':ests.std(), 'est_path':est_path,'times_path':times_path,
                "std_rel":std_rel,"std_rel_adj":std_rel*mean_calls,
                "lirpa_safe":lirpa_safe,
                'q_1':q_1,'q_3':q_3,'med_est':med_est,"lg_est_path":lg_est_path,
                "mean_log_est":mean_log_est,"std_log_est":std_log_est,
                "lg_q_1":lg_q_1,"lg_q_3":lg_q_3,"lg_med_est":lg_med_est,
                "log_path":log_path,"log_name":log_name,}
                results.update(simple_vars(method_config))
                results.update(simple_vars(exp_config))
                exp_res.append(results)
                results_df=pd.DataFrame([results])
                
                
                if exp_config.update_aggr_res:
                    if not os.path.exists(exp_config.aggr_res_path):
                        cols=['method_name','N','rho','ratio','n_rep','T','alpha','min_rate','mean_time','std_time','mean_est',
                        'bias','mean abs error','mean_rel_error','std_est','freq underest','gpu_name','cpu_name']
                        aggr_res_df= pd.DataFrame(columns=cols)
                    else:
                        aggr_res_df=pd.read_csv(exp_config.aggr_res_path)
                    aggr_res_df = pd.concat([aggr_res_df,results_df],ignore_index=True)
                    aggr_res_df.to_csv(exp_config.aggr_res_path,index=False)


    exp_df=pd.DataFrame(exp_res)
    exp_df.to_csv(os.path.join(exp_config.exp_log_path,'exp_results.csv'),index=False)  

if __name__ == '__main__':
    main()