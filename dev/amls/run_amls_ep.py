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
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from stat_reliability_measure.dev.utils import  float_to_file_float,str2bool,str2intList,str2floatList, str2list
from stat_reliability_measure.dev.utils import get_sel_df, print_config
import stat_reliability_measure.dev.mls.amls_uniform as amls_mls
from stat_reliability_measure.dev.amls.amls_utils import base_config
str2floatList=lambda x: str2list(in_str=x, type_out=float)
str2intList=lambda x: str2list(in_str=x, type_out=int)
low_str=lambda x: str(x).lower()


def run_amls_exp(model, X, y, epsilons=None, dataset_name = None,model_name=None, x_min=0,x_max=None, mask_opt=False,mask_vale=0,
                 log_hist_=False, agrr_res_path=None, 
                 log_txt_=False,dict_arg=None):
    
    config = base_config
    if dict_arg is not None:
        for k,v in dict_arg.items():
            setattr(config, k, v)
    if config.device is None:
        config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if config.verbose>=5:
            print(config.device)
        device=config.device
    else:
        device=config.device

    if not config.allow_multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
    torch.manual_seed(seed=config.torch_seed)
    np.random.seed(seed=config.np_seed)
    if config.aggr_res_path is None:
        aggr_res_path=os.path.join(config.log_dir,'agg_res.csv')
    else:
        aggr_res_path=config.aggr_res_path
    if not os.path.exists(ROOT_DIR+'/logs'):
        print('logs not found')
        os.mkdir(ROOT_DIR+'/logs')
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    raw_logs_path=os.path.join(config.log_dir,'raw_logs/'+config.method_name)
    if not os.path.exists(raw_logs_path):
        os.mkdir(raw_logs_path)
    loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
    log_name=config.method_name+'_'+'_'+loc_time
    exp_log_path=os.path.join(raw_logs_path,log_name)
    if os.path.exists(path=exp_log_path):
        exp_log_path = exp_log_path+'_'+str(np.random.randint(low=0,high=9))
    os.mkdir(path=exp_log_path)

    d = np.prod(X.shape[1:])
    config.d = d
    config_dict = print_config(config)
    config_path=os.path.join(exp_log_path,'config.json')
    with open(config_path,'w') as f:
        f.write(json.dumps(config_dict, indent = 4))
    
    X_correct,label_correct,accuracy,num_classes = t_u.get_x_y_accuracy_num_cl(X,y,model)
    if config.verbose>=2:
        print(f"model accuracy on test batch:{accuracy}")
    if config.use_attack:

        import foolbox as fb
        fmodel = fb.PyTorchModel(model, bounds=(x_min,x_max))
        attack=fb.attacks.LinfPGD()
        
    
        _, advs, success = attack(fmodel, X_correct[config.input_start:config.input_stop], 
        label_correct[config.input_start:config.input_stop], epsilons=config.epsilons)
    inp_indices=np.arange(start=config.input_start,stop=config.input_stop)
    i_exp=0
    param_ranges= [ inp_indices,config.T_range,config.N_range,config.s_range ,config.ratio_range,config.epsilons]
    lenghts=np.array([len(L) for L in param_ranges])
    nb_exps= np.prod(lenghts)
    method_name="MLS_SMC"
    if model_name is None:
        model_name=str(type(model))
    if dataset_name is None:
        dataset_name="dataset"
    for l in inp_indices:
        with torch.no_grad():
            x_0,y_0 = X_correct[l], label_correct[l]
        input_shape=x_0.shape
        #x_0.requires_grad=True
        for idx in range(len(config.epsilons)):
            
            
            epsilon = config.epsilons[idx]
            pgd_success= (success[idx][l]).item() if config.use_attack else None 
            p_l,p_u=None,None
            if config.lirpa_bounds:
                from stat_reliability_measure.dev.lirpa_utils import get_lirpa_bounds
                # Step 2: define perturbation. Here we use a Linf perturbation on input image.
                p_l,p_u=get_lirpa_bounds(x_0=x_0,y_0=y_0,model=model,epsilon=epsilon,
                num_classes=num_classes,noise_dist=config.noise_dist,a=config.a,device=config.device)
                p_l,p_u=p_l.item(),p_u.item()
            def prop(x):
                y = model(x)
                y_diff = torch.cat((y[:,:y_0], y[:,(y_0+1):]),dim=1) - y[:,y_0].unsqueeze(-1)
                y_diff, _ = y_diff.max(dim=1)
                return y_diff #.max(dim=1)
                
            for T in config.T_range:
                if T>=200:
                        N_range=config.N_range_alt
                else:
                    N_range=config.N_range
                for N in N_range: 
                    for s in config.s_range :
                        for ratio in config.ratio_range: 
                            i_exp+=1
                            aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                            if (not config.repeat_exp) and config.update_aggr_res and os.path.exists(aggr_res_path):
                                aggr_res_df = pd.read_csv(aggr_res_path)
                                same_exp_df = get_sel_df(df=aggr_res_df,triplets=[('method',method_name,'='),
                                ('model_name',model_name,'='),('epsilon',epsilon,'='),('image_idx',l,'='),('n_rep',config.n_rep,'='),
                    ('N',N,'='),('T',T,'='),('s',s,'='),('last_particle',config.last_particle,'=='),
                    ('ratio',ratio,'=')] )  
                                # if a similar experiment has been done in the current log directory we skip it
                                if len(same_exp_df)>0 and config.verbose>0:
                                    print(f"Skipping {method_name} run {i_exp}/{nb_exps}")
                                    print(f"with model: {model_name}, img_idx:{l},eps:{epsilon},T:{T},N:{N},s:{s},K:{K}")
                                    continue
                            loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
                            log_name=method_name+'_e_'+float_to_file_float(config.epsilons[idx])+'_N_'+str(N)+'_T_'+str(T)+'_s_'+float_to_file_float(s)
                            log_name=log_name+'_r_'+float_to_file_float(ratio)+'_'+'_'+loc_time
                            log_path=os.path.join(exp_log_path,log_name)
                            
                            print(f"Starting {method_name} run {i_exp}/{nb_exps}")
                                            
                            
                            K=int(N*ratio) if not config.last_particle else N-1
                            if config.verbose>0:
                                print(f"with model: {model_name}, img_idx:{l},eps:{epsilon},T:{T},N:{N},s:{s},K:{K}")
                            if config.verbose>3:
                                print(f"K/N:{K/N}")
                            times= []
                            rel_error= []
                            ests = [] 
                            log_ests=[]
                            calls=[]
                            if config.track_finish:
                                finish_flags=[]
                            for i in tqdm(range(config.n_rep)):
                                t=time()
                                lg_p,nb_calls,max_val,x,levels,dic=amls_mls.multilevel_uniform(prop=prop,
                                count_particles=N,count_mh_steps=T,x_min=x_min,x_max=x_max,
                                x_sample=x_0,sigma=epsilon,rho=ratio,CUDA=True,debug=(config.verbose>=1))
                                t=time()-t
                                # we don't need adversarial examples and highest score
                                del x
                                del max_val
                                log_ests.append(lg_p)
                                est=np.exp(lg_p)
                                if config.verbose:
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
                                if config.track_finish:
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
                            
                            if config.track_finish:
                                finish_flags=np.array(finish_flags)
                                freq_finished=finish_flags.mean()
                                freq_zero_est=(ests==0).mean()
                            else:
                                freq_zero_est,freq_finished=None,None
                            #finished=np.array(finish_flag)
                            if config.track_finish and freq_finished<1:
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
                            np.savetxt(fname=times_path,X=times)
                            est_path=os.path.join(log_path,'ests.txt')
                            np.savetxt(fname=est_path,X=ests)

                            std_log_est=log_ests.std()
                            mean_log_est=log_ests.mean()
                            lg_q_1,lg_med_est,lg_q_3=np.quantile(a=ests,q=[0.25,0.5,0.75])
                            lg_est_path=os.path.join(log_path,'lg_ests.txt')
                            np.savetxt(fname=lg_est_path,X=ests)

                            

                            plt.hist(times, bins=10)
                            plt.savefig(os.path.join(log_path,'times_hist.png'))
                            plt.hist(ests,bins=10)
                            plt.savefig(os.path.join(log_path,'ests_hist.png'))
                        
                            

                            #with open(os.path.join(log_path,'results.txt'),'w'):
                            results={'method':method_name,'gaussian_latent':str(config.gaussian_latent),
                            'image_idx':l,'dataset':config.dataset,
                                'epsilon':epsilon,"model_name":model_name,'n_rep':config.n_rep,'T':T,'ratio':ratio,'K':K,'s':s,
                            'min_rate':config.min_rate, "N":N, "mean_calls":calls.mean(),"std_calls":calls.std(),"std_adj":ests.std()*mean_calls,
                           'cores_number':config.cores_number,'g_target':config.g_target,"std_rel":std_rel, "std_rel_adj":std_rel_adj,
                            'freq_finished':freq_finished,'freq_zero_est':freq_zero_est,'unfinished_mean_time':unfinished_mean_time,
                            'unfinished_mean_est':unfinished_mean_est,"lg_est_path":lg_est_path,
                                "mean_log_est":mean_log_est,"std_log_est":std_log_est,
                                "lg_q_1":lg_q_1,"lg_q_3":lg_q_3,"lg_med_est":lg_med_est,
                            'np_seed':config.np_seed,'torch_seed':config.torch_seed,'pgd_success':pgd_success,'p_l':p_l,
                            'p_u':p_u,'noise_dist':config.noise_dist,'datetime':loc_time,
                            'q_1':q_1,'q_3':q_3,'med_est':med_est}
                            results_df=pd.DataFrame([results])
                            results_df.to_csv(os.path.join(log_path,'results.csv'),)
                            if config.aggr_res_path is None:
                                aggr_res_path=os.path.join(config.log_dir,'agg_res.csv')
                            else: 
                                aggr_res_path=config.aggr_res_path

                            if config.update_aggr_res:
                                if not os.path.exists(aggr_res_path):
                                    print(f'aggregate results csv file not found \n it will be build at {aggr_res_path}')
                                    cols=['method','gaussian_latent','N','rho','n_rep','T','epsilon','alpha','min_rate','mean_time','std_time','mean_est',
                                    'std_est','freq underest','g_target']
                                    cols+=['freq_finished','freq_zero_est','unfinished_mean_est','unfinished_mean_time']
                                    cols+=['pgd_success','p_l','p_u','gpu_name','cpu_name','np_seed','torch_seed','noise_dist','datetime']
                                    agg_res_df= pd.DataFrame(columns=cols)

                                else:
                                    agg_res_df=pd.read_csv(aggr_res_path)
                                agg_res_df = pd.concat([agg_res_df,results_df],ignore_index=True)
                                agg_res_df.to_csv(aggr_res_path,index=False)
    return config,results_df,agg_res_df
