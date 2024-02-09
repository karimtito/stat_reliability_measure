import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from time import time
from datetime import datetime
import stat_reliability_measure.dev.torch_utils as t_u
import stat_reliability_measure.dev.utils as utils
from stat_reliability_measure.dev.utils import  float_to_file_float,str2bool,str2intList,str2floatList, str2list
from stat_reliability_measure.dev.utils import get_sel_df, simple_vars, range_vars, range_dict_to_lists
import stat_reliability_measure.dev.mls.amls_uniform as amls_mls
from stat_reliability_measure.dev.amls.amls_config import MLS_SMC_Config
import stat_reliability_measure.dev.amls.amls_pyt as amls_pyt
from stat_reliability_measure.dev.imp_sampling.is_pyt import GaussianIS, gaussian_space_attack,hlrf
from stat_reliability_measure.dev.imp_sampling.is_config import IS_Config
import stat_reliability_measure.dev.mls.amls_uniform as amls_webb
from stat_reliability_measure.dev.mls.webb_config import MLS_Webb_Config
from stat_reliability_measure.dev.form.form_pyt import FORM_pyt as FORM_pyt
from stat_reliability_measure.dev.form.form_config import FORM_config
from stat_reliability_measure.dev.smc.smc_pyt import SamplerSMC, SamplerSmcMulti
from stat_reliability_measure.dev.smc.smc_pyt2 import SamplerSMC as SamplerSMC2
from stat_reliability_measure.dev.smc.smc_config import SMCSamplerConfig
from stat_reliability_measure.dev.smc.smc_config2 import SMCSamplerConfig as SMCSamplerConfig2
from stat_reliability_measure.dev.mc.mc_config import CrudeMC_Config
from stat_reliability_measure.dev.ce_is.ce_is import CrossEntropyIS
from stat_reliability_measure.dev.ce_is.ce_is_config import CE_IS_Config
from stat_reliability_measure.dev.line_sampling.ls_pyt import LineSampling
from stat_reliability_measure.dev.line_sampling.ls_config import LS_Config

from stat_reliability_measure.dev.mc.mc_pyt import MC_pf
from stat_reliability_measure.dev.hmls.hmls_config import HMLS_Config
from stat_reliability_measure.dev.hmls.hmls_pyt import HybridMLS
from stat_reliability_measure.config import ExpModelConfig
import scipy.stats as stats
from itertools import product as cartesian_product

method_config_dict={'amls':MLS_SMC_Config,'amls_webb':MLS_Webb_Config,'ls':LS_Config,'line_samp':LS_Config,
                    'line_sampling':LS_Config,
                    'mls_webb':MLS_Webb_Config,'ce_is':CE_IS_Config,
                    'hmls':HMLS_Config,'importance_sampling':IS_Config,
                    'imp_samp':IS_Config,'is':IS_Config,'mala2':SMCSamplerConfig2,
                    'mala':SMCSamplerConfig,'amls_batch':MLS_SMC_Config,
                    'mc':CrudeMC_Config,'crudemc':CrudeMC_Config,'crude_mc':CrudeMC_Config,
                    'form':FORM_config,'rw_smc':SMCSamplerConfig,
                    'smc_multi':SMCSamplerConfig,
                    
                    'hmc':SMCSamplerConfig,'smc':SMCSamplerConfig,}
method_func_dict={'amls':amls_pyt.ImportanceSplittingPyt,'mala':SamplerSMC,
                  'ls':LineSampling,'line_samp':LineSampling,'line_sampling':LineSampling,
                  'rw_smc':SamplerSMC,'ce_is': CrossEntropyIS,
                  'hmls':HybridMLS,'importance_sampling':GaussianIS,
                  'imp_samp':GaussianIS,'is':GaussianIS,'mala2':SamplerSMC2,
                  'mc':MC_pf,'crudemc':MC_pf,'crude_mc':MC_pf, 
                  'amls_webb': amls_webb.multilevel_uniform,'form':FORM_pyt,
                  'mls_webb':amls_webb.multilevel_uniform,
                    'hmc':SamplerSMC,'smc':SamplerSMC,'smc_multi':SamplerSmcMulti, 
                    'amls_batch':amls_pyt.ImportanceSplittingPytBatch,
                    'smc_multi':SamplerSMC}

weighted_methods = ['is','imp_samp','ce_is','importance sampling','importance_sampling',
                    'cross entropy is','cross_entropy_is']
mpp_methods = ['form','imp_samp','is','ls','line_samp','importance sampling','line sampling']
parametric_methods=['ce_is']
def run_est(model, X, y, method='mc', epsilon_range=[], fit_noise_to_input=False, p_target=1e-3,
                     noise_dist='uniform',dataset_name = 'dataset',data_dir='./',model_dir='./',
                 model_name='', plot_errorbar=False, allow_extra_arg=False,figsize=(13,8),
                 verbose=0, x_min=0,x_max=1., mask_opt=False,mask_idx=None, update_aggr_res=True,
                 mask_cond=None,p_ref=None, nb_points_errorbar=100, plot_example_img=False,
                 torch_seed=0,np_seed=0,random_seed=0,allow_unused=True,fontsize_exp=20,
                 alpha_CI=0.05, save_weights=True, shuffle=False,input_index=None,log_hist_=True,
                 log_plots_=True, aggr_res_path='',batch_opt=False, real_uniform=False,
                 log_txt_=True,exp_config=None,method_config=None,input_start=0,input_stop=None,
                 smc_multi=False,alt_functions=False,no_show=True,keep_mpp=False,mpp_hlrf=False,
                 fontsize_title=18,fontsize_label=18,fontsize_legend=18,check_mpp=True,theta_0=None,
                 theta_0_hlrf =False,nb_examples=2,
                 
                 
                 **kwargs):
    """ Running reliability experiments on neural network model with supervised data (X,y)
        values 
    """
    
    if dataset_name=='imagenet':
        #imagenet dataset is too large to be stored in memory and ordered by class
        #we need to shuffle the data for variety
        shuffle=True
        
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
        if input_index is None:
            input_index=input_start
        if input_stop is None:
            input_stop=input_start+1
    
        exp_config=ExpModelConfig(model=model,X=X,y=y,real_uniform=real_uniform,data_dir=data_dir, 
                                  model_dir=model_dir,update_aggr_res=update_aggr_res,
        dataset_name=dataset_name,model_name=model_name,epsilon_range=epsilon_range,
        input_start=input_start,input_stop=input_stop,noise_dist=noise_dist,
        aggr_res_path=aggr_res_path,x_min=x_min,x_max=x_max,mask_opt=mask_opt,
        mask_idx=mask_idx,mask_cond=mask_cond,verbose=verbose,shuffle=shuffle,
        torch_seed=torch_seed,np_seed=np_seed,)
    else:
        exp_config.update_aggr_res=update_aggr_res
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
    
        exp_config.torch_seed,exp_config.np_seed=torch_seed,np_seed
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(random_seed)
        
    exp_config.theta_0=theta_0
    if theta_0 is not None:
        method_config.used_theta_0=True
    for k,v in kwargs.items():
        if hasattr(method_config,k):
            if hasattr(exp_config,k) and  not ('config' in k):
                raise ValueError(f"parameter {k} is in both exp_config and method_config")
            setattr(method_config, k, v)
        elif hasattr(exp_config,k):
            setattr(exp_config, k, v)
        else:
            if allow_extra_arg:
                if verbose:
                    print(f"parameter {k} is not in exp_config or method_config and wont be used")
            else:
                if allow_unused: 
                    if verbose: 
                        print("")
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
        cols=['method_name','from_gaussian','N','rho','n_rep','T','epsilon','alpha','min_rate','mean_time','std_time','mean_est',
        'std_est','freq underest','g_target']
        cols+=['freq_finished','freq_zero_est','unfinished_mean_est','unfinished_mean_time']
        cols+=['pgd_success','p_l','p_u','gpu_name','cpu_name','np_seed','torch_seed','noise_dist','datetime']
        agg_res_df= pd.DataFrame(columns=cols)
    else:
        agg_res_df=pd.read_csv(aggr_res_path)
    estimation_func = method_func_dict[method]
    track_X = hasattr(method_config,'track_X') and method_config.track_X
    track_accept = hasattr(method_config,'track_accept') and method_config.track_accept
    track_beta = hasattr(method_config,'track_beta') and method_config.track_beta
    track_dt = hasattr(method_config,'track_dt') and method_config.track_dt
    save_rare= hasattr(method_config,'save_rare') and method_config.save_rare
    first_iter=True
    weights_list=[]
    p_ests = []
    if exp_config.input_stop>len(exp_config.X):
        exp_config.input_stop=len(exp_config.X)
    self_coverages = []
    ref_coverages = []
    if hasattr(method_config,'save_rare') and method_config.save_rare:
        save_rare=True
        rare_list = []
    else:
        save_rare=False
        rare_list=None
    print(range(exp_config.input_start, exp_config.input_stop))
    for l in range(exp_config.input_start, exp_config.input_stop):
        relative_index=l-exp_config.input_start
        
        dict_coverage = {}
        with torch.no_grad():
            exp_config.x_clean,exp_config.y_clean = exp_config.X[l], exp_config.y[l]
        print(f"method: {method_config.method_name}, input_index:{l},eps:{exp_config.epsilon}")
        exp_config.update(input_index=l,method_name=method_config.method_name)
        for idx in range(len(exp_config.epsilon_range)):
            # if fit_noise_to_input:
            #     if exp_config.noise_dist=='uniform':
            if hasattr(exp_config,'mpp'):
                del exp_config.mpp
                exp_config.mpp=None       
            
            if first_iter:
                first_iter=False
            else:
                exp_config.epsilon = exp_config.epsilon_range[idx]
                exp_config.update(method_name=method_config.method_name)
            
            pgd_success= (exp_config.attack_success[idx][l]).item() if exp_config.use_attack else None 
            if exp_config.lirpa_bounds:
                from stat_reliability_measure.dev.lirpa_utils import get_lirpa_bounds
                # Step 2: define perturbation. Here we use a Linf perturbation on input image.
                p_l,p_u=get_lirpa_bounds(x_clean=exp_config.x_clean,y_clean=exp_config.y_clean,model=model,epsilon=exp_config.epsilon,
                num_classes=exp_config.num_classes,noise_dist=exp_config.noise_dist,a=exp_config.a,device=exp_config.device)
                exp_config.p_l,exp_config.p_u=p_l.item(),p_u.item()
            if exp_config.lirpa_cert:
                from stat_reliability_measure.dev.lirpa_utils import get_lirpa_cert
                exp_config.lirpa_safe,exp_config.time_lirpa_safe=get_lirpa_cert(x_clean=exp_config.x_clean,y_clean=exp_config.y_clean,
                                epsilon = exp_config.epsilon, num_classes=exp_config.num_classes                                                
                                ,model=model, device=exp_config.device)
           
            if plot_example_img:
                
                #print('try plotting example image')
                x_0 = exp_config.x_clean.unsqueeze(0)
                uu = exp_config.gen(nb_examples)
                
                xx = exp_config.t_transform(uu)
                ll = exp_config.model(xx)
                yy = torch.argmax(ll,dim=1)
                
                t_u.plot_k_tensor(X=torch.cat([x_0,xx]),y=torch.cat([exp_config.y_clean.unsqueeze(0),yy]),dataset=exp_config.dataset,
                                  fontsize=fontsize_exp,)
                if not no_show:
                    plt.show()
                if noise_dist=='uniform':
                    plt.savefig(f"examples_noisy_img_idx_{l}_eps_{exp_config.epsilon}.pdf",bbox_inches='tight')
                else:
                    plt.savefig(f"examples_noisy_img_idx_{l}_sigma_{exp_config.sigma_noise}.pdf",bbox_inches='tight')
            lists_cart= cartesian_product(*method_param_lists)
            zeros_d = torch.zeros((1,exp_config.d)).to(exp_config.device)
            gradG_0,G_0 = exp_config.gradG_alt(zeros_d)
            norm_gradG_0 = torch.norm(gradG_0)
            fosm_est = exp_config.normal_dist.cdf(-G_0/norm_gradG_0)
            print(f'fosm_est:{fosm_est}')
            if mpp_hlrf:
                mpp_hlrf = hlrf(grad_f= exp_config.gradG_alt, zero_latent=zeros_d,num_iter=50,
                        step_size=0.8,stop_cond_type='beta',stop_eps=0.001)[0] 
                gradG_mpp_hlrf,G_mpp_hlrf = exp_config.gradG_alt(mpp_hlrf)
                norm_grad = torch.norm(gradG_mpp_hlrf)
                beta_mpp_hlrf = -(mpp_hlrf*(gradG_mpp_hlrf/norm_grad)).sum() + G_mpp_hlrf/norm_grad
                beta_naif = torch.norm(mpp_hlrf)
                form_est = exp_config.normal_dist.cdf(-beta_mpp_hlrf)
                form_naif = exp_config.normal_dist.cdf(-beta_naif)
                
                print(f"HLRF form est:{form_est}")
                print(f'HLRF form_naif:{form_naif}')
            for method_params in lists_cart:
                i_exp+=1
                vars(method_config).update(method_params)
                if check_mpp:
                    checked_mpp=False
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
                        print(f"Aggr. result path :{exp_config.aggr_res_path}")
                        
                        aggr_res_df = pd.read_csv(exp_config.aggr_res_path)
                        if len(aggr_res_df)>0:
                            same_method_df = get_sel_df(df=aggr_res_df,triplets=[('method_name',method_config.method_name,'=')])
                            
                            if len(same_method_df)>0:
                                print("Experiments already done for method: "+method_config.method_name)
                                try:
                                
                                    triplets=[('model_name',exp_config.model_name,'='),
                                    ('input_index',l,'='),('n_rep',exp_config.n_rep,'='),('noise_dist',exp_config.noise_dist,'=')]
                                
                                    if exp_config.noise_dist=='uniform':
                                        triplets.append(('epsilon',exp_config.epsilon,'='))
                                    else:
                                        triplets.append(('sigma_noise',exp_config.sigma_noise,'='))
                                
                                    same_exp_df = get_sel_df(df=same_method_df, cols=method_keys, vals=method_vals, 

                                    triplets =triplets)
                            
                                    
                                    # if a similar experiment has been done in the current log directory we skip it

                                    if len(same_exp_df)>0:
                                        print(f"Skipping {method_config.method_name} run {i_exp}/{nb_exps}, with model: {exp_config.model_name}, img_idx:{l},eps:{exp_config.epsilon},"+clean_method_args_str)
                                        p_est = same_exp_df['mean_est'].iloc[0]
                                        std_est = same_exp_df['std_est'].iloc[0]
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
                log_name=method_config.method_name+'_e_'+float_to_file_float(exp_config.epsilon_range[idx])+f'_n_{exp_config.n_rep}_datetime_'+loc_time
                
                log_path=os.path.join(exp_config.exp_log_path,log_name)
                if os.path.exists(log_path):
                    log_path=log_path+'_rand_'+str(np.random.randint(low=1,high=10000))
                os.mkdir(log_path)
                if exp_config.noise_dist=='uniform':
                    print(f"Starting {method_config.method_name} estimation {i_exp}/{nb_exps}, with model: {exp_config.model_name}, img_idx:{l}, eps:{exp_config.epsilon}, "+clean_method_args_str)
                else:
                    print(f"Starting {method_config.method_name} estimation {i_exp}/{nb_exps}, with model: {exp_config.model_name}, img_idx:{l}, sigma:{exp_config.sigma_noise}, "+clean_method_args_str)
                if exp_config.verbose>0:
                    print(f"with model: {model_name}, input_index:{l},eps:{exp_config.epsilon},")
              
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
                
                if exp_config.tqdm_opt:
                    if exp_config.notebook:
                        from tqdm.notebook import tqdm
                    else:
                        from tqdm import tqdm
                    iterator = tqdm(range(exp_config.n_rep))
                else: 
                    iterator = range(exp_config.n_rep)
                if track_accept:
                    accept_rates_list = []
                else:
                    accept_rates_list=None
                if track_dt:
                    dts_list=[]
                exp_required = {'verbose':exp_config.verbose}
                requires_keys = [k for k in simple_vars(method_config).keys() if 'requires_' in k]
                for k in requires_keys:
                    required_key = k.replace('requires_','')
                    exp_required[required_key]=getattr(exp_config,required_key)
                
                #selecting only method configuration variables relevent to the estimation function
                func_args_vars = {k:simple_vars(method_config)[k] for k in simple_vars(method_config).keys() if ('require' not in k) and ('name' not in k) and ('calls' not in k)}
                alt_func_list = ['V','G','h','gradG','gradV','gradh']
                if alt_functions:
                    #replacing the default functions by the alternative ones
                    for func in alt_func_list:
                        if func in exp_required.keys():
                            exp_required[func]  = getattr(exp_config,func+'_alt')
                args_dict = {**func_args_vars,**exp_required}
                method_config.update()
                var_ests=[]
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
                    if method in mpp_methods:
                        if keep_mpp:
                            exp_config.mpp = dict_out['mpp']
                        
                            
                        if check_mpp and not checked_mpp:
                            checked_mpp=True
                            mpp = torch.from_numpy(dict_out['mpp']).to(exp_config.device)
                            norm_mpp = torch.norm(mpp)
                            gradG_mpp,G_mpp = exp_config.gradG_alt(mpp)
                            cosine_G_mpp = torch.cosine_similarity(gradG_mpp, mpp,)
                            print(f"Norm mpp:{norm_mpp}, cosine(mpp,gradG(mpp)):{cosine_G_mpp}")
                            beta_mpp = -(mpp*(gradG_mpp/norm_mpp)).sum() + G_mpp/norm_mpp
                            del gradG_mpp
                            if not method_config.save_mpp:
                                del mpp
                            print(f"G_mpp:{G_mpp}")
                            form_mpp = exp_config.normal_dist.cdf(-beta_mpp)
                        if method_config.save_mpp:
                            mpp=dict_out['mpp']  
                            
                    if method in parametric_methods:
                            if method_config.save_thetas:
                                thetas_paths = os.path.join(log_path,f'thetas')
                                if not os.path.exists(thetas_paths):
                                    os.mkdir(thetas_paths)
                                thetas=dict_out['thetas']
                                thetas_path=os.path.join(thetas_paths,f'thetas_exp_{i}.txt')
                                np.savetxt(fname=thetas_path,X=thetas)
                            if method_config.save_theta:
                                theta_paths = os.path.join(log_path,f'thetas')
                                if not os.path.exists(theta_paths):
                                    os.mkdir(theta_paths)
                                theta=dict_out['theta']
                                theta_path=os.path.join(theta_paths,f'theta_exp_{i}.txt')
                                np.savetxt(fname=theta_path,X=theta)
                            if method_config.save_sigma:
                                sigma_paths = os.path.join(log_path,f'thetas')
                                if not os.path.exists(sigma_paths):
                                    os.mkdir(sigma_paths)
                                sigma=dict_out['sigma']
                                sigma_path=os.path.join(sigma_paths,f'sigma_exp_{i}.txt')
                                np.savetxt(fname=sigma_path,X=sigma)
                            if method_config.save_sigmas:
                                sigmas_paths = os.path.join(log_path,f'thetas')
                                if not os.path.exists(sigmas_paths):
                                    os.mkdir(sigmas_paths)
                                sigmas=dict_out['sigmas']
                                sigmas_path=os.path.join(sigmas_paths,f'sigmas_exp_{i}.txt')
                                np.savetxt(fname=sigmas_path,X=sigmas)                         
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
                            plt.savefig(os.path.join(accept_logs,f'accept_rates_{i}.png'),bbox_inches='tight')
                            plt.close()
                            exp_config.mean_acc_ratio = np.array(dict_out['acc_ratios']).mean()
                    if track_beta:
                        beta_list=dict_out['betas']
                        beta_logs=os.path.join(log_path,'beta_logs')
                        if not os.path.exists(beta_logs):
                            os.mkdir(path=beta_logs)
                        if log_txt_:
                            np.savetxt(fname=os.path.join(beta_logs,f'beta_{i}.txt'),X=beta_list)
                        if log_hist_:
                            x_T=np.arange(len(beta_list))
                            plt.plot(x_T,beta_list)
                            plt.savefig(os.path.join(beta_logs,f'beta_{i}.png'))
                            plt.close()
                    if track_dt:
                        dt_list=dict_out['dts']
                        dt_logs=os.path.join(log_path,'dt_logs')
                        if not os.path.exists(dt_logs):
                            os.mkdir(path=dt_logs)
                        if log_txt_:
                            np.savetxt(fname=os.path.join(dt_logs,f'dt_{i}.txt'),X=dt_list)
                        if log_hist_:
                            x_T=np.arange(len(dt_list))
                            plt.plot(x_T,dt_list)
                            plt.savefig(os.path.join(dt_logs,f'dt_{i}.png'))
                            plt.close()
                        
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
                    if save_rare:
                        rare_list.append(dict_out['u_rare' ])
                    if 'var_est' in dict_out.keys():
                        var_est=dict_out['var_est']
                        var_ests.append(var_est)
                    elif 'std_est' in dict_out.keys():
                        std_est=dict_out['std_est']
                        var_ests.append(std_est**2)
                    if save_weights and method in weighted_methods:
                        weights=dict_out['weights']
                        weights_paths = os.path.join(log_path,f'weights')
                        if not os.path.exists(weights_paths):
                            os.mkdir(weights_paths)
                        weights_path=os.path.join(weights_paths,f'weights_exp_{i}.txt')
                        np.savetxt(fname=weights_path,X=weights)
                        weights_list.append(weights)
                    del dict_out
                    
                times=np.array(times)
                ests = np.array(ests)
                log_ests=np.array(log_ests)
                mean_est=ests.mean()
                calls=np.array(calls)
                mean_calls=calls.mean()
                std_est=ests.std()
                if plot_errorbar:
                    
                    q_alpha = stats.norm.ppf(1-alpha_CI/2)
                    if nb_points_errorbar>exp_config.n_rep:
                        nb_points=exp_config.n_rep
                    else:
                        nb_points=nb_points_errorbar
                    if nb_points_errorbar>1:
                        errorbar_idx = np.arange(nb_points)
                    else:
                        errorbar_idx=[0]

                    self_coverage = (ests-std_est*q_alpha< mean_est) * (mean_est < ests+std_est*q_alpha)
                    self_coverage_rate = self_coverage.mean()
                    dict_coverage = {'self_coverage':self_coverage_rate}
                    self_coverages.append(self_coverage_rate)
                    trunc_ests = ests[errorbar_idx]
                    if len(var_ests)>0:
                        std_ests= np.sqrt(np.array(var_ests))
                        trunc_std_ests = std_ests[:nb_points]
                        plt.figure(figsize=figsize)
                        plt.errorbar(x=errorbar_idx,y=trunc_ests,yerr=q_alpha*trunc_std_ests,
                        fmt='o',errorevery=1,elinewidth=1.,capsize=2,
                        label = r'Est. Prob. +/- $q_{\alpha/2}\hat{\sigma}$')
                    else:
                        plt.figure(figsize=figsize)
                        plt.errorbar(x=errorbar_idx,y=trunc_ests,fmt='o',errorevery=1,elinewidth=1.
                        ,capsize=2, label = r'Est. Prob. +/- $q_{\alpha/2}\hat{\sigma}$')
                    if p_ref is not None:
                        
                        if type(p_ref)==list:
                            p_ref_ = p_ref[relative_index]
                        else:
                            p_ref_ = p_ref
                            
                        ref_coverage = (ests-std_est*q_alpha< p_ref_) * (p_ref_ < ests+std_est*q_alpha)
                        ref_coverage_rate = ref_coverage.mean()
                        dict_coverage['ref_coverage']=ref_coverage_rate
                        ref_coverages.append(ref_coverage_rate)
                        
                        plt.plot(np.arange(nb_points),np.ones(nb_points)*p_ref_,'r',label=f'Ref. Prob. = {p_ref_:.2E} ')
                    plt.plot(np.arange(nb_points),np.ones(nb_points)*mean_est,'g',label=f'Avg. Est.= {mean_est:.2E}, Std. Est. = {std_est:.2E}')
                    CI_low_empi = mean_est-q_alpha*std_est
                    CI_high_empi = mean_est+q_alpha*std_est
                    plt.xticks(fontsize=fontsize_label*0.9)
                    est_max = ests.max()
                    est_min = ests.min()
            
                    plt.yticks(fontsize=fontsize_label,)
                    plt.minorticks_on()
                
                    plt.plot(np.arange(nb_points),np.ones(nb_points)*CI_low_empi,'--',color='grey',label=fr'CI low = {CI_low_empi:.2E}, $\alpha$ = {alpha_CI}')
                    plt.plot(np.arange(nb_points),np.ones(nb_points)*CI_high_empi,'--',color='k',label=fr'CI high = {CI_high_empi:.2E}, $\alpha$ = {alpha_CI}')
                    plt.xlabel('Run index',fontsize=fontsize_label)
                    from matplotlib.ticker import FormatStrFormatter

                    plt.ylabel('Estimated Failure Probability',fontsize=fontsize_label)
                    plt.legend(loc='upper right',fontsize=fontsize_legend)
                    if exp_config.noise_dist=='uniform':
                        plt.title(fr'Estimation of failure probability with {method_config.method_name} using $\approx${mean_calls} model calls, $\varepsilon$={exp_config.epsilon}, input idx.={l}, {exp_config.n_rep} reps',
                        
                          fontsize=fontsize_title)
                    else:
                        plt.title(fr'Estimation of failure probability with {method_config.method_name} using $\approx${mean_calls} model calls, $\sigma$={exp_config.sigma_noise}, input idx.={l}, {exp_config.n_rep} reps',
                        
                          fontsize=fontsize_title)
                   
                    ploterror_path=os.path.join(log_path,'errorbar.png')
                    plt.savefig(ploterror_path,bbox_inches='tight')
                    if not no_show:
                        plt.show()
                    plt.close()

                    times_errorbar = times[errorbar_idx]
                    
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
                    if type(p_ref)==list:
                        p_ref_ = p_ref[relative_index]
                    else:
                        p_ref_ = p_ref
                    rel_error = np.abs(ests-p_ref_)/p_ref_
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
                result={"i_idx":l,'mean_calls':calls.mean(),'std_calls':calls.std()
                ,'mean_time':times.mean(),'std_time':times.std()
                ,'mean_est':ests.mean(),'std_est':ests.std(), 'est_path':est_path,'times_path':times_path,
                "std_rel":std_rel,"std_rel_adj":std_rel*mean_calls,
               'fosm_est':fosm_est,
                'q_1':q_1,'q_3':q_3,'med_est':med_est,"lg_est_path":lg_est_path,
                "mean_log_est":mean_log_est,"std_log_est":std_log_est,
                "lg_q_1":lg_q_1,"lg_q_3":lg_q_3,"lg_med_est":lg_med_est,
                "log_path":log_path,"log_name":log_name,
                }
                result.update({"fosm_est":fosm_est})
                if mpp_hlrf:
                    result.update({"form_hlrf_est":form_est,"form_hlrf_naif":form_naif,"G_mpp_hlrf":G_mpp_hlrf,})
                if method in mpp_methods:
                    if check_mpp:
                        result.update({"norm_mpp":norm_mpp,"cosine_G_mpp":cosine_G_mpp,"form_mpp":form_mpp,"G_mpp":G_mpp})
                result.update(dict_coverage)
                result.update({"method_name":method_config.method_name})
                if plot_errorbar:
                    result.update({"ploterror_path":ploterror_path})
                if track_finish:
                    result.update({"freq_finished":freq_finished,"freq_zero_est":freq_zero_est,
                    "unfinished_mean_est":unfinished_mean_est,"unfinished_mean_time":unfinished_mean_time})
           
                if track_accept:
                    result.update({"mean_acc_ratio":exp_config.mean_acc_ratio})
                if track_levels:
                    result.update({"mean_levels":mean_levels,"std_levels":std_levels})
                if track_beta:
                    result.update({"beta_path":beta_logs})
                if p_ref is not None:
                    if type(p_ref)==list:
                        p_ref_ = p_ref[relative_index]
                    else:
                        p_ref_ = p_ref
                    result.update({"rel_error":rel_error.mean(),"std_rel_error":rel_error.std(),
                                   "p_ref":p_ref_})
                result.update(simple_vars(method_config))
                print(f"method_name:{exp_config.method_name}")
                result.update(simple_vars(exp_config))
                
                result_df=pd.DataFrame([result])
                results_df = pd.concat([results_df,result_df],ignore_index=True)
                results_df.to_csv(os.path.join(exp_config.exp_log_path,'results.csv'),)
                
                if exp_config.update_aggr_res:
                    agg_res_df = pd.concat([agg_res_df,result_df],ignore_index=True)
                    agg_res_df.to_csv(aggr_res_path,index=False)

                
                p_ests.append(ests.mean())
                
    dict_out={}
    if len(results_df)>0:
        p_est = results_df['mean_est'].mean()
        std_est = results_df['std_est'].mean()
        q_alpha_CI = stats.norm.ppf(1-alpha_CI/2)
        dict_out['CI_low'] = p_est-q_alpha_CI*std_est
        dict_out['CI_high'] = p_est+q_alpha_CI*std_est
        dict_out['CI'] = np.array([dict_out['CI_low'],dict_out['CI_high']])
        if plot_errorbar:
            dict_out['ploterror_path']=ploterror_path
        
    dict_out.update({'results_df':results_df,'agg_res_df':agg_res_df,'std_est':std_est,
                'method_config':method_config, 'exp_config':exp_config})
    
    if track_X: 
        dict_out['X_list']=X_list
    if save_rare:
        if len(rare_list)>0:
            dict_out['rare_list']=rare_list
            dict_out['u_rare'] = rare_list[-1]
        else:
            dict_out['rare_list']=None
    if method in weighted_methods and save_weights:
        dict_out['weights_list']=weights_list
    if method in parametric_methods:
        if method_config.save_thetas:
            dict_out['thetas']=thetas
        if method_config.save_theta:
            dict_out['theta']=theta
        if method_config.save_sigma:
            dict_out['sigma']=sigma
        if method_config.save_sigmas:
            dict_out['sigmas']=sigmas
        
    if method in mpp_methods and method_config.save_mpp:
        dict_out['mpp'] = mpp
    
    dict_out['p_ests']=p_ests
    return p_est, dict_out