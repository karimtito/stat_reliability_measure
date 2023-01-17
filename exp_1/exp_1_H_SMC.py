import stat_reliability_measure.dev.smc.smc_pyt as smc_pyt
import scipy.stats as stat
import numpy as np
from tqdm import tqdm
from time import time
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import torch
import pandas as pd
import argparse
from stat_reliability_measure.dev.utils import str2bool,str2floatList,str2intList,float_to_file_float,dichotomic_search,get_sel_df
from stat_reliability_measure.dev.utils import print_config
from scipy.special import betainc
from stat_reliability_measure.home import ROOT_DIR



method_name="H_SMC"
class config:
    
    n_rep=200
    GV_opt=False
    L=10
    e_range=[0.75,0.85,0.95]
    p_range=[1e-6,1e-12]
    N_range=[32,64,128,256,512,1024]
    T_range=[10,20,50]
    only_duplicated=True
    v_min_opt=True
    
    
    
    
    L_range=[]
    min_rate=0.15
    
    alpha=0.2
    alpha_range=[]
    ess_alpha=0.8
    
    p_t=1e-6
    
    N=100
    T=1
    save_config=False 
    print_config=True
    d=1024
    verbose=0
    log_dir=ROOT_DIR+'/logs/linear_gaussian_tests'
    aggr_res_path = None
    update_agg_res=True
    sigma=1
    v1_kernel=True
    torch_seed=None
    gpu_name=None
    cpu_name=None
    cores_number=None
    track_gpu=False
    track_cpu=False
    device=None
    n_max=10000 
    allow_multi_gpu=True
    tqdm_opt=True
    allow_zero_est=True
    track_accept=True
    track_calls=False
    mh_opt=False
    adapt_dt=False
    adapt_dt_mcmc=False
    target_accept=0.574
    accept_spread=0.1
    dt_decay=0.999
    dt_gain=None
    dt_min=1e-3
    dt_max=0.5
    v_min_opt=True
    ess_opt=False
    only_duplicated=True
    np_seed=None
    lambda_0=0.5
    test2=False

    s_opt=False
    s=1
    clip_s=True
    s_min=1e-3
    s_max=3
    s_decay=0.95
    s_gain=1.0001

    track_dt=False
    mult_last=True
    linear=True

    track_ess=True
    track_beta=True
    track_dt=True
    track_v_means=True
    track_ratios=False

    kappa_opt=True

    adapt_func='ESS'
    M_opt = False
    adapt_step=True
    FT=True
    sig_dt=0.02
    L_min=1
    skip_mh=False
    repeat_exp = True 
    


parser=argparse.ArgumentParser()
parser.add_argument('--log_dir',default=config.log_dir)
parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--N',type=int,default=config.N)
parser.add_argument('--verbose',type=float,default=config.verbose)
parser.add_argument('--d',type=int,default=config.d)

parser.add_argument('--min_rate',type=float,default=config.min_rate)
parser.add_argument('--alpha',type=float,default=config.alpha)
parser.add_argument('--n_max',type=int,default=config.n_max)
parser.add_argument('--tqdm_opt',type=str2bool,default=config.tqdm_opt)

parser.add_argument('--save_config',type=str2bool, default=config.save_config)
#parser.add_argument('--update_agg_res',type=str2bool,default=config.update_agg_res)
#parser.add_argument('--aggr_res_path',type=str, default=config.aggr_res_path)
#parser.add_argument('--rho',type=float,default=config.rho)
parser.add_argument('--allow_multi_gpu',type=str2bool)
parser.add_argument('--track_gpu',type=str2bool,default=config.track_gpu)
parser.add_argument('--track_cpu',type=str2bool,default=config.track_cpu)
parser.add_argument('--device',type=str, default=config.device)
parser.add_argument('--allow_zero_est',type=str2bool, default=config.allow_zero_est)
parser.add_argument('--torch_seed',type=int, default=config.torch_seed)
parser.add_argument('--np_seed',type=int, default=config.np_seed)
parser.add_argument('--sigma', type=float,default=config.sigma)
parser.add_argument('--p_t',type=float,default=config.p_t)
parser.add_argument('--p_range',type=str2floatList,default=config.p_range)
parser.add_argument('--ess_alpha',type=float,default=config.ess_alpha)
parser.add_argument('--e_range',type=str2floatList,default=config.e_range)
parser.add_argument('--N_range',type=str2intList,default=config.N_range)
parser.add_argument('--T',type=int,default=config.T)
parser.add_argument('--T_range',type=str2intList,default=config.T_range)
parser.add_argument('--L',type=int,default=config.L)
parser.add_argument('--L_range',type=str2intList,default=config.L_range)
parser.add_argument('--alpha_range',type=str2floatList,default=config.alpha_range)
parser.add_argument('--v1_kernel',type=str2bool,default=config.v1_kernel)
parser.add_argument('--track_accept',type=str2bool,default=config.track_accept)
parser.add_argument('--track_calls',type=str2bool,default=config.track_calls)
parser.add_argument('--track_beta',type=str2bool,default=config.track_accept)
parser.add_argument('--track_ess',type=str2bool,default=config.track_calls)
parser.add_argument('--track_v_means',type=str2bool,default=config.track_v_means)
parser.add_argument('--track_ratios',type=str2bool,default=config.track_ratios)
parser.add_argument('--mh_opt',type=str2bool,default=config.mh_opt)
parser.add_argument('--adapt_dt',type=str2bool,default=config.adapt_dt)
parser.add_argument('--target_accept',type=float,default=config.target_accept)
parser.add_argument('--accept_spread',type=float,default=config.accept_spread)
parser.add_argument('--dt_decay',type=float,default=config.dt_decay)
parser.add_argument('--dt_gain',type=float,default=config.dt_gain)
parser.add_argument('--dt_min',type=float,default=config.dt_min)
parser.add_argument('--dt_max',type=float,default=config.dt_max)
parser.add_argument('--adapt_dt_mcmc',type=str2bool,default=config.adapt_dt_mcmc)
parser.add_argument('--update_agg_res',type=str2bool,default=config.update_agg_res)
parser.add_argument('--v_min_opt',type=str2bool,default=config.v_min_opt)
parser.add_argument('--ess_opt',type=str2bool,default=config.ess_opt)
parser.add_argument('--only_duplicated',type=str2bool,default=config.only_duplicated)
parser.add_argument('--lambda_0',type=float,default=config.lambda_0)
parser.add_argument('--test2',type=str2bool,default =config.test2)
parser.add_argument('--print_config',type=str2bool,default=config.print_config)
parser.add_argument('--track_dt',type=str2bool,default=config.track_dt)
parser.add_argument('--linear',type=str2bool,default=config.linear)
parser.add_argument('--adapt_func',type=str,default=config.adapt_func)
parser.add_argument('--M_opt',type=str2bool,default=config.M_opt)
parser.add_argument('--adapt_step',type=str2bool,default=config.adapt_step)
parser.add_argument('--FT',type=str2bool,default=config.FT)
parser.add_argument('--kappa_opt',type=str2bool,default=config.kappa_opt)
parser.add_argument('--sig_dt', type=float,default=config.sig_dt)
parser.add_argument('--L_min',type=int,default=config.L_min)
parser.add_argument('--skip_mh',type=str2bool,default=config.skip_mh)
parser.add_argument('--GV_opt',type=str2bool,default=config.GV_opt)
parser.add_argument('--repeat_exp',type=str2bool,default=config.repeat_exp)
args=parser.parse_args()

for k,v in vars(args).items():
    setattr(config, k, v)

def main():

    assert config.adapt_func.lower() in smc_pyt.supported_beta_adapt.keys(),f"select adaptive function in {smc_pyt.supported_beta_adapt.keys}"
    adapt_func=smc_pyt.supported_beta_adapt[config.adapt_func.lower()]

    if config.adapt_func.lower()=='simp_ess':
        adapt_func = lambda beta,v : smc_pyt.nextBetaSimpESS(beta_old=beta,v=v,lambda_0=config.lambda_0,max_beta=1e6)
    prblm_str='linear_gaussian' if config.linear else 'gaussian'

    if config.GV_opt:
        method_name="RW_SMC"
    elif config.L>1:
        method_name="H_SMC"
    else:
        method_name="MALA_SMC"
    if not config.linear:
        config.log_dir=config.log_dir.replace('linear_gaussian','gaussian')
    if len(config.p_range)==0:
        config.p_range= [config.p_t]

    if len(config.e_range)==0:
        config.e_range= [config.ess_alpha]


    if len(config.N_range)==0:
        config.N_range= [config.N]


    if len(config.T_range)==0:
        config.T_range= [config.T]


    if len(config.alpha_range)==0:
        config.alpha_range= [config.alpha]


    if not config.allow_multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"



    if config.torch_seed is None:
        config.torch_seed=int(time())
    torch.manual_seed(seed=config.torch_seed)

    if config.np_seed is None:
        config.np_seed=int(time())
    torch.manual_seed(seed=config.np_seed)



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


    if config.device is None:
        config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if config.verbose>=5:
            print(config.device)
        device=config.device
    else:
        device=config.device

    d=config.d
    #epsilon=config.epsilon


    if not os.path.exists(ROOT_DIR+'/logs'):
        os.mkdir(ROOT_DIR+'/logs')
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    results_path=ROOT_DIR+'/logs/'+ prblm_str+'_tests/results.csv' 
    if os.path.exists(results_path):
        results_g=pd.read_csv(results_path)
    else:
        results_g=pd.DataFrame(columns=['p_t','mean_est','mean_time','mean_err','stdtime','std_est','T','N','rho','alpha','n_rep','min_rate','method'])
        results_g.to_csv(results_path,index=False)
    raw_logs = os.path.join(config.log_dir,'raw_logs/')
    if not os.path.exists(raw_logs):
        os.mkdir(raw_logs)
    raw_logs_path=os.path.join(config.log_dir,'raw_logs/'+method_name)
    if not os.path.exists(raw_logs_path):
        os.mkdir(raw_logs_path)

    loc_time= datetime.today().isoformat().split('.')[0]
    log_name=method_name+'_'+'_'+loc_time
    exp_log_path=os.path.join(raw_logs_path,log_name)
    if os.path.exists(exp_log_path):
        exp_log_path = exp_log_path +'_'+ str(np.random.randint(low=0,high=9))
    os.mkdir(path=exp_log_path)
    config.json=vars(args)

    # if config.aggr_res_path is None:
    #     aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
    # else:
    #     aggr_res_path=config.aggr_res_path

    if config.dt_gain is None:
        config.dt_gain=1/config.dt_decay
    
    config_dict=print_config(config)
    path_config=os.path.join(exp_log_path,'config.json')
    with open(path_config,'w') as f:
        f.write(json.dumps(config_dict, indent = 4))

    param_ranges = [config.N_range,config.T_range,config.alpha_range,config.p_range,config.e_range]
    param_lens=np.array([len(l) for l in param_ranges])
    print(param_lens)
    nb_runs= np.prod(param_lens)

    #mh_str="adjusted" 
    #method=method_name
    method=method_name
    save_every = 1
    #adapt_func= smc_pyt.ESSAdaptBetaPyt if config.ess_opt else smc_pyt.SimpAdaptBetaPyt

    kernel_str='v1_kernel' if config.v1_kernel else 'v2_kernel'


    run_nb=0
    iterator= tqdm(range(config.n_rep))
    exp_res=[]
    L=config.L
    for p_t in config.p_range:
        if config.linear:
            
            get_c_norm= lambda p:stat.norm.isf(p)
            c=get_c_norm(p_t)
            if config.verbose>=1.:
                print(f'c:{c}')
            e_1= torch.Tensor([1]+[0]*(d-1)).to(device)
            V = lambda X: torch.clamp(input=c-X[:,0], min=0, max=None)
            
            gradV= lambda X: -torch.transpose(e_1[:,None]*(X[:,0]<c),dim0=1,dim1=0)
            
            norm_gen = lambda N: torch.randn(size=(N,d)).to(device)
        else:
            epsilon=1
            p_target_f=lambda h: 0.5*betainc(0.5*(d-1),0.5,(2*epsilon*h-h**2)/(epsilon**2))
            h,P_target = dichotomic_search(f=p_target_f,a=0,b=epsilon,thresh=p_t,n_max=100)
            c=epsilon-h
            print(f'c:{c}',f'P_target:{P_target}')
            e_1= torch.Tensor([1]+[0]*(d-1)).to(device)
            V = lambda X: torch.clamp(input=torch.norm(X,p=2,dim=-1)*c-X[:,0], min=0, max=None)
            
            gradV= lambda X: (c*X/torch.norm(X,p=2,dim=-1)[:,None] -e_1[None,:])*(X[:,0]<c*torch.norm(X,p=2,dim=1))[:,None]
            
            norm_gen = lambda N: torch.randn(size=(N,d)).to(device)

        for ess_t in config.e_range:
            if config.adapt_func.lower()=='ess':
                adapt_func = lambda beta,v : smc_pyt.nextBetaESS(beta_old=beta,v=v,ess_alpha=ess_t,max_beta=1e6)
            for T in config.T_range:
                for alpha in config.alpha_range:       
                    for N in config.N_range:
                        run_nb+=1
                        aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                        if (not config.repeat_exp) and config.update_agg_res and os.path.exists(aggr_res_path):
                            aggr_res_df = pd.read_csv(aggr_res_path)
                            same_exp_df = get_sel_df(df=aggr_res_df,triplets=[('method',method,'='),
                            ('p_t',p_t,'='),('n_rep',config.n_rep,'='),
                ('N',N,'='),('T',T,'='),('L',L,'='),('alpha',alpha,'='),
                ('ess_alpha',ess_t,'=')] )  
                            # if a similar experiment has been done in the current log directory we skip it
                            if len(same_exp_df)>0:
                                print(f"Skipping {method_name} run {run_nb}/{nb_runs}, with p_t:{p_t},ess_t:{ess_t},T:{T},alpha:{alpha},N:{N},L:{L}")
                                continue
                        loc_time= datetime.today().isoformat().split('.')[0]
                        log_name=method_name+f'_N_{N}_T_{T}_L_{L}_a_{float_to_file_float(alpha)}_ess_{float_to_file_float(ess_t)}'+'_'+loc_time.split('_')[0]
                        log_path=os.path.join(exp_log_path,log_name)
                        if os.path.exists(log_path):
                            log_path=log_path+'_'+str(np.random.randint(low=0,high=10))
                        

                        os.mkdir(path=log_path)
                        
                        print(f'Run {run_nb}/{nb_runs}')
                        times=[]
                        ests = []
                        calls=[]
                        finished_flags=[]
                        iterator= tqdm(range(config.n_rep)) if config.tqdm_opt else range(config.n_rep)
                        
                        print(f"Starting {method} simulations with p_t:{p_t},ess_t:{ess_t},T:{T},alpha:{alpha},N:{N},L:{L}")
                        for i in iterator:
                            t=time()
                            sampler=smc_pyt.SamplerSMC 
                            p_est,res_dict,=sampler(gen=norm_gen, V=V, gradV=gradV, adapt_func=adapt_func,min_rate=config.min_rate,N=N,T=T,L=L,
                            alpha=alpha, n_max=10000,
                            verbose=config.verbose, track_accept=config.track_accept,track_beta=config.track_beta,track_v_means=config.track_v_means,
                            track_ratios=config.track_ratios,track_ess=config.track_ess,kappa_opt=config.kappa_opt
                            ,gaussian =True,accept_spread=config.accept_spread, 
                            adapt_dt=config.adapt_dt, dt_decay=config.dt_decay,only_duplicated=config.only_duplicated,
                            dt_gain=config.dt_gain,dt_min=config.dt_min,dt_max=config.dt_max,
                            v_min_opt=config.v_min_opt, lambda_0= config.lambda_0,
                            track_dt=config.track_dt,M_opt=config.M_opt,adapt_step=config.adapt_step,FT=config.FT,
                            sig_dt=config.sig_dt,L_min=config.L_min,skip_mh=config.skip_mh,
                            GV_opt=config.GV_opt
                            )
                            t1=time()-t
                            if config.verbose>=2:
                                print(p_est)
                            #finish_flag=res_dict['finished']
                            
                            if config.track_accept:
                                accept_rates_mcmc=res_dict['accept_rates_mcmc']
                                np.savetxt(fname=os.path.join(log_path,f'accept_rates_mcmc_{i}.txt')
                                ,X=accept_rates_mcmc,)
                                x_T=np.arange(len(accept_rates_mcmc))
                                plt.plot(x_T,accept_rates_mcmc)
                                plt.savefig(os.path.join(log_path,f'accept_rates_mcmc_{i}.png'))
                                plt.close()
                                

                            if config.adapt_dt and config.track_dt:
                                dts=res_dict['dts']
                                np.savetxt(fname=os.path.join(log_path,f'dts_{i}.txt')
                                ,X=dts)
                                x_T=np.arange(len(dts))
                                plt.plot(x_T,dts)
                                plt.savefig(os.path.join(log_path,f'dts_{i}.png'))
                                plt.close()
                            
                            
                            times.append(t1)
                            ests.append(p_est)
                            calls.append(res_dict['calls'])
                        times=np.array(times)
                        ests = np.array(ests)
                        log_ests=np.log(np.clip(ests,a_min=1e-250,a_max=1))
                        std_log_est=log_ests.std()
                        mean_log_est=log_ests.mean()
                        lg_q_1,lg_med_est,lg_q_3=np.quantile(a=ests,q=[0.25,0.5,0.75])
                        lg_est_path=os.path.join(log_path,'lg_ests.txt')
                        np.savetxt(fname=lg_est_path,X=ests)
                        calls=np.array(calls)
                        mean_calls=calls.mean()
                    
                        abs_errors=np.abs(ests-p_t)
                        rel_errors=abs_errors/p_t
                        bias=np.mean(ests)-p_t
                        MSE=np.mean(abs_errors**2)
                        MSE_adj=MSE*mean_calls
                        MSE_rel=MSE/p_t**2
                        MSE_rel_adj=MSE_rel*mean_calls
                        
                        print(f"mean est:{ests.mean()}, std est:{ests.std()}")
                        print(f"mean rel error:{rel_errors.mean()}")
                        print(f"MSE rel:{MSE/p_t**2}")
                        print(f"MSE adj.:{MSE_adj}")
                        print(f"MSE rel. adj.:{MSE_rel_adj}")
                        print(f"mean calls:{calls.mean()}")

                        times=np.array(times)  
                        ests=np.array(ests)
                        q_1,med_est,q_3=np.quantile(a=ests,q=[0.25,0.5,0.75])
                        errs=np.abs(ests-p_t)
                        #fin = np.array(finished_flags)


                        np.savetxt(fname=os.path.join(log_path,'times.txt'),X=times)
                        np.savetxt(fname=os.path.join(log_path,'ests.txt'),X=ests)

                        plt.hist(times, bins=20)
                        plt.savefig(os.path.join(log_path,'times_hist.png'))
                        plt.close()
                        
                        plt.hist(times, bins=20)
                        plt.savefig(os.path.join(log_path,'times_hist.png'))
                        plt.close()

                        plt.hist(rel_errors,bins=20)
                        plt.savefig(os.path.join(log_path,'rel_errs_hist.png'))
                        plt.close()

                        #with open(os.path.join(log_path,'results.txt'),'w'):
                        results={"p_t":p_t,"method":method_name,'T':T,'N':N,'L':L,
                        "ess_alpha":ess_t,'alpha':alpha,'n_rep':config.n_rep,'min_rate':config.min_rate,'d':d,
                        "method":method,"kernel":kernel_str,'adapt_dt':config.adapt_dt,
                        'mean_calls':calls.mean(),'std_calls':calls.std()
                        ,'mean_time':times.mean(),'std_time':times.std()
                        ,'mean_est':ests.mean(),'bias':ests.mean()-p_t,'mean abs error':abs_errors.mean(),
                        'mean_rel_error':rel_errors.mean(),'std_est':ests.std(),'freq underest':(ests<p_t).mean(), 
                        "v_min_opt":config.v_min_opt,'GV_opt':config.GV_opt
                        ,'adapt_dt_mcmc':config.adapt_dt_mcmc,"adapt_dt":config.adapt_dt,
                        "adapt_dt_mcmc":config.adapt_dt_mcmc,"dt_decay":config.dt_decay,"dt_gain":config.dt_gain,
                        "target_accept":config.target_accept,"accept_spread":config.accept_spread, 
                        "mh_opt":config.mh_opt,'only_duplicated':config.only_duplicated,
                        "np_seed":config.np_seed,"torch_seed":config.torch_seed
                        ,'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,'cores_number':config.cores_number,
                        "d":config.d,"adapt_func":config.adapt_func,
                        "ess_opt":config.ess_opt, "linear":config.linear,
                        "dt_min":config.dt_min,"dt_max":config.dt_max, "FT":config.FT,
                        "M_opt":config.M_opt,"adapt_step":config.adapt_step,"sig_dt":config.sig_dt,
                        "L_min":config.L_min,"kappa_opt":config.kappa_opt,"skip_mh":config.skip_mh,
                        "MSE":MSE,"MSE adj":MSE_adj,"MSE_rel_adj":MSE_rel_adj,
                        'q_1':q_1,'q_3':q_3,'med_est':med_est,"lg_est_path":lg_est_path,
                            "mean_log_est":mean_log_est,"std_log_est":std_log_est,
                            "lg_q_1":lg_q_1,"lg_q_3":lg_q_3,"lg_med_est":lg_med_est,}
                        exp_res.append(results)
                        results_df=pd.DataFrame([results])
                        results_df.to_csv(os.path.join(log_path,'results.csv'),index=False)
                        if config.aggr_res_path is None:
                            aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                        else:
                            aggr_res_path=config.aggr_res_path
                        if config.update_agg_res:
                            if not os.path.exists(aggr_res_path):
                                cols=['p_t','method','N','rho','n_rep','T','alpha','min_rate','mean_time','std_time','mean_est',
                                'bias','mean abs error','mean_rel_error','std_est','freq underest','gpu_name','cpu_name','ratio']
                                aggr_res_df= pd.DataFrame(columns=cols)
                            else:
                                aggr_res_df=pd.read_csv(aggr_res_path)
                            aggr_res_df = pd.concat([aggr_res_df,results_df],ignore_index=True)
                            aggr_res_df.to_csv(aggr_res_path,index=False)


    exp_df=pd.DataFrame(exp_res)
    exp_df.to_csv(os.path.join(exp_log_path,'exp_results.csv'),index=False)                    

if __name__ == "__main__":
    main()