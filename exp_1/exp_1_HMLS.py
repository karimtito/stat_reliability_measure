import numpy as np 
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as stat
import argparse
from tqdm import tqdm
import torch
from stat_reliability_measure.home import ROOT_DIR
from datetime import datetime
import json
from stat_reliability_measure.dev.utils import  float_to_file_float,str2bool,str2intList,str2floatList, get_sel_df, print_config
import stat_reliability_measure.dev.hybrid_mls.hybrid_mls_pyt as hmls_pyt
from stat_reliability_measure.dev.torch_utils import adapt_verlet_mcmc,verlet_mcmc


method_name="HMLS"
class config:
    
    alpha=0.2
    alpha_range=[]
    n_rep=200
    T_range=[10,20,50]
    N_range=[32,64,128,256,512,1024]
    ratio_range=[0.1,0.85,0.95]
    p_range=[1e-6,1e-12]
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
    GV_opt=False
    
    
    
    verbose=0
    min_rate=0.2
    
    n_max=2000
    decay=0.95
    gain_rate=1.0001
    allow_zero_est=True
    
    N=40
    

    T=1
    L=1

    ratio=0.6

    s=1 
    s_range= []

    p_t=1e-15

    sig_dt=0.02
    d = 1024
    epsilon = 1
    
    track_dt=True
    tqdm_opt=True
    save_config = True
    print_config=True
    update_aggr_res=True
    aggr_res_path = None

    track_accept=False
    track_finish=True
    device = None

    torch_seed=0
    np_seed=0

    log_dir=ROOT_DIR+"/logs/linear_gaussian_tests"
    batch_opt=True
    allow_multi_gpu=True
    track_gpu=False
    track_cpu=False
    core_numbers=None
    gpu_name=None 
    cpu_name=None
    cores_number=None
    correct_T=False
    last_particle=False
    adapt_kernel = True
    repeat_exp = False
    FT=True

parser=argparse.ArgumentParser()

parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--verbose',type=float,default=config.verbose)
parser.add_argument('--min_rate',type=float,default=config.min_rate)

parser.add_argument('--n_max',type=int,default=config.n_max)
parser.add_argument('--allow_zero_est',type=str2bool, default=config.allow_zero_est)

parser.add_argument('--N',type=int,default=config.N)
parser.add_argument('--N_range',type=str2intList,default=config.N_range)
parser.add_argument('--T',type=int,default=config.T)
parser.add_argument('--L',type=int,default=config.L)
parser.add_argument('--T_range',type=str2intList,default=config.T_range)
parser.add_argument('--ratio',type=float,default=config.ratio)
parser.add_argument('--ratio_range',type=str2floatList,default=config.ratio_range)
parser.add_argument('--s',type=float,default=config.s)
parser.add_argument('--s_range',type=str2floatList,default=config.s_range)
parser.add_argument('--p_t',type=float,default=config.p_t)
parser.add_argument('--p_range',type=str2floatList,default=config.p_range)

parser.add_argument('--d',type=int,default=config.d)
parser.add_argument('--epsilon',type=float, default=config.epsilon)
parser.add_argument('--tqdm_opt',type=bool,default=config.tqdm_opt)
parser.add_argument('--save_config', type=bool, default=config.save_config)
parser.add_argument('--print_config',type=bool , default=config.print_config)
parser.add_argument('--update_aggr_res', type=bool,default=config.update_aggr_res)
parser.add_argument('--adapt_dt',type=str2bool,default=config.adapt_dt)
parser.add_argument('--target_accept',type=float,default=config.target_accept)
parser.add_argument('--accept_spread',type=float,default=config.accept_spread)
parser.add_argument('--dt_decay',type=float,default=config.dt_decay)
parser.add_argument('--dt_gain',type=float,default=config.dt_gain)
parser.add_argument('--dt_min',type=float,default=config.dt_min)
parser.add_argument('--dt_max',type=float,default=config.dt_max)
parser.add_argument('--alpha',type=float,default=config.alpha)
parser.add_argument('--adapt_dt_mcmc',type=str2bool,default=config.adapt_dt_mcmc)
parser.add_argument('--aggr_res_path',type=str, default=config.aggr_res_path)
parser.add_argument('--track_gpu',type=str2bool,default=config.track_gpu)
parser.add_argument('--track_cpu',type=str2bool,default=config.track_cpu)
parser.add_argument('--log_dir',default=config.log_dir)
parser.add_argument('--allow_multi_gpu',type=str2bool,default=config.allow_multi_gpu)
parser.add_argument('--batch_opt',type=str2bool,default=config.batch_opt)
parser.add_argument('--track_accept',type=str2bool,default= config.track_accept)
parser.add_argument('--track_finish',type=str2bool,default=config.track_finish)
parser.add_argument('--np_seed',type=int,default=config.np_seed)
parser.add_argument('--torch_seed',type=int,default=config.torch_seed)
parser.add_argument('--decay',type=float,default=config.decay)
parser.add_argument('--gain_rate',type=float,default=config.gain_rate)
parser.add_argument('--sig_dt',type=float,default=config.sig_dt)
parser.add_argument('--correct_T',type=str2bool,default=config.correct_T)
parser.add_argument('--last_particle',type=str2bool,default=config.last_particle)
parser.add_argument('--adapt_kernel',type=str2bool,default=config.adapt_kernel)
parser.add_argument('--repeat_exp',type=str2bool,default=config.repeat_exp)
parser.add_argument('--FT',type=str2bool,default=config.FT)
parser.add_argument('--GV_opt',type=str2bool,default=config.GV_opt)
parser.add_argument('--track_dt',type=str2bool,default=config.track_dt)
args=parser.parse_args()
for k,v in vars(args).items():
    setattr(config, k, v)

def main():
    #nb_runs=config.n_rep
    nb_runs=1
    if len(config.N_range)==0:
        config.N_range=[config.N]
    nb_runs*=len(config.N_range)
    if len(config.T_range)==0:
        config.T_range=[config.T]
    nb_runs*=len(config.T_range)
    if len(config.ratio_range)==0:
        config.ratio_range=[config.ratio]
    nb_runs*=len(config.ratio_range)
    if len(config.s_range)==0:
        config.s_range=[config.s]
    nb_runs*=len(config.s_range)
    if len(config.p_range)==0:
        config.p_range=[config.p_t]
    nb_runs*=len(config.p_range)

    if config.device is None:
        device= 'cuda:0' if torch.cuda.is_available() else 'cpu'


    if not config.allow_multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"

    if config.track_gpu:
        import GPUtil
        gpus=GPUtil.getGPUs()
        if len(gpus)>1:
            print("Multi gpus detected, only the first GPU will be tracked.")
        config.gpu_name=gpus[0].name

    if config.track_cpu:
        config.cpu_name=cpuinfo.get_cpu_info()[[key for key in cpuinfo.get_cpu_info().keys() if 'brand' in key][0]]
        config.cores_number=os.cpu_count()


    d=config.d

    if not os.path.exists(ROOT_DIR+'/logs'):
        os.mkdir(ROOT_DIR+'/logs')
        os.mkdir(config.log_dir)
    elif not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    raw_logs=os.path.join(config.log_dir,'raw_logs/')
    if not os.path.exists(raw_logs):
        os.mkdir(raw_logs)
    raw_logs_path=os.path.join(raw_logs,method_name)
    if not os.path.exists(raw_logs_path):
        os.mkdir(raw_logs_path)

    loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
    log_name=method_name+'_'+'_'+loc_time
    exp_log_path=os.path.join(raw_logs_path,log_name)

    os.mkdir(exp_log_path)
    exp_res = []
    config.json=vars(args)
    #if config.print_config:
    config_dict=print_config(config)
    path_config=os.path.join(exp_log_path,'config.json')
    with open(path_config,'w') as f:
        f.write(json.dumps(config_dict, indent = 4))
    # if config.save_confi
    # if config.save_config:
    #     with open(file=os.path.join(),mode='w') as f:
    #         f.write(config.json)

    
    
    get_c_norm= lambda p:stat.norm.isf(p)
    i_run=0


    for p_t in config.p_range:
        get_c_norm= lambda p:stat.norm.isf(p)
        c=get_c_norm(p_t)
        if config.verbose>=1.:
            print(f'c:{c}')
        e_1= torch.Tensor([1]+[0]*(d-1)).to(device)
        
        def V(X):
            return torch.clamp(input=c-X[:,0], min=0, max=None)
        def gradV(X):
            return -torch.transpose(e_1[:,None]*(X[:,0]<c),dim0=1,dim1=0)
        
        def norm_gen(N):
            return torch.randn(size=(N,d)).to(device)
        
        for T in config.T_range:
            for N in config.N_range: 
                for s in config.s_range:
                    for ratio in config.ratio_range: 
                        i_run+=1
                        aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                        
                        if (not config.repeat_exp) and config.update_aggr_res and os.path.exists(aggr_res_path):
                            aggr_res_df = pd.read_csv(aggr_res_path)
                            same_exp_df = get_sel_df(df=aggr_res_df,triplets=[('method',method_name,'='),
                            ('p_t',p_t,'='),('n_rep',config.n_rep,'='),('alpha',config.alpha,'='),
                            ('N',N,'='),
                            ('T',T,'='),('ratio',ratio,'='),('last_particle',config.last_particle,'==')] )  
                            # if a similar experiment has been done in the current log directory we skip it
                            if len(same_exp_df)>0:
                                K=int(N*ratio) if not config.last_particle else N-1
                                print(f"Skipping HMLS run {i_run}/{nb_runs}, with p_t= {p_t},N={N},K={K},T={T},ratio={ratio},alpha={config.alpha},n_rep={config.n_rep}")
                                continue
                        loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
                        

                        log_name=method_name+f'_N_{N}_T_{T}_s_{float_to_file_float(s)}_r_{float_to_file_float(ratio)}_t_'+loc_time.split('_')[0]
                        log_name=log_name+f"_r_{np.random.randint(low=0,high=10000)}"
                        log_path=os.path.join(exp_log_path,log_name)
                        os.mkdir(path=log_path)
                        
                        
                        gibbs_kernel = verlet_mcmc if not config.adapt_kernel else adapt_verlet_mcmc
                        K = int(N*(ratio)) #if not config.last_particle else int(N-1)
                        print(f"Starting Hybrid-MLS run {i_run}/{nb_runs}, with p_t= {p_t},N={N},K={K},T={T},ratio={ratio}")
                        times= []
                        ests = [] 
                        calls=[]
                        if config.track_finish:
                            finish_flags=[]
                        for i in tqdm(range(config.n_rep)):
                            t=time()
                            amls_res=hmls_pyt.HybridMLS(norm_gen, V=V,gradV=gradV
                             ,K=K, N=N, L=config.L,gibbs_kernel=gibbs_kernel,
                            tau=0 , n_max=config.n_max, T=T
                            ,verbose= config.verbose,
                            device=config.device,track_accept=config.track_accept,
                            accept_spread=config.accept_spread, 
                            adapt_dt=config.adapt_dt, dt_decay=config.dt_decay,
                            only_duplicated=config.only_duplicated,
                            dt_gain=config.dt_gain,dt_min=config.dt_min,dt_max=config.dt_max,
                            GV_opt=config.GV_opt,sig_dt=config.sig_dt,
                            alpha=config.alpha,track_dt=config.track_dt,
                            
                            )

                      
                                
                            t=time()-t
                            
                            est=amls_res[0]
                            
                            dict_out=amls_res[1]
                            if config.track_accept:
                                accept_logs=os.path.join(log_path,'accept_logs')
                                if not os.path.exists(accept_logs):
                                    os.mkdir(path=accept_logs)
                                # accept_rates=dict_out['accept_rates']
                                # np.savetxt(fname=os.path.join(accept_logs,f'accept_rates_{i}.txt')
                                # ,X=accept_rates)
                                # x_T=np.arange(len(accept_rates))
                                # plt.plot(x_T,accept_rates)
                                # plt.savefig(os.path.join(accept_logs,f'accept_rates_{i}.png'))
                                # plt.close()
                                accept_rates_mcmc=dict_out['accept_rates_mcmc']
                                x_T=np.arange(len(accept_rates_mcmc))
                                plt.plot(x_T,accept_rates_mcmc)
                                plt.savefig(os.path.join(accept_logs,f'accept_rates_mcmc_{i}.png'))
                                plt.close()
                                np.savetxt(fname=os.path.join(accept_logs,f'accept_rates_mcmc_{i}.txt')
                                ,X=accept_rates_mcmc)
                            if config.track_finish:
                                finish_flags.append(dict_out['finish_flag'])
                            if config.track_dt:
                                dt_logs=os.path.join(log_path,'dt_logs')
                                if not os.path.exists(dt_logs):
                                    os.mkdir(path=dt_logs)

                                dt_means = dict_out['dt_means']
                                dt_stds = dict_out['dt_stds']
                                np.savetxt(fname=os.path.join(dt_logs,f'dt_means_{i}.txt'),X=dt_means)
                                np.savetxt(fname=os.path.join(dt_logs,f'dt_stds_{i}.txt'),X=dt_stds)
                                x_T=np.arange(len(dt_means))
                                plt.errorbar(x_T,dt_means,yerr=dt_stds,label='dt')
                                plt.savefig(os.path.join(dt_logs,f'dt_{i}.png'))
                                plt.close()
                                
                            times.append(t)
                            ests.append(est)
                        calls.append(dict_out['Count_V'])


                        times=np.array(times)
                        ests = np.array(ests)
                        log_ests=np.log(np.clip(ests,a_min=1e-250,a_max=1))
                        std_log_est=log_ests.std()
                        mean_log_est=log_ests.mean()
                        lg_q_1,lg_med_est,lg_q_3=np.quantile(a=ests,q=[0.25,0.5,0.75])
                        lg_est_path=os.path.join(log_path,'lg_ests.txt')
                        np.savetxt(fname=lg_est_path,X=log_ests)
                    
                        abs_errors=np.abs(ests-p_t)
                        rel_errors=abs_errors/p_t
                        bias=np.mean(ests)-p_t

                        times=np.array(times)  
                        ests=np.array(ests)
                        calls=np.array(calls)



                        errs=np.abs(ests-p_t)
                        q_1,med_est,q_3=np.quantile(a=ests,q=[0.25,0.5,0.75])
                        mean_calls=calls.mean()
                        std_calls=calls.std()
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


                        np.savetxt(fname=os.path.join(log_path,'times.txt'),X=times)
                        np.savetxt(fname=os.path.join(log_path,'ests.txt'),X=ests)

                        plt.hist(times, bins=20)
                        plt.savefig(os.path.join(log_path,'times_hist.png'))
                        plt.close()
                        plt.hist(rel_errors,bins=20)
                        plt.savefig(os.path.join(log_path,'rel_errs_hist.png'))
                        plt.close()

                        #with open(os.path.join(log_path,'results.txt'),'w'):
                        results={'p_t':p_t,'method':method_name,
                        'N':N,'L':config.L,'n_rep':config.n_rep,'T':T,'ratio':ratio,'K':K,'lg_est_path':lg_est_path
                        ,'min_rate':config.min_rate,'mean_est':ests.mean(),'std_log_est':log_ests.std(),'mean_log_est':mean_log_est,
                        'lg_q_1':lg_q_1,'lg_q_3':lg_q_3,"lg_med_est":lg_med_est
                        ,'mean_time':times.mean(),'alpha':config.alpha
                        ,'std_time':times.std(),'MSE':MSE,'MSE_rel_adj':MSE_rel_adj,'MSE_rel':MSE_rel,
                        'mean_calls':mean_calls,'last_particle':config.last_particle,
                        'std_calls':std_calls
                        ,'bias':ests.mean()-p_t,'mean abs error':abs_errors.mean(),
                        'mean_rel_error':rel_errors.mean(),'std_est':ests.std(),'freq underest':(ests<p_t).mean()
                        ,'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,'cores_number':config.cores_number,
                        'batch_opt':config.batch_opt,"d":d, "correct_T":config.correct_T,
                        "np_seed":config.np_seed,"torch_seed":config.torch_seed,
                            'q_1':q_1,'q_3':q_3,'med_est':med_est,
                            "dt_min":config.dt_min,"dt_max":config.dt_max, "FT":config.FT,'sig_dt':config.sig_dt,
                        "GV_opt":config.GV_opt
                            }
                        exp_res.append(results)
                        results_df=pd.DataFrame([results])
                        results_df.to_csv(os.path.join(log_path,'results.csv'),index=False)
                        if config.aggr_res_path is None:
                            aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                        else:
                            aggr_res_path=config.aggr_res_path
                        if config.update_aggr_res:
                            if not os.path.exists(aggr_res_path):
                                cols=['p_t','method','N','rho','n_rep','T','alpha','min_rate','mean_time','std_time','mean_est',
                                'mean_calls','std_calls',
                                'bias','mean abs error','mean_rel_error','std_est','freq underest','gpu_name','cpu_name']
                                aggr_res_df= pd.DataFrame(columns=cols)
                            else:
                                aggr_res_df=pd.read_csv(aggr_res_path)
                            aggr_res_df = pd.concat([aggr_res_df,results_df],ignore_index=True)
                            aggr_res_df.to_csv(aggr_res_path,index=False)

        exp_df=pd.DataFrame(exp_res)
        exp_df.to_csv(os.path.join(exp_log_path,'exp_results.csv'),index=False)

if __name__ == "__main__":
    main()