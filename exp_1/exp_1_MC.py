import numpy as np 
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import GPUtil
import cpuinfo
import scipy.stats as stat
import argparse
from tqdm import tqdm
import torch
from stat_reliability_measure.home import ROOT_DIR
from datetime import datetime
from pathlib import Path
from stat_reliability_measure.dev.utils import  float_to_file_float, str2bool, str2intList, str2floatList, get_sel_df, print_config
import stat_reliability_measure.dev.mc.mc_pyt as mc_pyt


method_name="MC"
class config:
    

    n_rep=200
    N=int(1e5)
    N_range=[]
    batch_size=int(1e4)
    b_range=[]
    p_t=1e-15
    p_range=[]
    
    
    
    
    
    verbose=0


    allow_zero_est=True
    
    
    


    

    
    d = 1024
    epsilon = 1
    
    
    tqdm_opt=True
    save_config = True
    print_config=True
    update_aggr_res=True
    aggr_res_path = None

    track_advs=False
    track_finish=True
    device = None

    torch_seed=0
    np_seed=0

    log_dir=ROOT_DIR+'/logs/exp_1_linear_gaussian'
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
    adapt_kernel = False
    repeat_exp=True


parser=argparse.ArgumentParser()

parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--verbose',type=float,default=config.verbose)

parser.add_argument('--allow_zero_est',type=str2bool, default=config.allow_zero_est)

parser.add_argument('--N',type=int,default=config.N)
parser.add_argument('--N_range',type=str2intList,default=config.N_range)

parser.add_argument('--p_t',type=float,default=config.p_t)
parser.add_argument('--p_range',type=str2floatList,default=config.p_range)
parser.add_argument('--batch_size',type=int,default=config.batch_size)
parser.add_argument('--b_range',type=str2intList,default=config.b_range)


parser.add_argument('--d',type=int,default=config.d)
parser.add_argument('--epsilon',type=float, default=config.epsilon)
parser.add_argument('--tqdm_opt',type=bool,default=config.tqdm_opt)
parser.add_argument('--save_config', type=bool, default=config.save_config)
parser.add_argument('--print_config',type=bool , default=config.print_config)
parser.add_argument('--update_aggr_res', type=bool,default=config.update_aggr_res)

parser.add_argument('--aggr_res_path',type=str, default=config.aggr_res_path)
parser.add_argument('--track_gpu',type=str2bool,default=config.track_gpu)
parser.add_argument('--track_cpu',type=str2bool,default=config.track_cpu)
parser.add_argument('--log_dir',default=config.log_dir)
parser.add_argument('--allow_multi_gpu',type=str2bool,default=config.allow_multi_gpu)

parser.add_argument('--track_advs',type=str2bool,default= config.track_advs)
parser.add_argument('--track_finish',type=str2bool,default=config.track_finish)
parser.add_argument('--np_seed',type=int,default=config.np_seed)
parser.add_argument('--torch_seed',type=int,default=config.torch_seed)
parser.add_argument('--repeat_exp',type=str2bool,default= config.repeat_exp)

args=parser.parse_args()
for k,v in vars(args).items():
    setattr(config, k, v)

def main():
    #nb_runs=config.n_rep
    nb_runs=1
    if len(config.N_range)==0:
        config.N_range=[config.N]
    nb_runs*=len(config.N_range)
    if len(config.b_range)==0:
        config.b_range=[config.batch_size]
    nb_runs*=len(config.b_range)
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


    epsilon=config.epsilon
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
    log_name=method_name+'_'+'_'+loc_time+'_'+str(np.random.randint(0,10000))
    np.random.seed(config.np_seed)
    exp_log_path=os.path.join(raw_logs_path,log_name)
    os.mkdir(exp_log_path)
    exp_res = []
    config.json=vars(args)

    config_dict=print_config(config)
    config_path=os.path.join(exp_log_path,'config.json')
    with open(config_path,'w') as f:
        f.write(json.dumps(config_dict, indent = 4))
    # if config.save_confi
    # if config.save_config:
    #     with open(file=os.path.join(),mode='w') as f:
    #         f.write(config.json)

    get_c_norm= lambda p:stat.norm.isf(p)
    i_run=0

    for p_t in config.p_range:
        c=get_c_norm(p_t)
        P_target=stat.norm.sf(c)
        if config.verbose>=5:
            print(f"P_target:{P_target}")
        arbitrary_thresh=40 #pretty useless a priori but should not hurt results
        def v_batch_pyt(X,c=c):
            return torch.clamp(input=c-X[:,0],min=-arbitrary_thresh, max = None)
        amls_gen = lambda N: torch.randn(size=(N,d),device=config.device)
    
        h_V_batch_pyt= lambda x: -v_batch_pyt(x)


        for N in config.N_range: 
            for bs in config.b_range:
                i_run+=1
                aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                if (not config.repeat_exp) and config.update_aggr_res and os.path.exists(aggr_res_path):
                    aggr_res_df = pd.read_csv(aggr_res_path)
                    same_exp_df = get_sel_df(df=aggr_res_df,triplets=[('method',method_name,'='),
                    ('p_t',p_t,'='),('n_rep',config.n_rep,'='),('N',N,'='),
                    ('batch_size',bs,'=')] )  
                    # if a similar experiment has been done in the current log directory we skip it
                    if len(same_exp_df)>0:
                        
                        print(f"Skipping {method_name} run {i_run}/{nb_runs}, with p_t= {p_t},N={N},batch size={bs}")
                        continue
                loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
                log_name=method_name+'_'+'_'+loc_time
                log_name=method_name+f'_N_{N}_bs_{bs}_t_'+'_'+loc_time.split('_')[0]
                log_path=os.path.join(exp_log_path,log_name)
                os.mkdir(path=log_path)
                
                
                
                print(f"Starting Crude MC run {i_run}/{nb_runs}, with p_t= {p_t},N={N},batch size={bs}")
                
                times= []
                rel_error= []
                ests = [] 
                calls=[]
                if config.track_finish:
                    finish_flags=[]
                for i in tqdm(range(config.n_rep)):
                    t=time()
                    est = mc_pyt.MC_pf(gen=amls_gen, score=h_V_batch_pyt, N_mc=N,batch_size=bs,track_advs=config.track_advs)
                    t=time()-t
                    
                
                
                    times.append(t)
                    ests.append(est)
                    calls.append(N)


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
                results={'p_t':p_t,'method':method_name,'batch_size':bs,
                'N':N,'n_rep':config.n_rep,'lg_est_path':lg_est_path
                ,'mean_est':ests.mean(),'std_log_est':log_ests.std(),'mean_log_est':mean_log_est,
                'lg_q_1':lg_q_1,'lg_q_3':lg_q_3,"lg_med_est":lg_med_est
                ,'mean_time':times.mean()
                ,'std_time':times.std(),'MSE':MSE,'MSE_rel_adj':MSE_rel_adj,'MSE_rel':MSE_rel,
                'mean_calls':mean_calls,'last_particle':config.last_particle,
                'std_calls':std_calls
                ,'bias':ests.mean()-p_t,'mean abs error':abs_errors.mean(),
                'mean_rel_error':rel_errors.mean(),'std_est':ests.std(),'freq underest':(ests<p_t).mean()
                ,'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,'cores_number':config.cores_number,
                'batch_opt':config.batch_opt,"d":d, "correct_T":config.correct_T,
                "np_seed":config.np_seed,"torch_seed":config.torch_seed,
                    'q_1':q_1,'q_3':q_3,'med_est':med_est}
                exp_res.append(results)
                results_df=pd.DataFrame([results])
                results_df.to_csv(os.path.join(log_path,'results.csv'),index=False)
                if config.aggr_res_path is None:
                    aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                else:
                    aggr_res_path=config.aggr_res_path
                if config.update_aggr_res:
                    if not os.path.exists(aggr_res_path):
                        cols=['p_t','method','N','rho','n_rep','alpha','min_rate','mean_time','std_time','mean_est',
                        'mean_calls','std_calls','ratio', 'T',                      
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