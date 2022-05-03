import stat_reliability_measure.dev.torch_utils as t_u
import stat_reliability_measure.dev.smc.smc_pyt as smc_pyt
import scipy.stats as stat
import numpy as np
from tqdm import tqdm
from time import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import torch
import GPUtil
import foolbox as fb
import cpuinfo
import pandas as pd
import argparse
from stat_reliability_measure.dev.utils import str2bool,str2floatList,str2intList,float_to_file_float,dichotomic_search
method_name="smc_pyt"

#gaussian_linear
class config:
    dataset='imagenet'
    N=100
    N_range=[]
    T=1
    T_range=[]
    L=1
    L_range=[]
    min_rate=0.51
    
    alpha=0.002
    alpha_range=[]
    ess_alpha=0.9
    e_range=[]
   
    n_rep=10
    
    save_config=False 
    print_config=True
    
    x_min=0
    x_max=1
    x_mean=0
    x_std=1

    epsilons = None
    eps_max=0.3
    eps_min=0.1 
    eps_num=5
    model_arch='torchvision_resnet18'
    model_path=None
    export_to_onnx=False
    use_attack=True
    attack='PGD'
    lirpa_bounds=False
    download=True
    train_model=False
    
    
    noise_dist='uniform'
    d=None
    verbose=1
    log_dir=None
    aggr_res_path = None
    update_agg_res=False
    sigma=1
    v1_kernel=True
    torch_seed=None
    gpu_name=None
    cpu_name=None
    cores_number=None
    track_gpu=True
    track_cpu=True
    device=None
    n_max=10000 
    allow_multi_gpu=False
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
    dt_min=1e-5
    dt_max=1e-1
    v_min_opt=False
    ess_opt=False
    only_duplicated=False
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
    adapt_step=False
    FT=False
    sig_dt=0.015

    batch_opt=True
    track_finish=False
    lirpa_cert=False
    robust_model=False
    robust_eps=0.1
    load_batch_size=100 
    nb_epochs= 10
    adversarial_every=1
    data_dir="../../data/ImageNet/"
    p_ref_compute=False
    input_start=0
    input_stop=None

    gaussian_latent=True

    model_dir=None 
    L_min=1
    GK_opt=False
    GV_opt=False
    g_target=0.8
    skip_mh=False


parser=argparse.ArgumentParser()
parser.add_argument('--log_dir',type=str,default=config.log_dir)
parser.add_argument('--data_dir',type=str,default=config.data_dir)
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

parser.add_argument('--lambda_0',type=float,default=config.lambda_0)
parser.add_argument('--test2',type=str2bool,default =config.test2)
parser.add_argument('--print_config',type=str2bool,default=config.print_config)
parser.add_argument('--track_dt',type=str2bool,default=config.track_dt)
parser.add_argument('--linear',type=str2bool,default=config.linear)
parser.add_argument('--adapt_func',type=str,default=config.adapt_func)
parser.add_argument('--M_opt',type=str2bool,default=config.M_opt)
parser.add_argument('--adapt_step',type=str2bool,default=config.adapt_step)
parser.add_argument('--FT',type=str2bool,default=config.FT)
parser.add_argument('--sig_dt', type=float,default=config.sig_dt)

parser.add_argument('--load_batch_size',type=int,default=config.load_batch_size)
parser.add_argument('--model_arch',type=str,default = config.model_arch)
parser.add_argument('--robust_model',type=str2bool, default=config.robust_model)
parser.add_argument('--nb_epochs',type=int,default=config.nb_epochs)
parser.add_argument('--adversarial_every',type=int,default=config.adversarial_every)
parser.add_argument('--gaussian_latent',type=str2bool,default=config.gaussian_latent)

parser.add_argument('--eps_max',type=float,default=config.eps_max)
parser.add_argument('--eps_min',type=float,default=config.eps_min)
parser.add_argument('--eps_num',type=int,default=config.eps_num)
parser.add_argument('--epsilons',type=str2floatList,default=config.epsilons)
parser.add_argument('--L_min',type=int,default=config.L_min)
parser.add_argument('--GK_opt',type=str2bool,default=config.GK_opt)
parser.add_argument('--GV_opt',type=str2bool,default=config.GV_opt)
parser.add_argument('--skip_mh',type=str2bool,default=config.skip_mh)
parser.add_argument('--g_target',type=float,default=config.g_target)
parser.add_argument('--kappa_opt',type=str2bool,default=config.kappa_opt)
parser.add_argument('--only_duplicated',type=str2bool,default=config.only_duplicated)
parser.add_argument('--dataset',type=str, default=config.dataset)
args=parser.parse_args()

for k,v in vars(args).items():
    setattr(config, k, v)

if config.model_dir is None:
    config.model_dir=os.path.join("../../models/",config.dataset)
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)

config.d = t_u.datasets_dims[config.dataset]
color_dataset=config.dataset in ('cifar10','cifar100','imagenet') 
#assert config.adapt_func.lower() in smc_pyt.supported_beta_adapt.keys(),f"select adaptive function in {smc_pyt.supported_beta_adapt.keys}"
#adapt_func=smc_pyt.supported_beta_adapt[config.adapt_func.lower()]

if config.adapt_func.lower()=='simp_ess':
    adapt_func = lambda beta,v : smc_pyt.nextBetaSimpESS(beta_old=beta,v=v,lambda_0=config.lambda_0,max_beta=1e6)
elif config.adapt_func.lower()=='simp':
    adapt_func = lambda beta,v: smc_pyt.SimpAdaptBetaPyt(beta,v,config.g_target,v_min_opt=config.v_min_opt)
prblm_str=config.dataset




if len(config.e_range)==0:
    config.e_range= [config.ess_alpha]

if config.input_stop is None:
    config.input_stop=config.input_start+1

if len(config.N_range)==0:
    config.N_range= [config.N]

if config.noise_dist is not None:
    config.noise_dist=config.noise_dist.lower()

if config.noise_dist not in ['uniform','gaussian']:
    raise NotImplementedError("Only uniform and Gaussian distributions are implemented.")

if len(config.T_range)==0:
    config.T_range= [config.T]

if len(config.L_range)==0:
    config.L_range= [config.L]
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
    gpus=GPUtil.getGPUs()
    if len(gpus)>1:
        print("Multi gpus detected, only the first GPU will be tracked.")
    config.gpu_name=gpus[0].name

if config.track_cpu:
    config.cpu_name=cpuinfo.get_cpu_info()[[key for key in cpuinfo.get_cpu_info().keys() if 'brand' in key][0]]
    config.cores_number=os.cpu_count()


if config.device is None:
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    if config.verbose>=5:
        print(config.device)
    device=config.device
else:
    device=config.device

d=config.d
#epsilon=config.epsilon

if config.log_dir is None:
    config.log_dir=os.path.join('../../logs',config.dataset+'_tests')
if not os.path.exists('../../logs'):
    os.mkdir('../../logs')  
if not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir)

results_path=os.path.join(config.log_dir,'results.csv')
if os.path.exists(results_path):
    results_g=pd.read_csv(results_path)
else:
    results_g=pd.DataFrame(columns=['mean_est','mean_time','mean_err','stdtime','std_est','T','N','rho','alpha','n_rep','min_rate','method'])
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
os.mkdir(path=exp_log_path)
config.json=vars(args)

# if config.aggr_res_path is None:
#     aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
# else:
#     aggr_res_path=config.aggr_res_path

if config.dt_gain is None:
    config.dt_gain=1/config.dt_decay
config.json=vars(args)
if config.print_config:
    print(config.json)


if config.epsilons is None:
    log_min,log_max=np.log(config.eps_min),np.log(config.eps_max)
    log_line=np.linspace(start=log_min,stop=log_max,num=config.eps_num)
    config.epsilons=np.exp(log_line)

param_ranges = [config.N_range,config.T_range,config.L_range,config.e_range,config.alpha_range]
param_lens=np.array([len(l) for l in param_ranges])
nb_runs= np.prod(param_lens)

mh_str="adjusted" 
method=method_name+'_'+mh_str
save_every = 1
#adapt_func= smc_pyt.ESSAdaptBetaPyt if config.ess_opt else smc_pyt.SimpAdaptBetaPyt
num_classes=t_u.datasets_num_c[config.dataset.lower()]
print(f"Running reliability experiments on architecture {config.model_arch} trained on  {config.dataset}.")
print(f"Testing uniform noise perturbation with epsilon in {config.epsilons}")
test_loader = t_u.get_loader(train=False,data_dir=config.data_dir,download=config.download
,dataset=config.dataset,batch_size=config.load_batch_size,
           x_mean=None,x_std=None)

model,mean,std=t_u.get_model_imagenet(config.model_arch,model_dir=config.model_dir)
X_correct,label_correct,accuracy=t_u.get_correct_x_y(data_loader=test_loader,device=device,model=model)
if config.verbose>=2:
    print(f"model accuracy on test batch:{accuracy}")

x_min=torch.tensor(0)
x_max=torch.tensor(1)

if config.use_attack:
    fmodel = fb.PyTorchModel(model, bounds=(0,1),device=device)
    attack=fb.attacks.LinfPGD()
    #un-normalize data before performing attack
    #epsilons= np.array([0.0, 0.001, 0.01, 0.03,0.04,0.05,0.07,0.08,0.0825,0.085,0.086,0.087,0.09, 0.1, 0.3, 0.5, 1.0])
    _, advs, success = attack(fmodel, X_correct[config.input_start:config.input_stop], 
    label_correct[config.input_start:config.input_stop], epsilons=config.epsilons)



inp_indices=np.arange(start=config.input_start,stop=config.input_stop)
normal_dist=torch.distributions.Normal(loc=0, scale=1.)
run_nb=0
iterator= tqdm(range(config.n_rep))
exp_res=[]
model_name=config.model_arch
clip_min=torch.tensor(x_min).to(device)

clip_max=torch.tensor(x_max).to(device)
for l in inp_indices:
    with torch.no_grad():
    
        x_0,y_0 = X_correct[l], label_correct[l]

    x_0.requires_grad=True
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
        lirpa_safe=None
        if config.lirpa_cert:
            assert config.noise_dist.lower()=='uniform',"Formal certification only makes sense for uniform distributions"
            from stat_reliability_measure.dev.lirpa_utils import get_lirpa_cert
            lirpa_safe=get_lirpa_cert(x_0=x_0,y_0=y_0,model=model,epsilon=epsilon,
            num_classes=num_classes,device=config.device)

        
        
        if config.gaussian_latent:
            gen = lambda N: torch.randn(size=(N,d),device=config.device)
        else:
            gen= lambda N: (2*torch.rand(size=(N,d), device=device )-1)
        V_ = lambda X: t_u.V_pyt(X,x_0=x_0,model=model,epsilon=epsilon, target_class=y_0,gaussian_latent=config.gaussian_latent,clip_min=clip_min,
        clip_max=clip_max)
        gradV_ = lambda X: t_u.gradV_pyt(X,x_0=x_0,model=model, target_class=y_0,epsilon=epsilon, gaussian_latent=config.gaussian_latent,
        clip_min=clip_min,clip_max=clip_max)
        for ess_t in config.e_range:
            if config.adapt_func.lower()=='ess':
                adapt_func = lambda beta,v : smc_pyt.nextBetaESS(beta_old=beta,v=v,ess_alpha=ess_t,max_beta=1e6)
            for T in config.T_range:
                for L in config.L_range:
                    for alpha in config.alpha_range:       
                        for N in config.N_range:
                            loc_time= datetime.today().isoformat().split('.')[0]
                            log_name=method_name+f'_N_{N}_T_{T}_L_{L}_a_{float_to_file_float(alpha)}_ess_{float_to_file_float(ess_t)}'+'_'+loc_time.split('_')[0]
                            log_path=os.path.join(exp_log_path,log_name)
                            
                            
                            os.mkdir(path=log_path)
                            run_nb+=1
                            print(f'Run {run_nb}/{nb_runs}')
                            times=[]
                            ests = []
                            calls=[]
                            finished_flags=[]
                            iterator= tqdm(range(config.n_rep)) if config.tqdm_opt else range(config.n_rep)
                            print(f"Starting simulations with model:{model_name} img_idx:{l},eps={epsilon},ess_t:{ess_t},T:{T},alpha:{alpha},N:{N},L:{L}")
                            for i in iterator:
                                t=time()
                                p_est,res_dict,=smc_pyt.SamplerSMC(gen=gen,V= V_,gradV=gradV_,adapt_func=adapt_func,min_rate=config.min_rate,N=N,T=T,L=L,
                                alpha=alpha,n_max=config.n_max,L_min=config.L_min,
                                verbose=config.verbose, track_accept=config.track_accept,track_beta=config.track_beta,track_v_means=config.track_v_means,
                                track_ratios=config.track_ratios,track_ess=config.track_ess,kappa_opt=config.kappa_opt
                                ,gaussian =True,accept_spread=config.accept_spread, 
                                adapt_dt=config.adapt_dt, dt_decay=config.dt_decay,
                                dt_gain=config.dt_gain,dt_min=config.dt_min,dt_max=config.dt_max,
                                v_min_opt=config.v_min_opt,
                                track_dt=config.track_dt,M_opt=config.M_opt,adapt_step=config.adapt_step,FT=config.FT,
                                sig_dt=config.sig_dt, skip_mh=config.skip_mh,GV_opt=config.GV_opt
                                )
                                t1=time()-t

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
                                    

                                if config.track_dt:
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
                            calls=np.array(calls)
                        
                            

                            times=np.array(times)  
                            ests=np.array(ests)
                            
                            #fin = np.array(finished_flags)


                            np.savetxt(fname=os.path.join(log_path,'times.txt'),X=times)
                            np.savetxt(fname=os.path.join(log_path,'ests.txt'),X=ests)

                            plt.hist(times, bins=20)
                            plt.savefig(os.path.join(log_path,'times_hist.png'))
                            plt.close()
                            
                            plt.hist(times, bins=20)
                            plt.savefig(os.path.join(log_path,'times_hist.png'))
                            plt.close()

                          
                            #with open(os.path.join(log_path,'results.txt'),'w'):
                            results={"method":method_name,'T':T,'N':N,'L':L,
                            "ess_alpha":ess_t,'alpha':alpha,'n_rep':config.n_rep,'min_rate':config.min_rate,'d':d,
                            "method":method,'adapt_dt':config.adapt_dt,
                            'mean_calls':calls.mean(),'std_calls':calls.std()
                            ,'mean time':times.mean(),'std time':times.std()
                            ,'mean est':ests.mean(),'std est':ests.std(), 
                            "v_min_opt":config.v_min_opt
                            ,'adapt_dt_mcmc':config.adapt_dt_mcmc,"adapt_dt":config.adapt_dt,
                            "adapt_dt_mcmc":config.adapt_dt_mcmc,"dt_decay":config.dt_decay,"dt_gain":config.dt_gain,
                            "target_accept":config.target_accept,"accept_spread":config.accept_spread, 
                            "mh_opt":config.mh_opt,'only_duplicated':config.only_duplicated,
                            "np_seed":config.np_seed,"torch_seed":config.torch_seed
                            ,'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,'cores_number':config.cores_number,
                            "d":config.d,"adapt_func":config.adapt_func,
                            "ess_opt":config.ess_opt, "linear":config.linear,
                            "dt_min":config.dt_min,"dt_max":config.dt_max, "FT":config.FT,
                            "M_opt":config.M_opt,"adapt_step":config.adapt_step,
                            "noise_dist":config.noise_dist,"lirpa_safe":lirpa_safe,"L_min":config.L_min,
                            "skip_mh":config.skip_mh,"GV_opt":config.GV_opt}
                            exp_res.append(results)
                            results_df=pd.DataFrame([results])
                            results_df.to_csv(os.path.join(log_path,'results.csv'),index=False)
                            if config.aggr_res_path is None:
                                aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                            else:
                                aggr_res_path=config.aggr_res_path
                            if config.update_agg_res:
                                if not os.path.exists(aggr_res_path):
                                    cols=['method','N','rho','n_rep','T','alpha','min_rate','mean time','std time','mean est',
                                    'bias','mean abs error','mean rel error','std est','freq underest','gpu_name','cpu_name']
                                    aggr_res_df= pd.DataFrame(columns=cols)
                                else:
                                    aggr_res_df=pd.read_csv(aggr_res_path)
                                aggr_res_df = pd.concat([aggr_res_df,results_df],ignore_index=True)
                                aggr_res_df.to_csv(aggr_res_path,index=False)


exp_df=pd.DataFrame(exp_res)
exp_df.to_csv(os.path.join(exp_log_path,'exp_results.csv'),index=False)                    