from operator import mod
import torch

import foolbox as fb
import pandas as pd

import numpy as np
from tqdm import tqdm

from scipy.special import betainc
import GPUtil
import matplotlib.pyplot as plt
import cpuinfo
from torch import optim
import argparse
import os

from time import time
from datetime import datetime

from stat_reliability_measure.dev.torch_utils import get_model
import stat_reliability_measure.dev.torch_utils as t_u

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

#setting PRNG seeds for reproducibility

from stat_reliability_measure.dev.utils import  float_to_file_float,str2bool,str2intList,str2floatList, dichotomic_search, str2list
import stat_reliability_measure.dev.amls.amls_pyt as amls_pyt

str2floatList=lambda x: str2list(in_str=x, type_out=float)
str2intList=lambda x: str2list(in_str=x, type_out=int)
low_str=lambda x: str(x).lower()

method_name="amls_pyt"


class config:
    dataset="mnist"
    log_dir="../../logs/mnist_tests"
    model_dir="../../models/mnist"
    n_rep=10
    a=0
    verbose=0
    min_rate=0.51
    
    clip_s=True
   
    s_min=8e-3
    s_max=3
    n_max=2000
    x_min=None
    x_max=None
    x_mean=0
    x_std=1
    allow_zero_est=True
    
    N=40
    N_range=[]

    T=1
    T_range=[]

    ratio=0.6
    ratio_list=[]

    s=1
    s_list = []

    track_accept=False
    
    d = 784
    epsilon = 0.1
    
    n_max=2000
    tqdm_opt=True
    
    epsilons = []
    eps_max=0.3
    eps_min=0.1 
    eps_num=5
    allow_zero_est=True
    save_config=True
    print_config=True
    update_agg_res=True
    aggr_res_path=None
    gaussian_latent=True
    project_kernel=True
    allow_multi_gpu=False
    input_start=0
    input_stop=None
    g_target=None
    track_gpu=True
    track_cpu=True
    gpu_name=None
    cpu_name=None
    cores_number=None
    device=None
    torch_seed=0
    np_seed=0
    tf_seed=None
    model_arch='CNN_custom'
    model_path=None
    export_to_onnx=False
    use_attack=True
    attack='PGD'
    lirpa_bounds=False
    download=True
    train_model=False
    
    noise_dist='uniform'
   
    batch_opt=True
    track_finish=False
    lirpa_cert=False
    robust_model=False
    robust_eps=0.1
    load_batch_size=100 
    nb_epochs= 10
    adversarial_every=1
    data_dir="../../data"
    p_ref_compute=False
    force_train=False


parser=argparse.ArgumentParser()
parser.add_argument('--log_dir',default=config.log_dir)
parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--N',type=int,default=config.N)
parser.add_argument('--verbose',type=float,default=config.verbose)
parser.add_argument('--d',type=int,default=config.d)
parser.add_argument('--min_rate',type=float,default=config.min_rate)
parser.add_argument('--clip_s',type=str2bool,default=config.clip_s)
parser.add_argument('--s_min',type=float,default=config.s_min)
parser.add_argument('--s_max',type=float,default=config.s_max)
parser.add_argument('--n_max',type=int,default=config.n_max)
parser.add_argument('--tqdm_opt',type=str2bool,default=config.tqdm_opt)
parser.add_argument('--T',type=int,default=config.T)
parser.add_argument('--save_config',type=str2bool, default=config.save_config)
parser.add_argument('--update_agg_res',type=str2bool,default=config.update_agg_res)
parser.add_argument('--aggr_res_path',type=str, default=config.aggr_res_path)
parser.add_argument('--gaussian_latent',type=str2bool, default=config.gaussian_latent)
parser.add_argument('--allow_multi_gpu',type=str2bool)
parser.add_argument('--track_gpu',type=str2bool,default=config.track_gpu)
parser.add_argument('--track_cpu',type=str2bool,default=config.track_cpu)
parser.add_argument('--device',type=str, default=config.device)
parser.add_argument('--allow_zero_est',type=str2bool, default=config.allow_zero_est)
parser.add_argument('--torch_seed',type=int, default=config.torch_seed)
parser.add_argument('--np_seed',type=int, default=config.np_seed)
parser.add_argument('--export_to_onnx',type=str2bool, default=config.export_to_onnx)
parser.add_argument('--use_attack',type=str2bool,default=config.use_attack)
parser.add_argument('--attack',type=str,default=config.attack)
parser.add_argument('--epsilons',type=str2floatList,default=config.epsilons)
parser.add_argument('--input_start',type=int,default=config.input_start)
parser.add_argument('--input_stop',type=int,default=config.input_stop)
parser.add_argument('--lirpa_bounds',type=str2bool, default=config.lirpa_bounds)
parser.add_argument('--eps_min',type=float, default=config.eps_min)
parser.add_argument('--eps_max',type=float, default=config.eps_max)
parser.add_argument('--eps_num',type=int,default=config.eps_num)
parser.add_argument('--train_model',type=str2bool,default=config.train_model)
parser.add_argument('--noise_dist',type=str, default=config.noise_dist)
parser.add_argument('--data_dir',type=str, default=config.data_dir)
parser.add_argument('--a',type=float, default=config.a)
parser.add_argument('--N_range',type=str2intList,default=config.N_range)
parser.add_argument('--T_range',type=str2intList,default=config.T_range)
parser.add_argument('--download',type=str2bool, default=config.download)
parser.add_argument('--model_path',type=str,default=config.model_path)
parser.add_argument('--ratio',type=float,default=config.ratio)
parser.add_argument('--ratio_list',type=float,default=config.ratio_list)
parser.add_argument('--s',type=float,default=config.s)
parser.add_argument('--s_list ',type=str2floatList,default=config.s_list )
parser.add_argument('--track_finish',type=str2bool,default=config.track_finish)
parser.add_argument('--lirpa_cert',type=str2bool,default=config.lirpa_cert)
parser.add_argument('--load_batch_size',type=int,default=config.load_batch_size)
parser.add_argument('--model_arch',type=str,default = config.model_arch)
parser.add_argument('--robust_model',type=str2bool, default=config.robust_model)
parser.add_argument('--nb_epochs',type=int,default=config.nb_epochs)
parser.add_argument('--adversarial_every',type=int,default=config.adversarial_every)
parser.add_argument('--force_train',type=str2bool,default=config.force_train)
args=parser.parse_args()

for k,v in vars(args).items():
    setattr(config, k, v)

if config.dataset!='mnist':
    config.log_dir=config.log_dir.repalce('mnist',config.dataset)
    config.model_dir=config.model_dir.repalce('mnist',config.dataset)

if len(config.epsilons)==0:
    log_eps=np.linspace(start=np.log(config.eps_min),stop=np.log(config.eps_max),num=config.eps_num)
    config.epsilons=np.exp(log_eps)

if config.input_stop is None:
    config.input_stop=config.input_start+1

if config.noise_dist is not None:
    config.noise_dist=config.noise_dist.lower()

if config.noise_dist not in ['uniform','gaussian']:
    raise NotImplementedError("Only uniform and Gaussian distributions are implemented.")

if not config.allow_multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

if config.np_seed is None:
    config.np_seed=int(time.time())
np.random.seed(seed=config.np_seed)

if config.torch_seed is None:
    config.torch_seed=int(time.time())
torch.manual_seed(seed=config.torch_seed)

if len(config.T_range)<1:
    config.T_range=[config.T]
if len(config.N_range)<1:
    config.N_range=[config.N]
if len(config.s_list)<1:
    config.s_list=[config.s]
if len(config.ratio_list)<1:
    config.ratio_list=[config.ratio]


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


if not os.path.exists('../../logs'):
    print('logs not found')
    os.mkdir('../../logs')
if not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir)

raw_logs_path=os.path.join(config.log_dir,'raw_logs/'+method_name)
if not os.path.exists(raw_logs_path):
    os.mkdir(raw_logs_path)


if config.epsilons is None:
    log_min,log_max=np.log(config.eps_min),np.log(config.eps_max)
    log_line=np.linspace(start=log_min,stop=log_max,num=config.eps_num)
    config.epsilons=np.exp(log_line)

if config.aggr_res_path is None:
    aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
else:
    aggr_res_path=config.aggr_res_path



config.json=vars(config)
if config.print_config:
    print(', '.join("%s: %s" % item for item in config.json.items()))

#loading data


num_classes=10
test_loader = t_u.get_loader(train=False,data_dir=config.data_dir,download=config.download
,dataset=config.dataset,batch_size=config.load_batch_size,
           x_mean=None,x_std=None)

model, model_shape,model_name=t_u.get_model(config.model_arch, robust_model=config.robust_model, robust_eps=config.robust_eps,
    nb_epochs=config.nb_epochs,model_dir=config.model_dir,data_dir=config.data_dir,test_loader=test_loader,device=config.device,
    download=config.download,dataset=config.dataset,force_train=config.force_train)
X_correct,label_correct,accuracy=t_u.get_correct_x_y(data_loader=test_loader,device=device,model=model)
if config.verbose>=2:
    print(f"model accuracy on test batch:{accuracy}")

config.x_mean=t_u.datasets_means[config.dataset]
config.x_std=t_u.datasets_stds[config.dataset]





#X.requires_grad=True
normal_dist=torch.distributions.Normal(loc=0, scale=1.)


#inf=float('inf')

x_min=0
x_max=1
if config.use_attack:
    fmodel = fb.PyTorchModel(model, bounds=(x_min,x_max))
    attack=fb.attacks.LinfPGD()
    
   
    _, advs, success = attack(fmodel, X_correct[config.input_start:config.input_stop], 
    label_correct[config.input_start:config.input_stop], epsilons=config.epsilons)


inp_indices=np.arange(start=config.input_start,stop=config.input_stop)
i_exp=0
param_lists= [ inp_indices,config.T_range,config.N_range,config.s_list ,config.ratio_list,config.epsilons]
lenghts=np.array([len(L) for L in param_lists])
nb_exps= np.prod(lenghts)

for l in range(len(inp_indices)):
    with torch.no_grad():
    
        x_0,y_0 = X_correct[l], label_correct[l]
    input_shape=x_0.shape
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
            
        
        
        if config.gaussian_latent:
            amls_gen = lambda N: torch.randn(size=(N,d),device=config.device)
        else:
            amls_gen= lambda N: (2*torch.rand(size=(N,d), device=device )-1)
        
        
        
        amls_gen = lambda N: torch.randn(size=(N,d),device=config.device)
        normal_kernel =  lambda x,s : (x + s*torch.randn(size = x.shape,device=config.device))/np.sqrt(1+s**2) #normal law kernel, appliable to vectors 
        
            
        def h(x,gaussian_latent=config.gaussian_latent,eps=epsilon,x_min=x_min,x_max=x_max):
            x_m=x.reshape((x.shape[0],)+input_shape)
            if gaussian_latent:
                x_m=2*normal_dist.cdf(x_m)-1
            with torch.no_grad():

                input_m=torch.clamp(x_0+eps*x_m, min=x_min,max=x_max)
                
                y = model(input_m)
                y_diff = torch.cat((y[:,:y_0], y[:,(y_0+1):]),dim=1) - y[:,y_0].unsqueeze(-1)
                y_diff, _ = y_diff.max(dim=1)
            return y_diff #.max(dim=1)
        h_batch_pyt= lambda x: h(x).reshape((x.shape[0],1))
        
        for T in config.T_range:
            for N in config.N_range: 
                for s in config.s_list :
                    for ratio in config.ratio_list: 
                        loc_time= datetime.today().isoformat().split('.')[0]
                        log_name=method_name+'_e_'+float_to_file_float(config.epsilons[idx])+'_N_'+str(N)+'_T_'+str(T)+'_s_'+float_to_file_float(s)
                        log_name=log_name+'_r_'+float_to_file_float(ratio)+'_'+'_'+loc_time
                        log_path=os.path.join(raw_logs_path,log_name)
                        i_exp+=1
                        print(f"Starting experiment {i_exp}/{nb_exps}")
                                          

                        K=int(N*ratio)
                        if config.verbose>=0:
                            print(f"with model: {model_name}, img_idx:{l},eps:{epsilon},T:{T},N:{N},s:{s},K:{K}")
                        if config.verbose>3:
                            print(f"K/N:{K/N}")
                        times= []
                        rel_error= []
                        ests = [] 
                        calls=[]
                        if config.track_finish:
                            finish_flags=[]
                        for i in tqdm(range(config.n_rep)):
                            t=time()
                            if config.batch_opt:
                                amls_res=amls_pyt.ImportanceSplittingPytBatch(amls_gen, normal_kernel,K=K, N=N,s=s,  h=h_batch_pyt, 
                            tau=0 , n_max=config.n_max,clip_s=config.clip_s , T=T,
                            s_min= config.s_min, s_max =config.s_max,verbose= config.verbose,
                            device=config.device,track_accept=config.track_accept)

                            else:
                                amls_res = amls_pyt.ImportanceSplittingPyt(amls_gen, normal_kernel,K=K, N=N,s=s,  h=h_batch_pyt, 
                            tau=0 , n_max=config.n_max,clip_s=config.clip_s , T=T,
                            s_min= config.s_min, s_max =config.s_max,verbose= config.verbose,
                            device=config.device,)
                            t=time()-t
                            
                            est=amls_res[0]
                            print(f"Est:{est}")
                            dict_out=amls_res[1]
                            if config.track_accept:
                                accept_logs=os.path.join(log_path,'accept_logs')
                                if not os.path.exists(accept_logs):
                                    os.mkdir(path=accept_logs)
                                accept_rates=dict_out['accept_rates']
                                np.savetxt(fname=os.path.join(accept_logs,f'accept_rates_{i}.txt')
                                ,X=accept_rates)
                                x_T=np.arange(len(accept_rates))
                                plt.plot(x_T,accept_rates)
                                plt.savefig(os.path.join(accept_logs,f'accept_rates_{i}.png'))
                                plt.close()
                                accept_rates_mcmc=dict_out['accept_rates_mcmc']
                                x_T=np.arange(len(accept_rates_mcmc))
                                plt.plot(x_T,accept_rates_mcmc)
                                plt.savefig(os.path.join(accept_logs,f'accept_rates_mcmc_{i}.png'))
                                plt.close()
                                np.savetxt(fname=os.path.join(accept_logs,f'accept_rates_mcmc_{i}.txt')
                                ,X=accept_rates_mcmc)
                            if config.track_finish:
                                finish_flags.append(dict_out['finish_flag'])
                            times.append(t)
                            ests.append(est)
                            calls.append(dict_out['Count_h'])
            
        
                        times=np.array(times)
                        estimates = np.array(ests)
                        mean_est=estimates.mean()
                        calls=np.array(calls)
                        mean_calls=calls.mean()
                        std_est=estimates.std()
                        std_rel=std_est/mean_est
                        print(f"mean est:{estimates.mean()}, std est:{estimates.std()}")
                        print(f"mean calls:{calls.mean()}")
                        print(f"std. re.:{std_rel}")
                        print(f"std. rel. adj.:{std_rel*mean_calls}")
                        
                        if config.track_finish:
                            finish_flags=np.array(finish_flags)
                            freq_finished=finish_flags.mean()
                            freq_zero_est=(estimates==0).mean()
                        else:
                            freq_zero_est,freq_finished=None,None
                        #finished=np.array(finish_flag)
                        if config.track_finish and freq_finished<1:
                            unfinish_est=estimates[~finish_flags]
                            unfinish_times=times[~finish_flags]
                            unfinished_mean_est=unfinish_est.mean()
                            unfinished_mean_time=unfinish_times.mean()
                        else:
                            unfinished_mean_est,unfinished_mean_time=None,None
                        if os.path.exists(log_path):
                            log_path=log_path+'_rand_'+str(np.random.randint(low=0,high=9))
                        os.mkdir(log_path)
                        np.savetxt(fname=os.path.join(log_path,'times.txt'),X=times)
                        np.savetxt(fname=os.path.join(log_path,'estimates.txt'),X=estimates)

                        

                        plt.hist(times, bins=10)
                        plt.savefig(os.path.join(log_path,'times_hist.png'))
                        plt.hist(estimates,bins=10)
                        plt.savefig(os.path.join(log_path,'estimates_hist.png'))
                    
                        

                        #with open(os.path.join(log_path,'results.txt'),'w'):
                        results={'method':method_name,'gaussian_latent':str(config.gaussian_latent),'image_idx':l,
                            'epsilon':epsilon,"model_name":model_name,'n_rep':config.n_rep,'T':T,'ratio':ratio,'K':K,'s':s,
                        'min_rate':config.min_rate, "N":N, "mean_calls":calls.mean(),"std_calls":calls.std(),"std adj":std_rel*mean_calls,
                        'mean time':times.mean(),'std time':times.std(),'mean est':estimates.mean(),
                        'std est':estimates.std(),'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,
                        'cores_number':config.cores_number,'g_target':config.g_target,"std_rel":std_rel, 
                        'freq_finished':freq_finished,'freq_zero_est':freq_zero_est,'unfinished_mean_time':unfinished_mean_time,
                        'unfinished_mean_est':unfinished_mean_est
                        ,'np_seed':config.np_seed,'torch_seed':config.torch_seed,'pgd_success':pgd_success,'p_l':p_l,
                        'p_u':p_u,'noise_dist':config.noise_dist,'datetime':loc_time}
                        results_df=pd.DataFrame([results])
                        results_df.to_csv(os.path.join(log_path,'results.csv'),)
                        if config.aggr_res_path is None:
                            aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                        else: 
                            aggr_res_path=config.aggr_res_path

                        if config.update_agg_res:
                            if not os.path.exists(aggr_res_path):
                                print(f'aggregate results csv file not found /n it will be build at {aggr_res_path}')
                                cols=['method','gaussian_latent','N','rho','n_rep','T','epsilon','alpha','min_rate','mean time','std time','mean est',
                                'std est','freq underest','g_target']
                                cols+=['freq_finished','freq_zero_est','unfinished_mean_est','unfinished_mean_time']
                                cols+=['pgd_success','p_l','p_u','gpu_name','cpu_name','np_seed','torch_seed','noise_dist','datetime']
                                agg_res_df= pd.DataFrame(columns=cols)

                            else:
                                agg_res_df=pd.read_csv(aggr_res_path)
                            agg_res_df = pd.concat([agg_res_df,results_df],ignore_index=True)
                            agg_res_df.to_csv(aggr_res_path,index=False)