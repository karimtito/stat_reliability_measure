import numpy as np 
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import GPUtil
import cpuinfo
import scipy.stats as stat
import argparse
import json
from tqdm import tqdm
import torch
from stat_reliability_measure.home import ROOT_DIR
from datetime import datetime
from pathlib import Path
from stat_reliability_measure.dev.utils import float_to_file_float,str2bool,str2intList,str2floatList,print_config
import stat_reliability_measure.dev.torch_utils as t_u 

from stat_reliability_measure.dev.form.form_pyt import find_zero_gd_pyt 

method_name="FORM"

class config:
    n_rep=1
    dataset='mnist'
    epsilon=1e-15
    epsilons=[]
    model_arch=None 
    optim_steps=10
    opt_steps_list = []
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
    log_dir=None
    
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
    search_method="gd"
    step_size_range = []
    step_size=0.1
    tol=1e-3
    max_iter=100
    max_iter_range=[]
    random_init=False

    eps_max=0.3
    eps_min=0.2
    eps_num=5
    dataset='mnist'
    model_arch=None  
    epsilons = [0.15]
    save_config=False 
    print_config=True
    eps_max=0.3
    eps_min=0.2
    eps_num=5

    model_path=None
    export_to_onnx=False
    use_attack=False
    attack='PGD'
    lirpa_bounds=False
    download=True
    train_model=False
    train_model_epochs=10
    input_start=0
    input_stop=None

    gaussian_latent=True

    model_dir=None 
    noise_dist='uniform'
    d=None
    verbose=0
    log_dir=None
    aggr_res_path = None
    update_aggr_res=True
    repeat_exp=False
    data_dir=ROOT_DIR+"/data"
    load_batch_size=100
    robust_model=False
    robust_eps=0.3
    nb_epochs=10
    force_train=False
    lirpa_cert=False

parser=argparse.ArgumentParser()

parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--verbose',type=float,default=config.verbose)

parser.add_argument('--allow_zero_est',type=str2bool, default=config.allow_zero_est)
parser.add_argument('--random_init',type=str2bool, default=config.random_init)

parser.add_argument('--epsilon',type=float,default=config.epsilon)
parser.add_argument('--epsilons',type=str2floatList,default=config.epsilons)
parser.add_argument('--step_size',type=float,default=config.step_size)
parser.add_argument('--step_size_range',type=str2floatList,default=config.step_size_range)

parser.add_argument('--tol',type=float,default=config.tol)
parser.add_argument('--max_iter',type=int,default=config.max_iter)
parser.add_argument('--max_iter_range',type=str2intList,default=config.max_iter_range)
parser.add_argument('--d',type=int,default=config.d)
parser.add_argument('--optim_steps',type=float, default=config.optim_steps)
parser.add_argument('--opt_steps_list',type=str2floatList,default=config.opt_steps_list)
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
parser.add_argument('--input_start',type=int, default=config.input_start)
parser.add_argument('--input_stopt',type=int, default=config.input_stop)
parser.add_argument('--repeat_exp',type=str2bool,default=config.repeat_exp)

args=parser.parse_args()
for k,v in vars(args).items():
    setattr(config, k, v)

#gradient descent in 1 dimension to find zero of a function f
def find_zero_gd_pyt(f, grad_f, x0, obj='min', step_size=1e-2, max_iter=100, tol=1e-3,random_init=False):
    x = x0
    
    if random_init:
        x = x0 + step_size*torch.randn_like(x0)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    f_calls = 0
    sign= 1 if obj=='max' else -1
    for i in range(max_iter):
        x = x + sign*step_size * grad_f(x)
        f_calls += 2 # we count 2 calls for each gradient evaluation (forward and backward)
        if sign*f(x) > -tol:
            f_calls += 1 # we count one more call for the last function evaluation
            break
    if i == max_iter-1:
        print('Warning: max_iter reached in find_zero_gd')
    return x, f_calls

def main():
    #nb_runs=config.n_rep
    if config.model_dir is None:
        config.model_dir=os.path.join(ROOT_DIR+"/models/",config.dataset)

    if config.model_arch is None:
        config.model_arch = t_u.datasets_default_arch[config.dataset]

    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)

    if config.input_stop is None:
        config.input_stop=config.input_start+1
    else:
        assert config.input_start<config.input_stop,"/!\ input start must be strictly lower than input stop"
    config.d = t_u.datasets_dims[config.dataset]
    color_dataset=config.dataset in ('cifar10','cifar100','imagenet') 
    nb_runs=1

    if len(config.step_size_range)==0:
        config.step_size_range=[config.step_size]
    if len(config.max_iter_range)==0:
        config.max_iter_range=[config.max_iter]
    nb_runs*=len(config.step_size_range)
    if len(config.epsilons)==0:
        config.epsilons=[config.epsilon]
    nb_runs*=len(config.epsilons)

    if config.device is None:
        device= 'cuda:0' if torch.cuda.is_available() else 'cpu'
        config.device=device

    if not config.allow_multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"

    if len(config.opt_steps_list)==0:
        config.opt_steps_list = [config.optim_steps]
    nb_runs*=len(config.opt_steps_list)
    if config.log_dir is None:
        config.log_dir=os.path.join(ROOT_DIR+'/logs',config.dataset+'_tests')
    if config.noise_dist not in ['uniform','gaussian']:
        raise NotImplementedError("Only uniform and Gaussian distributions are implemented.")
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
    log_name=method_name+'_'+loc_time
    print(log_name)
    exp_log_path=os.path.join(raw_logs_path,log_name)
    os.mkdir(exp_log_path)
    exp_res = []
    config.json=vars(args)
    if config.print_config:
        print_config(config)
    #if config.save_config:
    #     with open(file=os.path.join(),mode='w') as f:
    #         f.write(config.json)

    epsilon=config.epsilon
    

    i_run=0
    if config.search_method.lower() in ('gd','gradient_descent'):
        dp_search_method= find_zero_gd_pyt
    else:
        raise NotImplementedError(f"Search method {config.search_method} not implemented")
    method=method_name
    
    num_classes=t_u.datasets_num_c[config.dataset.lower()]
    print(f"Running reliability experiments on architecture {config.model_arch} trained on {config.dataset}.")
    print(f"Testing uniform noise pertubatin with epsilon in {config.epsilons}")
    test_loader = t_u.get_loader(train=False,data_dir=config.data_dir,download=config.download
    ,dataset=config.dataset,batch_size=config.load_batch_size,
            x_mean=None,x_std=None)

    model, model_shape,model_name=t_u.get_model(config.model_arch, robust_model=config.robust_model, robust_eps=config.robust_eps,
        nb_epochs=config.nb_epochs,model_dir=config.model_dir,data_dir=config.data_dir,test_loader=test_loader,device=config.device,
        download=config.download,dataset=config.dataset, force_train=config.force_train)
    X_correct,label_correct,accuracy=t_u.get_correct_x_y(data_loader=test_loader,device=device,model=model)
    if config.verbose>=2:
        print(f"model accuracy on test batch:{accuracy}")

    config.x_mean=t_u.datasets_means[config.dataset]
    config.x_std=t_u.datasets_stds[config.dataset]
    x_min=0
    x_max=1
    if config.use_attack:
        import foolbox as fb
        fmodel = fb.PyTorchModel(model, bounds=(0,1),device=device)
        attack=fb.attacks.LinfPGD()
        #un-normalize data before performing attack
        _, advs, success = attack(fmodel, X_correct[config.input_start:config.input_stop], 
        label_correct[config.input_start:config.input_stop], epsilons=config.epsilons)


    config_dict=print_config(config)
    path_config=os.path.join(exp_log_path,'config.json')
    with open(path_config,'w') as f:
        f.write(json.dumps(config_dict, indent = 4))
    inp_indices=np.arange(start=config.input_start,stop=config.input_stop)
    normal_dist=torch.distributions.Normal(loc=0, scale=1.)
    run_nb=0
    iterator= tqdm(range(config.n_rep))
    exp_res=[]
    clip_min=0
    clip_max=1
    zero_d = torch.zeros(d, device=config.device)
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
                lirpa_safe=lirpa_safe.item()
                if config.verbose>=2:
                    print(f"lirpa safe:{lirpa_safe}")
            low=torch.max(x_0-epsilon, torch.tensor([x_min]).cuda())
            high=torch.min(x_0+epsilon, torch.tensor([x_max]).cuda())  
            V = lambda X: t_u.V_pyt(X,x_0=x_0,model=model,low=low,high=high,target_class=y_0,gaussian_latent=config.gaussian_latent)
            grad_V = lambda X: t_u.gradV_pyt(X,x_0=x_0,model=model,low=low,high=high, target_class=y_0,gaussian_latent=config.gaussian_latent)
            


            for max_iter in config.max_iter_range: 
                for step_size in config.step_size_range:
                
                    loc_time=datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
                    log_name=method_name+'_'+'_'+loc_time
                    log_name=method_name+f'_maxiter_{max_iter}_step_size_{step_size}_t_'.replace('.','_')+loc_time.split('.')[0]
                    log_path=os.path.join(exp_log_path,log_name)
                    os.mkdir(path=log_path)
                    i_run+=1
                    
                    
                    print(f"Starting FORM run {i_run}/{nb_runs} on {config.dataset} dataset, with model {model_name} and epsilon= {epsilon},Maxiter={max_iter},step size={step_size}")
                    
                    times= []
                    rel_error= []
                    ests = [] 
                    calls=[]
                    if config.track_finish:
                        finish_flags=[]
                    for _ in tqdm(range(config.n_rep)):
                        t=time()
                        approx_zero,f_calls = dp_search_method(f=V, grad_f=grad_V, x0=zero_d, step_size=step_size,
                                                            tol=config.tol, max_iter=max_iter,random_init=config.random_init,)
                        beta = torch.norm(approx_zero).detach().cpu().numpy()
                        est = stat.norm.cdf(-beta)
                        t=time()-t
                        if config.verbose>=1:
                            print(f"FORM finished in {t:.2f} seconds")
                            
                            print(f"FORM found p={est}, with beta={beta}, in {f_calls} function calls")
                    
                    
                        times.append(t)
                        ests.append(est)
                        calls.append(f_calls)


                    times=np.array(times)
                    ests = np.array(ests)
                    log_ests=np.log(np.clip(ests,a_min=1e-250,a_max=1))
                    std_log_est=log_ests.std()
                    mean_log_est=log_ests.mean()
                    lg_q_1,lg_med_est,lg_q_3=np.quantile(a=ests,q=[0.25,0.5,0.75])
                    lg_est_path=os.path.join(log_path,'lg_ests.txt')
                    np.savetxt(fname=lg_est_path,X=ests)
                
                    abs_errors=np.abs(ests-epsilon)
                    rel_errors=abs_errors/epsilon
                    bias=np.mean(ests)-epsilon

                    times=np.array(times)  
                    ests=np.array(ests)
                    calls=np.array(calls)
                    errs=np.abs(ests-epsilon)
                    q_1,med_est,q_3=np.quantile(a=ests,q=[0.25,0.5,0.75])
                    mean_calls=calls.mean()
                    std_calls=calls.std()
                    MSE=np.mean(abs_errors**2)
                    MSE_adj=MSE*mean_calls
                    MSE_rel=MSE/epsilon**2
                    MSE_rel_adj=MSE_rel*mean_calls
                    
                    print(f"mean est:{ests.mean()}, std est:{ests.std()}")
                    print(f"mean rel error:{rel_errors.mean()}")
                    print(f"MSE rel:{MSE/epsilon**2}")
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
                    results={'epsilon':epsilon,'method':method_name,'step_size':step_size, 'max_iter':max_iter,
                    'n_rep':config.n_rep,'lg_est_path':lg_est_path,'model_name':model_name,
                    'mean_est':ests.mean(),'std_log_est':log_ests.std(),'mean_log_est':mean_log_est,
                    'lg_q_1':lg_q_1,'lg_q_3':lg_q_3,"lg_med_est":lg_med_est
                    ,'mean_time':times.mean()
                    ,'std_time':times.std(),'MSE':MSE,'MSE_rel_adj':MSE_rel_adj,'MSE_rel':MSE_rel,
                    'mean_calls':mean_calls,'last_particle':config.last_particle,
                    'std_calls':std_calls
                    ,'bias':ests.mean()-epsilon,'mean abs error':abs_errors.mean(),
                    'mean_rel_error':rel_errors.mean(),'std_est':ests.std(),'freq underest':(ests<epsilon).mean()
                    ,'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,'cores_number':config.cores_number,
                    'batch_opt':config.batch_opt,"d":d, "correct_T":config.correct_T,
                    "np_seed":config.np_seed,"torch_seed":config.torch_seed,
                        'q_1':q_1,'q_3':q_3,'med_est':med_est,'beta':beta}
                    exp_res.append(results)
                    results_df=pd.DataFrame([results])
                    results_df.to_csv(os.path.join(log_path,'results.csv'),index=False)
                    if config.aggr_res_path is None:
                        aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                    else:
                        aggr_res_path=config.aggr_res_path
                    if config.update_aggr_res:
                        if not os.path.exists(aggr_res_path):
                            cols=['epsilon','method','N','rho','n_rep','alpha','min_rate','mean_time','std_time','mean_est',
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