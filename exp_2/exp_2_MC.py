import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os
import json
from stat_reliability_measure.home import ROOT_DIR
from time import time
from datetime import datetime
import stat_reliability_measure.dev.torch_utils as t_u
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#setting PRNG seeds for reproducibility
from stat_reliability_measure.dev.utils import  float_to_file_float,str2bool,str2intList,str2floatList, str2list
from stat_reliability_measure.dev.utils import get_sel_df,print_config
import stat_reliability_measure.dev.mc.mc_pyt as mc_pyt

str2floatList=lambda x: str2list(in_str=x, type_out=float)
str2intList=lambda x: str2list(in_str=x, type_out=int)
low_str=lambda x: str(x).lower()

method_name="MC"
class config:
    dataset="mnist"
    log_dir=None
    model_dir=None
    model_arch="dnn2"  
    epsilons = [0.15]

    N=int(1e6)
    N_range=[]
    batch_size=int(1e5)
    b_range=[]
    n_rep=100
    track_advs=False
    verbose=0

    print_img=False
    x_min=None
    x_max=None
    x_mean=0
    x_std=1
    allow_zero_est=True
    
   
    


    


    track_accept=False
    
    d = 784
    epsilon = 0.1
    
    n_max=2000
    tqdm_opt=True
    
    
    eps_max=0.3
    eps_min=0.1 
    eps_num=5
    allow_zero_est=True
    save_config=True
    print_config=True
    update_aggr_res=True
    aggr_res_path=None
    from_gaussian=True
    project_kernel=True
    allow_multi_gpu=True
    input_start=0
    input_stop=None
    g_target=None
    track_gpu=False
    track_cpu=False
    gpu_name=None
    cpu_name=None
    cores_number=None
    device=None
    torch_seed=0
    np_seed=0
    tf_seed=None

    model_path=None
    export_to_onnx=False
    use_attack=False
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
    data_dir=ROOT_DIR+"/data"
    p_ref_compute=False
    force_train=False
    repeat_exp=False


parser=argparse.ArgumentParser()
parser.add_argument('--log_dir',default=config.log_dir)
parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--N',type=int,default=config.N)
parser.add_argument('--verbose',type=float,default=config.verbose)
parser.add_argument('--d',type=int,default=config.d)
parser.add_argument('--b_range',type=str2intList,default=config.b_range)
parser.add_argument('--batch_size',type=str2bool,default=config.batch_size)
parser.add_argument('--tqdm_opt',type=str2bool,default=config.tqdm_opt)
parser.add_argument('--save_config',type=str2bool, default=config.save_config)
parser.add_argument('--update_aggr_res',type=str2bool,default=config.update_aggr_res)
parser.add_argument('--aggr_res_path',type=str, default=config.aggr_res_path)
parser.add_argument('--from_gaussian',type=str2bool, default=config.from_gaussian)
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
parser.add_argument('--N_range',type=str2intList,default=config.N_range)
parser.add_argument('--download',type=str2bool, default=config.download)
parser.add_argument('--model_path',type=str,default=config.model_path)
parser.add_argument('--track_finish',type=str2bool,default=config.track_finish)
parser.add_argument('--lirpa_cert',type=str2bool,default=config.lirpa_cert)
parser.add_argument('--load_batch_size',type=int,default=config.load_batch_size)
parser.add_argument('--model_arch',type=str,default = config.model_arch)
parser.add_argument('--dataset',type=str,default = config.dataset)
parser.add_argument('--robust_model',type=str2bool, default=config.robust_model)
parser.add_argument('--nb_epochs',type=int,default=config.nb_epochs)
parser.add_argument('--adversarial_every',type=int,default=config.adversarial_every)
parser.add_argument('--repeat_exp',type=str2bool,default=config.repeat_exp)
parser.add_argument('--force_train',type=str2bool,default=config.force_train)
parser.add_argument('--print_img',type=str2bool,default=config.print_img)
args=parser.parse_args()

for k,v in vars(args).items():
    setattr(config, k, v)


def main():
    if config.log_dir is None:
        config.log_dir=os.path.join(ROOT_DIR,f"logs/exp_2_{config.dataset}")

    if config.model_dir is None:
        config.model_dir=os.path.join(ROOT_DIR,f"models/{config.dataset}")

    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)

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


    if len(config.N_range)<1:
        config.N_range=[config.N]
    if len(config.b_range)<1:
        config.b_range=[config.batch_size]


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


    #epsilon=config.epsilon


    if not os.path.exists(ROOT_DIR+'/logs'):
        print('logs not found')
        os.mkdir(ROOT_DIR+'/logs')
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    raw_logs=os.path.join(config.log_dir,'raw_logs/')
    if not os.path.exists(raw_logs):
        os.mkdir(raw_logs)
    raw_logs_path=os.path.join(raw_logs,method_name)
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
    loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
    log_name=method_name+'_'+'_'+loc_time.replace('-','_').replace(':','_')
    log_name=method_name+'_'+'_'+loc_time
    exp_log_path=os.path.join(raw_logs_path,log_name)
    if os.path.exists(path=exp_log_path):
        exp_log_path = exp_log_path+'_'+str(np.random.randint(low=0,high=9))
    os.mkdir(path=exp_log_path)

    d=t_u.datasets_dims[config.dataset]
    config_dict=print_config(config)
    config_path=os.path.join(exp_log_path,'config.json')
    with open(config_path,'w') as f:
        f.write(json.dumps(config_dict, indent = 4, cls=utils.CustomEncoder))


    print(f"Input dimension: {d}")
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
    X_correct,label_correct=X_correct[:config.input_stop],label_correct[:config.input_stop]
    config.x_mean=t_u.datasets_means[config.dataset]
    config.x_std=t_u.datasets_stds[config.dataset]





    #X.requires_grad=True
    normal_dist=torch.distributions.Normal(loc=0, scale=1.)


    #inf=float('inf')

    x_min=0
    x_max=1
    if config.use_attack:

        import foolbox as fb
        fmodel = fb.PyTorchModel(model, bounds=(x_min,x_max))
        attack=fb.attacks.LinfPGD()
        
    
        _, advs, success = attack(fmodel, X_correct[config.input_start:config.input_stop], 
        label_correct[config.input_start:config.input_stop], epsilons=config.epsilons)


    inp_indices=np.arange(start=config.input_start,stop=config.input_stop)
    i_exp=0
    param_ranges= [inp_indices,config.b_range,config.N_range,config.epsilons]
    lenghts=np.array([len(L) for L in param_ranges])
    nb_exps= np.prod(lenghts)

    for l in inp_indices:
        with torch.no_grad():
            x_clean,y_clean = X_correct[l], label_correct[l]
        if config.print_img:
            plt.imshow(x_clean.detach().cpu().numpy().reshape((28,28)), cmap='gray')
            plt.title("Pred: {}".format(y_clean.argmax()))
            plt.show()

        input_shape=x_clean.shape
        #x_clean.requires_grad=True
        for idx in range(len(config.epsilons)):
            epsilon = config.epsilons[idx]
            low=torch.max(x_clean-epsilon, torch.tensor([x_min]).cuda())
            high=torch.min(x_clean+epsilon, torch.tensor([x_max]).cuda())
            if config.noise_dist=='gaussian' or config.from_gaussian:  
                gen = lambda N: torch.randn(size=(N,d),device=config.device)
            else:
                gen = lambda N: torch.randn(size=(N,d),device=config.device)
            epsilon = config.epsilons[idx]
            pgd_success= (success[idx][l]).item() if config.use_attack else None 
            p_l,p_u=None,None
            if config.lirpa_bounds:
                from stat_reliability_measure.dev.lirpa_utils import get_lirpa_bounds
                # Step 2: define perturbation. Here we use a Linf perturbation on input image.
                p_l,p_u=get_lirpa_bounds(x_clean=x_clean,y_clean=y_clean,model=model,epsilon=epsilon,
                num_classes=num_classes,noise_dist=config.noise_dist,a=config.a,device=config.device)
                p_l,p_u=p_l.item(),p_u.item()
            if config.noise_dist=='gaussian':
                gaussian_prior = True
                from_gaussian = False
            else:
                gaussian_prior = False
                from_gaussian = config.from_gaussian
            score = lambda X: -t_u.V_pyt(X,x_clean=x_clean,model=model,low=low,high=high,target_class=y_clean,
            from_gaussian=from_gaussian,
            gaussian_prior=gaussian_prior)
            for N in config.N_range: 
                for bs in config.b_range :
                    loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
                    log_name=method_name+'_'+'_'+loc_time.replace(':','_')
                    log_name=method_name+'_e_'+float_to_file_float(config.epsilons[idx])+'_N_'+str(N)+'_bs_'+float_to_file_float(bs)
                    log_name=log_name+'_'+loc_time
                    log_path=os.path.join(exp_log_path,log_name)
                    i_exp+=1
                    aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                    if (not config.repeat_exp) and config.update_aggr_res and os.path.exists(aggr_res_path):
                        aggr_res_df = pd.read_csv(aggr_res_path)
                        same_exp_df = get_sel_df(df=aggr_res_df,triplets=[('method',method_name,'='),
                        ('model_name',model_name,'='),
                        ('epsilon',epsilon,'='),('input_index',l,'='),('n_rep',config.n_rep,'='),('N',N,'='),
                        ('batch_size',bs,'=')] )  
                        # if a similar experiment has been done in the current log directory we skip it
                        if len(same_exp_df)>0:
                            
                            print(f"Skipping {method_name} run {i_exp}/{nb_exps}, model: {model_name}, img_idx:{l},eps:{epsilon},N={N},batch size={bs}")
                            continue
                    print(f"Starting {method_name} run {i_exp}/{nb_exps}")
                                        

                
                    if config.verbose>=0:
                        print(f"with model: {model_name}, img_idx:{l},eps:{epsilon},N:{N},batch_size:{bs}")
                
                    times= []
                    rel_error= []
                    ests = [] 
                    log_ests=[]
                    calls=[]
                    if config.track_finish:
                        finish_flags=[]
                    for i in tqdm(range(config.n_rep)):
                        t=time()
                        est = mc_pyt.MC_pf(gen=gen, score=score, N_mc=N,batch_size=bs,track_advs=config.track_advs).cpu()
                        t=time()-t
                        # we don't need adversarial examples and highest score
                        log_est = np.log(np.clip(a=est,a_min=1e-250,a_max=1))
                        log_ests.append(log_est)
                        nb_calls=N
                        if config.verbose:
                            print(f"Est:{est}")
                        # dict_out=amls_res[1]
                        
                
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
                    results={'method':method_name,'from_gaussian':str(config.from_gaussian),'input_index':l,
                        'epsilon':epsilon,"model_name":model_name,'dataset':config.dataset,'n_rep':config.n_rep,
                    'batch_size':bs, "N":N, "mean_calls":calls.mean(),"std_calls":calls.std(),"std_adj":ests.std()*mean_calls,
                    'mean_time':times.mean(),'std_time':times.std(),'mean_est':ests.mean(),'est_path':est_path,'times_path':times_path,
                    'std_est':ests.std(),'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,
                    'cores_number':config.cores_number,'g_target':config.g_target,"std_rel":std_rel, "std_rel_adj":std_rel_adj,
                    'freq_finished':freq_finished,'freq_zero_est':freq_zero_est,'unfinished_mean_time':unfinished_mean_time,
                    'unfinished_mean_est':unfinished_mean_est,"lg_est_path":lg_est_path,
                        "mean_log_est":mean_log_est,"std_log_est":std_log_est,
                        "lg_q_1":lg_q_1,"lg_q_3":lg_q_3,"lg_med_est":lg_med_est,
                    'np_seed':config.np_seed,'torch_seed':config.torch_seed,'pgd_success':pgd_success,'p_l':p_l,
                    'p_u':p_u,'noise_dist':config.noise_dist,'datetime':loc_time,
                    'q_1':q_1,'q_3':q_3,'med_est':med_est,
                    "log_path":log_path,}
                    results_df=pd.DataFrame([results])
                    results_df.to_csv(os.path.join(log_path,'results.csv'),)
                    if config.aggr_res_path is None:
                        aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
                    else: 
                        aggr_res_path=config.aggr_res_path

                    if config.update_aggr_res:
                        if not os.path.exists(aggr_res_path):
                            print(f'aggregate results csv file not found \n it will be build at {aggr_res_path}')
                            cols=['method','from_gaussian','N','rho','n_rep','T','epsilon','alpha','min_rate','mean_time','std_time','mean_est',
                            'std_est','freq underest','g_target','L','ratio','ess_apha']
                            cols+=['freq_finished','freq_zero_est','unfinished_mean_est','unfinished_mean_time']
                            cols+=['pgd_success','p_l','p_u','gpu_name','cpu_name','np_seed','torch_seed','noise_dist','datetime']
                            aggr_res_df= pd.DataFrame(columns=cols)

                        else:
                            aggr_res_df=pd.read_csv(aggr_res_path)
                        aggr_res_df = pd.concat([aggr_res_df,results_df],ignore_index=True)
                        aggr_res_df.to_csv(aggr_res_path,index=False)

if __name__ == '__main__':
    main()