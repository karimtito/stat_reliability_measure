from operator import mod
import torch

import foolbox as fb
import pandas as pd

import numpy as np
from tqdm import tqdm
from stat_reliability_measure.dev.utils import dichotomic_search
from scipy.special import betainc
import GPUtil
import matplotlib.pyplot as plt
import cpuinfo
from torch import optim
import argparse
import os
import stat_reliability_measure.dev.langevin_adapt.langevin_adapt_pyt as smc_pyt
from importlib import reload
from time import time
from datetime import datetime
from stat_reliability_measure.dev.torch_utils import project_ball_pyt, projected_langevin_kernel_pyt, multi_unsqueeze, compute_V_grad_pyt, compute_V_pyt
from stat_reliability_measure.dev.torch_utils import V_pyt, gradV_pyt, epoch

from stat_reliability_measure.dev.utils import str2bool,str2list,float_to_file_float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
import stat_reliability_measure.dev.torch_utils as t_u 
#setting PRNG seeds for reproducibility

str2floatList=lambda x: str2list(in_str=x, type_out=float)
str2intList=lambda x: str2list(in_str=x, type_out=int)
low_str=lambda x: str(x).lower()

method_name="langevin_adapt_pyt"

class config:
    log_dir="../../logs/mnist_tests"
    data_dir="../../data/ImageNet"
    n_rep=10
    N=40
    N_list=[]
    verbose=0
    min_rate=0.51
    T=1
    T_list=[]
    g_target=0.8
    g_list=[]
    alpha=0.025
    alpha_list=[]
    n_max=2000
    tqdm_opt=True
    d = 1024
    epsilons = None
    eps_max=0.3
    eps_min=0.1 
    eps_num=5
    allow_zero_est=True
    save_config=True
    print_config=True
    adapt_d_t=False
    track_accept=False
    update_agg_res=True
    aggr_res_path=None
    gaussian_latent=False
    project_kernel=True
    allow_multi_gpu=False
    input_start=0
    input_stop=None
    
    track_gpu=True
    track_cpu=True
    gpu_name=None
    cpu_name=None
    cores_number=None
    device=None
    torch_seed=0
    np_seed=0
    tf_seed=None
    model_path=None
    export_to_onnx=False
    use_attack=True
    attack='PGD'
    lirpa_bounds=False
    download=False
    train_model=False
    mh_opt=False
    noise_dist='uniform'
    a=0
    track_accept=False 
    adapt_d_t=False
    adapt_d_t_mcmc=False
    target_accept=0.574
    accept_spread=0.1
    d_t_decay=0.999
    d_t_gain=1/d_t_decay
    input_start=0
    input_stop=None
    split="val"
    model_name='resnet18'

    load_batch_size=100

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
parser.add_argument('--T',type=int,default=config.T)
parser.add_argument('--save_config',type=str2bool, default=config.save_config)
parser.add_argument('--update_agg_res',type=str2bool,default=config.update_agg_res)
parser.add_argument('--aggr_res_path',type=str, default=config.aggr_res_path)

parser.add_argument('--gaussian_latent',type=str2bool, default=config.gaussian_latent)
parser.add_argument('--allow_multi_gpu',type=str2bool)
parser.add_argument('--g_target',type=float,default=config.g_target)
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
parser.add_argument('--a',type=float, default=config.a)
parser.add_argument('--N_list',type=str2intList,default=config.N_list)
parser.add_argument('--T_list',type=str2intList,default=config.T_list)
parser.add_argument('--alpha_list',type=str2floatList,default=config.alpha_list)
parser.add_argument('--g_list',type=str2floatList,default=config.g_list)
parser.add_argument('--download',type=str2bool, default=config.download)
parser.add_argument('--model_path',type=str,default=config.model_path)

parser.add_argument('--mh_opt',type=str2bool,default=config.mh_opt)
parser.add_argument('--adapt_d_t',type=str2bool,default=config.adapt_d_t)
parser.add_argument('--target_accept',type=float,default=config.target_accept)
parser.add_argument('--accept_spread',type=float,default=config.accept_spread)
parser.add_argument('--d_t_decay',type=float,default=config.d_t_decay)
parser.add_argument('--d_t_gain',type=float,default=config.d_t_gain)
parser.add_argument('--adapt_d_t_mcmc',type=str2bool,default=config.adapt_d_t_mcmc)

parser.add_argument('--split',type=str,default=config.split)
parser.add_argument('--model_name',type=str,default=config.model_name)
parser.add_argument('--data_dir',type=str,default=config.data_dir)
args=parser.parse_args()

for k,v in vars(args).items():
    setattr(config, k, v)

if config.noise_dist is not None:
    config.noise_dist=config.noise_dist.lower()

if config.noise_dist not in ['uniform','gaussian']:
    raise NotImplementedError("Only uniform and Gaussian distributions are implemented.")

if not config.allow_multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

if config.input_stop is None:
    config.input_stop=config.input_start+1

if config.np_seed is None:
    config.np_seed=int(time.time())
np.random.seed(seed=config.np_seed)

if config.torch_seed is None:
    config.torch_seed=int(time.time())
torch.manual_seed(seed=config.torch_seed)




if len(config.T_list)<1:
    config.T_list=[config.T]
if len(config.N_list)<1:
    config.N_list=[config.N]

if len(config.alpha_list)<1:
    config.alpha_list=[config.alpha]
if len(config.g_list)==0:
    config.g_list=[config.g_target]

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



if not os.path.exists('../../logs'):
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


#loading data
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
dim=784
num_classes=10


assert config.download==False,"Automatic data downloading is not supported (anymore) for ImageNet dataset"

normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])



data_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

imagenet_dataset = datasets.ImageNet(config.data_dir, split=config.split, download=config.download, transform=data_transform)

data_loader = DataLoader(imagenet_dataset, batch_size = config.load_batch_size, shuffle=False)

supported_model_list=['resnet18']
  
if config.model_name  not in  supported_model_list:
    #instancing custom CNN model
    # model = CNN_custom()
    # model=model.to(device)
    # config.model_path="model_CNN.pt"
    # model_path=config.model_path
    # model_name=model_path.strip('.pt')
    raise NotImplementedError(f"Only models in {supported_model_list} are supported.")
    
else: 
    if config.model_name.lower()=='resnet18':
        model=models.resnet18(pretrained=True)

# if config.train_model:
#     mnist_train = datasets.MNIST(config.data_dir, train=True, download=config.download, transform=transforms.ToTensor())
#     train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True,)
#     opt = optim.SGD(model.parameters(), lr=1e-1)

#     print("Train Err", "Train Loss", "Test Err", "Test Loss", sep="\t")
#     for _ in range(10):
#         train_err, train_loss = epoch(train_loader, model, opt, device=config.device)
#         test_err, test_loss = epoch(data_loader, model, device=config.device)
#         print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
#     torch.save(model.state_dict(), model_path)



if config.export_to_onnx:
    batch_size=1
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True,device=device)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "model.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
for X,y in data_loader:
    X,y = X.to(device), y.to(device)
    break

X.requires_grad=True



normal_dist=torch.distributions.Normal(loc=0, scale=1.)

with torch.no_grad():
    logits=model(X)
    y_pred= torch.argmax(logits,-1)
    correct_idx=y_pred==y
    if config.verbose>0.1:
        print(f"Model accuracy on batch: {correct_idx.mean()}")
    
    X_correct, label_correct= X[correct_idx], y[correct_idx]


if config.use_attack:
    fmodel = fb.PyTorchModel(model, bounds=(0,1))
    attack=fb.attacks.LinfPGD()
    
    #epsilons= np.array([0.0, 0.001, 0.01, 0.03,0.04,0.05,0.07,0.08,0.0825,0.085,0.086,0.087,0.09, 0.1, 0.3, 0.5, 1.0])
    _, advs, success = attack(fmodel, X_correct[config.input_start:config.input_stop], 
    label_correct[config.input_start:config.input_stop], epsilons=config.epsilons)
exp_res=[]
for l in np.arange(start=config.input_start,stop=config.input_stop):
    with torch.no_grad():
        x_0,y_0 = X_correct[l], label_correct[correct_idx][l]
    for i in range(len(config.epsilons)):
        
        
        epsilon = config.epsilons[i]
        pgd_success= (success[i][l]).item() if config.use_attack else None 
        p_l,p_u=None,None
        if config.lirpa_bounds:
            from stat_reliability_measure.dev.lirpa_utils import get_lirpa_bounds
            # Step 2: define perturbation. Here we use a Linf perturbation on input image.
            p_l,p_u=get_lirpa_bounds(x_0=x_0,y_0=y_0,model=model,epsilon=epsilon,
            num_classes=num_classes,noise_dist=config.noise_dist,a=config.a,device=config.device)
            p_l,p_u=p_l.item(),p_u.item()
            
        
        
        normal_gen=lambda N: torch.randn(size=(N,dim),requires_grad=True).to(device)
        uniform_gen_eps = lambda N: epsilon*(2*torch.rand(size=(N,dim), device=device )-1)
        V_ = lambda X: V_pyt(X,x_0=x_0,model=model,epsilon=epsilon, target_class=y_0,gaussian_latent=True)
        gradV_ = lambda X: gradV_pyt(X,x_0=x_0,model=model, target_class=y_0,epsilon=epsilon, gaussian_latent=True)
        x_0.requires_grad=True
        print(config.g_list)
        for T in config.T_list: 
            for N in config.N_list: 
                for g_t in config.g_list:
                    for alpha in config.alpha_list:
                        ests,times,finish_flags = [],[],[]
                        iterator= tqdm(range(config.n_rep)) if config.tqdm_opt else range(config.n_rep)
                        for _ in iterator:
                            # Gaussian latent space version
                            t=time()
                            print(g_t)
                            p_est,dict_out=smc_pyt.LangevinSMCSimpAdaptPyt(gen=normal_gen, l_kernel=t_u.langevin_kernel_pyt,V=V_, gradV=gradV_,
                            g_target=g_t, min_rate=config.min_rate, track_accept=config.track_accept,mh_opt=config.mh_opt,
                             adapt_d_t=config.adapt_d_t,d_t_decay=config.d_t_decay,target_accept=config.target_accept,
                                alpha =alpha,N=N,T = T,n_max=config.n_max, verbose=config.verbose,allow_zero_est=config.allow_zero_est)
                            time_comp=time()-t
                            ests.append(p_est)
                            times.append(time_comp)
                            finish_flag=dict_out['finished']
                            finish_flags.append(finish_flag)

                        times=np.array(times)
                        estimates = np.array(ests)
                        finish_flags=np.array(finish_flags)
                        freq_finished=finish_flags.mean()
                        freq_zero_est=(estimates==0).mean()
                        #finished=np.array(finish_flag)
                        if freq_finished<1:
                            unfinish_est=estimates[~finish_flags]
                            unfinish_times=times[~finish_flags]
                            unfinished_mean_est=unfinish_est.mean()
                            unfinished_mean_time=unfinish_times.mean()
                        else:
                            unfinished_mean_est,unfinished_mean_time=None,None
                        loc_time= datetime.today().isoformat().split('.')[0]
                        log_name=method_name+'_e_'+float_to_file_float(config.epsilons[i])+'_N_'+str(N)+'_T_'+str(T)+'a'+float_to_file_float(alpha)
                        log_name=log_name+'g'+float_to_file_float(g_t)+'_'+'_'+loc_time
                        log_path=os.path.join(raw_logs_path,log_name)
                        os.mkdir(log_path)
                        np.savetxt(fname=os.path.join(log_path,'times.txt'),X=times)
                        np.savetxt(fname=os.path.join(log_path,'estimates.txt'),X=estimates)

                        

                        plt.hist(times, bins=10)
                        plt.savefig(os.path.join(log_path,'times_hist.png'))
                        plt.hist(estimates,bins=10)
                        plt.savefig(os.path.join(log_path,'estimates_hist.png'))
                    
                        

                        #with open(os.path.join(log_path,'results.txt'),'w'):
                        results={'method':method_name,'gaussian_latent':str(config.gaussian_latent),'image_idx':l,
                        'N':N,'g_target':g_t,'epsilon':epsilon,'n_rep':config.n_rep,'T':T,'alpha':alpha,
                        'min_rate':config.min_rate,
                        'mean time':times.mean(),'std time':times.std(),'mean est':estimates.mean(),
                        'std est':estimates.std(),'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,
                        'cores_number':config.cores_number,'g_target':config.g_target,
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
exp_df=pd.DataFrame(exp_res)
exp_df.to_csv(os.path.join(log_path,'exp_results.csv'),index=False)