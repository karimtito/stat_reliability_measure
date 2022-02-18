import torch

import foolbox as fb
import pandas as pd

import numpy as np
from tqdm import tqdm
from dev.utils import dichotomic_search
from scipy.special import betainc
import GPUtil
import matplotlib.pyplot as plt
import cpuinfo
from torch import optim
import argparse
import os
import dev.langevin_base.langevin_base_pyt as smc_pyt
from importlib import reload
from time import time
from datetime import datetime
from dev.torch_utils import project_ball_pyt, projected_langevin_kernel_pyt, multi_unsqueeze, compute_V_grad_pyt, compute_V_pyt
from dev.torch_utils import V_pyt, gradV_pyt, epoch
from dev.torch_arch import CNN_custom#,CNN,dnn2
from dev.utils import str2bool,str2list,float_to_file_float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
import dev.torch_utils as t_u 
#setting PRNG seeds for reproducibility

str2floatList=lambda x: str2list(in_str=x, type_out=float)
str2intList=lambda x: str2list(in_str=x, type_out=int)
low_str=lambda x: str(x).lower()

method_name="vanilla_mc"

class config:
    log_dir="./logs/mnist_tests"
    n_rep=10
    N=int(1e5)
    N_list=[]
    N_batch=int(1e3)
    N_b_list=[]
    verbose=0
    
    x_min=0
    x_max=1
    n_max=2000
    tqdm_opt=True
    d = 1024
    epsilons =[]
    eps_max=0.3
    eps_min=0.1 
    eps_num=5
    allow_zero_est=True
    save_config=True
    print_config=True
    update_agg_res=True
    aggr_res_path=None
    gaussian_latent=False
    project_kernel=True
    allow_multi_gpu=False
    input_index=0
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
    export_to_onnx=False
    use_attack=True
    attack='PGD'
    lirpa_bounds=False
    download=False
    train_model=False
    noise_dist='uniform'
    a=0
    model_path=None

parser=argparse.ArgumentParser()
parser.add_argument('--log_dir',default=config.log_dir)
parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--N',type=int,default=config.N)
parser.add_argument('--verbose',type=float,default=config.verbose)
parser.add_argument('--d',type=int,default=config.d)

parser.add_argument('--n_max',type=int,default=config.n_max)
parser.add_argument('--tqdm_opt',type=str2bool,default=config.tqdm_opt)

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
parser.add_argument('--input_index',type=int,default=config.input_index)
parser.add_argument('--lirpa_bounds',type=str2bool, default=config.lirpa_bounds)
parser.add_argument('--eps_min',type=float, default=config.eps_min)
parser.add_argument('--eps_max',type=float, default=config.eps_max)
parser.add_argument('--eps_num',type=int,default=config.eps_num)
parser.add_argument('--train_model',type=str2bool,default=config.train_model)
parser.add_argument('--noise_dist',type=str, default=config.noise_dist)
parser.add_argument('--a',type=float, default=config.a)
parser.add_argument('--N_list',type=str2intList,default=config.N_list)
parser.add_argument('--N_b_list',type=str2intList,default=config.T_list)
parser.add_argument('--x_min',type=float,default=config.x_min)
parser.add_argument('--x_max',type=float,default=config.x_max)
parser.add_argument('--download',type=str2bool, default=config.download)
parser.add_argument('--model_path',type=str,default=config.model_path)
args=parser.parse_args()

for k,v in vars(args).items():
    setattr(config, k, v)

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

if len(config.epsilons)==0:
    log_eps=np.linspace(start=np.log(config.eps_min),stop=np.log(config.eps_max),num=config.eps_num)
    config.epsilons=np.exp(log_eps)

if len(config.N_list)<1:
    config.N_list=[config.N]
if len(config.N_b_list)<1:
    config.N_b_list=[config.N_batch]



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


if not os.path.exists('./logs'):
    os.mkdir('./logs')
if not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir)

raw_logs_path=os.path.join(config.log_dir,'raw_logs')
if not os.path.exists(raw_logs_path):
    os.mkdir(raw_logs_path)


if config.epsilons is None:
    log_min,log_max=np.log(config.eps_min),np.log(config.eps_max)
    log_line=np.linspace(start=log_min,stop=log_max,num=config.eps_num)
    config.epsions=np.exp(log_line)

if config.aggr_res_path is None:
    aggr_res_path=os.path.join(config.log_dir,'aggr_res.csv')
else:
    aggr_res_path=config.aggr_res_path


#loading data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
dim=784
num_classes=10
if not os.path.exists("../data/MNIST"):
    config.download=True

config.json=vars(args)
if config.print_config:
    print(config.json)
mnist_test = datasets.MNIST("./data", train=False, download=config.download, transform=transforms.ToTensor())

test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

  

#instancing custom CNN model
if config.model_path is None:
    model = CNN_custom()
    model=model.to(device)
    model_path="model_CNN.pt"
else: 
    raise NotImplementedError("Testing of custom models is not yet implemented.")

if config.train_model:
    mnist_train = datasets.MNIST("./data", train=True, download=config.download, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True,)
    opt = optim.SGD(model.parameters(), lr=1e-1)

    print("Train Err", "Train Loss", "Test Err", "Test Loss", sep="\t")
    for _ in range(10):
        train_err, train_loss = epoch(train_loader, model, opt, device=config.device)
        test_err, test_loss = epoch(test_loader, model, device=config.device)
        print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
    torch.save(model.state_dict(), model_path)

model.load_state_dict(torch.load(model_path))
model.eval()

if config.export_to_onnx:
    batch_size=1
    x = torch.randn(batch_size, 1, 28, 28, requires_grad=True,device=device)
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
for X,y in test_loader:
    X,y = X.to(device), y.to(device)
    break

X.requires_grad=True

l=config.input_index

normal_dist=torch.distributions.Normal(loc=0, scale=1.)
with torch.no_grad():
    logits=model(X)
    y_pred= torch.argmax(logits,-1)
    correct_idx=y_pred==y

    x_0,y_0 = X[correct_idx][l], y[correct_idx][l]
    X_correct, label_correct= X[correct_idx], y[correct_idx]


if config.use_attack:
    fmodel = fb.PyTorchModel(model, bounds=(0,1))
    attack=fb.attacks.LinfPGD()
    
    #epsilons= np.array([0.0, 0.001, 0.01, 0.03,0.04,0.05,0.07,0.08,0.0825,0.085,0.086,0.087,0.09, 0.1, 0.3, 0.5, 1.0])
    _, advs, success = attack(fmodel, X_correct, label_correct, epsilons=config.epsilons)

if config.lirpa_bounds:
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import *
    import auto_LiRPA.operators
    with torch.no_grad():
        image=x_0.view(1,1,28,28)
    bounded_model=BoundedModule(model,global_input=torch.zeros_like(input=image),bound_opts={"conv_mode": "patches"})
            

for i in range(len(config.epsilons)):
    
    
    epsilon = config.epsilons[i]
    pgd_success= success[i][l] if config.use_attack else None 
    p_l,p_u=None,None

    if config.lirpa_bounds:
        from dev.lirpa_utils import get_lirpa_bounds
        # Step 2: define perturbation. Here we use a Linf perturbation on input image.
        p_l,p_u=get_lirpa_bounds(x_0=x_0,y_0=y_0,model=model,epsilon=config.epsilon,
        num_classes=num_classes,noise_dist=config.noise_dis,a=config.a,device=config.device)
            
            
    def h(x):
        y = model(x)
        y_diff = torch.cat((y[:,:y_0], y[:,(y_0+1):]),dim=1) - y[:,y_0].unsqueeze(-1)
        y_diff, _ = y_diff.max(dim=1)
        return y_diff #.max(dim=1)
    
    if config.noise_dist.lower()=='gaussian':
        gen=lambda N: epsilon*torch.randn(size=(N,dim),requires_grad=True).to(device)
    else:
        gen=lambda N: epsilon*(2*torch.rand(size=(N,dim), device=device )-1)
    normal_gen=lambda N: torch.randn(size=(N,dim),requires_grad=True).to(device)
    
    x_0.requires_grad=True
    
    for N in config.N_list: 
        for N_b in config.N_b_list:
                print(f"Starting experiment with N={N},N_b={N_b}")
                ests,times,finish_flags = [],[],[]
                iterator= tqdm(range(config.n_rep)) if config.tqdm_opt else range(config.n_rep)
                for _ in iterator:
                    # Gaussian latent space version
                    
                    t=time()
                    n_batches=N//N_b 
                    count_prop=0
                    for _ in range(n_batches):
                        X=gen(n_batches)
                        scores=h(X)
                        count_prop+=(scores>=0).sum().item()
                        
                    if N%N_b!=0:
                        X=gen(N%N_b)
                        count_prop+=(h(X)>=0).sum().item()
                    
                    p_est=count_prop/N
                    time_comp=time()-t
                    ests.append(p_est)
                    times.append(time_comp)


                times=np.array(times)
                estimates = np.array(ests)
                
                freq_finished=1
                
                #finished=np.array(finish_flag)
                if freq_finished<1:
                    unfinish_est=estimates[~finish_flags]
                    unfinish_times=times[~finish_flags]
                    unfinished_mean_est=unfinish_est.mean()
                    unfinished_mean_time=unfinish_times.mean()
                else:
                    unfinished_mean_est,unfinished_mean_time=None,None
                loc_time=datetime.today().isoformat().split('.')[0]
                log_name=method_name+'_eps_'+float_to_file_float(config.epsilons[i])+'_N_b_'+str(N_b)
                log_name=log_name+'_'+loc_time
                log_path=os.path.join(raw_logs_path,log_name)
                os.mkdir(log_path)
                np.savetxt(fname=os.path.join(log_path,'times.txt'),X=times)
                np.savetxt(fname=os.path.join(log_path,'estimates.txt'),X=estimates)

                

                plt.hist(times, bins=10)
                plt.savefig(os.path.join(log_path,'times_hist.png'))
                plt.hist(estimates,bins=10)
                plt.savefig(os.path.join(log_path,'estimates_hist.png'))
                
                

                #with open(os.path.join(log_path,'results.txt'),'w'):
                results={'method':method_name,'gaussian_latent':str(config.gaussian_latent),
                'N':N,'epsilon':config.epsilons[i],'n_rep':config.n_rep,
                'min_rate':config.min_rate,'mean_calls':N,'std_calls':0,
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