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
import dev.langevin_base_pyt as smc_pyt
from importlib import reload
from time import time
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

method_name="langevin_base_pyt"

class config:
    log_dir="./logs/mnist_tests"
    n_rep=10
    N=40
    N_list=[]
    verbose=0
    min_rate=0.51
    T=1
    T_list=[]
    rho=90
    rho_list=[]
    alpha=0.025
    alpha_list=[]
    n_max=2000
    tqdm_opt=True
    d = 1024
    epsilons = None
    eps_max=1
    eps_min=0.0001 
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
    lirpa_bounds=True
    download=False
    train_model=False
    noise_dist='uniform'
    a=0

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
parser.add_argument('--rho',type=float,default=config.rho)
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
parser.add_argument('--input_index',type=int,default=config.input_index)
parser.add_argument('--lirpa_bounds',type=str2bool, default=config.lirpa_bounds)
parser.add_argument('--eps_min',type=float, default=config.eps_min)
parser.add_argument('--eps_max',type=float, default=config.eps_max)
parser.add_argument('--eps_num',type=int,default=config.eps_num)
parser.add_argument('--train_model',type=str2bool,default=config.train_model)
parser.add_argument('--noise_dist',type=str, default=config.noise_dist)
parser.add_argument('--a',type=float, default=config.a)
parser.add_argument('--N_list',type=str2intList,default=config.N_list)
parser.add_argument('--T_list',type=str2intList,default=config.T_list)
parser.add_argument('--rho_list', type=str2floatList,default=config.rho_list)
parser.add_argument('--alpha_list',type=str2floatList,default=config.alpha_list)
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

if len(config.T_list)<1:
    config.T_list=[config.T]
if len(config.N_list)<1:
    config.N_list=[config.N]
if len(config.rho_list)<1:
    config.rho_list=[config.rho]
if len(config.alpha_list)<1:
    config.alpha_list=[config.alpha]

if config.track_gpu:
    gpus=GPUtil.getGPUs()
    if len(gpus)>1:
        print("Multi gpus detected, only the first GPU will be tracked.")
    config.gpu_name=gpus[0].name

if config.track_cpu:
    config.cpu_name=cpuinfo.get_cpu_info()['brand_raw']
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

mnist_train = datasets.MNIST("../data", train=True, download=config.download, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=config.download, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True,)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

  

#instancing custom CNN model
model_CNN = CNN_custom()
model_CNN=model_CNN.to(device)
model_path="model_CNN.pt"


if config.train_model:
    opt = optim.SGD(model_CNN.parameters(), lr=1e-1)

    print("Train Err", "Train Loss", "Test Err", "Test Loss", sep="\t")
    for _ in range(10):
        train_err, train_loss = epoch(train_loader, model_CNN, opt, device=config.device)
        test_err, test_loss = epoch(test_loader, model_CNN, device=config.device)
        print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
    torch.save(model_CNN.state_dict(), model_path)

model_CNN.load_state_dict(torch.load(model_path))
model_CNN.eval()

if config.export_to_onnx:
    batch_size=1
    x = torch.randn(batch_size, 1, 28, 28, requires_grad=True,device=device)
    torch_out = model_CNN(x)

    # Export the model
    torch.onnx.export(model_CNN,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "model_CNN.onnx",   # where to save the model (can be a file or file-like object)
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
    logits=model_CNN(X)
    y_pred= torch.argmax(logits,-1)
    correct_idx=y_pred==y

    x_0,y_0 = X[correct_idx][l], y[correct_idx][l]
    X_correct, label_correct= X[correct_idx], y[correct_idx]


if config.use_attack:
    fmodel = fb.PyTorchModel(model_CNN, bounds=(0,1))
    attack=fb.attacks.LinfPGD()
    
    #epsilons= np.array([0.0, 0.001, 0.01, 0.03,0.04,0.05,0.07,0.08,0.0825,0.085,0.086,0.087,0.09, 0.1, 0.3, 0.5, 1.0])
    _, advs, success = attack(fmodel, X_correct, label_correct, epsilons=config.epsilons)

if config.lirpa_bounds:
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import *
    import auto_LiRPA.operators
    with torch.no_grad():
        image=x_0.view(1,1,28,28)
    bounded_model=BoundedModule(model_CNN,global_input=torch.zeros_like(input=image),bound_opts={"conv_mode": "patches"})
            

for i in range(len(config.epsilons)):
    loc_time= float_to_file_float(time())
    log_name=method_name+'_'+float_to_file_float(config.epsilons[i])+loc_time
    log_path=os.path.join(raw_logs_path,log_name)
    os.mkdir(log_path)
    epsilon = config.epsilons[i]
    pgd_success= success[i][l] if config.use_attack else None 
    p_l,p_u=None,None
    if config.lirpa_bounds:
        # Step 2: define perturbation. Here we use a Linf perturbation on input image.
        
        norm = np.inf
        ptb = PerturbationLpNorm(norm = norm, eps = config.epsilons[i])
        # Input tensor is wrapped in a BoundedTensor object.
        bounded_image = BoundedTensor(image, ptb)
        # We can use BoundedTensor to get model prediction as usual. Regular forward/backward propagation is unaffected.
        print('Model prediction:', bounded_model(bounded_image))
        layer_nodes_names=[ ]
        for val in  bounded_model._modules.values():
            if type(val) not in [auto_LiRPA.operators.leaf.BoundInput,auto_LiRPA.operators.leaf.BoundParams]:
                layer_nodes_names.append(val.name)
        input_layer_name=layer_nodes_names[0]
        print(bounded_model._modules)
        last_layer_name=layer_nodes_names[-1]
        needed_dict={last_layer_name:[input_layer_name]}
        with torch.no_grad():
            lb, ub, A_dict = bounded_model.compute_bounds(x=(bounded_image,),return_A=True,return_b=False,needed_A_dict=needed_dict, b_dict=needed_dict, method='CROWN',need_A_only=False)

        first_to_last_data= A_dict[last_layer_name][input_layer_name]
        lA=first_to_last_data['lA']
        uA=first_to_last_data['uA']
        l_bias=first_to_last_data['lbias']
        u_bias=first_to_last_data['ubias']
        x_rshp=image.reshape((784,))
        lA_rshp=lA.reshape((10,784))
        uA_rshp=uA.reshape((10,784))
        l_bias=first_to_last_data['lbias']
        u_bias=first_to_last_data['ubias']
        mu_l=torch.matmul(lA_rshp,x_rshp)+l_bias.reshape((num_classes,))
        mu_u=torch.matmul(uA_rshp,x_rshp)+u_bias.reshape((num_classes,))
        tlA=torch.transpose(lA_rshp, dim0=1,dim1=0)
        tuA=torch.transpose(uA_rshp, dim0=1,dim1=0)
        if config.noise_dist=='gaussian':
            l_sigmas=torch.matmul(lA_rshp,tlA).diag()
            u_sigmas=torch.matmul(uA_rshp,tuA).diag()

            gamma_l=0.5-0.5*torch.erf(input= (-mu_l/(np.sqrt(2)*torch.sqrt(l_sigmas))))
            gamma_u=0.5-0.5*torch.erf(input= (-mu_u/(np.sqrt(2)*torch.sqrt(u_sigmas))))
        elif config.noise_dist=='uniform':
            pos_gap_l=(mu_l-config.a>=0).float()
            neg_gap_u=(mu_u-config.a<=0).float()
            gamma_l=  pos_gap_l*(1-torch.exp(-(mu_l-config.a)**2/(2*epsilon**2*torch.norm(input=lA_rshp,dim=1))))
            gamma_u= neg_gap_u*(torch.exp(-(mu_u-config.a)**2/(2*epsilon**2*torch.norm(input=uA_rshp, dim=1)))) +(1-neg_gap_u)
        else:
            raise NotImplementedError("Only uniform and Gaussian distributions are implemented for lirpa bounds.")
        
        y_0_flag= torch.arange(num_classes,device=config.device)==y_0
         
        pre_p_l=1-torch.prod(1-gamma_l)/(1-gamma_l[y_0]) 
        p_l=torch.clamp(pre_p_l,min=0,max=1) 
        p_u=torch.clamp(gamma_u.sum()-gamma_u[y_0],min=0,max=1)
            
    iterator= tqdm(range(config.n_rep)) if config.tqdm_opt else range(config.n_rep)
    
    normal_gen=lambda N: torch.randn(size=(N,dim),requires_grad=True).to(device)
    uniform_gen_eps = lambda N: epsilon*(2*torch.rand(size=(N,dim), device=device )-1)
    V_ = lambda X: V_pyt(X,x_0=x_0,model=model_CNN,epsilon=epsilon, target_class=y_0,gaussian_latent=True)
    gradV_ = lambda X: gradV_pyt(X,x_0=x_0,model=model_CNN, target_class=y_0,epsilon=epsilon, gaussian_latent=True)
    x_0.requires_grad=True
    
    for T in config.T_list: 
        for N in config.N_list: 
            for rho in config.rho_list:
                for alpha in config.alpha_list:
                    ests,times,finish_flags = [],[],[]
                    for _ in iterator:
                        # Gaussian latent space version
                        t=time()
                        p_est,finish_flag=smc_pyt.LangevinSMCBasePyt(gen=normal_gen, l_kernel=t_u.langevin_kernel_pyt,V=V_, gradV=gradV_,rho=rho,beta_0=0, min_rate=config.min_rate,
                            alpha =alpha,N=N,T = T,n_max=config.n_max, verbose=config.verbose,adapt_func=None,allow_zero_est=config.allow_zero_est)
                        time_comp=time()-t
                        ests.append(p_est)
                        times.append(time_comp)
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


                    np.savetxt(fname=os.path.join(log_path,'times.txt'),X=times)
                    np.savetxt(fname=os.path.join(log_path,'estimates.txt'),X=estimates)

                    plt.hist(times, bins=10)
                    plt.savefig(os.path.join(log_path,'times_hist.png'))
                    plt.hist(estimates,bins=10)
                    plt.savefig(os.path.join(log_path,'estimates_hist.png'))
                    

                    #with open(os.path.join(log_path,'results.txt'),'w'):
                    results={'method':method_name,'gaussian_latent':str(config.gaussian_latent),
                    'N':config.N,'rho':config.rho,'epsilon':config.epsilons[i],'n_rep':config.n_rep,'T':config.T,'alpha':config.alpha,'min_rate':config.min_rate,
                    'mean time':times.mean(),'std time':times.std(),'mean est':estimates.mean(),
                    'std est':estimates.std(),'gpu_name':config.gpu_name,'cpu_name':config.cpu_name,'cores_number':config.cores_number,'g_target':config.g_target,
                    'freq_finished':freq_finished,'freq_zero_est':freq_zero_est,'unfinished_mean_time':unfinished_mean_time,'unfinished_mean_est':unfinished_mean_est
                    , 'np_seed':config.np_seed,'torch_seed':config.torch_seed,'pgd_success':pgd_success.item(),'p_l':p_l.item(),'p_u':p_u.item(),
                    'noise_dist':config.noise_dist}
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
                            cols+=['pgd_success','p_l','p_u','gpu_name','cpu_name','np_seed','torch_seed','noise_dist']
                            agg_res_df= pd.DataFrame(columns=cols)

                        else:
                            agg_res_df=pd.read_csv(aggr_res_path)
                        agg_res_df = pd.concat([agg_res_df,results_df],ignore_index=True)
                        agg_res_df.to_csv(aggr_res_path,index=False)


    


