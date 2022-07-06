import torch

import torch.nn as nn

from stat_reliability_measure.dev.utils import float_to_file_float
from stat_reliability_measure.dev.torch_arch import CNN_custom,dnn2,dnn4,LeNet,ConvNet,DenseNet3
from torchvision import transforms,datasets,models as tv_models
from torch.utils.data import DataLoader
import timm
from torch import optim
import os
import math
import numpy as np



def TimeStepPyt(V,X,gradV,p=1,p_p=2):
    V_mean= V(X).mean()
    V_grad_norm_mean = ((torch.norm(gradV(X),dim = 1,p=p_p)**p).mean())**(1/p)
    with torch.no_grad():
        result=V_mean/V_grad_norm_mean
    return result

def project_ball_pyt(X, R):
    """projects input on hypersphere of radius R"""
    X_norms = torch.norm(X, dim =1)[:,None]
    X_in_ball = X_norms<=R
    return R*X/X_norms*(~X_in_ball)+(X_in_ball)*X

def projected_langevin_kernel_pyt(X,gradV,delta_t,beta, projection,device=None):
    """performs one step of langevin kernel with a projection"""
    if device is None:
        device=device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    G_noise = torch.randn(size = X.shape).to(device)
    grad=gradV(X)
    with torch.no_grad():
        X_new =projection(X-delta_t*grad+torch.sqrt(2*delta_t/beta)*G_noise)

    return X_new

def fgsm(model, X, y, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def epoch_adversarial(loader, model, attack, opt=None,device='cpu', **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, **kwargs)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def epoch(loader, model, opt=None,device='cpu'):
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def score_function(X,y_0,model):
    y = model(X)
    y_diff = torch.cat((y[:,:y_0], y[:,(y_0+1):]),dim=1) - y[:,y_0].unsqueeze(-1)
    s, _ = y_diff.max(dim=1)
    return s


def langevin_kernel_pyt(X,gradV,delta_t,beta,device=None,gaussian=True,gauss_sigma=1):
    """performs one step of overdamped langevin kernel with inverse temperature beta"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    
    grad=gradV(X)
    
    with torch.no_grad():
        G_noise = torch.randn(size = X.shape).to(device)
        if not gaussian:
            X_new =X-delta_t*grad+math.sqrt(2*delta_t/beta)*G_noise 
        else:
            X_new=(1-delta_t/beta)*X -delta_t*grad+math.sqrt(2*delta_t/beta)*G_noise # U(x)=beta*V(x)+(1/2)x^T*Sigma*x
    return X_new

def langevin_kernel_pyt3(X,gradV,delta_t,beta,device=None,gaussian=True,gauss_sigma=1):
    """performs one step of overdamped langevin kernel with inverse temperature beta"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    
    grad=gradV(X)
    
    with torch.no_grad():
        G_noise = torch.randn(size = X.shape).to(device)
        if not gaussian:
            X_new =X-beta*delta_t*grad+torch.sqrt(2*delta_t)*G_noise 
        else:
            X_new=(1-delta_t)*X -beta*delta_t*grad+torch.sqrt(2*delta_t)*G_noise # U(x)=beta*V(x)+(1/2)x^T*Sigma*x
    return X_new

def langevin_kernel_pyt2(X,gradV,delta_t,beta,device=None,gaussian=True,gauss_sigma=1):
    """performs one step of overdamped langevin kernel with inverse temperature beta"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    
    grad=gradV(X)
    
    with torch.no_grad():
        G_noise = torch.randn(size = X.shape,device=device)
        if not gaussian:
            X_new =X-beta*delta_t*grad+torch.sqrt(2*delta_t)*G_noise 
        else:
            rho_ = 1-delta_t
            X_new=rho_*X -beta*delta_t*grad+math.sqrt(1-rho_**2)*G_noise # U(x)=beta*V(x)+(1/2)x^T*Sigma*x
    return X_new


def verlet_kernel1(X, gradV, delta_t, beta,L,ind_L=None,p_0=None,lambda_=0, gaussian=True,kappa_opt=False,scale_M=None,GV=False):
    """ HMC (L>1) / Underdamped-Langevin (L=1) kernel with Verlet integration (a.k.a. Leapfrog scheme)

    """
    if ind_L is None:
        ind_L=L*torch.ones(size=(X.shape[0],),dtype=torch.int16)
    q_t = torch.clone(X)
    # if no initial momentum is given we draw it randomly from gaussian distribution
    if scale_M is None:
        scale_M=torch.eye(n=1,device=X.device)
    if p_0 is None:                        
        p_t=torch.sqrt(scale_M)*torch.randn_like(X) 

    else:
        p_t = p_0
    grad_q=lambda p,dt:dt*(p.data/scale_M)
    if kappa_opt:
        kappa= 2. / (1 + (1 - delta_t**2)**(1/2)) #if scale_M is 1 else 2. / (1 + torch.sqrt(1 - delta_t**2*(1/scale_M)))
       
    else:
        kappa=torch.ones_like(q_t)
    k=1
    i_k=ind_L>=k
    while (i_k).sum()>0:
        #I. Verlet scheme
        #computing half-point momentum
        #p_t.data = p_t-0.5*dt*gradV(X).detach() / norms(gradV(X).detach())
        
        if not GV:
            p_t.data[i_k] = p_t[i_k]-0.5*beta*delta_t[i_k]*gradV(q_t[i_k]).detach() 
        
        
        p_t.data[i_k] = p_t[i_k]- 0.5*delta_t[i_k]*kappa[i_k]*q_t.data[i_k]
        if p_t.isnan().any():
            print("p_t",p_t)
        #updating position
        q_t.data[i_k] = (q_t[i_k] + grad_q(p_t[i_k],delta_t[i_k]))
        assert not q_t.isnan().any(),"Nan values detected in q"
        #updating momentum again
        if not GV:
            p_t.data[i_k] = p_t[i_k] -0.5*beta*delta_t[i_k]*gradV(q_t[i_k]).detach()
        p_t.data[i_k] =p_t[i_k] -0.5*kappa[i_k]*delta_t[i_k]*q_t.data[i_k]
        #II. Optional smoothing of momentum memory
        p_t.data[i_k] = torch.exp(-lambda_*delta_t[i_k])*p_t[i_k]
        k+=1
        i_k=ind_L>=k
    return q_t.detach(),p_t

def verlet_kernel2(X, gradV, delta_t, beta,L,ind_L=None,p_0=None,lambda_=0, gaussian=True,kappa_opt=False,scale_M=None,GV=False):
    """ HMC (L>1) / Underdamped-Langevin (L=1) kernel with Verlet integration (a.k.a. Leapfrog scheme)

    """
    if ind_L is None:
        ind_L=L*torch.ones(size=(X.shape[0],),dtype=torch.int16)
    q_t = torch.clone(X)
    # if no initial momentum is given we draw it randomly from gaussian distribution
    if scale_M is None:
        scale_M=torch.Tensor([1])
    if p_0 is None:                        
        p_t=torch.sqrt(scale_M)*torch.randn_like(X) 

    else:
        p_t = p_0
    grad_q=lambda p,dt:dt*(p.data/scale_M)
    if kappa_opt:
        kappa= 2. / (1 + (1 - delta_t**2)**(1/2)) if scale_M == 1 else 2. / (1 + torch.sqrt(1 - delta_t**2*(1/scale_M)))
    else:
        kappa=1
    k=1
    i_k=ind_L>=k
    while (i_k).sum()>0:
        #I. Verlet scheme
        #computing half-point momentum
        #p_t.data = p_t-0.5*dt*gradV(X).detach() / norms(gradV(X).detach())
        if not GV:
            p_t.data[i_k] = p_t[i_k]-0.5*beta*delta_t[i_k]*kappa[i_k]*gradV(q_t[i_k]).detach() 
        
        p_t.data[i_k] -= 0.5*delta_t[i_k]*kappa[i_k]*q_t.data[i_k]
        #updating position
        q_t.data[i_k] = (q_t[i_k] + grad_q(p_t[i_k],delta_t[i_k]))
        #updating momentum again
        if not GV:
            p_t.data[i_k] = p_t[i_k] -0.5*beta*kappa[i_k]*delta_t[i_k]*gradV(q_t[i_k]).detach()
        p_t.data[i_k] -= 0.5*kappa[i_k]*delta_t[i_k]*q_t.data[i_k]
        #II. Optional smoothing of momentum memory
        p_t.data[i_k] = torch.exp(-lambda_*delta_t[i_k])*p_t[i_k]
        k+=1
        i_k=ind_L>=k
    return q_t.detach(),p_t




def multi_unsqueeze(input_,k,dim=-1):
    """"unsqueeze input multiple times"""
    for _ in range(k):
        input_=input_.unsqueeze(dim=dim)
    return input_


def compute_V_grad_pyt2(model, input_, target_class):
    """ Returns potentials and associated gradients for given inputs, model and target classes """
    if input_.requires_grad!=True:
        print("/!\\ input does not require gradient")
        input_.requires_grad=True
    #input_.retain_grad()
    
    logits = model(input_) 
    val, ind= torch.topk(logits,k=2)
    v_=val[:,0]-val[:,1]
    #v_.backward(gradient=torch.ones_like(v_),retain_graph=True)
    

    a_priori_grad=torch.autograd.grad(outputs=v_,inputs=input_,grad_outputs=torch.ones_like(v_),retain_graph=False)[0]
    with torch.no_grad():
        mask=torch.eq(ind[:,0],target_class)
        v=torch.where(condition=mask, input=v_,other=torch.zeros_like(v_))
        mask=multi_unsqueeze(mask,k=a_priori_grad.ndim- mask.ndim)
        grad=torch.where(condition=mask, input=a_priori_grad,other=torch.zeros_like(a_priori_grad))
    return v,grad

def compute_V_grad_pyt(model, input_, target_class,L=0):
    """ Returns potentials and associated gradients for given inputs, model and target classes """
    if input_.requires_grad!=True:
        print("/!\\ input does not require gradient")
        input_.requires_grad=True
    #input_.retain_grad()
    s=score_function(X=input_,y_0=target_class,model=model)
    v=torch.clamp(L-s,min=0)

    

    grad=torch.autograd.grad(outputs=v,inputs=input_,grad_outputs=torch.ones_like(v),retain_graph=False)[0]
   
    return v,grad


def compute_V_pyt2(model, input_, target_class,L=0):
    """Return potentials for given inputs, model and target classes"""
    with torch.no_grad():
        logits = model(input_) 
        val, ind= torch.topk(logits,k=2)
        output=val[:,0]-val[:,1]
        mask=ind[:,0]==target_class
        v=torch.where(condition=mask, input=output,other=torch.zeros_like(output))
    return v 

def compute_V_pyt(model, input_, target_class,L=0):
    """Return potentials for given inputs, model and target classes"""
    with torch.no_grad():
        s=score_function(X=input_, y_0=target_class, model=model)
        v=torch.clamp(L-s,min=0)
    return v 

def compute_h_pyt(model, input_, target_class):
    """Return potentials for given inputs, model and target classes"""
    with torch.no_grad():
        logits = model(input_) 
        val, ind= torch.topk(logits,k=2)
        output=val[:,0]-val[:,1]
        mask=ind[:,0]==target_class
        other=val[:,0]-logits[:,target_class]
        v=torch.where(condition=mask, input=output,other=other)
    return v 

normal_dist=torch.distributions.Normal(loc=0, scale=1.)

def V_pyt(x_,x_0,model,target_class,low,high,gaussian_latent=True,reshape=True,input_shape=None):
    with torch.no_grad():
        if input_shape is None:
            input_shape=x_0.shape
        if gaussian_latent:
            u=normal_dist.cdf(x_)
        else:
            u=x_
        if reshape:
            u=torch.reshape(u,(u.shape[0],)+input_shape)

        
        x_p = low+(high-low)*u
    v = compute_V_pyt(model=model,input_=x_p,target_class=target_class)
    return v

def gradV_pyt(x_,x_0,model,target_class,low,high,gaussian_latent=True,reshape=True,input_shape=None,gaussian_prior=False):
   
    if input_shape is None:
        input_shape=x_0.shape
    if gaussian_latent and not gaussian_prior:
        u=normal_dist.cdf(x_)
    else:
        u=x_
    if reshape:
        u=torch.reshape(u,(u.shape[0],)+input_shape)
    
    x_p = low+(high-low)*u
    _,grad_x_p = compute_V_grad_pyt(model=model,input_=x_p,target_class=target_class)
    grad_u=torch.reshape((high-low)*grad_x_p,x_.shape)
    if gaussian_latent:
        grad_x=torch.exp(normal_dist.log_prob(x_))*grad_u
    else:
        grad_x=grad_u
    return grad_x

def correct_min_max(x_min,x_max,x_mean,x_std):
    if x_min is None or x_max is None: 
        if x_min is None:
            x_min=0
        if x_max is None:
            x_max=1
    
    x_min=(x_min-x_mean)/x_std
    x_max=(x_max-x_mean)/x_std
    return x_min,x_max

supported_datasets=['mnist','cifar10','cifar100','imagenet']
datasets_in_shape={'mnist':(1,28,28),'cifar10':(3,32,32),'cifar100':(3,32,32),'imagenet':(3,224,224)}
datasets_dims={'mnist':784,'cifar10':3*1024,'cifar100':3*1024,'imagenet':3*224**2}
datasets_num_c={'mnist':10,'cifar10':10,'imagenet':1000}
datasets_means={'mnist':0,'cifar10':(0.4914, 0.4822, 0.4465),'cifar100':[125.3/255.0, 123.0/255.0, 113.9/255.0]}
datasets_stds={'mnist':1,'cifar10':(0.2023, 0.1994, 0.2010),'cifar100':[63.0/255.0, 62.1/255.0, 66.7/255.0]}
datasets_supp_archs={'mnist':{'dnn2':dnn2,'dnn4':dnn4,'cnn_custom':CNN_custom},
                    'cifar10':{'lenet':LeNet,'convnet':ConvNet},
                    'cifar100':{'densenet':DenseNet3}}
def get_loader(train,data_dir,download,dataset='mnist',batch_size=100,x_mean=None,x_std=None): 
    assert dataset in supported_datasets,f"support datasets are in {supported_datasets}"
    if dataset=='mnist':
        if x_mean is not None and x_std is not None: 
            assert x_std!=0, "Can't normalize with 0 std."
        
            transform_=transforms.Compose([
                                transforms.ToTensor(),
                                #transforms.Normalize((x_mean,), (x_std,)) we perform normalization at the model level
                            ])
        else: 
            transform_ =transforms.ToTensor()
        mnist_dataset = datasets.MNIST(data_dir, train=train, download=download,
         transform=transform_)
        data_loader = DataLoader(mnist_dataset, batch_size = batch_size, shuffle=train)
    elif dataset in ('cifar10','cifar100'):
        if train:
            data_transform = transforms.Compose([
                #transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #we perform normalization at the model level
                #transforms.Normalize(mean=datasets_stds['cifar10'], std=datasets_stds['cifar10']),
            ])
        else:
            data_transform=transforms.Compose([
                            transforms.ToTensor(),
                        #transforms.Normalize(mean=datasets_stds['cifar10'],std=datasets_stds['cifar10'])
                        ])
        cifar10_dataset = datasets.CIFAR10("../data", train=train, download=download, transform=data_transform)
        data_loader = DataLoader(cifar10_dataset , batch_size = batch_size, shuffle=train)  
    elif dataset=='cifar100':
        if train:
            data_transform = transforms.Compose([
                #transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #we perform normalization at the model level
                #transforms.Normalize(mean=datasets_stds['cifar10'], std=datasets_stds['cifar10']),
            ])
        else:
            data_transform=transforms.Compose([
                            transforms.ToTensor(),
                        #transforms.Normalize(mean=datasets_stds['cifar10'],std=datasets_stds['cifar10'])
                        ])
        cifar100_dataset = datasets.CIFAR100("../data", train=train, download=download, transform=data_transform)
        data_loader = DataLoader(cifar100_dataset , batch_size = batch_size, shuffle=train)  
   

    elif dataset=='imagenet':
        assert train==False,"Only validation data can be loaded for the ImageNet dataset."
        data_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            
        ])

        imagenet_dataset = datasets.ImageNet(data_dir, split="val", transform=data_transform)

        data_loader = DataLoader(imagenet_dataset, batch_size =batch_size, shuffle=False)



    
                    
    return data_loader


def get_correct_x_y(data_loader,device,model):
    for X,y in data_loader:
        X,y = X.to(device), y.to(device)
        break
    with torch.no_grad():
        logits=model(X)
        y_pred= torch.argmax(logits,-1)
        correct_idx=y_pred==y
    return X[correct_idx],y[correct_idx],correct_idx.float().mean()

supported_arch={'cnn_custom':CNN_custom,'dnn2':dnn2,'dnn4':dnn4,}
def get_model_imagenet(model_arch,model_dir):
    torch.hub.set_dir(model_dir)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalizer = transforms.Normalize(mean=mean, std=std)
    if model_arch.lower().startswith("torchvision"):
        model = getattr(tv_models, model_arch[len("torchvision_"):])(pretrained=True)
    else:
        model = timm.create_model(model_arch, pretrained=True)
        mean = model.default_cfg["mean"]
        std = model.default_cfg["std"]
        normalizer = transforms.Normalize(mean=mean, std=std)

    model = torch.nn.Sequential(normalizer, model).cuda(0).eval()
    return model,mean,std

def get_model(model_arch, robust_model, robust_eps,nb_epochs,model_dir,data_dir,test_loader, device ,
download,force_train=False,dataset='mnist',batch_size=100):
    
    input_shape=datasets_in_shape[dataset]
    print(f"input_shape:{input_shape}")
    if robust_model:
        c_robust_eps=float_to_file_float(robust_eps)
    model_name=model_arch +'_' + dataset if not robust_model else f"model_{model_arch}_{dataset}_robust_{c_robust_eps}"
    
    c_model_path=model_name+'.pt'
    model_path=os.path.join(model_dir,c_model_path)
    support_arch=datasets_supp_archs[dataset]
    assert model_arch.lower() in support_arch,f"/!\\Architectures supported for {dataset}:{list(support_arch.keys())}"
    network=support_arch[model_arch.lower()](dataset=dataset)
    normalizer=transforms.Normalize(mean=datasets_means[dataset], std=datasets_stds[dataset])
    if not os.path.exists(model_path) or force_train: 
        #if the model doesn't exist we retrain a model from scratch
        model=torch.nn.Sequential(normalizer, network)
        model.to(device)
        if force_train and os.path.exists(model_path):
            print("Retraining model from scratch")
            os.remove(model_path)
        else:
            print("Model not found: it will be trained from scratch.")
        train_loader=get_loader(train=True, data_dir=data_dir, download=download,dataset=dataset,batch_size=batch_size)
        opt = optim.SGD(model.parameters(), lr=1e-1)

        print("Train Err", "Train Loss", "Test Err", "Test Loss", sep="\t")
        for _ in range(nb_epochs):
            train_err, train_loss = epoch(train_loader, model, opt, device=device)
            if robust_model:
                train_err, train_loss = epoch_adversarial(train_loader, model, opt, device=device)
            test_err, test_loss = epoch(test_loader, model, device=device)
            print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
        torch.save(network.state_dict(), model_path)
    network.load_state_dict(torch.load(model_path))
    model=torch.nn.Sequential(normalizer, network)
    model.eval()
    model.to(device)
    return model, input_shape, model_name

def apply_l_kernel(Y,v_y,l_kernel,T:int,beta,gradV,V,delta_t:float,mh_opt:bool,track_accept:bool,
 gaussian:bool,device,v1_kernel:bool, adapt_d_t:bool,
target_accept:float, accept_spread:float,d_t_decay:float,d_t_gain:float,debug:bool=False,verbose:float =1
,track_delta_t=False,d_t_min=None,d_t_max=None,save_Y=False):
    """_summary_

    Args:
        Y (_type_): _description_
        v_y (_type_): _description_
        l_kernel (_type_): _description_
        T (int): _description_
        beta (_type_): _description_
        gradV (_type_): _description_
        V (function): _description_
        delta_t (float): _description_
        mh_opt (bool): _description_
        track_accept (bool): _description_
        gaussian (bool): _description_
        device (_type_): _description_
        v1_kernel (bool): _description_
        adapt_d_t (bool): _description_
        target_accept (float): _description_
        accept_spread (float): _description_
        d_t_decay (float): _description_
        d_t_gain (float): _description_
        debug (bool, optional): _description_. Defaults to False.
        verbose (float, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    nb=Y.shape[0]
    nb_calls=0
    local_accept_rates=[]
    if track_delta_t:
        delta_ts=[]
    if save_Y:
        Y_s=[Y]
    for _ in range(T):
            
        if track_accept or mh_opt:
            cand_Y=l_kernel(Y,gradV, delta_t,beta)
            with torch.no_grad():
                #cand_v=V(cand_Y).detach()
                cand_v_y = V(cand_Y)
                nb_calls+=nb
            log_a_high=-beta*(cand_v_y)
            log_a_low= -beta*v_y
            if verbose>=5:
                print(f"intermediate 1 log_diff:{log_a_high-log_a_low}")
            if v1_kernel:
                high_diff= (Y-cand_Y-delta_t*gradV(cand_Y)).detach()
                low_diff=(cand_Y-Y-delta_t*gradV(Y)).detach()
                if gaussian:
                    high_diff+=(delta_t/beta)*cand_Y
                    low_diff+=(delta_t/beta)*Y
              
        
                log_a_high-=(beta/(4*delta_t))*torch.norm(high_diff,p=2 ,dim=1)**2
                log_a_low-=(beta/(4*delta_t))*torch.norm(low_diff,p=2,dim=1)**2
                if verbose>=5: 
                    print(f"Intermediate 2 log_diff:{log_a_high-log_a_low}")

            #using 'Orstein-Uhlenbeck' version    
            else:
                high_diff= (Y-cand_Y-beta*delta_t*gradV(cand_Y)).detach()
                low_diff=(cand_Y-Y-beta*delta_t*gradV(Y)).detach()
                if gaussian:
                    high_diff+=delta_t*cand_Y.detach()
                    low_diff+=delta_t*Y.detach()
                log_a_high-=(1/(delta_t*(2*delta_t)))*torch.norm(high_diff,p=2 ,dim=1)**2
                log_a_low-=(1/(delta_t*(2*delta_t)))*torch.norm(low_diff,p=2 ,dim=1)**2
                if verbose>=5: 
                    print(f"Intermediate 2 log_diff:{log_a_high-log_a_low}")
            nb_calls+= 2*nb
            
            
            
            
            if gaussian: 
                log_a_high-=0.5*torch.sum(cand_Y**2,dim=1)
                log_a_low-=0.5*torch.sum(Y**2,dim=1)

            if verbose>=5: 
                print(f"intermediate 3 log_diff:{log_a_high-log_a_low}")
            #alpha=torch.clamp(input=torch.eYp(log_a_high-log_a_low),max=1) /!\ clamping is not necessary
            alpha_=torch.exp(log_a_high-log_a_low)
            if verbose>=5: 
                print(f"alpha_={alpha_}")
            b_size= nb
            U=torch.rand(size=(b_size,),device=device) 
            accept=U<alpha_
            accept_rate=accept.float().mean().item()
            if track_accept:
                if verbose>=3:
                    print(f"Local accept rate: {accept_rate}")
                #accept_rates.append(accept_rate)
                local_accept_rates.append(accept_rate)
            if adapt_d_t:
                if accept_rate>target_accept+accept_spread:
                    delta_t*=d_t_gain
                    

                elif accept_rate<target_accept-accept_spread: 
                    delta_t*=d_t_decay
                if d_t_min is not None:
                    delta_t=torch.clamp(delta_t,min=d_t_min)
                if d_t_max is not None:
                    delta_t=torch.clamp(delta_t,max=d_t_max)
                if verbose>=2.5:
                    print(f"New delta_t:{delta_t.item()}")

                if track_delta_t:
                    delta_ts.append(delta_t.item())
                    if verbose>=2:
                        print(delta_ts)
            if mh_opt:
                with torch.no_grad():
                    Y=torch.where(accept.unsqueeze(-1),input=cand_Y,other=Y)
                    
                    v_y=torch.where(accept, input=cand_v_y,other=v_y)
                    nb_calls+=nb
                    if debug:
                        
                        v2 = V(Y)
                        #nb_calls+= N
                        assert torch.equal(v_y,v2),"/!\ error in potential computation"
            else:
                Y=cand_Y
                v_y=cand_v_y
        else:
            Y=l_kernel(Y, gradV, delta_t, beta)
        if save_Y:
            Y_s.append(Y)    
        nb_calls= nb_calls+nb

    if not track_accept or mh_opt:
        with torch.no_grad():
            v_y=V(Y)

    dict_out={}
    if track_accept:
        dict_out['local_accept_rates']=local_accept_rates
    if adapt_d_t:
        dict_out['delta_t']=delta_t
        if track_delta_t: 
            dict_out['delta_ts']=delta_ts
    if save_Y:
        dict_out['Y_s'] =Y_s
    return Y,v_y,nb_calls,dict_out


def normal_kernel(x,s):
    return (x + s*torch.randn_like(x))/math.sqrt(1+s**2)

def gaussian_kernel(x,dt,scale_M=1):
    kappa= 1. / (1 + torch.sqrt(1 - dt**2*(1/scale_M)))
    return x-dt**2*kappa*(1/scale_M)*x+dt*((scale_M)**(-1/2))*torch.randn_like(x)

def apply_gaussian_kernel(Y,v_y,T:int,beta:float,dt, V,
 gaussian:bool,device, adapt_dt:bool=False,
 debug:bool=False,verbose=1
,kernel_pass=0, track_accept:bool=False,
save_Ys=False,scale_M=1):
    nb=Y.shape[0]
    Ys=[Y]
    VY=v_y 
    nb_calls=0
    l_accept_rates=[]
    l_kernel_pass=0
    for _ in range(T):
        Z = gaussian_kernel(Y,dt,scale_M=scale_M) # propose a refreshed samples
        kernel_pass+=nb
        l_kernel_pass+=nb
        # compute their scores
        VZ= V(Z)
        nb_calls+=nb
        log_a_high=-beta*VZ#-(s**2/(1+s**2))*torch.norm(high_diff,p=2 ,dim=1)**2
        log_a_low= -beta*VY#-(s**2/(1+s**2))*torch.norm(low_diff,p=2,dim=1)**2
        if gaussian: 
            log_a_high-= 0.5*torch.sum(Z**2,dim=1)
            log_a_low-= 0.5*torch.sum(Y**2,dim=1)
        #alpha=torch.clamp(input=torch.eYp(log_a_high-log_a_low),max=1) /!\ clamping is not necessary
        alpha_=torch.exp(log_a_high-log_a_low)
        b_size= nb
        U=torch.rand(size=(b_size,),device=device) 
        accept_flag=U<alpha_
        if verbose>=2.5:
            print(accept_flag.float().mean().item())
        Y=torch.where(accept_flag.unsqueeze(-1),input=Z,other=Y)
        Ys.append(Y)
        VY=torch.where(accept_flag,input=VZ,other=VY)
        if track_accept:
            l_accept_rate=accept_flag.float().mean().item()
            if verbose>=3:
                print(f"local accept_rate:{l_accept_rate}")
            l_accept_rates.append(l_accept_rate)
    dict_out={'l_kernel_pass':l_kernel_pass}
    if adapt_dt:
        dict_out['dt']=dt
    if track_accept:
        dict_out['acc_rate']=np.array(l_accept_rates).mean()
    return Y,VY,nb_calls,dict_out


def apply_simp_kernel(Y,v_y,simp_kernel,T:int,beta:float,s:float, V,
 gaussian:bool,device, decay:float,clip_s:bool,s_min:float,s_max:float,
 debug:bool=False,verbose:float =1,rejection_rate:float=0
,kernel_pass=0, track_accept:bool=False,reject_ctrl:bool=True,reject_thresh:float=0.9,
save_Ys=False):
   
    nb=Y.shape[0]
    Ys=[Y]
    VY=v_y 
    nb_calls=0
    l_accept_rates=[]
    l_kernel_pass=0
    for _ in range(T):
        Z = simp_kernel(Y,s) # propose a refreshed samples
        kernel_pass+=nb
        l_kernel_pass+=nb
        # compute their scores
        VZ= V(Z)
        nb_calls+=nb
        #accept_flag= SZ>L_j              
        #nb_calls+= 2*nb
        # The gaussian kernel used here is reversible thus Q(x|y)=Q(y|x)
        #high_diff=(Z-(1/math.sqrt(1+s**2))*Y)
        #low_diff=(Y-(1/math.sqrt(1+s**2))*Z)
        
        log_a_high=-beta*VZ#-(s**2/(1+s**2))*torch.norm(high_diff,p=2 ,dim=1)**2
        log_a_low= -beta*VY#-(s**2/(1+s**2))*torch.norm(low_diff,p=2,dim=1)**2


        if gaussian: 
            log_a_high-= 0.5*torch.sum(Z**2,dim=1)
            log_a_low-= 0.5*torch.sum(Y**2,dim=1)
        #alpha=torch.clamp(input=torch.eYp(log_a_high-log_a_low),max=1) /!\ clamping is not necessary
        alpha_=torch.exp(log_a_high-log_a_low)
        b_size= nb
        U=torch.rand(size=(b_size,),device=device) 
        accept_flag=U<alpha_
        if verbose>=2.5:
            print(accept_flag.float().mean().item())
        Y=torch.where(accept_flag.unsqueeze(-1),input=Z,other=Y)
        Ys.append(Y)
        VY=torch.where(accept_flag,input=VZ,other=VY)
        rejection_rate  = (kernel_pass-(nb))/kernel_pass*rejection_rate+(1./kernel_pass)*((1-accept_flag.float()).sum().item())
        
        if track_accept:
            l_accept_rate=accept_flag.float().mean().item()
            if verbose>=3:
                print(f"local accept_rate:{l_accept_rate}")
            l_accept_rates.append(l_accept_rate)
        if reject_ctrl and rejection_rate>=reject_thresh:
            
            s = s*decay if not clip_s else np.clip(s*decay,a_min=s_min,a_max=s_max)
            if verbose>1:            
                print('Strength of kernel diminished!')
                print(f's={s.item()}')
    
    dict_out={'l_kernel_pass':l_kernel_pass}
    if track_accept:
        dict_out['local_accept_rates']=l_accept_rates
    if reject_ctrl:
        dict_out['s']=s
        dict_out['rejection_rate']=rejection_rate
    return Y,VY,nb_calls,dict_out

        


# def compute_V_grad_pyt3(model, input_, target_class):
#     """ Returns potentials and associated gradients for given inputs, model and target classes """
#     if input_.requires_grad!=True:
#         print("/!\\ Input does not require gradient (we assume it is a leaf node).")
#         input_.requires_grad=True
#     input_.retain_grad()
#     logits = model(input_) 
#     val, ind= torch.topk(logits,k=2)
#     v_=val[:,0]-val[:,1]
#     v_.backward(gradient=torch.ones_like(v_),retain_graph=True)
    

#     a_priori_grad=input_.grad
#     mask=torch.eq(ind[:,0],target_class)
#     v=torch.where(condition=mask, input=v_,other=torch.zeros_like(v_))
#     mask=multi_unsqueeze(mask,k=a_priori_grad.ndim- mask.ndim)
#     grad=torch.where(condition=mask, input=a_priori_grad,other=torch.zeros_like(a_priori_grad))
#     return v,grad


def Hamiltonian(X,p,V,beta,scale_M=1,gaussian=True):
    with torch.no_grad():
        U = beta*V(X) +0.5*torch.sum(X**2,dim=1) if gaussian else beta*V(X)
    H = U + 0.5*torch.sum((1/scale_M)*p**2,dim=1)
    return H





def apply_v_kernel(Y,v_y,v_kernel,T:int,beta,gradV,V,delta_t:float,mh_opt:bool,track_accept:bool,
 gaussian:bool,device, adapt_d_t:bool,
target_accept:float, accept_spread:float,d_t_decay:float,d_t_gain:float,debug:bool=False,verbose:float =1
,track_delta_t=False,d_t_min=None,d_t_max=None,lambda_=0,L=1,gamma=0,track_H=False,
save_y=False,kappa_opt=True):
    """_summary_

    Args:
        Y (_type_): _description_
        v_y (_type_): _description_
        l_kernel (_type_): _description_
        T (int): _description_
        beta (_type_): _description_
        gradV (_type_): _description_
        V (function): _description_
        delta_t (float): _description_
        mh_opt (bool): _description_
        track_accept (bool): _description_
        gaussian (bool): _description_
        device (_type_): _description_
        v1_kernel (bool): _description_
        adapt_d_t (bool): _description_
        target_accept (float): _description_
        accept_spread (float): _description_
        d_t_decay (float): _description_
        d_t_gain (float): _description_
        debug (bool, optional): _description_. Defaults to False.
        verbose (float, optional): _description_. Defaults to 1.
        L (int, optional): number of Hamiltonian dynamics iteration 
        Track_H (bool, optional): tracking Halmitonian values
    Returns:
        _type_: _description_
    """
    nb=Y.shape[0]
    nb_calls=0
    local_accept_rates=[]
    if track_delta_t:
        delta_ts=[]
    p_0 = torch.randn_like(Y)
    H=Hamiltonian(X=Y, p=p_0,V=V,beta=beta,gaussian=gaussian,)
    H_s=[]
    if track_H:
        H_s=[H.cpu().mean().item()]
    if save_y:
        y_s=[Y]
    nb_calls+=nb
    for _ in range(T):
            
        if track_accept or mh_opt:
            cand_Y,cand_p=v_kernel(Y,gradV,p_0 =p_0, delta_t=delta_t,beta=beta,lambda_=lambda_,L=L,kappa_opt=kappa_opt)
            nb_calls+=L*nb
            cand_H=Hamiltonian(X=cand_Y,p=cand_p,V=V,beta=beta,gaussian =gaussian)
            nb_calls+=nb
            alpha_=min(torch.exp(-cand_H+H),1)
            if verbose>=5: 
                print(f"alpha_={alpha_}")
            b_size= nb
            U=torch.rand(size=(b_size,),device=device) 
            accept=U<alpha_
            accept_rate=accept.float().mean().item()
            if track_accept:
                if verbose>=3:
                    print(f"Local accept rate: {accept_rate}")
                #accept_rates.append(accept_rate)
                local_accept_rates.append(accept_rate)
            if adapt_d_t:
                if accept_rate>target_accept+accept_spread:
                    delta_t*=d_t_gain
                    

                elif accept_rate<target_accept-accept_spread: 
                    delta_t*=d_t_decay
                if d_t_min is not None:
                    delta_t=torch.clamp(delta_t,min=d_t_min)
                if d_t_max is not None:
                    delta_t=torch.clamp(delta_t,max=d_t_max)
                if verbose>=2.5:
                    print(f"New delta_t:{delta_t.item()}")

                if track_delta_t:
                    delta_ts.append(delta_t.item())
                    if verbose>=2:
                        print(delta_ts)
            if mh_opt:
                with torch.no_grad():
                    Y=torch.where(accept.unsqueeze(-1),input=cand_Y,other=Y)
                    
                    H=torch.where(accept, input=cand_H,other=H)
                    if track_H:
                        H_s.append(H.cpu().mean().item())

                    p_0=torch.where(accept.unsqueeze(-1),input=cand_p,other=p_0)
                    p_0=gamma*p_0+(1-gamma)*torch.randn_like(p_0)
                    nb_calls+=nb
                    if debug:
                        
                        v2 = V(Y)
                        #nb_calls+= N
                        assert torch.equal(v_y,v2),"/!\ error in potential computation"
            else:
                Y=cand_Y
                H=cand_H
                H_s.append(H.cpu().mean().item())
        else:
            Y,p_0=v_kernel(Y, gradV, delta_t, beta)
            p_0=gamma*p_0+(1-gamma)*torch.randn_like(p_0)
            
        if save_y:
            y_s.append(Y)    
        nb_calls= nb_calls+nb

    
    with torch.no_grad():
        v_y=V(Y)
    nb_calls+=nb

    dict_out={}
    if track_accept:
        dict_out['local_accept_rates']=local_accept_rates
    if adapt_d_t:
        dict_out['delta_t']=delta_t
        if track_delta_t: 
            dict_out['delta_ts']=delta_ts
    if track_H:
        H_s=np.array(H_s)
        dict_out['H_stds']=H_s.std()
        dict_out['H_means']=H_s.mean()
    if save_y:
        dict_out['Y_s'] = y_s
    return Y,v_y,nb_calls,dict_out



def verlet_mcmc(q,beta:float,gaussian:bool,V,gradV,T:int,delta_t,L:int=1,
kappa_opt:bool=True,save_H=True,save_func=None,device='cpu',scale_M=None,gaussian_verlet=False,ind_L=None):
    """ Simple implementation of Hamiltonian dynanimcs MCMC 

    Args:
        q (_type_): _description_
        p (_type_): _description_
        beta (_type_): _description_
        gaussian (_type_): _description_
        V (_type_): _description_
        gradV (_type_): _description_
        T (_type_): _description_
        delta_t (_type_): _description_
        kappa_opt (bool, optional): _description_. Defaults to True.
        save_H (bool, optional): _description_. Defaults to True.
        save_Q (bool, optional): _description_. Defaults to True.
        save_func (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    acc  = 0
    p=torch.randn_like(q)
    if scale_M is not None:
        sqrt_M = torch.sqrt(scale_M)
        p=sqrt_M*torch.randn_like(q)
    (N,d)=q.shape
    H_old = Hamiltonian(X=q,p=p,beta=beta, gaussian=gaussian,V=V)
    nb_calls=N
    if save_H:
        H_ = np.zeros(T+1)
        H_[0]=H_old.mean()

    if save_func is not None:
        saved=[save_func(q,p)]
    for i  in range(T):
        q_trial,p_trial=verlet_kernel1(X=q,gradV=gradV, p_0=p,delta_t=delta_t,beta=0,L=L,kappa_opt=kappa_opt,
        scale_M=scale_M,ind_L=ind_L,GV=gaussian_verlet)
        nb_calls+=ind_L.sum().item()
        H_trial= Hamiltonian(X=q_trial, p=p_trial,V=V,beta=beta,gaussian=gaussian)
        nb_calls+=N
        
        alpha=torch.rand(size=(N,),device=device )
        delta_H=torch.clamp(-(H_trial-H_old),max=0)
        accept=torch.exp(delta_H)>alpha
        acc+=accept.sum()
        q=torch.where(accept.unsqueeze(-1),input=q_trial,other=q)
        p = torch.randn_like(q)
        if scale_M is not None:
            p = sqrt_M*p
        H_old= Hamiltonian(X=q, p=p,V=V,beta=beta,gaussian=gaussian)
        if save_H:
            H_[i+1] = H_old.mean()
        nb_calls+=N
        

        if save_func is not None:
            saved.append(save_func(q,p))
    
    dict_out={'acc_rate':acc/(T*N)}

    if save_H:
        dict_out['H']=H_
    if save_func is not None:
        dict_out['saved']=saved
    v_q=V(q)
    nb_calls+=N
    return q,v_q,nb_calls,dict_out


def adapt_verlet_mcmc(q,v_q,ind_L,beta:float,gaussian:bool,V,gradV,T:int,delta_t,L:int=1,
kappa_opt:bool=True,save_H=True,save_func=None,device='cpu',scale_M=None,alpha_p:float=0.1,prop_d=0.1,FT=False,dt_max=None,dt_min=None,sig_dt=0.015,
verbose=0,L_min=1,gaussian_verlet=False,skip_mh=False):
    """ Simple implementation of Hamiltonian dynanimcs MCMC 

    Args:
        q (_type_): _description_
        p (_type_): _description_
        beta (_type_): _description_
        gaussian (_type_): _description_
        V (_type_): _description_
        gradV (_type_): _description_
        T (_type_): _description_
        delta_t (_type_): _description_
        kappa_opt (bool, optional): _description_. Defaults to True.
        save_H (bool, optional): _description_. Defaults to True.
        save_Q (bool, optional): _description_. Defaults to True.
        save_func (_type_, optional): _description_. Defaults to None.
        alpha_p 

    Returns:
        _type_: _description_
    """
    acc  = 0
    T_max=T
    if scale_M is None:
        scale_M=1
        sqrt_M=1
    else:
        sqrt_M = torch.sqrt(scale_M) 
    p=sqrt_M*torch.randn_like(q)
    if FT:
        
        maha_dist = lambda x,y: ((1/scale_M)*(x-y)**2).sum(1)
        ones_L=torch.ones_like(ind_L)
    (N,d)=q.shape
    o_old=q+q**2
    mu_old,sig_old=(o_old).mean(0),(o_old).std(0)
    H_old= Hamiltonian(X=q, p=p,V=V,beta=beta,gaussian=gaussian,scale_M=scale_M)
    nb_calls=0 #we can reuse the potential value v_q of previous iteration: no new potential computation
    if save_H:
        H_ = np.zeros(T+1)
        H_[0]=H_old.mean()

    if save_func is not None:
        saved=[save_func(q,p)]
    prod_correl=torch.ones(size=(d,),device=q.device)
    i=0
    #torch.multinomial(input=)
    while (prod_correl>alpha_p).sum()>=prop_d*d and i<T_max:
        q_trial,p_trial=verlet_kernel1(X=q,gradV=gradV, p_0=p,delta_t=delta_t,beta=beta,L=L,kappa_opt=kappa_opt,
        scale_M=scale_M, ind_L=ind_L,GV=gaussian_verlet)
        
        nb_calls+=4*ind_L.sum().item()-N # for each particle each vertlet integration step requires two oracle calls (gradients)
        H_trial= Hamiltonian(X=q_trial, p=p_trial,V=V,beta=beta,gaussian=gaussian,scale_M=scale_M)
        nb_calls+=N # N new potentials are computed 
        delta_H=torch.clamp(-(H_trial-H_old),max=0)
        if FT:
            exp_weight=torch.exp(torch.clamp(delta_H,max=0))
            m_distances= maha_dist(x=q,y=q_trial)/ind_L.float().to(device)
            lambda_i=m_distances*exp_weight+1e-8*torch.ones_like(m_distances)
            if lambda_i.isnan().any():
                print(f"NaN values in lambda_i:")
                print(lambda_i)
            elif (lambda_i==0).sum()>0:
                print(f"zero values in lambda_i")
                print(f"lambda_i")
                print(f"exp_weight:{exp_weight}")
                print(f"m_distances:{m_distances}")
                if (m_distances==0).sum()>0:
                    print(f"q_trial:{q_trial}")
                    print(f"q:{q}")
                print(f"ind_L:{ind_L}")
            
            sel_ind=torch.multinomial(input=lambda_i,num_samples=N,replacement=True,)
            
            delta_t = torch.clamp(delta_t[sel_ind]+sig_dt*torch.randn_like(delta_t),min=dt_min,max=dt_max,)
            noise_L=torch.rand(size=(N,),device=ind_L.device)
            
            ind_L= torch.clamp(ind_L[sel_ind]+((noise_L>=(2/3)).float()-(noise_L<=1/3).float())*ones_L,min=L_min,max=L+1e-3)
            
        if skip_mh:
            q=q_trial
            nb_accept=N
        else:
            alpha=torch.rand(size=(N,),device=device)
            accept=torch.exp(delta_H)>alpha
            nb_accept=accept.sum().item()
            acc+=nb_accept
            q=torch.where(accept.unsqueeze(-1),input=q_trial,other=q)
        if nb_accept>0:
            o_new=q+q**2
            mu_new,sig_new=o_new.mean(0),o_new.std(0)
            correl=((o_new-mu_new)*(o_old-mu_old)).mean(0)/(sig_old*sig_new)
            prod_correl*=correl
    
        p = torch.randn_like(q)
        if scale_M is not None:
            p = sqrt_M*p
        H_old= Hamiltonian(X=q, p=p,V=V,beta=beta,gaussian=gaussian,scale_M=scale_M)
        #no new potential is computed 
        if save_H:
            H_[i+1] = H_old.mean()
        
        

        if save_func is not None:
            saved.append(save_func(q,p))
        i+=1
    if verbose:
        print(f"T_final={i}")
    dict_out={'acc_rate':acc/(i*N)}

    if save_H:
        dict_out['H']=H_
    if save_func is not None:
        dict_out['saved']=saved
    if FT:
        dict_out['dt']=delta_t
        dict_out['ind_L']=ind_L
    v_q=V(q)
    #no new potential computation 
    
    return q,v_q,nb_calls,dict_out


def Hamiltonian2(X,p,beta,v_x=None,V=None,scale_M=1,gaussian=True):
    if v_x is None:
        assert V is not None
        with torch.no_grad():
            U = beta*V(X) +0.5*torch.sum(X**2,dim=1) if gaussian else beta*V(X)
    else:
        U = beta*v_x +0.5*torch.sum(X**2,dim=1) if gaussian else beta*v_x
    H = U + 0.5*torch.sum((1/scale_M)*p**2,dim=1)
    return H



def adapt_verlet_mcmc2(q,v_q,ind_L,beta:float,gaussian:bool,V,gradV,T:int,delta_t,L:int=1,
kappa_opt:bool=True,save_H=True,save_func=None,device='cpu',scale_M=None,alpha_p:float=0.1,prop_d=0.1,FT=False,dt_max=None,dt_min=None,sig_dt=0.015,
verbose=0,L_min=1,gaussian_verlet=False,skip_mh=False):
    """ Simple implementation of Hamiltonian dynanimcs MCMC 

    Args:
        q (_type_): _description_
        p (_type_): _description_
        beta (_type_): _description_
        gaussian (_type_): _description_
        V (_type_): _description_
        gradV (_type_): _description_
        T (_type_): _description_
        delta_t (_type_): _description_
        kappa_opt (bool, optional): _description_. Defaults to True.
        save_H (bool, optional): _description_. Defaults to True.
        save_Q (bool, optional): _description_. Defaults to True.
        save_func (_type_, optional): _description_. Defaults to None.
        alpha_p 

    Returns:
        _type_: _description_
    """
    acc  = 0
    T_max=T
    if scale_M is None:
        scale_M=1
        sqrt_M=1
    else:
        sqrt_M = torch.sqrt(scale_M) 
    p=sqrt_M*torch.randn_like(q)
    if FT:
        
        maha_dist = lambda x,y: ((1/scale_M)*(x-y)**2).sum(1)
        ones_L=torch.ones_like(ind_L)
    (N,d)=q.shape
    o_old=q+q**2
    mu_old,sig_old=(o_old).mean(0),(o_old).std(0)
    
    U = beta*v_q +0.5*torch.sum(q**2,dim=1)
    H_old=U+ torch.sum((1/scale_M)*p**2,dim=1)
    nb_calls=0 #we reuse the potential value v_q of previous iteration
    if save_H:
        H_ = np.zeros(T+1)
        H_[0]=H_old.mean()

    if save_func is not None:
        saved=[save_func(q,p)]
    prod_correl=torch.ones(size=(d,),device=q.device)
    i=0
    while (prod_correl>alpha_p).sum()>=prop_d*d and i<T_max:
        q_trial,p_trial=verlet_kernel1(X=q,gradV=gradV, p_0=p,delta_t=delta_t,beta=beta,L=L,kappa_opt=kappa_opt,
        scale_M=scale_M, ind_L=ind_L,GV=gaussian_verlet)
        # for each particle each vertlet integration step requires two gradient computations
        nb_calls+=4*(ind_L).sum().item()-N 
        v_trial=V(q_trial)
        nb_calls+=N             # we compute the potential of the new particle                        
        H_trial= Hamiltonian2(X=q_trial, p=p_trial,v_x=v_trial,beta=beta,gaussian=gaussian,scale_M=scale_M)

        delta_H=torch.clamp(-(H_trial-H_old),max=0)          
        if FT:
            exp_weight=torch.exp(torch.clamp(delta_H,max=0))
            m_distances= maha_dist(x=q,y=q_trial)/ind_L.float().to(device)
            lambda_i=m_distances*exp_weight+1e-8*torch.ones_like(m_distances)
            if lambda_i.isnan().any():
                print(f"NaN values in lambda_i:")
                print(lambda_i)
            elif (lambda_i==0).sum()>0:
                print(f"zero values in lambda_i")
                print(f"lambda_i")
                print(f"exp_weight:{exp_weight}")
                print(f"m_distances:{m_distances}")
                if (m_distances==0).sum()>0:
                    print(f"q_trial:{q_trial}")
                    print(f"q:{q}")
                print(f"ind_L:{ind_L}")
            
            sel_ind=torch.multinomial(input=lambda_i,num_samples=N,replacement=True,)
            
            delta_t = torch.clamp(delta_t[sel_ind]+sig_dt*torch.randn_like(delta_t),min=dt_min,max=dt_max,)
            noise_L=torch.rand(size=(N,),device=ind_L.device)
            
            ind_L= torch.clamp(ind_L[sel_ind]+((noise_L>=(2/3)).float()-(noise_L<=1/3).float())*ones_L,min=L_min,max=L+1e-3)
            
        alpha=torch.rand(size=(N,),device=device)
        accept=torch.exp(delta_H)>alpha
        nb_accept=accept.sum().item()
        acc+=nb_accept
        if skip_mh:
            q=q_trial
            v_q=v_trial
            nb_accept=N
        else:
            q=torch.where(accept.unsqueeze(-1),input=q_trial,other=q)
            v_q=torch.where(accept,input=v_trial,other=v_q)
        if nb_accept>0:
            o_new=q+q**2
            mu_new,sig_new=o_new.mean(0),o_new.std(0)
            correl=((o_new-mu_new)*(o_old-mu_old)).mean(0)/(sig_old*sig_new)
            prod_correl*=correl
    
        
        


        p = torch.randn_like(q)
        if scale_M is not None:
            p = sqrt_M*p
        H_old= Hamiltonian2(X=q, p=p,V=V,beta=beta,gaussian=gaussian,scale_M=scale_M)
        
        if save_H:
            H_[i+1] = H_old.mean()
        

        if save_func is not None:
            saved.append(save_func(q,p))
        i+=1
    if verbose:
        print(f"T_final={i}")
    dict_out={'acc_rate':acc/(i*N),'T_final':i}

    if save_H:
        dict_out['H']=H_
    if save_func is not None:
        dict_out['saved']=saved
    if FT:
        dict_out['dt']=delta_t
        dict_out['ind_L']=ind_L
    
    
    return q,v_q,nb_calls,dict_out


