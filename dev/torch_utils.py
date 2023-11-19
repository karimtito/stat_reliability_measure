import torch
import torch.nn as nn
from stat_reliability_measure.dev.utils import float_to_file_float
from stat_reliability_measure.dev.torch_arch import CNN_custom,dnn2,dnn4,LeNet,ConvNet,DenseNet3
from torchvision import transforms,datasets,models as tv_models
from torch.utils.data import DataLoader
import scipy.stats as stat
from stat_reliability_measure.home import ROOT_DIR
from torch import optim
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
def norm_batch_tensor(x,d):
    y = x.reshape(x.shape[:1]+(d,))
    return y.norm(dim=-1)

def TimeStepPyt(v,grad_v,p=1,p_p=2):
    """computes the initial time step for the langevin/hmc kernel"""
    V_mean= v.mean()
    V_grad_norm_mean = ((torch.norm(grad_v,dim = 1,p=p_p)**p).mean())**(1/p)
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

def quantize_img(image_tensor, num_bits=8,x_min=0.,x_max=1.,rescale =True,
    rebias=True):
    """Quantizes an image tensor with given bit depth."""
    image_tensor= (image_tensor-x_min)/(x_max-x_min)
    image_tensor = image_tensor.mul(num_bits).clamp(0, num_bits).round()
    if num_bits < 8:
        image_tensor = torch.floor_divide(image_tensor, 2**(8 - num_bits))
    if rescale:
        image_tensor = image_tensor.div(2**num_bits - 1)
        if rebias:
            image_tensor = image_tensor.mul(x_max - x_min).add(x_min)
    return image_tensor

def score_function(X,y_clean,model):
    y = model(X)
    y_diff = torch.cat((y[:,:y_clean], y[:,(y_clean+1):]),dim=1) - y[:,y_clean].unsqueeze(-1)
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


def verlet_kernel(X, gradV, delta_t, beta,L,v_q=None,grad_v_q=None,
ind_L=None,p_0=None,lambda_=0, kappa_opt=False,
gaussian=True,scale_M=None,GV=False):
    """ HMC (L>1) / Underdamped-Langevin (L=1) kernel with Verlet integration (a.k.a. Leapfrog scheme)

    """
    grad_calls=0
    if ind_L is None:
        ind_L=L*torch.ones(size=(X.shape[0],),dtype=torch.int16)
    #assert ind_L.shape[0]==X.shape[0]
    assert torch.min(ind_L).item()>=1 # all samples are in the first iteration of the Verlet scheme
    q_t = torch.clone(X)
    # if no initial momentum is given we draw it randomly from gaussian distribution
    if kappa_opt:
        kappa= 2. / (1 + (1 - delta_t**2)**(1/2))
    #scale_M=torch.eye(n=1,device=X.device)
    if p_0 is not None:
        p_t=torch.randn_like(X).detach()

   
    
    grad_q=lambda p,dt:dt*(p.data)
    
    if not GV and (grad_v_q is None or v_q is None):
        
        grad_v_q,v_q=gradV(q_t)

        grad_calls+=X.shape[0]
    
    k=1
    i_k=ind_L>=k
    assert i_k.sum().item() == X.shape[0] # all samples are in the first iteration of the Verlet scheme
    while i_k.sum().item()>0:
        #I. Verlet scheme
        #computing half-point momentum
        #p_t.data = p_t-0.5*dt*gradV(X).detach() / norms(gradV(X).detach())
        if not GV:
            p_t.data[i_k] = p_t[i_k]-0.5*beta*delta_t[i_k]*grad_v_q[i_k]
        
        
        p_t.data[i_k] = p_t[i_k]- 0.5*delta_t[i_k]*q_t.data[i_k]
        if p_t.isnan().any():
            print("p_t",p_t)
        #updating position
        q_t.data[i_k] = (q_t[i_k] + grad_q(p_t[i_k],delta_t[i_k]))
        #assert not q_t.isnan().any(),"Nan values detected in q"
        #updating momentum again
       
        if not GV:
            grad_v_q.data[i_k],v_q.data[i_k]=gradV(q_t[i_k])
            grad_calls+=i_k.sum().item()
            p_t.data[i_k] = p_t[i_k] -0.5*beta*delta_t[i_k]*grad_v_q[i_k]
        p_t.data[i_k] =p_t[i_k] -0.5*delta_t[i_k]*q_t.data[i_k]
        #II. Optional smoothing of momentum memory
        p_t.data[i_k] = torch.exp(-lambda_*delta_t[i_k])*p_t[i_k]

        k+=1
        i_k=ind_L>=k
    return q_t.detach(),p_t.detach(),grad_calls,grad_v_q,v_q

def verlet_kernel_kappa_opt(X, gradV, delta_t, beta,L, grad_v_q=None,v_q=None,
    ind_L=None,p_0=None,lambda_=0, 
    scale_M=None,GV=False):
    """ HMC (L>1) / Underdamped-Langevin (L=1) kernel with Verlet integration (a.k.a. Leapfrog scheme)

    """
    grad_calls = 0
    if ind_L is None:
        ind_L=L*torch.ones(size=(X.shape[0],),dtype=torch.int16)
    q_t = torch.clone(X)
    # if no initial momentum is given we draw it randomly from gaussian distribution
   
    scale_M=torch.Tensor([1])
    if p_0 is None:                        
        p_t=torch.sqrt(scale_M)*torch.randn_like(X) 

    else:
        p_t = p_0
    grad_q=lambda p,dt:dt*(p.data)
    
    kappa= 2. / (1 + (1 - delta_t**2)**(1/2))
    
    
    if not GV and (grad_v_q is None or v_q is None):
        grad_v_q, v_q = gradV(q_t)
        grad_calls += X.shape[0]
    k=1
    i_k=ind_L>=k
    while (i_k).sum().item()>0:
        #I. Verlet scheme
        #computing half-point momentum
        #p_t.data = p_t-0.5*dt*gradV(X).detach() / norms(gradV(X).detach())
        if not GV:
            p_t.data[i_k] = p_t[i_k]-0.5*beta*delta_t[i_k]*kappa[i_k]*grad_v_q[i_k]
        
        p_t.data[i_k] -= 0.5*delta_t[i_k]*kappa[i_k]*q_t.data[i_k]
        #updating position
        
        q_t.data[i_k] = (q_t[i_k] + grad_q(p_t[i_k],delta_t[i_k]))
        #updating momentum again
        if not GV:
            grad_v_q.data[i_k],v_q.data[i_k]=gradV(q_t[i_k])
            grad_calls+=i_k.sum().item()
            p_t.data[i_k] = p_t[i_k] -0.5*beta*kappa[i_k]*delta_t[i_k]*grad_v_q[i_k]
        p_t.data[i_k] -= 0.5*kappa[i_k]*delta_t[i_k]*q_t.data[i_k]
        #II. Optional smoothing of momentum memory
        p_t.data[i_k] = torch.exp(-lambda_*delta_t[i_k])*p_t[i_k]
        k+=1
        i_k=ind_L>=k
    return q_t.detach(),p_t,grad_calls,grad_v_q,v_q




def multi_unsqueeze(input_,k,dim=-1):
    """"unsqueeze input multiple times"""
    for _ in range(k):
        input_=input_.unsqueeze(dim=dim)
    return input_




def compute_V_grad_pyt(model, input_, target_class,L=0):
    """ Returns potentials and associated gradients for given inputs, model and target classes """
    if input_.requires_grad!=True:
        input_.requires_grad=True
    #input_.retain_grad()
    s=score_function(X=input_,y_clean=target_class,model=model)
    v=torch.clamp(L-s,min=0)
    # print(f"v={v},input_={input_}")
    # print(f"v.shape={v.shape},input_.shape={input_.shape}")
    # print(f"v.requires_grad={v.requires_grad},input_.requires_grad={input_.requires_grad}")
    # print(f"v.grad={v.grad},input_.grad={input_.grad}")
    grad=torch.autograd.grad(outputs=v,inputs=input_,grad_outputs=torch.ones_like(v),retain_graph=False)[0]
    return v.detach(),grad.detach()

def compute_h_grad_pyt(model, input_, target_class,L=0):
    """ Returns potentials and associated gradients for given inputs, model and target classes """
    if input_.requires_grad!=True:
        input_.requires_grad=True
    #input_.retain_grad()
    s=score_function(X=input_,y_clean=target_class,model=model)
    grad=torch.autograd.grad(outputs=s,inputs=input_,grad_outputs=torch.ones_like(s),retain_graph=False)[0]
    return s.detach(),grad.detach()

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
        s=score_function(X=input_, y_clean=target_class, model=model)
    v=torch.clip(L-s,min=0)
    return v

def compute_h_pyt(model, input_, target_class,L=0):
    """Return potentials for given inputs, model and target classes"""
    with torch.no_grad():
        s=score_function(X=input_, y_clean=target_class, model=model)
        
    return s
normal_dist=torch.distributions.Normal(loc=0, scale=1.)
norm_dist_pyt = torch.distributions.Normal

def h_pyt(x_,x_clean,model,target_class,low,high,from_gaussian=True,reshape=True,input_shape=None,
           noise_dist='gaussian',
          noise_scale=1.0,):
    gaussian_prior=noise_dist=='gaussian' or noise_dist=='normal'
    with torch.no_grad():
        if input_shape is None:
            input_shape=x_clean.shape
        if from_gaussian and not gaussian_prior:
            u=normal_dist.cdf(x_)
        else:
            u=noise_scale*x_+x_clean
        if reshape:
            u=torch.reshape(u,(u.shape[0],)+input_shape)

        
        x_p = u if gaussian_prior else low+(high-low)*u 
        if gaussian_prior:
            x_p=torch.maximum(x_p,low.view(low.size()+(1,1)))
            x_p=torch.minimum(x_p,high.view(high.size()+(1,1)))
    h = compute_h_pyt(model=model,input_=x_p,target_class=target_class)
    return h

def gradh_pyt(x_,x_clean,model,target_class,low,high,from_gaussian=True,reshape=True,input_shape=None, noise_dist='gaussian',
              noise_scale=1.0,):
    gaussian_prior=noise_dist=='gaussian' or noise_dist=='normal'
    if input_shape is None:
        input_shape=x_clean.shape
    if from_gaussian and not gaussian_prior:
        u=normal_dist.cdf(x_)
    else:
        u=x_
    if reshape:
        u=torch.reshape(u,(u.shape[0],)+input_shape)
    
    x_p = u if gaussian_prior else low+(high-low)*u 
    if gaussian_prior:
        x_p=torch.maximum(x_p,low.view(low.size()+(1,1)))
        x_p=torch.minimum(x_p,high.view(high.size()+(1,1)))
    h,grad_x_p = compute_h_grad_pyt(model=model,input_=x_p,target_class=target_class)
    grad_u=torch.reshape(grad_x_p,x_.shape) if gaussian_prior else torch.reshape((high-low)*grad_x_p,x_.shape)
    with torch.no_grad():
        if from_gaussian and not gaussian_prior:
            grad_x=torch.exp(normal_dist.log_prob(x_))*grad_u
        else:
            grad_x=grad_u*noise_scale
    return grad_x.detach(),h.detach()

normal_dist=torch.distributions.Normal(loc=0, scale=1.)
def gaussian_to_image(gaussian_sample,low,high,input_shape=(28,28)):
    image = low+(high-low)*normal_dist.cdf(gaussian_sample).view(-1,*input_shape)
    return image

def V_pyt(x_,x_clean,model,target_class,low,high,from_gaussian=True,
          reshape=True,input_shape=None,
           noise_dist='gaussian',
          noise_scale=1.0,):
    
    gaussian_prior = noise_dist=='gaussian' or noise_dist=='normal'
    with torch.no_grad():
        if input_shape is None:
            input_shape=x_clean.shape
        if from_gaussian and not gaussian_prior:
            u=normal_dist.cdf(x_)
        else:
            u=noise_scale*x_+x_clean
        if reshape:
            u=torch.reshape(u,(u.shape[0],)+input_shape)

        
        x_p = u if gaussian_prior else low+(high-low)*u 
        if gaussian_prior:
            x_p=torch.maximum(x_p,low.view(low.size()+(1,1)))
            x_p=torch.minimum(x_p,high.view(high.size()+(1,1)))
    v = compute_V_pyt(model=model,input_=x_p,target_class=target_class)
    return v

def gradV_pyt(x_,x_clean,model,target_class,low,high,from_gaussian=True,reshape=True,input_shape=None, noise_dist='gaussian',
              noise_scale=1.0,):
    gaussian_prior = noise_dist=='gaussian' or noise_dist=='normal'
    if input_shape is None:
        input_shape=x_clean.shape
    if from_gaussian and not gaussian_prior:
        u=normal_dist.cdf(x_)
    else:
        u=x_
    if reshape:
        u=torch.reshape(u,(u.shape[0],)+input_shape)
    
    x_p = u if gaussian_prior else low+(high-low)*u 
    if gaussian_prior:
        x_p=torch.maximum(x_p,low.view(low.size()+(1,1)))
        x_p=torch.minimum(x_p,high.view(high.size()+(1,1)))
    v,grad_x_p = compute_V_grad_pyt(model=model,input_=x_p,target_class=target_class)
    grad_u=torch.reshape(grad_x_p,x_.shape) if gaussian_prior else torch.reshape((high-low)*grad_x_p,x_.shape)
    with torch.no_grad():
        if from_gaussian and not gaussian_prior:
            grad_x=torch.exp(normal_dist.log_prob(x_))*grad_u
        else:
            grad_x=grad_u*noise_scale
    return grad_x.detach(),v.detach()

def correct_min_max(x_min,x_max,x_mean,x_std):
    if x_min is None or x_max is None: 
        if x_min is None:
            x_min=0
        if x_max is None:
            x_max=1
    if x_mean is not None and x_std is not None:
        x_min=(x_min-x_mean)/x_std
        x_max=(x_max-x_mean)/x_std
    return x_min,x_max

supported_datasets=['mnist','fashion-mnist','cifar10','cifar100','imagenet']
datasets_in_shape={'mnist':(1,28,28),'cifar10':(3,32,32),'cifar100':(3,32,32),'imagenet':(3,224,224)}
datasets_dims={'mnist':784,'cifar10':3*1024,'cifar100':3*1024,'imagenet':3*224**2}
datasets_num_c={'mnist':10,'cifar10':10,'imagenet':1000}
datasets_means={'mnist':0,'cifar10':(0.4914, 0.4822, 0.4465),'cifar100':[125.3/255.0, 123.0/255.0, 113.9/255.0], 
                'imagenet':(0.485, 0.456, 0.406)}

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
datasets_stds={'mnist':1,'cifar10':(0.2023, 0.1994, 0.2010),'cifar100':[63.0/255.0, 62.1/255.0, 66.7/255.0],
               'imagenet':(0.229, 0.224, 0.225)}
datasets_supp_archs={'mnist':{'dnn2':dnn2,'dnn_2':dnn2,'dnn_4':dnn4,'dnn4':dnn4,'cnn_custom':CNN_custom},
                    'cifar10':{'lenet':LeNet,'convnet':ConvNet,'dnn2':dnn2},
                    'cifar100':{'densenet':DenseNet3}}
datasets_default_arch={'mnist':'dnn2', 'cifar10':'convnet', 'cifar100':'densenet','imagenet':'resnet18'}
defaults_datasets=['mnist','cifar10','cifar100','imagenet']
def get_loader(train,data_dir,download,dataset='mnist',batch_size=100,x_mean=None,x_std=None,shuffle=False): 
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

        data_loader = DataLoader(imagenet_dataset, batch_size =batch_size, shuffle=shuffle)
                         
    return data_loader

def plot_tensor(x,cmap='gray'):
    if 'cuda' in str(x.device):
        x=x.cpu()
    if len(x.shape)==4:
        x=x.squeeze(0)
    if len(x.shape)==3:
        x=x.permute(1,2,0)
    plt.imshow(x.detach(),cmap=cmap)
    plt.show()

def plot_k_tensor(X,figsize=(10,10),x_0=None,img_size=(28,28)):
    """plots k tensors representing images in a row of 4 columns"""

    k = X.shape[0]
    nrows = k // 4 if k%4==0 else k // 4 +1
    if x_0 is not None:
        plt.imshow(x_0.view(*img_size).detach().cpu().numpy(),cmap='gray')
    plt.figure(figsize=(figsize[0]*k,figsize[1]))
    for i in range(k):
        plt.subplot(nrows,4,i+1)
        plt.imshow(X[i].view(*img_size).detach().cpu().numpy(),cmap='gray')


def gaussian_to_image(gaussian_sample,normal_layer,image_shape=(28,28)):
    image = normal_layer(gaussian_sample.view(-1,*image_shape))
    return image

def build_h_cos(p_target, device, dim=2,verbose=0):
    """ builds a cosinus score function with parameters p_target"""
    c = stat.f.isf(p_target, 1, dim-1)
    q = np.sqrt(c/(dim-1+c))
    h = lambda X: X[:,0] - q*X.norm(dim=1)
    return h,q

def build_gradh_cos(p_target, device, dim=2,verbose=0):
    """ builds gradient function of a cosinus score with parameters p_target"""
    c = stat.f.isf(p_target, 1, dim-1)
    
    q = np.sqrt(c/(dim-1+c))
    e_1 = torch.Tensor([1.]+[0]*(dim-1)).to(device)
    def gradh(X):
        h_ = X[:,0] - q*X.norm(dim=1)
        grad_value=e_1.unsqueeze(0)-q*X/X.norm(dim=1)[:,None]
        return grad_value.detach(),h_.detach()
    return gradh,q

def build_V_cos(p_target, device, dim=2,verbose=0):
    """ builds a cosinus potential function with parameters p_target and dim"""
    c = stat.f.isf(p_target, 1, dim-1)
    q = np.sqrt(c/(dim-1+c))
    def V(X):
        with torch.no_grad():
            v = torch.clamp(-X[:,0] + q*X.norm(dim=1),min=0,max=None)
        return v
    return V,q

def build_gradV_cos(p_target, device, dim=2, verbose=0):
    assert dim==2
    c = stat.f.isf(p_target, 1, dim-1)
    q = np.sqrt(c/(dim-1+c))
    e_1 = torch.Tensor([1.]+[0]*(dim-1)).to(device)
    def gradV(X):
        v = torch.clamp(-X[:,0] + q*X.norm(dim=1),min=0,max=None)
        grad_value=-e_1.unsqueeze(0)+q*X/X.norm(dim=1)[:,None]
        grad_cond = v>0
        return (grad_value*grad_cond[:,None]).detach(),v.detach()
    return gradV,q

def sig(v,r0,a=.1,dv=0.4):
    """ Generic sigmoid function 
    a: min,  b: mav[0], r_0**2: critical value, dv: steepness
    v: input value
    """
    return a+2*(1-a)/(1.0+torch.exp( (r0**2-v)/dv )) 

def grad_sig(v,r0,a=.1,dv=0.4):
    b=2.-a
    return (b-a)*torch.exp((r0**2-v)/dv)/(dv*(1+torch.exp((r0**2-v)/dv))**2)
def sigma(x):
    return 1/(1+torch.exp(-x))

def grad_sigma(x):
    return torch.exp(-x)/(1+torch.exp(-x))**2

f = lambda v,r0,dim: (v[0]**2 + sig(v[0]**2+v[1]**2,r0) * v[1]**2)/(dim-1)
def plot_f(f,r0,dim=2):
    l = 1.4*r0
    x = torch.linspace(-l, l,1000)
    y = torch.linspace(-l, l,1000)
    xx, yy = torch.meshgrid(x, y)
    zz = f([xx,yy],r0,dim=dim)
    max_level=zz.max()
    min_level=zz.min()
    contour_levels = list(np.linspace(5,max_level*1.1,15))
    contour_levels.append(f(torch.Tensor([r0,0.]),r0,dim=dim).numpy())
    contour_levels.sort()
    contour_levels = np.array(contour_levels)
    plt.figure(figsize=(6,6))
    contours = plt.contour(xx,yy,zz,levels=contour_levels,cmap='viridis')
    plt.clabel(contours, inline=True, fontsize=7)
    plt.axis('scaled')
    plt.colorbar(contours, shrink=0.8, extend='both')

f_alt = lambda v,r0,dim: (v[0]**2+ sig(v[0]**2+v[1]**2,r0) * v[1]**2)/(dim-1)

def plot_f_alt(f,r0,dim, min_level=1.,axis_mode = 'scaled',thresh_mult=1.5):
    l = thresh_mult*r0/np.sqrt(dim-1)
    x_md_norm = torch.linspace(0, l,1000)
    x_d_norm = torch.linspace(0, l,1000) 
    xx,yy = torch.meshgrid(x_md_norm,x_d_norm)
    zz = f([np.sqrt(dim-1)*xx,yy],r0,dim=dim)  
    max_level=zz.max()
    contour_levels = list(np.linspace(min_level,max_level*1.,15))
    contour_levels.append(f_alt(torch.Tensor([0,r0]),r0,dim=dim).numpy())
    contour_levels.sort()
    contour_levels = np.array(contour_levels)
    plt.figure(figsize=(6,6))
    contours = plt.contour(xx,yy,zz,levels=contour_levels,cmap='viridis')
    plt.clabel(contours, inline=True, fontsize=7,fmt='%1.1f')
    plt.xlabel('|X_{-d}|/sqrt(d-1)')
    plt.ylabel('|X_d|')
    plt.axis(axis_mode)
    #plot the critical value level
    x_critical = np.linspace(0, r0/np.sqrt(dim-1),1000)
    y_critical = np.sqrt((r0**2-(dim-1)*x_critical**2))
    plt.plot(x_critical,y_critical,'r--',label='critical value')
    plt.legend()
    plt.colorbar(contours, shrink=0.8, extend='both')









def build_h_quad(p_target, device, dim=2, verbose=0):
    """ builds a sigmoid score function with parameters r0,a,dv"""
    r0 = np.sqrt(stat.chi2.isf(p_target, dim))
    h = lambda X: X[:,:dim-1].square().sum(1) +sig(X.square().sum(1),r0)*X[:,dim-1]**2-r0**2
    return h,r0

def build_gradh_quad(p_target, device, dim=2, dv=0.4,a=.1, verbose=0):
    """ builds the gradient of a sigmoid score function with parameters r0,a,dv"""
    r0 = np.sqrt(stat.chi2.isf(p_target, dim))
    def gradh(X):
        h_ = X[:,:dim-1].square().sum(1) +sig(X.square().sum(1),r0)*X[:,dim-1]**2-r0**2
        X_1,X_d = torch.zeros_like(X), torch.zeros_like(X)
        X_1[:,:dim-1] = X[:,:dim-1]
        X_1[:,dim-1] = 0.
        X_d[:,:dim-1] = 0.
        X_d[:,dim-1] = X[:,dim-1]
        grad_h = 2*X_1
        grad_h += 2*a*X_d
        grad_h += 2*(1-a)*(2/dv)*(X[:,dim-1]**2).unsqueeze(-1)*grad_sigma((X.square().sum(1)-r0**2)/dv)[:,None]*X
        grad_h += 4*(1-a)*sigma((X.square().sum(1)-r0**2)/dv).unsqueeze(-1)*X_d
        del X_1,X_d
        return grad_h.detach(),h_.detach()
    return gradh

def build_V_quad(p_target, device,dim=2 ,verbose=0,a=.1,dv=0.4):
    """ builds a sigmoid potential with parameters r0,a,dv"""
    r0 = np.sqrt(stat.chi2.isf(p_target, dim))
    V = lambda X: torch.clip(r0**2-X[:,:dim-1].square().sum(1)-sig(X.square().sum(1),r0)*X[:,dim-1]**2, min=0)
    return V,r0

def build_gradV_quad(p_target,device, verbose=0,dim=2,a=.1,dv=0.4):
    """ builds the gradient of a sigmoid potential with parameters r0,a,dv"""
    r0 = np.sqrt(stat.chi2.isf(p_target, dim))
    def gradV(X):
        X_1,X_d = torch.zeros_like(X), torch.zeros_like(X)
        X_1[:,:dim-1] = X[:,:dim-1]
        X_1[:,dim-1] = 0.
        X_d[:,:dim-1] = 0.
        X_d[:,dim-1] = X[:,dim-1]
        v = torch.clip(r0**2-X_1.square().sum(1)-sig(X.square().sum(1),r0)*X[:,dim-1]**2, min=0)
        grad_h = 2*X_1
        grad_h += 2*a*X_d
        grad_h += 2*(1-a)*(2/dv)*(X[:,dim-1]**2).unsqueeze(-1)*grad_sigma((X.square().sum(1)-r0**2)/dv)[:,None]*X
        grad_h += 4*(1-a)*sigma((X.square().sum(1)-r0**2)/dv).unsqueeze(-1)*X_d
        del X_1,X_d
        grad_v = -grad_h*(X.square().sum(1)>r0**2)[:,None]
        return grad_v.detach(),v.detach()
    return gradV,r0

def build_h_lin(p_target,device, dim=2, verbose=0):
    """ builds q linear score function with parameters p_target"""
    c = stat.norm.isf(p_target)
    def h(X):
        with torch.no_grad():
            return X[:,0]-c
    return h,c

def build_gradh_lin(p_target,device, dim=2, verbose=0):
    """ builds the gradient of a linear score function with parameters p_target"""
    c = stat.norm.isf(p_target)
    e_1 = torch.Tensor([1.]+[0]*(dim-1)).to(device)
    def gradh(X):
        gradh = (e_1.unsqueeze(0)*torch.ones_like(X))
        h_ = X[:,0]-c
        return (e_1.unsqueeze(0)*torch.ones_like(X)).detach(),h_.detach()
    return gradh,c

def build_V_lin(p_target, device,dim=2, verbose=0):
    c = stat.norm.isf(p_target)
    def V(X):
        with torch.no_grad():
            return torch.clamp(input=c-X[:,0], min=0, max=None)
    if verbose:
        print("V_lin built")
        print("c=",c)
        P_target=stat.norm.sf(c)
        print("P_target=",P_target)
    return V,c

def build_gradV_lin(p_target,device, verbose=0,dim=2,):
    c = stat.norm.isf(p_target)
    e_1 = torch.Tensor([1.]+[0]*(dim-1)).to(device)
    def gradV(X):
        v = torch.clamp(input=c-X[:,0], min=0, max=None)
        gradv= (-e_1*(v>0)[:,None])
        return gradv.detach(),v.detach()
    return gradV,c




def get_correct_x_y(data_loader,device,model):
    for X,y in data_loader:
        X,y = X.to(device), y.to(device)
        break
    with torch.no_grad():
        logits=model(X)
        y_pred= torch.argmax(logits,-1)
        correct_idx=y_pred==y
    return X[correct_idx],y[correct_idx],correct_idx.float().mean().item()

def get_x_y_accuracy_num_cl(X,y,model):
    with torch.no_grad():
        logits=model(X)
        num_classes=logits.shape[-1]
        y_pred= torch.argmax(logits,-1)
        correct_idx=y_pred==y
    return X[correct_idx],y[correct_idx],correct_idx.float().mean(),num_classes

def get_model_accuracy(X,y,model):
    """return model accuracy on supervised data (X,y)"""
    with torch.no_grad():
        logits=model(X)
        y_pred= torch.argmax(logits,-1)
        correct_idx=y_pred==y
    return correct_idx.float().mean()



supported_arch={'cnn_custom':CNN_custom,'dnn2':dnn2,'dnn4':dnn4,}
def get_model_imagenet(model_arch,model_dir):
    torch.hub.set_dir(model_dir)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalizer = transforms.Normalize(mean=mean, std=std)
    if model_arch.lower().startswith("torchvision"):
        model = getattr(tv_models, model_arch[len("torchvision_"):])(weights="IMAGENET1K_V2")
    else:
        import timm
        model = timm.create_model(model_arch, pretrained=True)
        mean = model.default_cfg["mean"]
        std = model.default_cfg["std"]
        normalizer = transforms.Normalize(mean=mean, std=std)

    model = torch.nn.Sequential(normalizer, model).cuda(0).eval()
    return model,mean,std

def get_model(model_arch, test_loader, device=None , robust_model=False, robust_eps=0.1,nb_epochs=10,model_dir='./',data_dir='./',
download=True,force_train=False,dataset='mnist',batch_size=100,lr=1E-1):
    if device is None:
        device=device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
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
        opt = optim.SGD(model.parameters(), lr=lr)

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
    return (x + s*torch.randn_like(x))/torch.math.sqrt(1+s**2)

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

def Hmlt(q,p,v_q,beta):
    u = beta*v_q + 0.5*q.square().sum(dim=1)
    return u + 0.5*p.square().sum(dim=1)


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



def verlet_mcmc(q,v_q,beta:float,gaussian:bool,V,gradV,T:int,delta_t,
                grad_v_q=None,L:int=1,
        kappa_opt:bool=True,save_H=False,save_func=None,device='cpu',scale_M=None,
gaussian_verlet=False,ind_L=None,mult_grad_calls=2.):
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
    
    (N,d)=q.shape
    H_old = Hmlt(q=q,p=p,v_q=v_q,beta=beta)
    nb_calls=N
    if save_H:
        H_ = np.zeros(T+1)
        H_[0]=H_old.mean()

    if save_func is not None:
        saved=[save_func(q,p)]
    for i  in range(T):
        q_trial,p_trial,grad_calls,grad_v_q_trial,v_q_trial=verlet_kernel(X=q,gradV=gradV, p_0=p,delta_t=delta_t,beta=beta,L=L,kappa_opt=kappa_opt,
        scale_M=scale_M,ind_L=ind_L,GV=gaussian_verlet)
        nb_calls+=mult_grad_calls*grad_calls # we count each gradient call as 2 function calls
        H_trial= Hmlt(q=q_trial,p=p_trial,v_q=v_q_trial,beta=beta)
        nb_calls+=N
        alpha=torch.rand(size=(N,),device=device )
        delta_H=torch.clamp(-(H_trial-H_old),max=0)
        accept=torch.exp(delta_H)>alpha
        acc+=accept.sum()
        q=torch.where(accept.unsqueeze(-1),input=q_trial,other=q)
        v_q=torch.where(accept,input=v_q_trial,other=v_q)
        if not gaussian_verlet:
            grad_v_q=torch.where(accept.unsqueeze(-1),input=grad_v_q_trial,other=grad_v_q)
        p = torch.randn_like(q)
        
        H_old= Hmlt(q=q,p=p,v_q=v_q,beta=beta)
        if save_H:
            H_[i+1] = H_old.mean()

        if save_func is not None:
            saved.append(save_func(q,p))
    
    dict_out={'acc_rate':acc.cpu()/(T*N)}

    if save_H:
        dict_out['H']=H_
    if save_func is not None:
        dict_out['saved']=saved
    
    return q,v_q,grad_v_q,nb_calls,dict_out


def adapt_verlet_mcmc(q,v_q,ind_L,beta:float,gaussian:bool,V,gradV,T:int,delta_t,
                      grad_v_q=None,L:int=1,
kappa_opt:bool=True,save_H=True,save_func=None,device='cpu',scale_M=None,alpha_p:float=0.1,prop_d=0.1,FT=False,dt_max=None,dt_min=None,sig_dt=0.015,
verbose=0,L_min=1,gaussian_verlet=False,skip_mh=False,corrected_kernel=False,
grad_calls_mult=2.,save_T=True):
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

    verlet_mixer=verlet_kernel_kappa_opt if kappa_opt else verlet_kernel
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
    H_old= Hmlt(q=q, p=p,v_q=v_q,beta=beta)
    nb_calls=0 #we can reuse the potential value v_q of previous iteration: no new potential computation
    if save_H:
        H_ = np.zeros(T+1)
        H_[0]=H_old.mean()
    if FT:
        H_0 = H_old
    if save_func is not None:
        saved=[save_func(q,p)]
    prod_correl=torch.ones(size=(d,),device=q.device)
    i=0
    #torch.multinomial(input=)
    while (prod_correl>alpha_p).sum()>=prop_d*d and i<T_max:
        q_trial,p_trial,grad_calls,grad_v_q_trial,v_q_trial=verlet_mixer(X=q,v_q=v_q,gradV=gradV,grad_v_q=grad_v_q, 
        p_0=p,delta_t=delta_t,beta=beta,
        L=L, ind_L=ind_L,GV=gaussian_verlet)
        
        nb_calls+= grad_calls_mult*grad_calls
        H_trial= Hmlt(q=q_trial, p=p_trial,v_q=v_q_trial,beta=beta)
        nb_calls+=N # N new potentials are computed 
        delta_H=torch.clamp(-(H_trial-H_old),max=0)
        
            
        if skip_mh:
            q=q_trial
            v_q=v_q_trial
            grad_v_q=grad_v_q_trial
            nb_accept=N
        else:
            alpha=torch.rand(size=(N,),device=device)
            accept=torch.exp(delta_H)>alpha
            nb_accept=accept.sum().item()
            acc+=nb_accept 
            q=torch.where(accept.unsqueeze(-1),input=q_trial,other=q)
            v_q = torch.where(accept,input=v_q_trial,other=v_q)
            if not gaussian_verlet:
                grad_v_q=torch.where(accept.unsqueeze(-1),input=grad_v_q_trial,other=grad_v_q)
        
        if nb_accept>0:
            o_new=q+q**2
            mu_new,sig_new=o_new.mean(0),o_new.std(0)
            correl=((o_new-mu_new)*(o_old-mu_old)).mean(0)/(sig_old*sig_new)
            prod_correl*=correl
            mu_old, sig_old, o_old = mu_new, sig_new, o_new
    
        p = torch.randn_like(q)
        H_old= Hmlt(q=q, p=p,v_q=v_q,beta=beta)

        #no new potential is computed 
        if save_H:
            H_[i+1] = H_old.mean()
        if save_func is not None:
            saved.append(save_func(q,p))
        i+=1
    
    if FT:
        delta_H_final = H_old-H_0
        exp_weight=torch.exp(torch.clamp(delta_H_final,max=0))
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
        
        ind_L= torch.clamp(ind_L[sel_ind.to(ind_L.device)]+((noise_L>=(2/3)).float()-(noise_L<=1/3).float())*ones_L,min=L_min,max=L+1e-3)

    if verbose:
        print(f"T_final={i}")
    dict_out={'acc_rate':acc/(i*N)}
    if FT:
        dict_out['dt'] = delta_t
        dict_out['ind_L'] = ind_L
    if save_H:
        dict_out['H']=H_
    if save_func is not None:
        dict_out['saved']=saved
    
        
    
    return q,v_q,grad_v_q,nb_calls,dict_out

def get_imagenet_simple_labels():
    json_path = Path(ROOT_DIR) / "data/ImageNet/imagenet-simple-labels.json"
    with open(json_path, 'r') as f:
        imagenet_dict = json.load(f)
    return imagenet_dict

def get_imagenet_dict():
    json_path = Path(ROOT_DIR) / "data/ImageNet/imagenet-simple-labels.json"
    with open(json_path, 'r') as f:
        imagenet_dict = json.load(f)
    return imagenet_dict

def idx_zero(tensor):
    """ returns idx where tensor is null"""
    return (tensor==0.)
def idx_negative(tensor):
    """ returns idx where tensor is negative"""
    return (tensor<0)

class NormalCDFLayer(torch.nn.Module):
    def __init__(self,device='cpu',mu=0.,sigma=1.,x_clean = 0.,epsilon = 0.1,x_min=0., x_max=1., offset=0.,
                 no_atoms=True):
        super(NormalCDFLayer, self).__init__()
        self.device=device
        self.norm = torch.distributions.Normal(0, 1)
        self.mu = mu
        self.sigma = sigma
        self.x_clean = x_clean
        self.epsilon = epsilon
        self.x_min = x_min
        self.x_max = x_max
        self.offset =offset
        
    def forward(self,x):
        return torch.clip(self.offset+self.epsilon*(2*self.norm.cdf(x)-1),min=self.x_min,max=self.x_max)
    def inverse(self,x):
        return self.norm.icdf(((x-self.offset)/self.epsilon+1.)/2.)
    def string(self):
        return f"NormalCDFLayer(mu={self.mu},sigma={self.sigma})"

class NormalToUnifLayer(torch.nn.Module):
    def __init__(self,device='cpu',input_shape=None,low=0.,high=1.,epsilon = None, x_clean = None,x_min=0., x_max=1.,mu=0.,sigma=1.,
                 ):
        super(NormalToUnifLayer, self).__init__()
        
        self.device=device
        self.norm = torch.distributions.Normal(0, 1)
        
        self.low = low
        self.high = high
        self.epsilon = epsilon
        self.x_clean = x_clean
        self.x_min = x_min
        self.x_max = x_max
        self.mu = mu
        self.sigma = sigma
        if low is None:
            assert self.x_clean is not None, "x_clean must be specified if low is None"
            assert self.epsilon is not None, "epsilon must be specified if low is None"
            assert input_shape is not None or x_clean is not None, "input_shape or x_clean must be specified"
            self.low=torch.maximum(self.x_clean-self.epsilon, self.x_min.view(-1,*self.input_shape).to(self.device))
        if high is None:
            assert self.x_clean is not None, "x_clean must be specified if high is None"
            assert self.epsilon is not None, "epsilon must be specified if low is None"
            assert input_shape is not None or x_clean is not None, "input_shape or x_clean must be specified"
            self.high=torch.minimum(self.x_clean+self.epsilon, self.x_max.view(-1,*self.input_shape).to(self.device))
        self.range=self.high-self.low
    def forward(self,x):
        return self.low + self.range*self.norm.cdf(x)  
    def inverse(self,x):
        return self.norm.icdf(((x-self.low)/self.range))
    def string(self):
        return f"NormalToUnifLayer(mu={self.mu},sigma={self.sigma})"

imagenet_simple_labels = get_imagenet_simple_labels()
def plot_imagenet_pictures(X,y=None, nb_rows = 5, nb_cols = None,text_labels=imagenet_simple_labels):
    """plots the first nb_pictures pictures of the imagenet dataset with subfigures and saves them in the exp_config.exp_log_path + '/imagenet_pictures/' folder"""
    #using suplots to plot in an approximate square shape
     
    nb_rows = nb_rows
    nb_cols = nb_cols if nb_cols is not None else nb_rows
    if nb_rows*nb_cols > X.shape[0]:
        nb_rows = int(np.sqrt(X.shape[0]))
        nb_cols = nb_rows
    nb_pictures = nb_rows*nb_cols
    print(f"Plotting {nb_pictures} ImageNet pictures")
    fig, axs = plt.subplots(nrows=nb_rows,ncols=nb_cols, figsize=(15, 15))
    fig.subplots_adjust(hspace = .5, wspace=.001)
    #making sure that axs is a 1D array
    axs = axs.ravel()
    # plotting the pictures
    for i in range(nb_pictures):
        #permuting the dimensions of the picture to get the right format
        X_numpy = X[i].permute(1,2,0).detach().cpu().numpy()
        axs[i].imshow(X[i].permute(1,2,0).detach().cpu().numpy())
        del X_numpy
        if y is not None:
            numpy_label=y[i].detach().cpu().numpy()
            if text_labels is not None:
                text_label = text_labels[numpy_label]
            else:
                text_label = numpy_label
            axs[i].set_title(f"Label: {text_label}")
        axs[i].set_axis_off()
    # saving the pictures   
    plt.show()
    plt.savefig('imagenet_pictures.png')
    plt.close()


