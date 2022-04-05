from random import gauss
import torch
import torch.nn as nn
from stat_reliability_measure.dev.utils import float_to_file_float
from stat_reliability_measure.dev.torch_arch import CNN_custom,dnn2,dnn4,LeNet
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
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
            X_new =X-delta_t*grad+torch.sqrt(2*delta_t/beta)*G_noise 
        else:
            X_new=(1-delta_t/beta)*X -delta_t*grad+torch.sqrt(2*delta_t/beta)*G_noise # U(x)=beta*V(x)+(1/2)x^T*Sigma*x
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


def verlet_kernel1(X, gradV, delta_t, beta,L,p_0=None,lambda_=0, gaussian=True):
    """ HMC (L>1) / Underdamped-Langevin (L=1) kernel with Verlet integration (a.k.a. Leapfrog scheme)

    """
    q_t = X
    # if no initial momentum is given we draw it randomly from gaussian distribution
    if p_0 is None:                        
        p_t = torch.randn_like(X)
    else:
        p_t = p_0
    

    for _ in range(L):
        #I. Verlet scheme
        #computing half-point momentum
        #p_t.data = p_t-0.5*dt*gradV(X).detach() / norms(gradV(X).detach())
        p_t.data = p_t-0.5*delta_t*beta*gradV(q_t).detach()
        if gaussian:
            p_t.data -= 0.5*delta_t*q_t.data
        #updating position
        q_t.data = (q_t + delta_t*p_t.data)
        #updating momentum again
        p_t.data = p_t -0.5*delta_t*beta*gradV(q_t).detach()
        if gaussian:
            p_t.data -= 0.5*delta_t*q_t.data
        #II. Optional smoothing of momentum memory
        p_t.data = math.exp(-lambda_*delta_t)*p_t


        
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

def V_pyt(x_,x_0,model,target_class,epsilon=0.05,gaussian_latent=True,clipping=True, clip_min=0, clip_max=1,reshape=True,input_shape=None):
    if input_shape is None:
        input_shape=x_0.shape
    if gaussian_latent:
        u=epsilon*(2*normal_dist.cdf(x_)-1)
    else:
        u=x_
    if reshape:
        u=torch.reshape(u,(u.shape[0],)+input_shape)
    x_p = x_0+u if not clipping else torch.clamp(x_0+u,min=clip_min,max=clip_max)
    v = compute_V_pyt(model=model,input_=x_p,target_class=target_class)
    return v

def gradV_pyt(x_,x_0,model,target_class,epsilon=0.05,gaussian_latent=True,clipping=True, clip_min=0, clip_max=1,reshape=True,input_shape=None):
    if input_shape is None:
        input_shape=x_0.shape
    if gaussian_latent:
        u=epsilon*(2*normal_dist.cdf(x_)-1)
    else:
        u=x_
    if reshape:
        u=torch.reshape(u,(u.shape[0],)+input_shape)
    x_p = x_0+u if not clipping else torch.clamp(x_0+u,min=clip_min,max=clip_max)
    _,grad_u = compute_V_grad_pyt(model=model,input_=x_p,target_class=target_class)
    grad_u=torch.reshape(grad_u,x_.shape)
    if gaussian_latent:
        grad_x=torch.exp(normal_dist.log_prob(x_))*grad_u/(2*epsilon)
    else:
        grad_x=grad_u
    return grad_x


supported_datasets=['mnist','cifar10']
datasets_idx={'mnist':0,'cifar10':1}
datasets_in_shape=[(1,28,28),(3,32,32)]

def get_loader(train,data_dir,download,dataset='mnist',batch_size=100): 
    assert dataset in supported_datasets,f"support datasets are in {supported_datasets}"
    if dataset=='mnist':
        mnist_dataset = datasets.MNIST(data_dir, train=train, download=download, transform=transforms.ToTensor())
        data_loader = DataLoader(mnist_dataset, batch_size = batch_size, shuffle=train,)
    elif dataset=='cifar10':
        data_transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                      ])
        cifar10_dataset = datasets.CIFAR10("../data", train=train, download=download, transform=data_transform)
        data_loader = DataLoader(cifar10_dataset , batch_size = 100, shuffle=train)    
    
                    
    return data_loader

def get_model(model_arch, robust_model, robust_eps,nb_epochs,model_dir,data_dir,test_loader, device ,
download,model_shape=(1,28,28),force_train=False,dataset='mnist',batch_size=100):

    model_shape=datasets_in_shape[datasets_idx[dataset]]
    if robust_model:
        c_robust_eps=float_to_file_float(robust_eps)
    model_name="model_"+model_arch +'_' + dataset if not robust_model else f"{model_arch}_robust_{c_robust_eps}"
    
    c_model_path=model_name+'.pt'
    model_path=os.path.join(model_dir,c_model_path)
    if model_arch.lower()=='cnn_custom':
        model=CNN_custom()
    elif model_arch.lower()=='dnn2':
        model=dnn2()
    elif model_arch.lower()=='dnn4':
        model=dnn4()
    elif model_arch.lowwer()=='lenet':
        model=LeNet()
    else:
        raise RuntimeError("Model architecture not supported.")
    if not os.path.exists(model_path) or force_train: 
        #if the model doesn't exist we retrain a model from scratch
        model.to(device)
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
        torch.save(model.state_dict(), model_path)
    
        
        
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    return model, model_shape, model_name

def apply_l_kernel(Y,v_y,l_kernel,T:int,beta,gradV,V,delta_t:float,mh_opt:bool,track_accept:bool,
 gaussian:bool,device,v1_kernel:bool, adapt_d_t:bool,
target_accept:float, accept_spread:float,d_t_decay:float,d_t_gain:float,debug:bool=False,verbose:float =1
,track_delta_t=False,d_t_min=None,d_t_max=None):
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
    return Y,v_y,nb_calls,dict_out


def normal_kernel(x,s):
    return (x + s*torch.randn_like(x))/math.sqrt(1+s**2)

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
        
            
        
        

                        
        nb_calls+= 2*nb
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


def Hamiltonian(X,p,V,beta,gaussian=True):
    with torch.no_grad():
        U = beta*V(X) +torch.sum(X**2,dim=1) if gaussian else beta*V(X)
    H = U + torch.sum(p**2,dim=1)
    return H





def apply_v_kernel(Y,v_y,v_kernel,T:int,beta,gradV,V,delta_t:float,mh_opt:bool,track_accept:bool,
 gaussian:bool,device, adapt_d_t:bool,
target_accept:float, accept_spread:float,d_t_decay:float,d_t_gain:float,debug:bool=False,verbose:float =1
,track_delta_t=False,d_t_min=None,d_t_max=None,lambda_=0,L=1,gamma=0,track_H=False):
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
    H=Hamiltonian(X=Y, p=p_0,V=V,beta=beta,gaussian=gaussian)
    H_s=[]
    if track_H:
        H_s=[H.cpu().mean().item()]
    nb_calls+=nb
    for _ in range(T):
            
        if track_accept or mh_opt:
            cand_Y,cand_p=v_kernel(Y,gradV,p_0 =p_0, delta_t=delta_t,beta=beta,lambda_=lambda_,L=L)
            nb_calls+=L*nb
            cand_H=Hamiltonian(X=cand_Y,p=cand_p,V=V,beta=beta,gaussian =gaussian)
            nb_calls+=nb
            alpha_=torch.exp(-cand_H+H)
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
                    H_s.append(H.cpu().mean().item())

                    p_0=torch.where(accept.unsqueeze(-1),input=cand_p,other=p_0)
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
    return Y,v_y,nb_calls,dict_out