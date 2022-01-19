import torch
import torch.nn as nn

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

def langevin_kernel_pyt(X,gradV,delta_t,beta,device=None):
    """performs one step of langevin kernel with inverse temperature beta"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    G_noise = torch.randn(size = X.shape).to(device)
    grad=gradV(X)
    with torch.no_grad():
        X_new =X-delta_t*grad+torch.sqrt(2*delta_t/beta)*G_noise
    return X_new

def multi_unsqueeze(input_,k,dim=-1):
    """"unsqueeze input multiple times"""
    for _ in range(k):
        input_=input_.unsqueeze(dim=dim)
    return input_

def compute_V_grad_pyt2(model, input_, target_class):
    """ Returns potentials and associated gradients for given inputs, model and target classes """
    if input_.requires_grad!=True:
        print("/!\\ Input does not require gradient (we assume it is a leaf node).")
        input_.requires_grad=True
    input_.retain_grad()
    logits = model(input_) 
    val, ind= torch.topk(logits,k=2)
    v_=val[:,0]-val[:,1]
    v_.backward(gradient=torch.ones_like(v_),retain_graph=True)
    

    a_priori_grad=input_.grad
    mask=torch.eq(ind[:,0],target_class)
    v=torch.where(condition=mask, input=v_,other=torch.zeros_like(v_))
    mask=multi_unsqueeze(mask,k=a_priori_grad.ndim- mask.ndim)
    grad=torch.where(condition=mask, input=a_priori_grad,other=torch.zeros_like(a_priori_grad))
    return v,grad

def compute_V_grad_pyt(model, input_, target_class):
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

def compute_V_pyt(model, input_, target_class):
    """Return potentials for given inputs, model and target classes"""
    with torch.no_grad():
        logits = model(input_) 
        val, ind= torch.topk(logits,k=2)
        output=val[:,0]-val[:,1]
        mask=ind[:,0]==target_class
        v=torch.where(condition=mask, input=output,other=torch.zeros_like(output))
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