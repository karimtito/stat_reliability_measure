import numpy as np
from math import pi
import torch
from stat_reliability_measure.dev.torch_utils import score_function

# PROCEDURE : X_normalised = (X - X_mean)/X_range
# X = [rho, theta, psi, v_own, v_int] (in that order)

X_mean = np.array( [1.9791091*1e4, 0.0, 0.0, 650.0, 600.0] )
X_range = np.array( [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0] )

X_dom = np.array([ [0.0,    -pi, -pi, 100.0,     0.0],
                   [60760.0, pi,  pi, 1200.0, 1200.0] ])

# u = np.true_divide(x - X_mean, X_range) # element-wise division

normal_dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

# PROPERTY 1

IP1_dom = [np.array( [ [55947.691, None, None, 1145, None],
                       [None     , None, None, None, 60] ])]

def dom_to_bounds(domain,device):
    """ Convert domain to bounds for torch. """
    x_min_real =np.where(domain[0, :] != None, domain[0, :],domain[0], X_dom[0, :])[0]
    x_max_real = np.where(domain[1, :] != None, domain[1, :],domain[1], X_dom[1, :])[0] 
    x_min = torch.tensor(real_to_scaled(x_min_real),dtype=torch.float32,device=device)
    x_max = torch.tensor(real_to_scaled(x_max_real),dtype=torch.float32,device=device)
    x_clean = (x_min+x_max)/2
    return x_min,x_max

def real_to_scaled(x):
    """ Convert real values to scaled values. """
    x = np.true_divide(x - X_mean, X_range)
    return(x)

def GenP1(x):
    """ Generate uniformaly samples to check property 1. """
    x = np.random.uniform(low=0.0, high=1.0, size=(5,))
    x[0] = x[0]*(60760.0 - 55947.691) + 55947.691
    x[1] = x[1]*(pi - (-pi)) + (-pi)
    x[2] = x[2]*(pi - (-pi)) + (-pi)
    x[3] = x[3]*(1200.0 - 1145.0) + 1145.0
    x[4] = x[4] * (60.0 - 0.0) + 0.0
    return(x)

def IP1(x):
    """ Check if inputs of property 1 are satisfied. """
    return( (x[0] >= 55947.691) and (x[3] >= 1145) and (x[4] <= 60) )
    
def OP1(x, coc_mean=7.5188840201005975, coc_range=373.94992):
    """ Check if outputs of property 1 are satisfied. """
    return(x[0] <= (1500 - coc_mean)/coc_range)

coc_thresh = (1500 - 7.5188840201005975)/373.94992

def h1(x,model,from_gaussian=True,x_min=0.,x_max=1.,normalize=True):
    """ score function for output property 1. """
    with torch.no_grad():
        x = normal_dist.cdf(x) if from_gaussian else x
        x = x*(x_max-x_min) + x_min if normalize else x
        logits = model(x)
        score = logits[0]-coc_thresh
    return score

def h1_batch(x,model,from_gaussian=True,x_min=0.,x_max=1.,normalize=True):
    """ score function for output property 1. """
    with torch.no_grad():
        x = normal_dist.cdf(x) if from_gaussian else x
        x = x*(x_max-x_min) + x_min if normalize else x
        logits = model(x)
        score = logits[:,0]-coc_thresh
    return score

def V1(x,model,from_gaussian=True,x_min=0,x_max=0, normalize=True):
    """ potential function for output property 1. """
    with torch.no_grad( ):
        x = normal_dist.cdf(x) 
        x = x*(x_max-x_min) + x_min 
        logits = model(x)
        v = torch.clamp(coc_thresh-logits[0],maxin=0.)
    return v

def V1_batch(x,model,from_gaussian=True,x_min =0.,x_max=0.,normalize=True):
    with torch.no_grad():
        """ potential function for output property 1. """
        x = normal_dist.cdf(x)
        x = x*(x_max-x_min) + x_min
        logits = model(x)
        v = torch.clamp(coc_thresh-logits[:,0],min=0.)
    return v

def gradV1_batch(x, model,from_gaussian=True,normalize=True,x_min=0.,x_max=0.):
    """ gradient of potential function for output property 1. """
    u = normal_dist.cdf(x) 
    inputs = u*(x_max-x_min) + x_min
    outputs = model(inputs)
    v = torch.clamp(coc_thresh-outputs[:,0],min=0.)
    grad_x = torch.autograd.grad(outputs=v,inputs=u,grad_outputs=torch.ones_like(v),retain_graph=False)[0]
    with torch.no_grad():
        grad_x = torch.exp(normal_dist.log_prob(x))*grad_x
        grad_x = grad_x*(x_max-x_min)
    return grad_x.detach(), v.detach()

# PROPERTY 2

IP2_dom = [np.array( [ [55947.691, None, None, 1145, None],
                       [None     , None, None, None, 60] ])]

def IP2(x):
    return( (x[0] >= 55947.691) and (x[3] >= 1145) and (x[4] <= 60) )

def OP2(x):
    """ COC score is not the maximal score. """
    return(np.argmax(x) != 0)

def h2(x,model, from_gaussian=True,x_min=0.,x_max=1.,normalize= True):
    if from_gaussian:
        x = normal_dist.cdf(x)*(x_max-x_min) + x_min
    outputs = model(x)
    score = outputs[0] - outputs[1:].max(dim=0)
    return score.detach()

def h2_batch(x,model, from_gaussian=True,x_min=0.,x_max=1.,normalize=True):
    """ score function for output property 2. """
    with torch.no_grad():
        if from_gaussian:
            x = normal_dist.cdf(x)
        if normalize:
            x = x*(x_max-x_min) + x_min
        outputs = model(x)
        score = outputs[:,0] - outputs[:,1:].max(dim=1)
    return score.detach()

def V2(x, model, from_gaussian=True,x_min=0.,x_max=1.):
    """ potential function for output property 2. """
    with torch.no_grad():
        if from_gaussian:
            x = normal_dist.cdf(x)*(x_max-x_min) + x_min
        outputs = model(x)
        v = torch.clamp(-(outputs[0] - outputs[1:].max(dim=0)),min=0.)
    return v

def V2_batch(x,model, from_gaussian=True, x_min=0., x_max=1.):
    """ potential function for output property 2. """
    with torch.no_grad():
        inputs = normal_dist.cdf(x)*(x_max-x_min) + x_min
        outputs = model(inputs)
        v = torch.clamp(-(outputs[:,0] - outputs[:,1:].max(dim=1)),min=0.)
    return v

def gradV2_batch(x, model, from_gaussian=True,x_min =0.,x_max=1.):
    """ batch gradient of potential function for output property 2. """
   
    inputs = normal_dist.cdf(x) * (x_max-x_min) + x_min
    outputs = model(inputs)
    v = torch.clamp(-(outputs[:,0] - outputs[:,1:].max(dim=1)),min=0.)
    grad_x = torch.autograd.grad(outputs=v,inputs=inputs,grad_outputs=torch.ones_like(v),retain_graph=False)[0]
    with torch.no_grad():
        grad_x = torch.exp(normal_dist.log_prob(x))*grad_x
        grad_x = grad_x*(x_max-x_min)
    return grad_x.detach(),v.detach()



# PROPERTY 3

IP3_dom = [np.array( [ [1500.0, -0.06, 3.10, 980.0, 960.0],
                       [1800.0,  0.06, None,  None,  None] ])]

def IP3(x):
    return( (1500.0 <= x[0] <= 1800.0) and (-0.06 <= x[1] <= 0.06) and (3.10 <= x[2]) and (980.0 <= x[3]) and (960.0 <= x[4]) )

def OP3(x):
    """ COC score is not the minimal score. """
    return(np.argmin(x) != 0)


def h3_batch(x,model, from_gaussian=True,x_min=0.,x_max=1., normalize =True):
    """ score function for output property 2. """
    with torch.no_grad():
        x = normal_dist.cdf(x) if from_gaussian else x
        x = x*(x_max-x_min) + x_min if normalize else x
        outputs = model(x)
        score = outputs[:,1:].min(dim=1)-outputs[:,0] 
    return score

def V3_batch(x,model,from_gaussian=True, x_min=0., x_max=1., normalize =True):
    """ potential function for output property 3. """
    with torch.no_grad():
        x = normal_dist.cdf(x) 
        x = x*(x_max-x_min) + x_min 
        outputs = model(x)
        v = torch.clamp(outputs[:,0]-outputs[:,1:].min(),min=0.)
    return v

def gradV3_batch(x,model,from_gaussian=True, x_min=0., x_max=1., normalize =True):
    """ gradient of potential function for output property 3."""
    u = normal_dist.cdf(x) 
    inputs = u*(x_max-x_min) + x_min 
    outputs = model(inputs)
    v = torch.clamp(outputs[:,0]-outputs[:,1:].min(dim=1),min=0.)
    grad_x = torch.autograd.grad(outputs=v,inputs=inputs,grad_outputs=torch.ones_like(v),retain_graph=False)[0]
    with torch.no_grad():
        grad_x = torch.exp(normal_dist.log_prob(x))*grad_x
        grad_x = grad_x*(x_max-x_min)
    return grad_x.detach(), v.detach()

# PROPERTY 4

IP4_dom = [np.array( [ [1500.0, -0.06, 0.0, 1000.0, 700.0],
                          [1800.0,  0.06, 0.0,  None,  800.0] ])]

def IP4(x):
    return((1500.0 <= x[0] <= 1800.0) and (-0.06 <= x[1] <= 0.06) and (x[2] == 0) and (1000.0 <= x[3]) and (700.0 <= x[4] <= 800.0) )

def OP4(x):
    """ COC score is not the minimal score. """
    return(np.argmin(x) != 0)

h4_bathc = h3_batch
V4_batch = V3_batch
gradV4_batch = gradV3_batch


# PROPERTY 5

IP5_dom = [np.array( [ [250.0, 0.2, -3.141592, 100.0, 0.0],
                            [400.0, 0.4, -3.141592+0.005, 400.0, 400.0] ])]
def IP5(x):
    return( (250.0 <= x[0] <= 400.0) and (0.2 <= x[1] <= 0.4) and (-3.141592 <= x[2] <= -3.141592 + 0.005) and (100.0 <= x[3] <= 400.0) and (0.0 <= x[4] <= 400.0) )

def OP5(x):
    """ SL score is the minimal score. """
    return(np.argmin(x) == 4)

def h5_batch(x, model, from_gaussian=True,x_min=0.,x_max=1., normalize =True):
    """ score function for output property 5.
        SRC score is the minimal score. """
    with torch.no_grad():
        x = normal_dist.cdf(x) if from_gaussian else x
        x = x*(x_max-x_min) + x_min if normalize else x
        outputs = model(x)
        score = outputs[:,4] - torch.concatenate((outputs[:,0:4],outputs[:,5:]),dim=1).min(dim=1)
    return score

def V5_batch(x, model, from_gaussian=True,x_min=0.,x_max=1., normalize =True):
    """ score function for output property 5.
        SL score is the minimal score. """
    with torch.no_grad():
        x = normal_dist.cdf(x) if from_gaussian else x
        x = x*(x_max-x_min) + x_min if normalize else x
        outputs = model(x)
        v = torch.clip(torch.concatenate((outputs[:,0:4],outputs[:,5:]),dim=1).min(dim=1)-outputs[:,4],min=0.)
    return v

def gradV5_batch(x, model, from_gaussian=True,x_min=0.,x_max=1., normalize =True ):
    """ score function for output property 5. 
        SL score is the minimal score."""
    u = normal_dist.cdf(x) 
    inputs = u*(x_max-x_min) + x_min
    outputs = model(inputs)
    v = torch.clip(-(outputs[:,4] - torch.concatenate((outputs[:,0:4],outputs[:,5:]),dim=1).max(dim=1)),min=0.)
    grad_x = torch.autograd.grad(outputs=v,inputs=inputs,grad_outputs=torch.ones_like(v),retain_graph=False)[0]
    with torch.no_grad():
        grad_x = normal_dist.cdf(grad_x) 
        grad_x = grad_x*(x_max-x_min) 
    return grad_x.detach(), v.detach()

     
    

# PROPERTY 6
IP6_dom = [np.array( [ [12000.0, 0.7, -3.141592, 100.0, 0.0],
                            [62000.0, 3.141592, -3.141592+0.005, 1200.0, 1200.0] ])]
def IP6(x):
    return( (12000.0 <= x[0] <= 62000.0) and ((0.7 <= x[1] <= 3.141592) or (-3.141592 <= x[1] <= -0.7)) and (-3.141592 <= x[2] <= -3.141592+0.005) and (100.0 <= x[3] <= 1200.0) and (0.0 <= x[4] <= 1200.0) )

def OP6(x):
    """ COC score is the minimal score. """
    return(np.argmin(x) == 0)

def h6_batch(x,model, from_gaussian=True,x_min=0.,x_max=1., normalize =True):
    """ score function for output property 6. """
    with torch.no_grad():
        u = normal_dist.cdf(x)
        inputs = u*(x_max-x_min) + x_min if normalize else u
        outputs = model(inputs)
        score = outputs[:,0] - outputs[:,1:].min(dim=1)
    return score

def V6_batch(x, model , from_gaussian=True,x_min=0.,x_max=1., normalize =True):
    """ potential function for output property 6. """
    with torch.no_grad():
        u = normal_dist.cdf(x)
        inputs = u*(x_max-x_min) + x_min if normalize else u
        outputs = model(inputs)
        v = torch.clamp(outputs[:,1:].min(dim=1)-outputs[:,0],min=0.)
    return v

def gradV6_batch(x,model,from_gaussian=True,x_min=0.,x_max=1., normalize =True):
    u = normal_dist.cdf(x)
    inputs = u*(x_max-x_min) + x_min
    outputs = model(inputs)
    v = torch.clamp(outputs[:,1:].min(dim=1)-outputs[:,0],min=0.)
    grad_x = torch.autograd.grad(outputs=v,inputs=inputs,grad_outputs=torch.ones_like(v),retain_graph=False)[0]
    with torch.no_grad():
        grad_x = torch.exp(normal_dist.log_prob(x))*grad_x
        grad_x = grad_x*(x_max-x_min)
    return grad_x.detach(), v.detach()

# PROPERTY 7
IP7_dom = [np.array( [ [12000.0, -3.141592, -3.141592, 100.0, 0.0],
                            [60760.0, 3.141592, 3.141592, 1200.0, 1200.0] ])]

def IP7(x):
    return( (12000.0 <= x[0] <= 60760.0) and (-3.141592 <= x[1] <= 3.141592) and (-3.141592 <= x[2] <= 3.141592) and (100.0 <= x[3] <= 1200.0) and (0.0 <= x[4] <= 1200.0) )

def OP7(x):
    """ SR score and SL score are not the minimal scores. """
    min_score_ind = np.argmin(x)
    return( (min_score_ind != 3) and (min_score_ind != 4) )

def h7(x,model,from_gaussian=True,normalize=True,x_min=0.,x_max=1.):
    """ score function for output property 7. """
    with torch.no_grad():
        u = normal_dist.cdf(x) if from_gaussian else x
        inputs = u*(x_max-x_min) + x_min if normalize else u
        outputs = model(inputs)
        score = outputs[:,0:3].min(dim=1) - outputs[:,3:5].min(dim=1)
    return score

def V7_batch(x, model, from_gaussian=True,normalize=True,x_min=0.,x_max=1.):
    """ potential function for output property 7. """
    with torch.no_grad():
        u = normal_dist.cdf(x) if from_gaussian else x
        inputs = u*(x_max-x_min) + x_min if normalize else u
        outputs = model(inputs)
        v = torch.clamp(outputs[:,3:5].min(dim=1)-outputs[:,0:3].min(dim=1),min=0.)
    return v

def gradV7_batch(x, model, from_gaussian=True,normalize=True,x_min=0., x_max=1.):
    """ gradient of potential function for output property 7. """
    u = normal_dist.cdf(x) 
    inputs = u*(x_max-x_min) + x_min 
    outputs = model(inputs)
    v = torch.clamp(outputs[:,3:5].min(dim=1)-outputs[:,0:3].min(dim=1),min=0.)
    grad_x = torch.autograd.grad(outputs=v,inputs=inputs,grad_outputs=torch.ones_like(v),retain_graph=False)[0]
    with torch.no_grad():
        grad_x = torch.exp(normal_dist.log_prob(x))*grad_x
        grad_x = grad_x*(x_max-x_min)
    return grad_x.detach(), v.detach()

# PROPERTY 8
IP8_dom = [np.array( [ [0.0, -3.141592, -0.75*3.141592, 600.0, 600.0],
                            [60760.0, -0.75*3.141592, 0.0, 1200.0, 1200.0] ])]

def IP8(x):
    return( (0.0 <= x[0] <= 60760.0) and (-3.141592 <= x[1] <= -0.75*3.141592) and (-0.1 <= x[2] <= 0.1) and (600.0 <= x[3] <= 1200.0) and (600.0 <= x[4] <= 1200.0) )

def OP8(x):
    """ COC score or WR score is the minimal score. """
    min_score_ind = np.argmin(x)
    return( (min_score_ind == 0) or (min_score_ind == 1) )

def h8_batch(x,model,from_gaussian=True,normalize=True,x_min=0.,x_max=1.):
    """ score function for output property 8. """
    with torch.no_grad():
        x = normal_dist.cdf(x) if from_gaussian else x
        x = x*(x_max-x_min) + x_min if normalize else x
        outputs= model(x)
        score = outputs[:,0:2].min(dim=1) - outputs[:,2:].min(dim=1)
    return score

def V8_batch(x,model, from_gaussian=True,normalize=True,x_min=0.,x_max=1.):
    """ potential function for output property 8. """
    with torch.no_grad():
        x = normal_dist.cdf(x) if from_gaussian else x
        x = x*(x_max-x_min) + x_min if normalize else x
        outputs = model(x)
        v = torch.clamp(outputs[:,2:].min(dim=1)-outputs[:,0:2].min(dim=1),min=0.)
    return v

def gradV8_batch(x, model, from_gaussian=True, normalize=True,x_min =0.,x_max=1.):
    u = normal_dist.cdf(x)
    inputs = u*(x_max-x_min) + x_min
    outputs = model(inputs)
    v = torch.clamp(outputs[:,2:].min(dim=1)-outputs[:,0:2].min(dim=1),min=0.)
    grad_x = torch.autograd.grad(outputs=v,inputs=inputs,grad_outputs=torch.ones_like(v),retain_graph=False)[0]
    with torch.no_grad():
        grad_x = torch.exp(normal_dist.log_prob(x))*grad_x
        grad_x = grad_x*(x_max-x_min)
    return grad_x.detach(), v.detach()



# PROPERTY 9
IP9_dom = [np.array( [ [2000.0, -0.4, -3.141592, 100.0, 0.0],
                            [7000.0, -0.14, -3.141592+0.01, 150.0, 150.0] ])]
def IP9(x):
    return( (2000.0 <= x[0] <= 7000.0) and (-0.4 <= x[1] <= -0.14) and (-3.141592 <= x[2] <= -3.141592+0.01) and (100.0 <= x[3] <= 150.0) and (0.0 <= x[4] <= 150.0) )

def OP9(x):
    """ SL score is the minimal score. """
    return(np.argmin(x) == 4)

def h9_batch(x,model,from_gaussian=True,normalize=True,x_min=0.,x_max=1.):
    """ score function for output property 9. """
    with torch.no_grad():
        x = normal_dist.cdf(x) if from_gaussian else x
        x = x*(x_max-x_min) + x_min if normalize else x
        outputs = model(x)
        score = outputs[:,4] - outputs[:,0:4].min(dim=1)
    return score

def V9_batch(x,model,from_gaussian=True,normalize=True,x_min=0.,x_max=1.):
    """ potential function for output property 9. """
    with torch.no_grad():
        x = normal_dist.cdf(x) if from_gaussian else x
        x = x*(x_max-x_min) + x_min if normalize else x
        outputs = model(x)
        v = torch.clamp(outputs[:,0:4].min(dim=1)-outputs[:,4],min=0.)
    return v

def gradV9_batch(x, model, from_gaussian=True,normalize=True,x_min=0.,x_max=1.):
    u = normal_dist.cdf(x)
    inputs = u*(x_max-x_min) + x_min
    outputs = model(inputs)
    v = torch.clamp(outputs[:,0:4].min(dim=1)-outputs[:,4],min=0.)
    grad_x = torch.autograd.grad(outputs=v,inputs=inputs,grad_outputs=torch.ones_like(v),retain_graph=False)[0]
    with torch.no_grad():
        grad_x = torch.exp(normal_dist.log_prob(x))*grad_x
        grad_x = grad_x*(x_max-x_min)
    return grad_x.detach(), v.detach()



# PROPERTY 10
IP10_dom = [np.array( [ [36000.0, 0.7, -3.141592, 900.0, 600.0],
                            [60760.0, 3.141592, -3.141592+0.01, 1200.0, 1200.0] ])]
def IP10(x):
    return( (36000.0 <= x[0] <= 60760.0) and (0.7 <= x[1] <= 3.141592) and (-3.141592 <= x[2] <= -3.141592+0.01) and (900.0 <= x[3] <= 1200.0) and (600.0 <= x[4] <= 1200.0) )

def OP10(x):
    """ COC score is the minimal score. """
    return(np.argmin(x) == 0)


def h10_batch(x,model,from_gaussian=True,normalize=True,x_min=0.,x_max=1.):
    """ score function for output property 10. """
    with torch.no_grad():
        x = normal_dist.cdf(x) if from_gaussian else x
        x = x*(x_max-x_min) + x_min if normalize else x
        outputs = model(x)
        score = outputs[:,0] - outputs[:,1:].min(dim=1)
    return score

def V10_batch(x,model,from_gaussian=True,normalize=True,x_min=0.,x_max=1.):
    """ potential function for output property 10. """
    with torch.no_grad():
        x = normal_dist.cdf(x) 
        x = x*(x_max-x_min) + x_min 
        outputs = model(x)
        v = torch.clamp(outputs[:,1:].min(dim=1)-outputs[:,0],min=0.)
    return v

def gradV10_batch(x, model, from_gaussian=True,normalize=True,x_min=0.,x_max=1.):
    u = normal_dist.cdf(x)
    inputs = u*(x_max-x_min) + x_min
    outputs = model(inputs)
    v = torch.clamp(outputs[:,1:].min(dim=1)-outputs[:,0],min=0.)
    grad_x = torch.autograd.grad(outputs=v,inputs=inputs,grad_outputs=torch.ones_like(v),retain_graph=False)[0]
    with torch.no_grad():
        grad_x = torch.exp(normal_dist.log_prob(x))*grad_x
        grad_x = grad_x*(x_max-x_min)
    return grad_x.detach(), v.detach()

