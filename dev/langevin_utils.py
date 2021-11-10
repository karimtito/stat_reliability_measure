import numpy as np 

def project_ball(X, R):
    X_norms = np.linalg.norm(X, axis =1)[:,None]
    X_in_ball = X_norms<=R
    return R*X/X_norms*(~X_in_ball)+(X_in_ball)*X

def projected_langevin_kernel(X,gradV,delta_t,beta, projection):
    G_noise = np.random.normal(size = X.shape)
    X_new =projection(X-delta_t*gradV(X)+np.sqrt(2*delta_t/beta)*G_noise)
    return X_new

def TimeStep(V,X,gradV,p=1):
    V_mean= V(X).mean()
    V_grad_norm_mean = ((np.linalg.norm(gradV(X),axis = 1)**p).mean())**(1/p)
    return V_mean/V_grad_norm_mean
