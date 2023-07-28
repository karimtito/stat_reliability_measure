from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import auto_LiRPA.operators
from time import time


def get_lirpa_bounds(x_clean,y_clean,model,epsilon,num_classes,noise_dist,a,device):
    t = time()
    with torch.no_grad():
        image=x_clean.view(1,1,28,28)
        bounded_model=BoundedModule(model,global_input=torch.zeros_like(input=image),bound_opts={"conv_mode": "patches"})
    # Step 2: define perturbation. Here we use a Linf perturbation on input image.
        
        norm = np.inf
        ptb = PerturbationLpNorm(norm = norm, eps = epsilon)
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
        if noise_dist=='gaussian':
            l_sigmas=torch.matmul(lA_rshp,tlA).diag()
            u_sigmas=torch.matmul(uA_rshp,tuA).diag()

            gamma_l=0.5-0.5*torch.erf(input= (-mu_l/(np.sqrt(2)*torch.sqrt(l_sigmas))))
            gamma_u=0.5-0.5*torch.erf(input= (-mu_u/(np.sqrt(2)*torch.sqrt(u_sigmas))))
        elif noise_dist=='uniform':
            pos_gap_l=(mu_l-a>=0).float()
            neg_gap_u=(mu_u-a<=0).float()
            gamma_l=  pos_gap_l*(1-torch.exp(-(mu_l-a)**2/(2*epsilon**2*torch.norm(input=lA_rshp,dim=1))))
            gamma_u= neg_gap_u*(torch.exp(-(mu_u-a)**2/(2*epsilon**2*torch.norm(input=uA_rshp, dim=1)))) +(1-neg_gap_u)
        else:
            raise NotImplementedError("Only uniform and Gaussian distributions are implemented for lirpa bounds.")
        
        #y_clean_flag= torch.arange(num_classes,device=device)==y_clean
         
        pre_p_l=1-torch.prod(1-gamma_l)/(1-gamma_l[y_clean]) 
        p_l=torch.clamp(pre_p_l,min=0,max=1) 
        p_u=torch.clamp(gamma_u.sum()-gamma_u[y_clean],min=0,max=1)
        time_lirpa=time()-t
        return (p_l,p_u), time_lirpa


#TODO complete lirpa certification 
def get_lirpa_cert(x_clean,y_clean,model,epsilon,num_classes,device,method='CROWN'):
    t=time()
    N=x_clean.shape[0]
    image=x_clean.view(N,1,28,28)
    bounded_model=BoundedModule(model,global_input=torch.zeros_like(input=image),bound_opts={"conv_mode": "patches"})
# Step 2: define perturbation. Here we use a Linf perturbation on input image.
    
    norm = np.inf
    ptb = PerturbationLpNorm(norm = norm, eps = epsilon)
    # Input tensor is wrapped in a BoundedTensor object.
    bounded_image = BoundedTensor(image, ptb)
    with torch.no_grad():
        lb,ub = bounded_model.compute_bounds(x=(bounded_image),method=method)
    
    lirpa_safe=lb[:,y_clean]>=ub.max(1) 
    time_lirpa = time()-t
    return lirpa_safe,time_lirpa