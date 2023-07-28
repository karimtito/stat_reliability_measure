import torch
import dev.torch_utils as t_u
import rawpy
import numpy as np
from tqdm import tqdm_notebook as tqdm



def MC_simulation_from_raw(raw_name,model,N=10,batch_size=5,c_1=1.15,c_2=1150):
    """ MC simulation for raw images
     description:
      - raw_name: path to the raw image (string)
       - model: pytorch model
        - N: number of MC simulations (int)
         - batch_size: batch size for MC simulations (int)
          - c_1: constant for noise scale (float)
           - c_2: constant for noise scale (float)
            
             
     return:    
        - scores: list of scores (list)
         - img_norm_diffs: list of image norm differences (list)
            - img_rgb_diffs: list of image rgb differences (list)
             - y_clean: label of the raw image (int)
              - score_0: score of the raw image (float)
                
    """
    linear_params = rawpy.Params(rawpy.DemosaicAlgorithm.LINEAR, half_size=False,)
    raw = rawpy.imread(raw_name)
    img = np.copy(raw.raw_image.astype(np.float32)) 
    img_rgb = raw.postprocess(linear_params)
    image_tensor=torch.from_numpy(img_rgb/255.).to(torch.float32).permute(2,0,1).unsqueeze(0)
    image_tensor=torch.nn.functional.interpolate(image_tensor, size=224, mode='bilinear').cuda()
    logits= model(image_tensor).detach().cpu() 
    y_clean = torch.argmax(logits)
    score_0 = t_u.score_function(model=model,X=image_tensor,y_clean=y_clean)
    d = img.size
    input_shape=img.shape # (H,W)
    noise_scale=np.sqrt(np.clip(1.15*img-1150,a_min=0,a_max=None))
    nb_iter= N//batch_size
    rest = N%batch_size
    img_norm_diffs=[]
    img_rgb_diffs=[]
    tensor_diffs=[]
    scores=[]
    for i in tqdm(range(nb_iter)):
        x = torch.randn((batch_size,d),device='cuda')
        x = x.reshape(x.shape[0],*input_shape).cpu().numpy()
        for j in range(batch_size):
            x_np = x[j]
            new_img = np.clip(img+x_np*noise_scale,a_min=0,a_max=2**14-1)
            raw.raw_image[:,:] = new_img
            img_norm_diffs.append(np.linalg.norm(new_img-img))
            del new_img
            new_img_rgb = raw.postprocess(linear_params)
            new_tensor = torch.from_numpy(new_img_rgb/255.).to(torch.float32).permute(2,0,1).unsqueeze(0)
            img_rgb_diffs.append(np.linalg.norm(new_img_rgb-img_rgb))
            del new_img_rgb
            new_tensor=torch.nn.functional.interpolate(new_tensor, size=224, mode='bilinear').cuda()
            score = t_u.score_function(model=model,X=new_tensor,y_clean=y_clean)
            scores.append(score.item())
            tensor_diffs.append(torch.norm(new_tensor-image_tensor))
            del new_tensor
    x = torch.randn((rest,d),device='cuda')
    x = x.reshape(x.shape[0],*input_shape).cpu().numpy()
    for j in range(rest):
        x_np = x[j]
        new_img = np.clip(img+x_np*noise_scale,a_min=0,a_max=2**14-1)
        raw.raw_image[:,:] = new_img
        img_norm_diffs.append(np.linalg.norm(new_img-img))
        del new_img
        new_img_rgb = raw.postprocess(linear_params)
        new_tensor = torch.from_numpy(new_img_rgb/255.).to(torch.float32).permute(2,0,1).unsqueeze(0)
        img_rgb_diffs.append(np.linalg.norm(new_img_rgb-img_rgb))
        del new_img_rgb
        new_tensor=torch.nn.functional.interpolate(new_tensor, size=224, mode='bilinear').cuda()
        score = t_u.score_function(model=model,X=new_tensor,y_clean=y_clean)
        scores.append(score.detach().cpu().numpy())
        tensor_diffs.append(torch.norm(new_tensor-image_tensor))
        del new_tensor   
    return scores,img_norm_diffs,img_rgb_diffs,y_clean,score_0




    