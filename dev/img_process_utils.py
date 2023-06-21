
import torch
import numpy as np
import cv2
from torch.nn.functional import conv2d


class Preprocessing_Layer(torch.nn.Module):
    """
    This is an added layer to the pytorch model, it allows to back-propagate gradients through the preprocessing of the images
    and then work on [0,255] images instead of the preprocessed domain. -> model = nn.Sequential(Preprocessing_Layer(), model)
    This preprocessing works on ResNet models as well as EfficientNet (when not trained with adv prop) with the following values:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    For EfficientNet trained with adv_prop, see Preprocessing_Layer_robust
    """
    def __init__(self,  im_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 interpolation_method='none', antialiasing_method='none'):
        super(Preprocessing_Layer, self).__init__()
        self.mean = mean
        self.std = std
        self.resizer = im_size
        self.interpolation = interpolation_method
        self.antialiasing = antialiasing_method
        # print("resizing at: {} with {} interpolation ({} antialiasing)".format(self.resizer, self.interpolation, self.antialiasing))

    def interpolate_antialiasing(self, image):
        if self.interpolation!='none':
            model_img_size = self.resizer
            if self.antialiasing!='none':
                filter_size = 600//model_img_size
                if filter_size%2==0:
                  kernel_size = filter_size+1
                else:
                  kernel_size = filter_size

                if self.antialiasing=='gaussian':
                    filter_zeros = np.zeros((kernel_size, kernel_size, 1))
                    filter_zeros[filter_size//2, filter_size//2,:] = 1

                    gaussian_filter = cv2.GaussianBlur(filter_zeros, (kernel_size, kernel_size),1.6)
                    gaussian_filter = torch.tensor(gaussian_filter).unsqueeze(0).unsqueeze(0)
                    gaussian_filter = gaussian_filter/(gaussian_filter.sum())

                    gaussian_filter = gaussian_filter.to(image.device)
                    used_filter = gaussian_filter

                else:
                    filter = torch.ones(1,1,filter_size,filter_size)*1/(filter_size**2)
                    filter = filter.double().to(image.device)
                    used_filter = filter

                r,g,b = conv2d(image[:,0,:,:].unsqueeze(1).double(), used_filter,stride=1), conv2d(image[:,1,:,:].unsqueeze(1).double(), used_filter,stride=1), conv2d(image[:,2,:,:].unsqueeze(1).double(), used_filter,stride=1)
                output = torch.zeros((r.shape[0],3,r.shape[2],r.shape[3]), device=image.device)
                r = r[:,0,:,:]
                g = g[:,0,:,:]
                b = b[:,0,:,:]
                output[:,0,:,:],output[:,1,:,:],output[:,2,:,:] = r,g,b

                image = output
            image = torch.nn.functional.interpolate(image, size=self.resizer, mode=self.interpolation)
        return(image)