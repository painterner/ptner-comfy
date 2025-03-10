import cv2
import gradio as gr
import os
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import gdown
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# os.system("git clone https://github.com/xuebinqin/DIS")
# os.system("mv DIS/IS-Net/* .")

# project imports
from .data_loader_cache import normalize, im_reader, im_preprocess 
from .models import *

#Helpers
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Download official weights
if not os.path.exists("saved_models"):
    os.mkdir("saved_models")
    os.system("mv isnet.pth saved_models/")
    
class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,image):
        image = normalize(image,self.mean,self.std)
        return image


transform =  transforms.Compose([GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0])])

def load_image(im, hypar):
    # im = im_reader(im_path)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im,255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0) # make a batch of image, shape


def build_model(hypar,device):
    net = hypar["model"]#GOSNETINC(3,1)

    # convert to half precision
    if(hypar["model_digit"]=="half"):
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)
    
    import pathlib
    pf = str(pathlib.Path(__file__).parent.absolute())

    if(hypar["restore_model"]!=""):
        net.load_state_dict(torch.load(pf+"/"+hypar["model_path"]+"/"+hypar["restore_model"], map_location=device))
        net.to(device)
    net.eval()  
    return net

    
def predict(net,  inputs_val, shapes_val, hypar, device):
    '''
    Given an Image, predict the mask
    '''
    net.eval()

    if(hypar["model_digit"]=="full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

  
    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device) # wrap inputs in Variable
   
    ds_val = net(inputs_val_v)[0] # list of 6 results

    pred_val = ds_val[0][0,:,:,:] # B x 1 x H x W    # we want the first one which is the most accurate prediction

    ## recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(shapes_val[0][0],shapes_val[0][1]),mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val-mi)/(ma-mi) # max = 1

    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy()*255).astype(np.uint8) # it is the mask we need
    
# Set Parameters
hypar = {} # paramters for inferencing


hypar["model_path"] ="./saved_models" ## load trained weights from this path
hypar["restore_model"] = "isnet.pth" ## name of the to-be-loaded weights
hypar["interm_sup"] = False ## indicate if activate intermediate feature supervision

##  choose floating point accuracy --
hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
hypar["seed"] = 0

hypar["cache_size"] = [1024, 1024] ## cached input spatial resolution, can be configured into different size

## data augmentation parameters ---
hypar["input_size"] = [1024, 1024] ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
hypar["crop_size"] = [1024, 1024] ## random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation

hypar["model"] = ISNetDIS()

 # Build Model
net = build_model(hypar, device)

def post_process(mask: np.ndarray) -> np.ndarray:
    """
    Post Process the mask for a smooth boundary by applying Morphological Operations
    Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
    args:
        mask: Binary Numpy Mask
    """
    from cv2 import (
        BORDER_DEFAULT,
        MORPH_ELLIPSE,
        MORPH_OPEN,
        GaussianBlur,
        getStructuringElement,
        morphologyEx,
    )
    kernel = getStructuringElement(MORPH_ELLIPSE, (3, 3))
    mask = morphologyEx(mask, MORPH_OPEN, kernel)
    mask = GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=BORDER_DEFAULT)
    mask = np.where(mask < 127, 0, 255).astype(np.uint8)  # type: ignore
    return mask

def inference(image, pilImage):
  
  image_tensor, orig_size = load_image(image, hypar) 
  mask = predict(net, image_tensor, orig_size, hypar, device)
  
#   pil_mask = Image.fromarray(mask).convert('L')
  pil_mask = Image.fromarray(post_process(np.array(mask))).convert('L')
  im_rgb = pilImage.convert("RGB")
  
  im_rgba = im_rgb.copy()
  im_rgba.putalpha(pil_mask)
  
  def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
  
  return [pil2tensor(im_rgba), pil2tensor(pil_mask)]


class PtRembackgroundIsNet:
    @classmethod
    def INPUT_TYPES(s):
        import pathlib
        pf = str(pathlib.Path(__file__).parent.absolute())
        return {"required":
                    {"images": ("IMAGE",),
                     "model": (os.listdir(pf+"/"+hypar["model_path"]), )
                     },
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, images, model):
        hypar["restore_model"] = model
        resultimages = []
        resultmasks = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            pilimg = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img = np.array(pilimg)
            result = inference(img, pilimg)
            resultimages.append(result[0])
            resultmasks.append(result[1])
        return (torch.cat(resultimages, dim=0), torch.cat(resultmasks, dim=0))
    
NODE_CLASS_MAPPINGS = {
    "Pt-RembackgroundIsNet": PtRembackgroundIsNet
}
