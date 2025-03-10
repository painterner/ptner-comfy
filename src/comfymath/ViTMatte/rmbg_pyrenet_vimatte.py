# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
from typing import List

import os
from os import path
from os.path import join as opj
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from transparent_background import Remover

import torch
from torchvision.ops import box_convert
import torchvision.transforms.functional as F

from detectron2.engine import default_argument_parser
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
#from google.colab.patches import cv2_imshow

#from detectron2.config import LazyConfig, instantiate
from detectron2.config.lazy import LazyConfig
from detectron2.config.instantiate import instantiate

from huggingface_hub import hf_hub_download

from comfy.utils import ProgressBar
import comfy

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class LoadPViMatteModelType:
    def __init__(self) -> None:
        pass
    
bmtype = 'pt-LoadPViMatteModelType'
pmtype = 'pt-LoadPyrenetModelType'

PYRENET_CKPT = "/data/ComfyUI/custom_nodes/Painterner/src/comfymath/checkpoint/latest_from_huggingface.pth"

VITMATTE_CKPT = '/data/ComfyUI/custom_nodes/Painterner/src/comfymath/checkpoint/ViTMatte_B_DIS_from_huggingface.pth'

VITMATTE_SOURCE_PATH = '/data/ComfyUI/custom_nodes/Painterner/src/comfymath/ViTMatte'
VITMATTE_CONFIG_MODEL = '/data/ComfyUI/custom_nodes/Painterner/src/comfymath/ViTMatte/configs/common/model.py'

# sys.path.insert(0, VITMATTE_SOURCE_PATH)
sys.path.append(VITMATTE_SOURCE_PATH)

class LoadPViMatteModel:
    @classmethod
    def INPUT_TYPES(s):
        return {}

    RETURN_TYPES = (bmtype,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    
    def init_model(self, model, checkpoint, device):
        """
        Initialize the model.
        Input:
            config: the config file of the model
            checkpoint: the checkpoint of the model
        """
        assert model in ['vitmatte-s', 'vitmatte-b']
        
        if model == 'vitmatte-s':
            #sys.path.insert(0, "ViTMatte")
            config = VITMATTE_CONFIG_MODEL
            cfg = LazyConfig.load(config)
            model = instantiate(cfg.model)
            model.to(device)
            model.eval()
            DetectionCheckpointer(model).load(checkpoint)
        elif model == 'vitmatte-b':
            #sys.path.insert(0, "ViTMatte")
            config = VITMATTE_CONFIG_MODEL
            cfg = LazyConfig.load(config)
            cfg.model.backbone.embed_dim = 768
            cfg.model.backbone.num_heads = 12
            cfg.model.decoder.in_chans = 768
            model = instantiate(cfg.model)
            model.to(device)
            model.eval()
            DetectionCheckpointer(model).load(checkpoint)
            
        return model
            
    def load(self):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        vitmatte_model = self.init_model(model='vitmatte-b', checkpoint=VITMATTE_CKPT, device=device) #상대경로 확인 

        return (vitmatte_model, )
    
class LoadPyrenetModel:
    @classmethod
    def INPUT_TYPES(s):
        return {}

    RETURN_TYPES = (pmtype,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    def load(self):
        remover = Remover(mode='base', device='cuda:0', ckpt=PYRENET_CKPT)
        return (remover, )

class RembgPViMatte():
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "images": ("IMAGE", ),
                        "remover": (pmtype, ),
                        "vitmodel": (bmtype, ),
                        "background": ("STRING", {"multiline": False,}),
                        "threshold": ("INT", {"default": 50, "min": 0, "max": 65536}),
                    },
                }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("IMAGE", "replace count")
    FUNCTION = "predict"

    CATEGORY = "image"

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        if path.exists("./checkpoint/ViTMatte_B_DIS.pth"):
            pass
        else:
            hf_hub_download(repo_id="nielsr/vitmatte-checkpoints", filename="ViTMatte_B_DIS.pth", local_dir="./checkpoint")
            # gdown.download(id="1d97oKuITCeWgai2Tf3iNilt6rMSSYzkW", output="./checkpoint/ViTMatte_B_DIS.pth", quiet=False)
            
        if path.exists(PYRENET_CKPT):
            pass
        else:
            hf_hub_download(repo_id="jyoung105/inspyrenet", filename="latest.pth", local_dir="./checkpoint")

    def ispyrenet(self, input_image):

        # Load model
        ispyrenet_output = self.remover.process(input_image)
        # ispyrenet_output = remover.process(input_image, threshold=0.5) # use threhold parameter for hard prediction.
                
        return ispyrenet_output

    
    def generate_checkerboard_image(self, height, width, num_squares):
        num_squares_h = num_squares
        square_size_h = height // num_squares_h
        square_size_w = square_size_h
        num_squares_w = width // square_size_w

        new_height = num_squares_h * square_size_h
        new_width = num_squares_w * square_size_w
        image = np.zeros((new_height, new_width), dtype=np.uint8)

        for i in range(num_squares_h):
            for j in range(num_squares_w):
                start_x = j * square_size_w
                start_y = i * square_size_h
                color = 255 if (i + j) % 2 == 0 else 200
                image[start_y:start_y + square_size_h, start_x:start_x + square_size_w] = color

        image = cv2.resize(image, (width, height))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image


    def init_vitmatte(self, model_type):
        """
        Initialize the vitmatte with model_type in ['vit_s', 'vit_b']
        """

        #import sys
        #sys.path.append("/usr/local/lib/python3.10/dist-packages/detectron2/config")

        cfg = LazyConfig.load(vitmatte_config[model_type])
        vitmatte = instantiate(cfg.model)
        vitmatte.to(device)
        vitmatte.eval()
        DetectionCheckpointer(vitmatte).load(vitmatte_models[model_type])

        return vitmatte


    def generate_trimap(self, mask, erode_kernel_size=10, dilate_kernel_size=10):
        erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        eroded = cv2.erode(mask, erode_kernel, iterations=5)
        dilated = cv2.dilate(mask, dilate_kernel, iterations=5)
        trimap = np.zeros_like(mask)
        trimap[dilated==255] = 128
        trimap[eroded==255] = 255
        
        return trimap
    

    def store_img(self, img):
        return img, []  # when new image is uploaded, `selected_points` should be empty


    def convert_pixels(self, gray_image, boxes):
        converted_image = np.copy(gray_image)

        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            converted_image[y1:y2, x1:x2][converted_image[y1:y2, x1:x2] == 1] = 0.5

        return converted_image


    def covert_tracer_output_to_vitmatte_input(self, mask_path): #image path: '~~/~~/~~.png'
        # Load your background-removed image
        image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # Assuming the alpha channel (if present) represents transparency (0 = removed, 255 = retained)
        if image.shape[2] == 4:  # Check if the image has an alpha channel
            alpha_channel = image[:, :, 3]
            mask = (alpha_channel == 0)  # Create a mask where removed part is True

            # Invert the mask if necessary (to have True for removed part and False for foreground)
            mask = ~mask

            # Convert the mask to a 1024x1024 list
            mask_list = mask.tolist()  # Convert the NumPy array to a list
            # Now mask_list contains the True/False values for removed/background parts

            return mask_list


    def covert_tracer_output_to_vitmatte_input2(self, mask_image): #image path: '~~/~~/~~.png'
        # Load your background-removed image
        #image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # Assuming the alpha channel (if present) represents transparency (0 = removed, 255 = retained)
        if mask_image.shape[2] == 4:  # Check if the image has an alpha channel
            alpha_channel = mask_image[:, :, 3]
            mask = (alpha_channel == 0)  # Create a mask where removed part is True

            # Invert the mask if necessary (to have True for removed part and False for foreground)
            mask = ~mask

            return mask


    #def cal_foreground(image_dir, alpha_dir):
    def cal_foreground(self, image_dir, alpha):
        """
        Calculate the foreground of the image.
        Input:
            image_dir: the directory of the image
            alpha_dir: the directory of the alpha matte
        Output:
            foreground: the foreground of the image, numpy array
        """
        image = Image.open(image_dir).convert('RGB')
        #alpha = Image.open(alpha_dir).convert('L')
        alpha = alpha.convert('L')
        alpha = F.to_tensor(alpha).unsqueeze(0)
        image = F.to_tensor(image).unsqueeze(0)
        foreground = image * alpha + (1 - alpha)
        foreground = foreground.squeeze(0).permute(1, 2, 0).numpy()

        return foreground


    def cal_foreground2(self, image_RGBA, alpha_np):
        """
        Calculate the foreground of the image with transparent background.
        Input:
            image_dir: the directory of the image
            alpha: the alpha mask as a PIL Image (1024, 1024)
        Output:
            foreground: the foreground of the image, PIL Image
        """
        #image = Image.open(image_dir).convert('RGBA')

        # Create a new blank RGBA image with the same size as the original image
        foreground = Image.new('RGBA', image_RGBA.size)

        # Set alpha values based on the provided alpha mask
        for y in range(image_RGBA.size[1]):
            for x in range(image_RGBA.size[0]):
                r, g, b, a_original = image_RGBA.getpixel((x, y))  # Get RGBA values from the original image

                # Multiply alpha channel with the provided alpha mask
                a = int(a_original * alpha_np[y, x] / 255)  # Scale alpha values to [0, 255]

                # Set the RGBA values for each pixel in the new image
                foreground.putpixel((x, y), (r, g, b, a))

        return foreground


    def infer_one_image(self, model, input, save_dir=None):
        """
        Infer the alpha matte of one image.
        Input:
            model: the trained model
            image: the input image
            trimap: the input trimap
        """
        with torch.no_grad():
            output = model(input)['phas'].flatten(0, 2)
            output = F.to_pil_image(output) #tensor to pil

            #output.save(opj(save_dir))
            #output.save(save_dir)

        #return None
        return output

#def get_data(image_dir, trimap_dir):
    def get_data(self, image, trimap):
        """
        Get the data of one image.
        Input:
            image_dir: the directory of the image
            trimap_dir: the directory of the trimap
        """
        #image = Image.open(image_dir).convert('RGB')
        image = Image.fromarray(image).convert('RGB')
        image = F.to_tensor(image).unsqueeze(0)
        #trimap = Image.open(trimap_dir).convert('L')
        trimap = Image.fromarray(trimap).convert('L')
        trimap = F.to_tensor(trimap).unsqueeze(0)

        return {
            'image': image,
            'trimap': trimap
        }


    def merge_new_bg(self, image_dir, bg_dir, alpha_dir):
        """
        Merge the alpha matte with a new background.
        Input:
            image_dir: the directory of the image
            bg_dir: the directory of the new background
            alpha_dir: the directory of the alpha matte
        """
        image = Image.open(image_dir).convert('RGB')
        bg = Image.open(bg_dir).convert('RGB')
        alpha = Image.open(alpha_dir).convert('L')
        image = F.to_tensor(image)
        bg = F.to_tensor(bg)
        bg = F.resize(bg, image.shape[-2:])
        alpha = F.to_tensor(alpha)
        new_image = image * alpha + bg * (1 - alpha)

        new_image = new_image.squeeze(0).permute(1, 2, 0).numpy()
        
        return new_image


    def run_vitmatte(self, model, input_x,input_x_RGBA, masks, erode_kernel_size, dilate_kernel_size, fg_box_threshold, fg_text_threshold, fg_caption, tr_box_threshold, tr_text_threshold, tr_caption = "glass, lens, crystal, diamond, bubble, bulb, web, grid"):
        #set_image(input_x, "RGB")

        # generate alpha matte
        torch.cuda.empty_cache()
        mask = masks.astype(np.uint8)*255
        trimap = self.generate_trimap(mask, erode_kernel_size, dilate_kernel_size).astype(np.float32)

        #input = get_data(input_x, trimap_dir) #save_dir = where to save alpha(mask)
        input = self.get_data(input_x, trimap)

        torch.cuda.empty_cache()
        alpha = self.infer_one_image(model, input) #alpha: PIL Image

        # Convert alpha to a numpy array
        alpha_np = alpha.convert('L')  # Convert to grayscale
        alpha_np = np.array(alpha_np)  # Convert to numpy array

        fg2 = self.cal_foreground2(Image.fromarray(input_x_RGBA), alpha_np) #vitmatte 결과이지만 배경이 흰색이 아닌 투명

        return fg2


    def vit(self, input_image, input_image_RGBA, mask):
        print("[Debug] torch.cuda.is_available: ",torch.cuda.is_available())
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        # #vitmatte_model = 'vit_b'
        # vitmatte_model = self.init_model(model='vitmatte-b', checkpoint='./checkpoint/ViTMatte_B_DIS.pth', device=device) #상대경로 확인 

        erode_kernel_size =10
        dilate_kernel_size=10
        
        fg_box_threshold=0.5
        fg_text_threshold=0.5
        fg_caption="subject"
        
        tr_box_threshold=0.5
        tr_text_threshold=0.5
        tr_caption= "glass, lens, crystal, diamond, bubble, bulb, web, grid"

        input_image_arr = np.array(input_image)
        input_image_arr_RGBA = np.array(input_image_RGBA)
        mask_image = np.array(mask)

        # Process the images
        # Your code to perform the operations using input_path and mask_path goes here
        
        # Example:
        mask = self.covert_tracer_output_to_vitmatte_input2(mask_image) #이때, 입력값 mask_image는 검흰 아직 아님..배경 제거만 된 실제 이미지
        foreground_alpha2 = self.run_vitmatte(self.vitmatte_model, input_image_arr, input_image_arr_RGBA, mask, erode_kernel_size, dilate_kernel_size, fg_box_threshold, fg_text_threshold, fg_caption, tr_box_threshold, tr_text_threshold, tr_caption)

        return foreground_alpha2


    def predict(
        self,
        images,
        remover,
        vitmodel,
        background,
        threshold
    ):
        self.remover = remover
        self.vitmatte_model = vitmodel

        resultimages = []
        pbar = ProgressBar(len(images))
        
        count = 0
        for (batch_number, image) in enumerate(images):
            # print("sizsss", len(images), image.size())
            i = 255. * image.cpu().numpy()
            image = np.clip(i, 0, 255).astype(np.uint8)
            
            # 打开图片
            image = Image.fromarray(image)
            image_RGBA = image.convert("RGBA")
            
            ispyrenet_output = self.ispyrenet(image)
            new_image = self.vit(image, image_RGBA, ispyrenet_output)
            resultimages.append(pil2tensor(new_image))
            
            count = count + 1
            pbar.update(1)
            if (count - 1) % 20 == 0:
                server = comfy.utils.get_server()
                # server.send_json('pt-RemoveBriai', {"value": count, "total": len(images)})
                server.send_sync('pt-RemovePViMatte', {"value": count, "total": len(images)}, server.client_id)
                
        return (torch.cat(resultimages, dim=0), count, )
        
        return output_path
    
NODE_CLASS_MAPPINGS = {
    "Pt-RembgPViMatte": RembgPViMatte,
    'Pt-LoadPViMatteModel': LoadPViMatteModel,
    'Pt-LoadPyrenetModel': LoadPyrenetModel,
}
