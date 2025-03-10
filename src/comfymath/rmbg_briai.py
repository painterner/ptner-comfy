import os
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from comfy.utils import ProgressBar
import comfy
import numpy as np
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo

torch.set_float32_matmul_precision(["high", "highest"][0])

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
  

def process(image, birefnet, transform_image):    
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to('cuda')
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    
    # mask_array = np.array(mask)
    # print("get mask_array", mask_array)
    # revert_mask_array = 255 - mask_array
    # revert_mask = Image.fromarray(revert_mask_array)
    
    revert_image = image.copy()
    image.putalpha(mask)
    
    # revert_image.putalpha(revert_mask)
    
    black_image = Image.new("RGB", revert_image.size, (0, 0, 0))
    revert_image.paste(black_image, mask=mask)
    return image, mask, revert_image

class LoadBriaiModelType:
    def __init__(self) -> None:
        pass
    
bmtype = 'pt-LoadBriaiModelType'
    
class LoadBriaiModel:
    @classmethod
    def INPUT_TYPES(s):
        return {}

    RETURN_TYPES = (bmtype,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    def load(self):
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", trust_remote_code=True
        )
        birefnet.to("cuda")

        return (birefnet, )
    
class RemoveBriai:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "images": ("IMAGE", ),
                        "model": (bmtype, ),
                        "background": ("STRING", {"multiline": False,}),
                        "threshold": ("INT", {"default": 50, "min": 0, "max": 65536}),
                    },
                }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "REVERT IMAGE", "replace count")
    FUNCTION = "remove_similar_colors"

    CATEGORY = "image"
    
    # 移除与背景颜色相近的颜色
    def remove_similar_colors(self, images, model, background, threshold):
        resultimages = []
        resultimages_mask = []
        resultimages_revert = []
        pbar = ProgressBar(len(images))
        
        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        count = 0
        for (batch_number, image) in enumerate(images):
            # print("sizsss", len(images), image.size())
            i = 255. * image.cpu().numpy()
            image = np.clip(i, 0, 255).astype(np.uint8)
            
            # 打开图片
            image = Image.fromarray(image)
            
            new_image, mask, revert_image = process(image, model, transform_image)
            resultimages.append(pil2tensor(new_image))
            resultimages_mask.append(pil2tensor(mask))
            resultimages_revert.append(pil2tensor(revert_image))
            
            count = count + 1
            pbar.update(1)
            if (count - 1) % 20 == 0:
                server = comfy.utils.get_server()
                # server.send_json('pt-RemoveBriai', {"value": count, "total": len(images)})
                server.send_sync('pt-RemoveBriai', {"value": count, "total": len(images)}, server.client_id)
                
        return (
            torch.cat(resultimages, dim=0), 
            torch.cat(resultimages_mask, dim=0), 
            torch.cat(resultimages_revert, dim=0), 
            count)

NODE_CLASS_MAPPINGS = {
    "Pt-RemoveBriai": RemoveBriai,
    'Pt-LoadBriaiModel': LoadBriaiModel
}
