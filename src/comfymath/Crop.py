from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision.transforms as transforms
from comfy.utils import ProgressBar

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
def pil2tensor(image, squeeze):
    if squeeze:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
    else:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
  

class CircularCropNode:
    def __init__(self):
        # 这里可以初始化一些默认参数
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
          "required": {
              "images": ("IMAGE",),
              "clipFrames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
              "background": ("IMAGE", ),
              "cx": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100000.0, "step": 0.001}),
              "cy": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100000.0, "step": 0.001}),
              "resizeWidth": ("INT", {"default": 256, "min": 0, "max": 100000, "step": 1}),
              "resizeHeight": ("INT", {"default": 256, "min": 0, "max": 100000, "step": 1}),
              "radius": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100000.0, "step": 0.001}),
          },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK", "BBOX",)
    RETURN_NAMES = ("crop_image", "IMAG2", "IMAGE4", "IMAGE8", "crop_mask", "bbox",)
    FUNCTION = "crop"
    CATEGORY = "Painterner/Image"
    
    def processByTorch(self, image, center_x, center_y, radius):
        # Convert image to tensor
        transform_to_tensor = transforms.ToTensor()
        image_tensor = transform_to_tensor(image)

        # Create a mask
        height, width = image_tensor.shape[1], image_tensor.shape[2]
        Y, X = torch.meshgrid(torch.arange(height), torch.arange(width))
        dist_from_center = torch.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        mask = dist_from_center <= radius

        # Apply mask
        mask = mask.unsqueeze(0).expand_as(image_tensor)
        cropped_tensor = image_tensor * mask

        # Convert back to PIL Image
        transform_to_pil = transforms.ToPILImage()
        cropped_image = transform_to_pil(cropped_tensor)

        # Crop to bounding box of the circle
        bbox = (int(center_x - radius), int(center_y - radius), int(center_x + radius), int(center_y + radius))
        cropped_image = cropped_image.crop(bbox)

        return [cropped_image, mask, bbox]
    
    def process(self, image, background, center_x, center_y, resizeWidth, resizeHeight, radius):
        # Convert image to RGBA
        image = image.convert("RGBA")

        # Create a mask
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=255)

        # Apply mask
        # result = Image.new('RGBA', image.size)
        # result.paste(image, (0, 0), mask)

        # Crop to bounding box of the circle
        bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
        result = image.crop(bbox)
        
        # bgrSize = background.size
        # bgmask = Image.new('L', bgrSize.size, 0)
        # bgdraw = ImageDraw.Draw(bgmask)
        # bgdraw.ellipse((0, 0, bgrSize.size[0], bgrSize.size[1]), fill=255)
        # bgr = Image.new('RGBA', bgrSize.size)
        # bgr.paste(background, (0, 0), bgmask)
        # background = bgr
        
        result = result.resize((resizeWidth*2, resizeHeight*2), Image.Resampling.LANCZOS)
        background = background.resize((resizeWidth*2, resizeHeight*2), Image.Resampling.LANCZOS)
        combined = Image.alpha_composite(background, result)
        result = combined
        
        bgrSize = result
        bgmask = Image.new('L', bgrSize.size, 0)
        bgdraw = ImageDraw.Draw(bgmask)
        bgdraw.ellipse((2, 2, bgrSize.size[0]-2, bgrSize.size[1]-2), fill=255)
        bgr = Image.new('RGBA', bgrSize.size)
        bgr.paste(result, (0, 0), bgmask)
        result = bgr
        
        result1 = result.resize((resizeWidth, resizeHeight), Image.Resampling.LANCZOS)
        
        result2 = result.resize((resizeWidth//2, resizeHeight//2), Image.Resampling.LANCZOS)
        
        result3 = result.resize((resizeWidth//4, resizeHeight//4), Image.Resampling.LANCZOS)
        
        result4 = result.resize((resizeWidth//8, resizeHeight//8), Image.Resampling.LANCZOS)
        
        # 也裁剪mask
        cropedMask =  tensor2pil(self.mask2image(pil2tensor(mask, False))).crop(bbox)
        
        result1 = pil2tensor(result1, False)
        result2 = pil2tensor(result2, False)
        result3 = pil2tensor(result3, False)
        result4 = pil2tensor(result4, False)
        cropedMask = pil2tensor(cropedMask, False)
        return [result1, result2, result3, result4, cropedMask, bbox]
    
    def mask2image(self, mask):
        return mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    
    def crop(self, images, background, clipFrames, cx, cy, resizeWidth, resizeHeight, radius):
        bounding_boxes = []
        cropped_images = []
        cropped_images2 = []
        cropped_images3 = []
        cropped_images4 = []
        masks = []
        
        pbar = ProgressBar(images.shape[0])

        self.max_bbox_width = 0
        self.max_bbox_height = 0
        
        length = images.shape[0]
        images = images[clipFrames:(length - clipFrames)]
        
        background = tensor2pil(background[0])
        background = background.convert("RGBA")

        for i, (img, ) in enumerate(zip(images)): # (img, ) 与 (img ) 不一样. 因为 (img)等价于 img, 相当于非tuble了
            pilImg = tensor2pil(img)
            bg = background.copy()
            [result, result2, result3, result4, mask, bbox] = self.process(pilImg, bg, cx, cy, resizeWidth, resizeHeight, radius)
            cropped_images.append(result)
            cropped_images2.append(result2)
            cropped_images3.append(result3)
            cropped_images4.append(result4)
            bounding_boxes = [bbox]
            masks = [mask]
            pbar.update(1)
            
        cropped_image_out = torch.stack(cropped_images, dim=0)
        cropped_image_out2 = torch.stack(cropped_images2, dim=0)
        cropped_image_out3 = torch.stack(cropped_images3, dim=0)
        cropped_image_out4 = torch.stack(cropped_images4, dim=0)
        cropped_mask_out = torch.stack(masks, dim=0)   
        return (cropped_image_out, cropped_image_out2, cropped_image_out3, cropped_image_out4, cropped_mask_out[:, :, :, 0], bounding_boxes)

    # Example usage
    def example_usage():
        # Load an image
        image = Image.open("path_to_your_image.jpg")

        # Create node
        node = CircularCropNode()

        # Define crop parameters
        center_x, center_y, radius = 100, 100, 50

        # Process image
        cropped_image = node.process(image, center_x, center_y, radius)

        # Save or display the result
        cropped_image.show()

    # Uncomment to run the example
    # example_usage()


NODE_CLASS_MAPPINGS = {
    "Pt-CropCirlcle": CircularCropNode
}
