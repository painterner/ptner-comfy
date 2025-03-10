import torch
import numpy as np
from torchvision.transforms import Resize, CenterCrop
from PIL import Image
import math

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class imageCropFromMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
          "required": {
              "image": ("IMAGE",),
              "mask": ("MASK",),
              "image_crop_multi": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
              "mask_crop_multi": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
              "bbox_smooth_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
          },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOX",)
    RETURN_NAMES = ("crop_image", "crop_mask", "bbox",)
    FUNCTION = "crop"
    CATEGORY = "EasyUse/Image"

    def cropimage(self, original_images, masks, crop_size_mult, bbox_smooth_alpha):

      bounding_boxes = []
      cropped_images = []

      self.max_bbox_width = 0
      self.max_bbox_height = 0

      # First, calculate the maximum bounding box size across all masks
      curr_max_bbox_width = 0
      curr_max_bbox_height = 0
      for i, (mask, img) in enumerate(zip(masks, original_images)):
        _mask = tensor2pil(mask)
        non_zero_indices = np.nonzero(np.array(_mask))
        min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
        min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
        min_x = np.max([min_x-10, 0])
        max_x = np.min([max_x+10, img.shape[1]-1])
        min_y = np.max([min_y-10, 0])
        max_y = np.min([max_y+10, img.shape[0]-1])
        
        width = max_x - min_x
        height = max_y - min_y
        curr_max_bbox_width = max(curr_max_bbox_width, width)
        curr_max_bbox_height = max(curr_max_bbox_height, height)
        
        print("imageCropFromMask croped", min_y, min_x, max_y, max_x)
        cropped_img = img[min_y:max_y, min_x:max_x, :]
        bounding_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))
        cropped_images.append(cropped_img)

      return cropped_images, bounding_boxes
  
    def mask2image(self, mask):
        return mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

    def crop(self, image, mask, image_crop_multi, mask_crop_multi, bbox_smooth_alpha):
        cropped_images, bounding_boxes = self.cropimage(image, mask, image_crop_multi, bbox_smooth_alpha)
        cropped_mask_image, _ = self.cropimage(self.mask2image(mask), mask, mask_crop_multi, bbox_smooth_alpha)

        cropped_image_out = torch.stack(cropped_images, dim=0)
        cropped_mask_out = torch.stack(cropped_mask_image, dim=0)

        return (cropped_image_out, cropped_mask_out[:, :, :, 0], bounding_boxes)

    
NODE_CLASS_MAPPINGS = {
    "Pt-CropByMasks": imageCropFromMask
}
