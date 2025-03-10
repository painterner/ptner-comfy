from typing import Any, Mapping

import os
import sys
import json
import hashlib
import traceback
import math
import time
import random
import logging

import torch

from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo

import numpy as np
import safetensors.torch

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview
import node_helpers

import base64
from io import BytesIO

class Textbox:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"a": ("STRING", {"multiline": True, "dynamicPrompts": True})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "func"
    CATEGORY = "math/conversion"

    def func(self, a: str) -> tuple[str]:
        return (a,)
    
class CheckpointLoaderDynamic:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": ("STRING",  {"multiline": False,}),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name):
        print("dynamic loading from", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_name, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]

class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image):
        if not folder_paths.exists_annotated_filepath(image):
            img = node_helpers.pillow(Image.open, BytesIO(base64.b64decode(image)))
        else:
            image_path = folder_paths.get_annotated_filepath(image)
            img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        m = hashlib.sha256()
        if not folder_paths.exists_annotated_filepath(image):
            m.update(base64.b64decode(image))
        else:
            image_path = folder_paths.get_annotated_filepath(image)   
            with open(image_path, 'rb') as f:
                m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        # if not folder_paths.exists_annotated_filepath(image):
        #     return "Invalid image file: {}".format(image)

        return True


class SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
        self.is_changed = True

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(s, latent):
        # 强制刷新，这在api请求时候总会在ws返回executed 事件。
        # 需要禁止rgtree的替换优化,否则会冲突失效。
        self.is_changed = not self.is_changed
        return self.is_changed

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }


class PreviewImage(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
        
class ReplaceSvgViewBox:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"svg_content": ("STRING",  {"multiline": True,}),
                     "new_width": ("INT", {"default": 1024, "min": 0, "max": 65536}),
                     "new_height": ("INT", {"default": 1024, "min": 0, "max": 65536})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("STRING",)
    FUNCTION = "modify_svg"
        
    def modify_svg(self, svg_content, new_width, new_height):
        import re
        # 匹配 width 和 height 属性 (假设它们的顺序不变)
        width_pattern = r'width="(\d+)"'
        height_pattern = r'height="(\d+)"'

        width_match = re.search(width_pattern, svg_content)
        height_match = re.search(height_pattern, svg_content)

        if width_match and height_match:
            original_width = width_match.group(1)
            original_height = height_match.group(1)

            # 替换 width 和 height 属性
            svg_content = re.sub(width_pattern, f'width="{new_width}"', svg_content, 1)
            svg_content = re.sub(height_pattern, f'height="{new_height}"', svg_content, 1)

            # 增加 viewBox 属性
            viewBox = f'viewBox="0 0 {original_width} {original_height}"'
            svg_content = re.sub(r'<svg([^>]*)>', rf'<svg\1 {viewBox}>', svg_content, 1)

            print("SVG文件已成功修改。")
        else:
            print("未能匹配到 width 或 height 属性。")
            
        return (svg_content, )

class CalcToCondition:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "width": ("INT", {"default": 1024, "min": 0, "max": 65536}),
                     "height": ("INT", {"default": 1024, "min": 0, "max": 65536})},
                }
    CATEGORY = "image"

    RETURN_TYPES = ("INT",)
    FUNCTION = "modify_svg"
        
    def modify_svg(self, width, height):
        if(width >= height):
            return (1, )
        else:
            return (2, )

NODE_CLASS_MAPPINGS = {
    "Text box": Textbox,
    "Text": Textbox,
    "Pt-Save Image": SaveImage,
    "Pt-Preview Image": PreviewImage,
    "Pt-Load Image": LoadImage,
    "Pt-Load Checkpoint": CheckpointLoaderDynamic,
    "Pt-ReplaceSvgViewBox": ReplaceSvgViewBox,
}
