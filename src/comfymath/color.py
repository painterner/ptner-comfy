import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
import json
import torch

from colorspacious import cspace_convert
from colorspacious import deltaE


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
  

# https://zenn.dev/inaturam/articles/2b6b58de75c27f
class RgbToLab:
    def __init__(self):
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.
        return None

    def rgb2xyz(self, rgb):  # rgb from [0,1]
        # xyz_from_rgb = np.array([
        # [0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]
        # ])
        mask = (rgb > .04045).type(torch.FloatTensor)
        if (rgb.is_cuda):
            mask = mask.cuda()
        rgb = (((rgb + .055) / 1.055)**2.4) * mask + rgb / 12.92 * (1 - mask)
        x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] + .180423 * rgb[:, 2, :, :]
        y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]
        z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] + .950227 * rgb[:, 2, :, :]
        out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)
        return out

    def xyz2lab(self, xyz):
        # 0.95047, 1., 1.08883 # white
        sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
        if (xyz.is_cuda):
            sc = sc.cuda()
        xyz_scale = xyz / sc
        mask = (xyz_scale > .008856).type(torch.FloatTensor)
        if (xyz_scale.is_cuda):
            mask = mask.cuda()
        xyz_int = xyz_scale**(1 / 3.) * mask + (7.787 * xyz_scale + 16. / 116.) * (1 - mask)
        L = 116. * xyz_int[:, 1, :, :] - 16.
        a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
        b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
        out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)
        return out

    def rgb2lab(self, rgb):
        lab = self.xyz2lab(self.rgb2xyz(rgb))
        l_rs = (lab[:, [0], :, :] - self.l_cent) / self.l_norm
        ab_rs = lab[:, 1:, :, :] / self.ab_norm
        out = torch.cat((l_rs, ab_rs), dim=1)
        return out
    
class LabToRgb():
    def __init__(self):
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.
        return None

    def lab2xyz(self, lab):
        y_int = (lab[:, 0, :, :] + 16.) / 116.
        x_int = (lab[:, 1, :, :] / 500.) + y_int
        z_int = y_int - (lab[:, 2, :, :] / 200.)
        if (z_int.is_cuda):
            z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
        else:
            z_int = torch.max(torch.Tensor((0,)), z_int)
        out = torch.cat(
            (x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1)
        mask = (out > .2068966).type(torch.FloatTensor)
        if (out.is_cuda):
            mask = mask.cuda()
        out = (out**3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)
        sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
        sc = sc.to(out.device)
        out = out * sc
        return out

    def xyz2rgb(self, xyz):
        # array([[ 3.24048134, -1.53715152, -0.49853633],
        #        [-0.96925495,  1.87599   ,  0.04155593],
        #        [ 0.05564664, -0.20404134,  1.05731107]])
        r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
        g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + .04155593 * xyz[:, 2, :, :]
        b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]
        rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
        # sometimes reaches a small negative number, which causes NaNs
        rgb = torch.max(rgb, torch.zeros_like(rgb))
        mask = (rgb > .0031308).type(torch.FloatTensor)
        if (rgb.is_cuda):
            mask = mask.cuda()
        rgb = (1.055 * (rgb**(1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)
        return rgb

    def lab2rgb(self, lab_rs):
        l = lab_rs[:, [0], :, :] * self.l_norm + self.l_cent
        ab = lab_rs[:, 1:, :, :] * self.ab_norm
        lab = torch.cat((l, ab), dim=1)
        out = self.xyz2rgb(self.lab2xyz(lab))
        return out
    
rgb2lab = RgbToLab()
lab2rgb = LabToRgb()
        
        

class GetImageColorsMedian:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "images": ("IMAGE", ),
                        "top_n": ("INT", {"default": 10, "min": 0, "max": 65536}),
                    },
                "optional": {
                    "MASK": ("MASK",),
                 },
                }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("STRING", "TOP1")
    FUNCTION = "main_colors"

    CATEGORY = "image"
    
    def main_colors(self, images, MASK, top_n=10):
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu()
            i = i.numpy()
            image = np.clip(i, 0, 255).astype(np.uint8)
            
            image = torch.tensor(image, dtype=torch.float32)
            
            image = rgb2lab.rgb2lab(image.unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(0)
            
            if MASK is None:
                pass
            else:
                image = image[(MASK.permute(1,2,0) > 0.5).squeeze()] # 本应该MASK === 1.0, 但是浮点数利用 > 0.5就好

            # Reshape the image to be a list of pixels
            pixels = image.reshape((-1, 3))
            
            median_color = torch.tensor(np.median(pixels, axis=0))
            
            rgb = lab2rgb.lab2rgb(median_color.unsqueeze(0).unsqueeze(0).unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(0)
            
            rgb = rgb.squeeze()
                    
            # Convert the sorted colors to hex
            hex_colors = [rgb_to_hex(color) for color in [rgb]]
            # print("get i color clusters", hex_colors)
            
            return (json.dumps(hex_colors), hex_colors[0], )
    
class GetImageColors:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "images": ("IMAGE", ),
                        "top_n": ("INT", {"default": 10, "min": 0, "max": 65536}),
                    },
                "optional": {
                    "MASK": ("MASK",),
                 },
                }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("STRING", "TOP1")
    FUNCTION = "main_colors"

    CATEGORY = "image"
    
    def main_colors(self, images, MASK, top_n=10):
        MASK = MASK[0:1]
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu()
            
            # print(MASK.permute(1,2,0) > 0.5)
            # print((MASK.permute(1,2,0) > 0.5).shape)
            # print("ssss", image.shape)
            
            if MASK is None:
                image = i
            else:
                image = i[(MASK.permute(1,2,0) > 0.5).squeeze()] # 本应该MASK === 1.0, 但是浮点数利用 > 0.5就好
                
            i = image.numpy()
            image = np.clip(i, 0, 255).astype(np.uint8)
            
            # Reshape the image to be a list of pixels
            pixels = image.reshape((-1, 3))
            
            if pixels.shape[0] == 0:
                continue
            
            # print(MASK)
            # print(pixels)
            
            # Use KMeans to find top_n clusters in the pixel data
            kmeans = KMeans(n_clusters=top_n)
            kmeans.fit(pixels)
            
            # Get the colors (cluster centers) and sort by frequency
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Count the occurrence of each cluster label
            counts = np.bincount(labels)
            
            # Sort colors by count in descending order
            sorted_indices = np.argsort(-counts)
            sorted_colors = colors[sorted_indices]
            
            # Convert the sorted colors to hex
            hex_colors = [rgb_to_hex(color) for color in sorted_colors]
            # print("get i color clusters", hex_colors)
            
            return (json.dumps(hex_colors), hex_colors[0], )
        
        # 如果到头了还是全部被移除
        return (json.dumps(['#000000']), '#000000', )
    
class RGBA2RGB:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "images": ("IMAGE", ),
                    },
                }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "convert"

    CATEGORY = "image"
    
    def convert(self, images):
        resultimages = []
        for (batch_number, image) in enumerate(images):
            resultimages.append(image[..., :3])
            
        return (torch.cat(resultimages, dim=0).unsqueeze(0), )
            
             
    
class RemovePostProcessingThreshold:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "images": ("IMAGE", ),
                        "background": ("STRING", {"multiline": False,}),
                        "threshold": ("INT", {"default": 50, "min": 0, "max": 65536}),
                        "threshold2": ("INT", {"default": 43, "min": 0, "max": 65536}),
                    },
                }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("IMAGE", "color rgb", "replace count")
    FUNCTION = "remove_similar_colors"

    CATEGORY = "image"
    
    def hex_to_rgb(self, hex_color):
        # 去掉开头的字符#
        hex_color = hex_color.lstrip('#')

        # 将每对字符转成整数
        rgb_tuple = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        return rgb_tuple
    
    def rgb_to_lab(rgb_array):
        """
        将整个 RGB 数组转换为 CIE Lab 颜色空间
        :param rgb_array: RGB 图像数组，大小为 (height, width, 3)
        :return: 转换后的 Lab 数组
        """
        # 将 RGB 数组归一化到 [0, 1] 范围
        rgb_normalized = rgb_array / 255.0

        # 使用 colorspacious 库批量转换 RGB 到 Lab
        lab_array = cspace_convert(rgb_normalized, 'sRGB1', 'XYZ100'
        )
        # or by
        # lab_array = cspace_convert(rgb_array, 'sRGB255', 'XYZ100'
        # )
        
        return lab_array
    
    #     # 计算 CIE2000 色差
    # def cie2000_delta_e(lab1, lab2):
    #     """
    #     计算 CIE2000 色差
    #     :param lab1: 第一个 Lab 颜色
    #     :param lab2: 第二个 Lab 颜色
    #     :return: 两个 Lab 颜色之间的 CIE2000 色差
    #     """
    #     delta_e = deltaE(lab1, lab2, uniform_space="CIELab")
    #     return delta_e
    
    
    
    # 计算颜色之间的欧几里得距离
    def e_distance(self, c1, c2):
        # return np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2))
        return torch.sqrt(torch.sum((c1 - c2) ** 2, dim=-1))
        
    # 计算颜色之间的欧几里得距离
    def color_distance(self, c1, c2):
        # c1 = self.rgb_to_lab(c1)
        # c2 = self.rgb_to_lab(c2)
        
        r = deltaE(c1, c2, input_space="sRGB255")
        return r
        

    # 移除与背景颜色相近的颜色
    def remove_similar_colors(self, images, background, threshold, threshold2):
        resultimages = []
        resultimages2 = []
        
        count = 0
        for (batch_number, image) in enumerate(images):
            # print("background", background, self.hex_to_rgb(background))
            background_color = self.hex_to_rgb(background)
            jch = cspace_convert(background_color, "sRGB255", "JCh")
            lightness = jch[0] 
            # # 在 dell micro_frontend 中运行 streamlit run streamlit_colorspacious.py
            if lightness > threshold2:
                jchcolor = (255, 255, 255)
            else:
                jchcolor = (0, 0, 0)
            # print("sizsss", len(images), image.size())
            i = 255. * image.cpu().numpy()
            image = np.clip(i, 0, 255).astype(np.uint8)
            
            # 打开图片
            image = Image.fromarray(image).convert("RGBA")
            pixels = np.array(image)

    
            # # 遍历每个像素
            # for i in range(pixels.shape[0]):
            #     for j in range(pixels.shape[1]):
            #         current_color = pixels[i, j][:3]  # 忽略alpha通道

            #         # 计算当前颜色和背景颜色的色差
            #         dist = self.color_distance(current_color, background_color)

            #         # 如果色差小于阈值，将其设置为透明
            #         if dist < threshold:
            #             pixels[i, j][3] = 0  # 将alpha通道设置为0，变为透明
            #             count = count + 1
            
                # 将像素数据转换为tensor并传输到指定设备
            # pixels = torch.tensor(pixels, dtype=torch.float32, device='cuda')
            # background_color = torch.tensor(background_color, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
            # deltaE计算无法使用cuda ?
            pixels = torch.tensor(pixels, dtype=torch.float32, device='cuda')
            background_color = torch.tensor(background_color, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
            # print("sizsss", background_color.size(), pixels.size())
            background_color = background_color.expand(pixels.shape[0], pixels.shape[1], 3)
            blab = rgb2lab.rgb2lab(background_color[..., 0:3].unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(0)
            plab = rgb2lab.rgb2lab(pixels[..., 0:3].unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(0)
            
            # 提取颜色通道（忽略alpha通道）
            pixel_colors = pixels[:, :, :3]
            
            # 计算每个像素颜色与背景颜色的距离
            # distances = self.color_distance(pixel_colors, background_color)
            distances = self.e_distance(blab, plab)

            # 创建遮罩，判断哪些颜色与背景颜色的距离小于阈值
            mask = distances < threshold
            
            pis2 = pixels.clone()
            
            # 将与背景相近的颜色的alpha通道设置为0（透明）
            pis2[mask] = torch.tensor([*jchcolor, 1.], device='cuda')  # 将alpha通道设置为0
            pixels[mask] = 0  # 将alpha通道设置为0
            # pixels[mask, 3] = 0  # 错误 这样写报错,  why ?
            # pixels[mask][:, 3] = 0  # 错误 这样写 赋值不会被影响, why ?

            pis2 = pis2.cpu().numpy().astype(np.uint8)
            pixels = pixels.cpu().numpy().astype(np.uint8)
            # print(pixels.shape)

            # 创建新的图片
            new_image2 = Image.fromarray(pis2)
            new_image2 = new_image2.convert("RGB")
            new_image = Image.fromarray(pixels)
            
            resultimages2.append(pil2tensor(new_image2))
            resultimages.append(pil2tensor(new_image))
            
        return (torch.cat(resultimages, dim=0), torch.cat(resultimages2, dim=0), count, )

class CombineTwoVideos:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "images": ("IMAGE", ),
                        "images2": ("IMAGE", )
                    },
                }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("IMAGE", "replace count")
    FUNCTION = "start"

    CATEGORY = "image"
    
    # 将images 贴到 images2 上去 (images 内容显示优先法)
    def start(self, images, images2):
        resultimages = []
        
        count = 0
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            image = np.clip(i, 0, 255).astype(np.uint8)
            image = Image.fromarray(image).convert("RGBA")
            
            i2 = 255. * images2[batch_number].cpu().numpy()
            image2 = np.clip(i2, 0, 255).astype(np.uint8)
            image2 = Image.fromarray(image2).convert("RGBA")
            
            image2 = Image.alpha_composite(image2, image)
            # image2.paste(image) 这样操作结果image2 == image, 不对
            
            # 创建新的图片
            new_image = image2
            resultimages.append(pil2tensor(new_image))
            
        count = batch_number + 1
        return (torch.cat(resultimages, dim=0), count, )
    
    
NODE_CLASS_MAPPINGS = {
    "Pt-Get RGBA2RGB": RGBA2RGB,
    "Pt-Get Colors": GetImageColors,
    "Pt-Get Colors Median": GetImageColorsMedian,
    # 应用色差理论移除所有与背景色相似的颜色
    "Pt-RemovePostProcessingThreshold": RemovePostProcessingThreshold,
    'Pt-CombineTwoVideos': CombineTwoVideos
}
