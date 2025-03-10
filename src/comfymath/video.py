import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
import json
import torch
import folder_paths
import psutil
import itertools
import cv2
import os
import comfy
import nodes
from comfy.utils import common_upscale, ProgressBar
from .utils import BIGMAX, DIMMAX, calculate_file_hash, get_sorted_dir_files_from_directory,\
        lazy_get_audio, hash_path, validate_path, strip_path, try_download_video, is_url, imageOrLatent, ffmpeg_path

video_extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov']

def cv_frame_generator(video, force_rate, frame_load_cap, skip_first_frames,
                       select_every_nth, meta_batch=None, unique_id=None):
    video_cap = cv2.VideoCapture(video)
    if not video_cap.isOpened() or not video_cap.grab():
        raise ValueError(f"{video} could not be loaded with cv.")

    # extract video metadata
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    width = 0

    if width <=0 or height <=0:
        _, frame = video_cap.retrieve()
        height, width, _ = frame.shape

    # set video_cap to look at start_index frame
    total_frame_count = 0
    total_frames_evaluated = -1
    frames_added = 0
    base_frame_time = 1 / fps
    prev_frame = None

    if force_rate == 0:
        target_frame_time = base_frame_time
    else:
        target_frame_time = 1/force_rate

    if total_frames > 0:
        if force_rate != 0:
            yieldable_frames = int(total_frames / fps * force_rate)
        else:
            yieldable_frames = total_frames
        if select_every_nth:
            yieldable_frames //= select_every_nth
        if frame_load_cap != 0:
            yieldable_frames =  min(frame_load_cap, yieldable_frames)
    else:
        yieldable_frames = 0
    yield (width, height, fps, duration, total_frames, target_frame_time, yieldable_frames)
    pbar = ProgressBar(yieldable_frames)
    time_offset=target_frame_time
    while video_cap.isOpened():
        if time_offset < target_frame_time:
            is_returned = video_cap.grab()
            # if didn't return frame, video has ended
            if not is_returned:
                break
            time_offset += base_frame_time
        if time_offset < target_frame_time:
            continue
        time_offset -= target_frame_time
        # if not at start_index, skip doing anything with frame
        total_frame_count += 1
        if total_frame_count <= skip_first_frames:
            continue
        else:
            total_frames_evaluated += 1

        # if should not be selected, skip doing anything with frame
        if total_frames_evaluated%select_every_nth != 0:
            continue

        # opencv loads images in BGR format (yuck), so need to convert to RGB for ComfyUI use
        # follow up: can videos ever have an alpha channel?
        # To my testing: No. opencv has no support for alpha
        unused, frame = video_cap.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # convert frame to comfyui's expected format
        # TODO: frame contains no exif information. Check if opencv2 has already applied
        frame = np.array(frame, dtype=np.float32)
        torch.from_numpy(frame).div_(255)
        if prev_frame is not None:
            inp  = yield prev_frame
            if inp is not None:
                #ensure the finally block is called
                return
        prev_frame = frame
        frames_added += 1
        if pbar is not None:
            pbar.update_absolute(frames_added, yieldable_frames)
        # if cap exists and we've reached it, stop processing frames
        if frame_load_cap > 0 and frames_added >= frame_load_cap:
            break
    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id)
        meta_batch.has_closed_inputs = True
    if prev_frame is not None:
        yield prev_frame

def target_size(width, height, force_size, custom_width, custom_height, downscale_ratio=8) -> tuple[int, int]:
    if downscale_ratio is None:
        downscale_ratio = 8
    if force_size == "Disabled":
        pass
    elif force_size == "Custom Width" or force_size.endswith('x?'):
        height *= custom_width/width
        width = custom_width
    elif force_size == "Custom Height" or force_size.startswith('?x'):
        width *= custom_height/height
        height = custom_height
    else:
        width = custom_width
        height = custom_height
    width = int(width/downscale_ratio + 0.5) * downscale_ratio
    height = int(height/downscale_ratio + 0.5) * downscale_ratio
    return (width, height)

def resized_cv_frame_gen(custom_width, custom_height, force_size, downscale_ratio, **kwargs):
    gen = cv_frame_generator(**kwargs)
    info =  next(gen)
    width, height = info[0], info[1]
    frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
    if kwargs.get('meta_batch', None) is not None:
        frames_per_batch = min(frames_per_batch, kwargs['meta_batch'].frames_per_batch)
    if force_size != "Disabled" or downscale_ratio is not None:
        new_size = target_size(width, height, force_size, custom_width, custom_height, downscale_ratio)
        yield (*info, new_size[0], new_size[1], False)
        if new_size[0] != width or new_size[1] != height:
            def rescale(frame):
                s = torch.from_numpy(np.fromiter(frame, np.dtype((np.float32, (height, width, 3)))))
                s = s.movedim(-1,1)
                s = common_upscale(s, new_size[0], new_size[1], "lanczos", "center")
                return s.movedim(1,-1).numpy()
            yield from itertools.chain.from_iterable(map(rescale, batched(gen, frames_per_batch)))
            return
    else:
        yield (*info, info[0], info[1], False)
    yield from gen
   
def batched(it, n):
    while batch := tuple(itertools.islice(it, n)):
        yield batch
         
def batched_vae_encode(images, vae, frames_per_batch):
    for batch in batched(images, frames_per_batch):
        image_batch = torch.from_numpy(np.array(batch))
        yield from vae.encode(image_batch).numpy()

def load_video(out_name, meta_batch=None, unique_id=None, memory_limit_mb=None, vae=None,
               generator=resized_cv_frame_gen, **kwargs):
    kwargs['video'] = strip_path(kwargs['video'])
    downscale_ratio = getattr(vae, "downscale_ratio", 8) if vae is not None else None
    if meta_batch is None or unique_id not in meta_batch.inputs:
        gen = generator(meta_batch=meta_batch, unique_id=unique_id, downscale_ratio=downscale_ratio, **kwargs)
        (width, height, fps, duration, total_frames, target_frame_time, yieldable_frames, new_width, new_height, alpha) = next(gen)

        if meta_batch is not None:
            meta_batch.inputs[unique_id] = (gen, width, height, fps, duration, total_frames, target_frame_time, yieldable_frames, new_width, new_height, alpha)
            if yieldable_frames:
                meta_batch.total_frames = min(meta_batch.total_frames, yieldable_frames)

    else:
        (gen, width, height, fps, duration, total_frames, target_frame_time, yieldable_frames, new_width, new_height, alpha) = meta_batch.inputs[unique_id]

    memory_limit = None
    if memory_limit_mb is not None:
        memory_limit *= 2 ** 20
    else:
        #TODO: verify if garbage collection should be performed here.
        #leaves ~128 MB unreserved for safety
        try:
            memory_limit = (psutil.virtual_memory().available + psutil.swap_memory().free) - 2 ** 27
            print("memory limit cal", memory_limit, psutil.virtual_memory().available, psutil.swap_memory().free)
        except:
            print("Failed to calculate available memory. Memory load limit has been disabled")
    if memory_limit is not None:
        if vae is not None:
            #space required to load as f32, exist as latent with wiggle room, decode to f32
            max_loadable_frames = int(memory_limit//(width*height*3*(4+4+1/10)))
        else:
            #TODO: use better estimate for when vae is not None
            #Consider completely ignoring for load_latent case?
            max_loadable_frames = int(memory_limit//(width*height*3*(.1)))
        if meta_batch is not None:
            if meta_batch.frames_per_batch > max_loadable_frames:
                raise RuntimeError(f"Meta Batch set to {meta_batch.frames_per_batch} frames but only {max_loadable_frames} can fit in memory")
            gen = itertools.islice(gen, meta_batch.frames_per_batch)
        else:
            original_gen = gen
            gen = itertools.islice(gen, max_loadable_frames)
       
    gen = itertools.islice(gen, 24*5)
    
    frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
    if vae is not None:
        gen = batched_vae_encode(gen, vae, frames_per_batch)
        vw,vh = new_width//downscale_ratio, new_height//downscale_ratio
        channels = getattr(vae, 'latent_channels', 4)
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (channels,vh,vw)))))
    else:
        #Some minor wizardry to eliminate a copy and reduce max memory by a factor of ~2
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (new_height, new_width, 4 if alpha else 3)))))
    # if meta_batch is None and memory_limit is not None:
    #     try:
    #         next(original_gen)
    #         raise RuntimeError(f"Memory limit hit after loading {len(images)} frames. Stopping execution.")
    #     except StopIteration:
    #         pass
    if len(images) == 0:
        raise RuntimeError("No frames generated")

    if 'start_time' in kwargs:
        start_time = kwargs['start_time']
    else:
        start_time = kwargs['skip_first_frames'] * target_frame_time
    target_frame_time *= kwargs.get('select_every_nth', 1)
    #Setup lambda for lazy audio capture
    audio = lazy_get_audio(kwargs['video'], start_time, kwargs['frame_load_cap']*target_frame_time)
    #Adjust target_frame_time for select_every_nth
    video_info = {
        "source_fps": fps,
        "source_frame_count": total_frames,
        "source_duration": duration,
        "source_width": width,
        "source_height": height,
        "loaded_fps": 1/target_frame_time,
        "loaded_frame_count": len(images),
        "loaded_duration": len(images) * target_frame_time,
        "loaded_width": new_width,
        "loaded_height": new_height,
    }
    if vae is None:
        return (images, len(images), audio, video_info)
    else:
        return ({"samples": images}, len(images), audio, video_info)


class LoadVideoUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1].lower() in video_extensions):
                    files.append(f)
        return {"required": {
                    "video": (sorted(files),),
                     "force_rate": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                     "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                     "custom_width": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                     "custom_height": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                     "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                     "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                     "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                     },
                "optional": {
                    "meta_batch": ("VHS_BatchManager",),
                    "vae": ("VAE",),
                },
                "hidden": {
                    "unique_id": "UNIQUE_ID"
                },
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = (imageOrLatent, "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")

    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        kwargs['video'] = folder_paths.get_annotated_filepath(strip_path(kwargs['video']))
        return load_video(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        image_path = folder_paths.get_annotated_filepath(video)
        return calculate_file_hash(image_path)

    @classmethod
    def VALIDATE_INPUTS(s, video, force_size, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True


class LoadVideoPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "isGif": ("BOOLEAN", {"default": False}),
                "video": ("STRING", {"placeholder": "X://insert/path/here.mp4", "vhs_path_extensions": video_extensions}),
                "out_name": ("STRING", {"placeholder": "here.mp4"}),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                 "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                 "custom_width": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                 "custom_height": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "Pt-Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = (imageOrLatent, "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")

    FUNCTION = "load_video"

    def load_video(self, isGif, **kwargs):
        print("pt load video", kwargs["video"])
        if kwargs['video'] is None or validate_path(kwargs['video']) != True:
            raise Exception("video is not a valid path: " + kwargs['video'])
        if is_url(kwargs['video']):
            print("is url, try downloading...")
            kwargs['video'] = try_download_video(kwargs['video'], kwargs['out_name']) or kwargs['video']
            
        if isGif:
            gifs = self.load_gif(kwargs["video"])
            os.remove(kwargs["video"])
            return (gifs, len(gifs), None, None)
        return load_video(**kwargs)
    
    def load_gif(self, video):
          # 打开 GIF 文件
        gif = Image.open(video)
        frames = []
        
        # 遍历 GIF 的每一帧
        try:
            while True:
                # 将帧转换为 RGBA 格式（确保包含 Alpha 通道）
                frame = gif.convert("RGBA")
                # 转换为 NumPy 数组
                frame_np = np.array(frame)
                # 转换为 PyTorch 张量，形状为 [height, width, channels]
                frame_tensor = torch.from_numpy(frame_np.astype(np.float32) / 255.0)
                # frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0  # 归一化到 [0, 1]
                frames.append(frame_tensor)
                # 移动到下一帧
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass  # GIF 的帧已全部处理完毕
        
        # 堆叠帧，形成四维张量 [frames, channels, height, width]
        gif_tensor = torch.stack(frames, dim=0)
        return gif_tensor

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        return validate_path(video, allow_none=True)

class RemoveVideoBackground:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "url": ("STRING", {"multiline": False} ),
                        "background": ("STRING", {"multiline": False,}),
                        "threshold": ("INT", {"default": 50, "min": 0, "max": 65536}),
                    },
                }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("IMAGE", "replace count")
    FUNCTION = "remove_similar_colors"

    CATEGORY = "image"
    
    def hex_to_rgb(self, hex_color):
        # 去掉开头的字符#
        hex_color = hex_color.lstrip('#')

        # 将每对字符转成整数
        rgb_tuple = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        return rgb_tuple
        
    # 计算颜色之间的欧几里得距离
    def color_distance(self, c1, c2):
        # return np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2))
        return torch.sqrt(torch.sum((c1 - c2) ** 2, dim=-1))

    # 移除与背景颜色相近的颜色
    def remove_similar_colors(self, images, background, threshold):
        
        background_color = self.hex_to_rgb(background)
        resultimages = []
        
        count = 0
        for (batch_number, image) in enumerate(images):
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
            pixels = torch.tensor(pixels, dtype=torch.float32, device='cuda')
            background_color = torch.tensor(background_color, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
            background_color = background_color.expand(pixels.shape[0], pixels.shape[1], 3)
            
            # 提取颜色通道（忽略alpha通道）
            pixel_colors = pixels[:, :, :3]
            
            # 计算每个像素颜色与背景颜色的距离
            distances = self.color_distance(pixel_colors, background_color)

            # 创建遮罩，判断哪些颜色与背景颜色的距离小于阈值
            mask = distances < threshold
            
            # 将与背景相近的颜色的alpha通道设置为0（透明）
            pixels[mask] = 0  # 将alpha通道设置为0
            # pixels[mask, 3] = 0  # 错误 这样写报错,  why ?
            # pixels[mask][:, 3] = 0  # 错误 这样写 赋值不会被影响, why ?

            pixels = pixels.cpu().numpy().astype(np.uint8)
            print(pixels.shape)

            # 创建新的图片
            new_image = Image.fromarray(pixels)
            resultimages.append(pil2tensor(new_image))
            
        return (torch.cat(resultimages, dim=0), count, )

  
class EmptyHunyuanLatentVideoMergeImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                              "latent": ("LATENT",),
                              "width": ("INT", {"default": 848, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "length": ("INT", {"default": 25, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/video"

    def generate(self, latent, width, height, length, batch_size=1):
        latent1 = latent["samples"]
        latent1 = latent1.repeat(1, 1, 3, 1, 1)
        shape = latent1.shape
        # length/4 意思是对时间进行下采样，因为一般视频都是有冗余的，况且要进行video to video 这样做可以让模型自己再预测4帧?
        latent2 = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, shape[3], shape[4]], device=comfy.model_management.intermediate_device())
        # latent2 = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        # comfy empty execution is 4, 但是comfy vae 编码出来的是16，vae编码出来可以需要更多的维度来表示输入的图像，但是empty image只需要4就够了
        # latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device) 
        
        print("hunyun latent shape merge", latent1.shape, latent2.shape)
        latent = torch.cat((latent1, latent2), dim=2)
        return ({"samples":latent}, )

    
NODE_CLASS_MAPPINGS = {
    "Pt-Load Video (Path)": LoadVideoPath,
    "Pt-Image + Emptys Latent": EmptyHunyuanLatentVideoMergeImage
    # "RemoveVideoBackground": RemoveVideoBackground
}