import os,sys
import torch
import hashlib
import numpy as np
from PIL import Image
import subprocess
import folder_paths as comfy_paths
from typing import Optional, Union, List


class cstr(str):
    class color:
        END = '\33[0m'
        BOLD = '\33[1m'
        ITALIC = '\33[3m'
        UNDERLINE = '\33[4m'
        BLINK = '\33[5m'
        BLINK2 = '\33[6m'
        SELECTED = '\33[7m'

        BLACK = '\33[30m'
        RED = '\33[31m'
        GREEN = '\33[32m'
        YELLOW = '\33[33m'
        BLUE = '\33[34m'
        VIOLET = '\33[35m'
        BEIGE = '\33[36m'
        WHITE = '\33[37m'

        BLACKBG = '\33[40m'
        REDBG = '\33[41m'
        GREENBG = '\33[42m'
        YELLOWBG = '\33[43m'
        BLUEBG = '\33[44m'
        VIOLETBG = '\33[45m'
        BEIGEBG = '\33[46m'
        WHITEBG = '\33[47m'

        GREY = '\33[90m'
        LIGHTRED = '\33[91m'
        LIGHTGREEN = '\33[92m'
        LIGHTYELLOW = '\33[93m'
        LIGHTBLUE = '\33[94m'
        LIGHTVIOLET = '\33[95m'
        LIGHTBEIGE = '\33[96m'
        LIGHTWHITE = '\33[97m'

        GREYBG = '\33[100m'
        LIGHTREDBG = '\33[101m'
        LIGHTGREENBG = '\33[102m'
        LIGHTYELLOWBG = '\33[103m'
        LIGHTBLUEBG = '\33[104m'
        LIGHTVIOLETBG = '\33[105m'
        LIGHTBEIGEBG = '\33[106m'
        LIGHTWHITEBG = '\33[107m'

        @staticmethod
        def add_code(name, code):
            if not hasattr(cstr.color, name.upper()):
                setattr(cstr.color, name.upper(), code)
            else:
                raise ValueError(f"'cstr' object already contains a code with the name '{name}'.")

    def __new__(cls, text):
        return super().__new__(cls, text)

    def __getattr__(self, attr):
        if attr.lower().startswith("_cstr"):
            code = getattr(self.color, attr.upper().lstrip("_cstr"))
            modified_text = self.replace(f"__{attr[1:]}__", f"{code}")
            return cstr(modified_text)
        elif attr.upper() in dir(self.color):
            code = getattr(self.color, attr.upper())
            modified_text = f"{code}{self}{self.color.END}"
            return cstr(modified_text)
        elif attr.lower() in dir(cstr):
            return getattr(cstr, attr.lower())
        else:
            raise AttributeError(f"'cstr' object has no attribute '{attr}'")

    def print(self, **kwargs):
        print(self, **kwargs)

#! MESSAGE TEMPLATES
cstr.color.add_code("msg", f"{cstr.color.BLUE}WAS Node Suite: {cstr.color.END}")
cstr.color.add_code("warning", f"{cstr.color.BLUE}WAS Node Suite {cstr.color.LIGHTYELLOW}Warning: {cstr.color.END}")
cstr.color.add_code("error", f"{cstr.color.RED}WAS Node Suite {cstr.color.END}Error: {cstr.color.END}")


MODELS_DIR =  comfy_paths.models_dir

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# PIL Hex
def pil2hex(image):
    return hashlib.sha256(np.array(tensor2pil(image)).astype(np.uint16).tobytes()).hexdigest()

# PIL to Mask
def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask

# Mask to PIL
def mask2pil(mask):
    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask_np = mask.cpu().numpy().astype('uint8')
    mask_pil = Image.fromarray(mask_np, mode="L")
    return mask_pil

# Tensor to SAM-compatible NumPy
def tensor2sam(image):
    # Convert tensor to numpy array in HWC uint8 format with pixel values in [0, 255]
    sam_image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    # Transpose the image to HWC format if it's in CHW format
    if sam_image.shape[0] == 3:
        sam_image = np.transpose(sam_image, (1, 2, 0))
    return sam_image

# SAM-compatible NumPy to tensor
def sam2tensor(image):
    # Convert the image to float32 and normalize the pixel values to [0, 1]
    float_image = image.astype(np.float32) / 255.0
    # Transpose the image from HWC format to CHW format
    chw_image = np.transpose(float_image, (2, 0, 1))
    # Convert the numpy array to a tensor
    tensor_image = torch.from_numpy(chw_image)
    return tensor_image

# Freeze PIP modules
def packages(versions=False):
    import sys
    import subprocess
    try:
        result = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], stderr=subprocess.STDOUT)
        lines = result.decode().splitlines()
        return [line if versions else line.split('==')[0] for line in lines]
    except subprocess.CalledProcessError as e:
        print("An error occurred while fetching packages:", e.output.decode())
        return []
    
def install_package(package, uninstall_first: Union[List[str], str] = None):
    if os.getenv("WAS_BLOCK_AUTO_INSTALL", 'False').lower() in ('true', '1', 't'):
        cstr(f"Preventing package install of '{package}' due to WAS_BLOCK_INSTALL env").msg.print()
    else:
        if uninstall_first is None:
            return

        if isinstance(uninstall_first, str):
            uninstall_first = [uninstall_first]

        cstr(f"Uninstalling {', '.join(uninstall_first)}..")
        subprocess.check_call([sys.executable, '-s', '-m', 'pip', 'uninstall', *uninstall_first])
        cstr("Installing package...").msg.print()
        subprocess.check_call([sys.executable, '-s', '-m', 'pip', '-q', 'install', package])

# IMAGE REMBG
# Sam model needs additional input, may need to be new node entirely
#   See: https://github.com/danielgatis/rembg/blob/main/USAGE.md#using-input-points
# u2net_cloth_seg model needs additional inputs, may create a new node
# An undocumented feature "putaplha" changes how alpha is applied, but does not appear to make a difference

class WAS_Remove_Rembg:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "transparency": ("BOOLEAN", {"default": True},),
                "model": (["u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use", "isnet-anime",
                           "sam", "birefnet-general", "birefnet-general-lite", "birefnet-portrait", "birefnet-dis",
                           "birefnet-hrsod", "birefnet-cod", "birefnet-massive"],),
                "post_processing": ("BOOLEAN", {"default": False}),
                "only_mask": ("BOOLEAN", {"default": False},),
                "alpha_matting": ("BOOLEAN", {"default": False},),
                "alpha_matting_foreground_threshold": ("INT", {"default": 240, "min": 0, "max": 255}),
                "alpha_matting_background_threshold": ("INT", {"default": 10, "min": 0, "max": 255}),
                "alpha_matting_erode_size": ("INT", {"default": 10, "min": 0, "max": 255}),
                "background_color": (["none", "black", "white", "magenta", "chroma green", "chroma blue"],),
                # "putalpha": ("BOOLEAN", {"default": True},),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "image_rembg"

    CATEGORY = "WAS Suite/Image/AI"

     # A helper function to convert from strings to logical boolean
     # Conforms to https://docs.python.org/3/library/stdtypes.html#truth-value-testing
     # With the addition of evaluating string representations of Falsey types
    def __convertToBool(self, x):

        # Evaluate string representation of False types
        if type(x) == str:
            x = x.strip()
            if (x.lower() == 'false'
                or x.lower() == 'none'
                or x == '0'
                or x == '0.0'
                or x == '0j'
                or x == "''"
                or x == '""'
                or x == "()"
                or x == "[]"
                or x == "{}"
                or x.lower() == "decimal(0)"
                or x.lower() == "fraction(0,1)"
                or x.lower() == "set()"
                or x.lower() == "range(0)"
            ):
                return False
            else:
                return True

        # Anything else will be evaluated by the bool function
        return bool(x)

    def image_rembg(
            self,
            images,
            transparency=True,
            model="u2net",
            alpha_matting=False,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            post_processing=False,
            only_mask=False,
            background_color="none",
            # putalpha = False,
    ):

        # ComfyUI will allow strings in place of booleans, validate the input.
        transparency = transparency if type(transparency) is bool else self.__convertToBool(transparency)
        alpha_matting = alpha_matting if type(alpha_matting) is bool else self.__convertToBool(alpha_matting)
        post_processing = post_processing if type(post_processing) is bool else self.__convertToBool(post_processing)
        only_mask = only_mask if type(only_mask) is bool else self.__convertToBool(only_mask)

        if "rembg" not in packages():
            install_package("rembg")

        from rembg import remove, new_session

        os.environ['U2NET_HOME'] = os.path.join(MODELS_DIR, 'rembg')
        os.makedirs(os.environ['U2NET_HOME'], exist_ok=True)

        # Set bgcolor
        bgrgba = None
        if background_color == "black":
            bgrgba = [0, 0, 0, 255]
        elif background_color == "white":
            bgrgba = [255, 255, 255, 255]
        elif background_color == "magenta":
            bgrgba = [255, 0, 255, 255]
        elif background_color == "chroma green":
            bgrgba = [0, 177, 64, 255]
        elif background_color == "chroma blue":
            bgrgba = [0, 71, 187, 255]
        else:
            bgrgba = None

        if transparency and bgrgba is not None:
            bgrgba[3] = 0

        batch_tensor = []
        for image in images:
            image = tensor2pil(image)
            batch_tensor.append(pil2tensor(
                remove(
                    image,
                    session=new_session(model),
                    post_process_mask=post_processing,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_size=alpha_matting_erode_size,
                    only_mask=only_mask,
                    bgcolor=bgrgba,
                    # putalpha = putalpha,
                )
                .convert(('RGBA' if transparency else 'RGB'))))
        batch_tensor = torch.cat(batch_tensor, dim=0)

        return (batch_tensor,)
