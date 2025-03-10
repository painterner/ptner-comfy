from .src.comfymath.convert import NODE_CLASS_MAPPINGS as convert_NCM
from .src.comfymath.color import NODE_CLASS_MAPPINGS as convert_COLORS
from .src.comfymath.masks import NODE_CLASS_MAPPINGS as convert_MASKS
from .src.comfymath.video import NODE_CLASS_MAPPINGS as convert_videos
from .src.comfymath.video_combine import NODE_CLASS_MAPPINGS as convert_video_combine
from .src.comfymath.rmbg_briai import NODE_CLASS_MAPPINGS as convert_BRIAI
from .src.comfymath.ViTMatte.rmbg_pyrenet_vimatte import NODE_CLASS_MAPPINGS as convert_pyrenet_vimatte
from .src.comfymath.Crop import NODE_CLASS_MAPPINGS as convert_crop

from .dis_background_removal.app_comfyui import NODE_CLASS_MAPPINGS as convert_ISNET



NODE_CLASS_MAPPINGS = {
    **convert_NCM,
    **convert_COLORS,
    **convert_MASKS,
    **convert_ISNET,
    **convert_videos,
    **convert_BRIAI,
    **convert_pyrenet_vimatte,
    **convert_video_combine,
    **convert_crop
}


def remove_cm_prefix(node_mapping: str) -> str:
    return node_mapping


NODE_DISPLAY_NAME_MAPPINGS = {key: key for key in NODE_CLASS_MAPPINGS}
