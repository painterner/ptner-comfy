from .src.comfymath.convert import NODE_CLASS_MAPPINGS as convert_NCM




NODE_CLASS_MAPPINGS = {
    **convert_NCM,
}


def remove_cm_prefix(node_mapping: str) -> str:
    return node_mapping


NODE_DISPLAY_NAME_MAPPINGS = {key: key for key in NODE_CLASS_MAPPINGS}
