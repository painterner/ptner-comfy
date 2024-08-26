from typing import Any, Mapping

class Textbox:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"a": ("STRING", {"multiline": True, "dynamicPrompts": True})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "func"
    CATEGORY = "math/conversion"

    def func(self, a: str) -> tuple[str]:
        return (a,)


NODE_CLASS_MAPPINGS = {
    "Text box": Textbox,
    "Text": Textbox,
}
