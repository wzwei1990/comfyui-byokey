"""Reve AI image generation nodes."""
from .utils import bytes_to_tensor, download_bytes, http_request, run_async, tensor_to_base64

REVE_API_BASE = "https://api.reve.art/v1"


class BYOKey_Reve_Image:
    """Reve AI image creation. Output: IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": ("STRING", {"default": "reve-v1"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "style": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Reve AI image creation."

    def execute(self, api_key, prompt, model="reve-v1",
                aspect_ratio="1:1", style="", seed=0):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {"prompt": prompt, "model": model, "aspect_ratio": aspect_ratio}
        if style:
            body["style"] = style
        if seed > 0:
            body["seed"] = seed
        resp = run_async(http_request("POST", f"{REVE_API_BASE}/images/create", headers, body))
        images = resp.get("images", [])
        if not images:
            raise RuntimeError(f"No images: {resp}")
        img_url = images[0].get("url")
        if not img_url:
            raise RuntimeError(f"No URL: {images[0]}")
        return (bytes_to_tensor(run_async(download_bytes(img_url))),)
