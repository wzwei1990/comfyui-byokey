"""Recraft image generation nodes."""
import base64

import torch

from .utils import bytes_to_tensor, download_bytes, http_request, run_async, tensor_to_base64

RECRAFT_API_BASE = "https://external.api.recraft.ai/v1"


class BYOKey_Recraft_Image:
    """Recraft text-to-image. Output: IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "style": (["realistic_image", "digital_illustration", "logo_raster", "any"], {"default": "realistic_image"}),
                "size": (["1024x1024", "1536x1024", "1024x1536", "1820x1024", "1024x1820"], {"default": "1024x1024"}),
                "model": ("STRING", {"default": "recraftv3"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Recraft text-to-image generation."

    def execute(self, api_key, prompt, style="realistic_image",
                size="1024x1024", model="recraftv3", seed=0):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {"prompt": prompt, "style": style, "size": size, "model": model}
        resp = run_async(http_request("POST", f"{RECRAFT_API_BASE}/images/generations", headers, body))
        data = resp.get("data", [])
        if not data:
            raise RuntimeError(f"No data: {resp}")
        img_url = data[0].get("url")
        if not img_url:
            raise RuntimeError(f"No URL: {data[0]}")
        return (bytes_to_tensor(run_async(download_bytes(img_url))),)
