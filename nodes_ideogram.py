"""Ideogram image generation nodes."""
import base64

import torch

from .utils import bytes_to_tensor, download_bytes, http_request, run_async

IDEOGRAM_API_BASE = "https://api.ideogram.ai"


class BYOKey_Ideogram:
    """Ideogram image generation. Output: IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["V_3", "V_2", "V_1"], {"default": "V_3"}),
                "aspect_ratio": (["ASPECT_1_1", "ASPECT_16_9", "ASPECT_9_16", "ASPECT_4_3", "ASPECT_3_4"], {"default": "ASPECT_1_1"}),
                "magic_prompt": (["AUTO", "ON", "OFF"], {"default": "AUTO"}),
                "style_type": (["AUTO", "GENERAL", "REALISTIC", "DESIGN", "RENDER_3D", "ANIME"], {"default": "AUTO"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Ideogram image generation."

    def execute(self, api_key, prompt, model="V_3", aspect_ratio="ASPECT_1_1",
                magic_prompt="AUTO", style_type="AUTO", negative_prompt="", seed=0):
        headers = {"Api-Key": api_key, "Content-Type": "application/json"}
        body = {
            "image_request": {
                "prompt": prompt, "model": model, "aspect_ratio": aspect_ratio,
                "magic_prompt_option": magic_prompt, "style_type": style_type,
            }
        }
        if negative_prompt:
            body["image_request"]["negative_prompt"] = negative_prompt
        if seed > 0:
            body["image_request"]["seed"] = seed

        resp = run_async(http_request("POST", f"{IDEOGRAM_API_BASE}/generate", headers, body))
        data = resp.get("data", [])
        if not data:
            raise RuntimeError(f"No data in Ideogram response: {resp}")
        images = []
        for item in data:
            img_url = item.get("url")
            if img_url:
                images.append(bytes_to_tensor(run_async(download_bytes(img_url))))
        if not images:
            raise RuntimeError("No images returned")
        return (torch.cat(images, dim=0),)
