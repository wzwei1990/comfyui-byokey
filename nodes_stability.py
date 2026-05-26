"""Stability AI image generation nodes."""
import base64

import torch

from .utils import bytes_to_tensor, http_request, run_async, tensor_to_base64

STABILITY_API_BASE = "https://api.stability.ai/v2beta"


class BYOKey_StabilityImage:
    """Stability AI image generation (SD3.5, Stable Image Ultra, etc).
    Output type: IMAGE (compatible with official Stability AI nodes).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Stability AI API key"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "endpoint": (["stable-image/generate/ultra", "stable-image/generate/sd3", "stable-image/generate/core"], {"default": "stable-image/generate/ultra"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21", "3:2", "2:3", "5:4", "4:5"], {"default": "1:1"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "img2img strength"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Stability AI image generation."

    def execute(self, api_key, prompt, endpoint="stable-image/generate/ultra",
                negative_prompt="", aspect_ratio="1:1", seed=0,
                image=None, strength=0.7):
        import aiohttp
        url = f"{STABILITY_API_BASE}/{endpoint}"
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

        # Stability uses multipart/form-data
        import asyncio
        import aiohttp
        from io import BytesIO

        async def _call():
            form = aiohttp.FormData()
            form.add_field("prompt", prompt)
            form.add_field("output_format", "png")
            form.add_field("aspect_ratio", aspect_ratio)
            if seed > 0:
                form.add_field("seed", str(seed))
            if negative_prompt:
                form.add_field("negative_prompt", negative_prompt)
            if image is not None:
                from PIL import Image as PILImage
                import numpy as np
                img_tensor = image[0] if len(image.shape) == 4 else image
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_img = PILImage.fromarray(img_np)
                buf = BytesIO()
                pil_img.save(buf, format="PNG")
                buf.seek(0)
                form.add_field("image", buf, filename="image.png", content_type="image/png")
                form.add_field("strength", str(strength))
                form.add_field("mode", "image-to-image")

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=form,
                                       timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        raise RuntimeError(f"Stability API error (HTTP {resp.status}): {text[:1000]}")
                    data = await resp.json()
                    return data

        resp = run_async(_call())
        if "image" in resp:
            img_bytes = base64.b64decode(resp["image"])
            return (bytes_to_tensor(img_bytes),)
        if "artifacts" in resp and resp["artifacts"]:
            img_bytes = base64.b64decode(resp["artifacts"][0]["base64"])
            return (bytes_to_tensor(img_bytes),)
        raise RuntimeError(f"No image in Stability response: {list(resp.keys())}")
