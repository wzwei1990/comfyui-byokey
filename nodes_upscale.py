"""Image upscale nodes (Magnific, Topaz, WaveSpeed)."""
import base64

from .utils import bytes_to_tensor, http_request, run_async, tensor_to_base64


class BYOKey_Magnific:
    """Magnific AI image upscale. Output: IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
            },
            "optional": {
                "scale": (["2", "4", "8", "16"], {"default": "2"}),
                "creativity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Magnific AI image upscale."

    def execute(self, api_key, image, scale="2", creativity=0.5, prompt=""):
        import aiohttp
        from io import BytesIO
        from PIL import Image as PILImage
        import numpy as np

        async def _call():
            url = "https://api.magnific.ai/v1/upscale"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            b64 = tensor_to_base64(image)
            body = {"image": f"data:image/png;base64,{b64}", "scale": int(scale), "creativity": creativity}
            if prompt:
                body["prompt"] = prompt
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=body,
                                       timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status >= 400:
                        raise RuntimeError(f"Magnific error: {await resp.text()}")
                    data = await resp.json()
                    return data

        resp = run_async(_call())
        if resp.get("image"):
            img_data = resp["image"]
            if img_data.startswith("data:"):
                img_data = img_data.split(",", 1)[1]
            return (bytes_to_tensor(base64.b64decode(img_data)),)
        raise RuntimeError(f"No image in Magnific response: {list(resp.keys())}")
