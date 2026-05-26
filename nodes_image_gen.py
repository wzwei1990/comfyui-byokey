"""Image generation node (OpenAI Images API compatible)."""
import base64

import torch

from .utils import bytes_to_tensor, download_bytes, http_request, run_async, tensor_to_base64


class BYOKey_ImageGen:
    """Image generation via OpenAI Images API (DALL-E 3, GPT Image, etc).
    Drop-in replacement for OpenAI DALL-E / GPT Image nodes — same output type (IMAGE).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "model": ("STRING", {"default": "gpt-image-2"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "size": (["auto", "1024x1024", "1024x1536", "1536x1024", "2048x2048", "2048x1152", "1152x2048", "3840x2160", "2160x3840", "1792x1024", "1024x1792"], {"default": "auto"}),
                "quality": (["auto", "low", "medium", "high", "standard", "hd"], {"default": "auto"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey"
    DESCRIPTION = "Image generation via OpenAI-compatible Images API."

    def execute(self, api_key, base_url, model, prompt,
                size="1024x1024", quality="auto", n=1, image=None, seed=0):
        url = f"{base_url.rstrip('/')}/images/generations"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {"model": model, "prompt": prompt, "n": n}
        if size != "auto":
            data["size"] = size
        if quality != "auto":
            data["quality"] = quality

        if image is not None:
            url = f"{base_url.rstrip('/')}/images/edits"
            data["image"] = f"data:image/png;base64,{tensor_to_base64(image)}"

        resp = run_async(http_request("POST", url, headers, data))
        images = []
        for item in resp.get("data", []):
            img_tensor = run_async(self._get_image(item))
            images.append(img_tensor)

        if not images:
            raise RuntimeError("No images returned from API")
        return (torch.cat(images, dim=0),)

    async def _get_image(self, item: dict) -> torch.Tensor:
        if item.get("b64_json"):
            img_bytes = base64.b64decode(item["b64_json"])
            return bytes_to_tensor(img_bytes)
        img_url = item.get("url", "")
        if not img_url:
            raise RuntimeError("API returned neither b64_json nor url")
        img_bytes = await download_bytes(img_url)
        return bytes_to_tensor(img_bytes)
