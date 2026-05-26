"""Additional providers: Bria, HitPaw, Quiver, WaveSpeed, Topaz, ByteDance LLM, Sonilo."""
import base64
import os
import time
import uuid

import torch

from .utils import bytes_to_tensor, download_bytes, http_request, run_async, tensor_to_base64


class BYOKey_ByteDance_LLM:
    """ByteDance Seed LLM (Doubao). Output: STRING."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "doubao-1.5-pro-32k"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "base_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3"}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "images": ("IMAGE",),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 128000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey"
    DESCRIPTION = "ByteDance Seed/Doubao LLM."

    def execute(self, api_key, model, prompt, base_url="https://ark.cn-beijing.volces.com/api/v3",
                system_prompt="", images=None, max_tokens=4096, temperature=0.7, seed=0):
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if images is not None:
            content = [{"type": "text", "text": prompt}]
            for i in range(images.shape[0]):
                b64 = tensor_to_base64(images[i].unsqueeze(0))
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        data = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        resp = run_async(http_request("POST", url, headers, data))
        return (resp["choices"][0]["message"]["content"],)


class BYOKey_WaveSpeed_Upscale:
    """WaveSpeed image/video upscale. Output: IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
            },
            "optional": {
                "model": ("STRING", {"default": "wavespeed-ai/aura-sr"}),
                "scale": ("INT", {"default": 4, "min": 2, "max": 8}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 30}),
                "max_wait": ("INT", {"default": 120, "min": 30, "max": 600}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "WaveSpeed image upscale."

    def execute(self, api_key, image, model="wavespeed-ai/aura-sr", scale=4,
                poll_interval=3, max_wait=120):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        b64 = tensor_to_base64(image)
        body = {"input": {"image": f"data:image/png;base64,{b64}", "upscaling_factor": scale}}
        resp = run_async(http_request("POST", f"https://api.wavespeed.ai/api/v3/{model}", headers, body))
        request_id = resp.get("data", {}).get("id") or resp.get("id")
        if not request_id:
            raise RuntimeError(f"No request_id: {resp}")

        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            r = run_async(http_request("GET", f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result", headers))
            status = r.get("data", {}).get("status", "").lower()
            if status in ("failed", "error"):
                raise RuntimeError(f"WaveSpeed failed: {r}")
            if status in ("completed", "succeeded"):
                outputs = r.get("data", {}).get("outputs", [])
                if outputs:
                    img_bytes = run_async(download_bytes(outputs[0]))
                    return (bytes_to_tensor(img_bytes),)
                raise RuntimeError(f"No output: {r}")
        raise RuntimeError(f"WaveSpeed timed out after {max_wait}s")


class BYOKey_Bria_Image:
    """Bria AI image generation. Output: IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": ("STRING", {"default": "bria-2.3"}),
                "num_results": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Bria AI image generation."

    def execute(self, api_key, prompt, model="bria-2.3", num_results=1, seed=0):
        headers = {"api_token": api_key, "Content-Type": "application/json"}
        body = {"prompt": prompt, "num_results": num_results}
        if seed > 0:
            body["seed"] = seed
        resp = run_async(http_request("POST", f"https://engine.prod.bria-api.com/v1/text-to-image/{model}", headers, body))
        results = resp.get("result", [])
        if not results:
            raise RuntimeError(f"No results: {resp}")
        images = []
        for item in results:
            url = item.get("urls", [None])[0] or item.get("url")
            if url:
                images.append(bytes_to_tensor(run_async(download_bytes(url))))
        if not images:
            raise RuntimeError("No image URLs in response")
        return (torch.cat(images, dim=0),)
