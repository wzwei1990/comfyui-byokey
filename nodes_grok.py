"""Grok (xAI) image and video generation nodes."""
import base64
import os
import time
import uuid

import torch

from .utils import bytes_to_tensor, download_bytes, http_request, run_async, tensor_to_base64

GROK_API_BASE = "https://api.x.ai/v1"


class BYOKey_Grok_Image:
    """Grok image generation. Output: IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": ("STRING", {"default": "grok-2-image"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "xAI Grok image generation."

    def execute(self, api_key, prompt, model="grok-2-image", n=1, seed=0):
        url = f"{GROK_API_BASE}/images/generations"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {"model": model, "prompt": prompt, "n": n, "response_format": "b64_json"}
        resp = run_async(http_request("POST", url, headers, body))
        images = []
        for item in resp.get("data", []):
            if item.get("b64_json"):
                images.append(bytes_to_tensor(base64.b64decode(item["b64_json"])))
            elif item.get("url"):
                images.append(bytes_to_tensor(run_async(download_bytes(item["url"]))))
        if not images:
            raise RuntimeError(f"No images in Grok response: {resp}")
        return (torch.cat(images, dim=0),)


class BYOKey_Grok_Video:
    """Grok video generation. Output: STRING (video path)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": ("STRING", {"default": "grok-2-video"}),
                "image": ("IMAGE",),
                "duration": ("INT", {"default": 5, "min": 3, "max": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "xAI Grok video generation."

    def execute(self, api_key, prompt, model="grok-2-video", image=None,
                duration=5, seed=0, poll_interval=10, max_wait=600):
        import time
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {"model": model, "prompt": prompt, "duration": duration}
        if image is not None:
            body["image"] = f"data:image/png;base64,{tensor_to_base64(image)}"
        resp = run_async(http_request("POST", f"{GROK_API_BASE}/videos/generations", headers, body))
        task_id = resp.get("id")
        if not task_id:
            raise RuntimeError(f"No task id: {resp}")

        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            r = run_async(http_request("GET", f"{GROK_API_BASE}/videos/generations/{task_id}", headers))
            status = r.get("status", "").lower()
            if status in ("failed",):
                raise RuntimeError(f"Grok video failed: {r}")
            if status in ("completed", "succeeded"):
                video_url = r.get("url") or r.get("video_url")
                if video_url:
                    filepath = os.path.join(
                        __import__("folder_paths").get_output_directory(),
                        f"grok_{uuid.uuid4().hex[:8]}.mp4"
                    )
                    with open(filepath, "wb") as f:
                        f.write(run_async(download_bytes(video_url)))
                    return (filepath,)
                raise RuntimeError(f"No video URL: {r}")
        raise RuntimeError(f"Grok video timed out after {max_wait}s")
