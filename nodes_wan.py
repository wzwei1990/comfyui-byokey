"""Wan image and video generation nodes (via Wavespeed/Fal)."""
import os
import time
import uuid

import torch

from .utils import bytes_to_tensor, download_bytes, http_request, run_async, tensor_to_base64

WAN_API_BASE = "https://api.wavespeed.ai/api/v3"


class BYOKey_Wan_Text2Video:
    """Wan text-to-video. Output: STRING (video path)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "WaveSpeed/Wan API key"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": ("STRING", {"default": "wan2.1-t2v-480p"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "size": (["832x480", "480x832", "624x624"], {"default": "832x480"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Wan text-to-video generation."

    def execute(self, api_key, prompt, model="wan2.1-t2v-480p",
                negative_prompt="", size="832x480", seed=0,
                poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {"prompt": prompt, "size": size}
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if seed > 0:
            body["seed"] = seed
        resp = run_async(http_request("POST", f"{WAN_API_BASE}/{model}", headers, body))
        task_id = resp.get("data", {}).get("id") or resp.get("id")
        if not task_id:
            raise RuntimeError(f"No task_id: {resp}")
        return (self._poll(task_id, headers, poll_interval, max_wait),)

    def _poll(self, task_id, headers, poll_interval, max_wait):
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            resp = run_async(http_request("GET", f"{WAN_API_BASE}/predictions/{task_id}/result", headers))
            status = resp.get("data", {}).get("status", "").lower()
            if status in ("failed", "error"):
                raise RuntimeError(f"Wan failed: {resp}")
            if status in ("completed", "succeeded"):
                outputs = resp.get("data", {}).get("outputs", [])
                if outputs:
                    return self._save(outputs[0])
                raise RuntimeError(f"No output: {resp}")
        raise RuntimeError(f"Wan timed out after {max_wait}s")

    def _save(self, url):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, f"wan_{uuid.uuid4().hex[:8]}.mp4")
        with open(filepath, "wb") as f:
            f.write(run_async(download_bytes(url)))
        return filepath


class BYOKey_Wan_Image2Video:
    """Wan image-to-video. Output: STRING (video path)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": ("STRING", {"default": "wan2.1-i2v-480p"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Wan image-to-video."

    def execute(self, api_key, image, prompt, model="wan2.1-i2v-480p",
                seed=0, poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        b64 = tensor_to_base64(image)
        body = {"prompt": prompt, "image": f"data:image/png;base64,{b64}"}
        if seed > 0:
            body["seed"] = seed
        resp = run_async(http_request("POST", f"{WAN_API_BASE}/{model}", headers, body))
        task_id = resp.get("data", {}).get("id") or resp.get("id")
        if not task_id:
            raise RuntimeError(f"No task_id: {resp}")
        t2v = BYOKey_Wan_Text2Video()
        return (t2v._poll(task_id, headers, poll_interval, max_wait),)
