"""Vidu video generation nodes."""
import os
import time
import uuid

from .utils import download_bytes, http_request, run_async, tensor_to_base64

VIDU_API_BASE = "https://api.vidu.com/v1"


class BYOKey_Vidu_Text2Video:
    """Vidu text-to-video. Output: STRING (video path)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["vidu-2.0", "vidu-1.5"], {"default": "vidu-2.0"}),
                "duration": ("INT", {"default": 4, "min": 4, "max": 8}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Vidu text-to-video generation."

    def execute(self, api_key, prompt, model="vidu-2.0", duration=4,
                resolution="720p", aspect_ratio="16:9", seed=0,
                poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
        body = {"model": model, "prompt": prompt, "duration": duration,
                "resolution": resolution, "aspect_ratio": aspect_ratio}
        if seed > 0:
            body["seed"] = seed
        resp = run_async(http_request("POST", f"{VIDU_API_BASE}/tasks", headers, body))
        task_id = resp.get("id") or resp.get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id: {resp}")
        return (self._poll(task_id, headers, poll_interval, max_wait),)

    def _poll(self, task_id, headers, poll_interval, max_wait):
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            resp = run_async(http_request("GET", f"{VIDU_API_BASE}/tasks/{task_id}", headers))
            state = resp.get("state", "").lower()
            if state in ("failed", "error"):
                raise RuntimeError(f"Vidu failed: {resp}")
            if state in ("success", "completed"):
                results = resp.get("results", [])
                if results and results[0].get("url"):
                    return self._save(results[0]["url"])
                raise RuntimeError(f"No video URL: {resp}")
        raise RuntimeError(f"Vidu timed out after {max_wait}s")

    def _save(self, url):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, f"vidu_{uuid.uuid4().hex[:8]}.mp4")
        with open(filepath, "wb") as f:
            f.write(run_async(download_bytes(url)))
        return filepath


class BYOKey_Vidu_Image2Video:
    """Vidu image-to-video. Output: STRING (video path)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["vidu-2.0", "vidu-1.5"], {"default": "vidu-2.0"}),
                "duration": ("INT", {"default": 4, "min": 4, "max": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Vidu image-to-video."

    def execute(self, api_key, image, prompt, model="vidu-2.0",
                duration=4, seed=0, poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
        b64 = tensor_to_base64(image)
        body = {"model": model, "prompt": prompt, "duration": duration,
                "image": f"data:image/png;base64,{b64}"}
        if seed > 0:
            body["seed"] = seed
        resp = run_async(http_request("POST", f"{VIDU_API_BASE}/tasks", headers, body))
        task_id = resp.get("id") or resp.get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id: {resp}")
        t2v = BYOKey_Vidu_Text2Video()
        return (t2v._poll(task_id, headers, poll_interval, max_wait),)
