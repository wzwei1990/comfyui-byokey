"""PixVerse video generation nodes."""
import os
import time
import uuid

from .utils import download_bytes, http_request, run_async, tensor_to_base64

PIXVERSE_API_BASE = "https://api.pixverse.ai/v2"


class BYOKey_PixVerse_Text2Video:
    """PixVerse text-to-video. Output: STRING (video path)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": ("STRING", {"default": "v3.5"}),
                "quality": (["360p", "540p", "720p", "1080p"], {"default": "720p"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 8}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3"], {"default": "16:9"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "PixVerse text-to-video generation."

    def execute(self, api_key, prompt, model="v3.5", quality="720p",
                duration=5, aspect_ratio="16:9", seed=0, poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {"prompt": prompt, "model_version": model, "quality": quality,
                "duration": duration, "aspect_ratio": aspect_ratio}
        if seed > 0:
            body["seed"] = seed
        resp = run_async(http_request("POST", f"{PIXVERSE_API_BASE}/video/text/generate", headers, body))
        task_id = resp.get("Resp", {}).get("task_id") or resp.get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id: {resp}")
        return (self._poll(task_id, headers, poll_interval, max_wait),)

    def _poll(self, task_id, headers, poll_interval, max_wait):
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            resp = run_async(http_request("GET", f"{PIXVERSE_API_BASE}/video/result/{task_id}", headers))
            status = (resp.get("Resp", {}).get("status") or resp.get("status", "")).lower()
            if status in ("failed", "error"):
                raise RuntimeError(f"PixVerse failed: {resp}")
            if status in ("successful", "completed", "done"):
                video_url = resp.get("Resp", {}).get("url") or resp.get("url")
                if not video_url:
                    raise RuntimeError(f"No video URL: {resp}")
                return self._save(video_url)
        raise RuntimeError(f"PixVerse timed out after {max_wait}s")

    def _save(self, url):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, f"pixverse_{uuid.uuid4().hex[:8]}.mp4")
        with open(filepath, "wb") as f:
            f.write(run_async(download_bytes(url)))
        return filepath
