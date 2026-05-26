"""LTXV video generation nodes."""
import os
import time
import uuid

from .utils import download_bytes, http_request, run_async, tensor_to_base64

LTXV_API_BASE = "https://api.ltx.studio/v1"


class BYOKey_LTXV_Text2Video:
    """LTX Video text-to-video. Output: STRING (video path)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": ("STRING", {"default": "ltx-video"}),
                "width": ("INT", {"default": 768, "min": 256, "max": 1920, "step": 64}),
                "height": ("INT", {"default": 512, "min": 256, "max": 1080, "step": 64}),
                "num_frames": ("INT", {"default": 97, "min": 1, "max": 257}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "LTX Video text-to-video."

    def execute(self, api_key, prompt, model="ltx-video",
                width=768, height=512, num_frames=97, seed=0,
                poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {"prompt": prompt, "model": model, "width": width,
                "height": height, "num_frames": num_frames}
        if seed > 0:
            body["seed"] = seed
        resp = run_async(http_request("POST", f"{LTXV_API_BASE}/generations", headers, body))
        task_id = resp.get("id") or resp.get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id: {resp}")

        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            resp = run_async(http_request("GET", f"{LTXV_API_BASE}/generations/{task_id}", headers))
            status = resp.get("status", "").lower()
            if status in ("failed", "error"):
                raise RuntimeError(f"LTXV failed: {resp}")
            if status in ("completed", "succeeded"):
                video_url = resp.get("video_url") or resp.get("output", {}).get("url")
                if video_url:
                    return (self._save(video_url),)
                raise RuntimeError(f"No video URL: {resp}")
        raise RuntimeError(f"LTXV timed out after {max_wait}s")

    def _save(self, url):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, f"ltxv_{uuid.uuid4().hex[:8]}.mp4")
        with open(filepath, "wb") as f:
            f.write(run_async(download_bytes(url)))
        return filepath
