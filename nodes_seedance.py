"""ByteDance Seedance 2.0 video generation nodes."""
import logging
import os
import time
import uuid

from .utils import download_bytes, http_request, run_async, tensor_to_base64

logger = logging.getLogger(__name__)

BYTEPLUS_API_BASE = "https://api.byteplus.com/api/v3/contents/generations/tasks"


class BYOKey_Seedance2_Text2Video:
    """Seedance 2.0 text-to-video generation.
    Output type: STRING (video file path), compatible with video load nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "BytePlus/Volcengine API key"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["Seedance 2.0", "Seedance 2.0 Fast"], {"default": "Seedance 2.0"}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 10}),
                "ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], {"default": "16:9"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "watermark": ("BOOLEAN", {"default": False}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("Path to generated video file",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "ByteDance Seedance 2.0 text-to-video generation."

    MODEL_IDS = {
        "Seedance 2.0": "dreamina-seedance-2-0-260128",
        "Seedance 2.0 Fast": "dreamina-seedance-2-0-fast-260128",
    }

    def execute(self, api_key, prompt, model="Seedance 2.0", resolution="720p",
                duration=5, ratio="16:9", seed=0, watermark=False,
                poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        model_id = self.MODEL_IDS[model]
        body = {
            "model": model_id,
            "content": [{"type": "text", "text": prompt}],
            "resolution": resolution,
            "ratio": ratio,
            "duration": duration,
            "seed": seed,
            "watermark": watermark,
        }
        resp = run_async(http_request("POST", BYTEPLUS_API_BASE, headers, body))
        task_id = resp.get("id") or resp.get("data", {}).get("id")
        if not task_id:
            raise RuntimeError(f"No task_id in response: {resp}")

        return (self._poll_and_download(task_id, headers, poll_interval, max_wait),)

    def _poll_and_download(self, task_id, headers, poll_interval, max_wait):
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            resp = run_async(http_request("GET", f"{BYTEPLUS_API_BASE}/{task_id}", headers))
            status = resp.get("status", "").lower()
            if status in ("failed", "error"):
                raise RuntimeError(f"Seedance task failed: {resp}")
            if status in ("succeed", "completed", "done"):
                video_url = None
                content = resp.get("content", {})
                if isinstance(content, dict):
                    video_url = content.get("video_url")
                if not video_url:
                    raise RuntimeError(f"No video_url in response: {resp}")
                return self._save_video(video_url)
        raise RuntimeError(f"Timed out after {max_wait}s")

    def _save_video(self, url):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filename = f"seedance_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(output_dir, filename)
        video_bytes = run_async(download_bytes(url))
        with open(filepath, "wb") as f:
            f.write(video_bytes)
        return filepath


class BYOKey_Seedance2_Image2Video:
    """Seedance 2.0 image-to-video generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["Seedance 2.0", "Seedance 2.0 Fast"], {"default": "Seedance 2.0"}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("Path to generated video file",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "ByteDance Seedance 2.0 image-to-video."

    MODEL_IDS = BYOKey_Seedance2_Text2Video.MODEL_IDS

    def execute(self, api_key, image, prompt, model="Seedance 2.0",
                resolution="720p", duration=5, seed=0, poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        model_id = self.MODEL_IDS[model]
        b64 = tensor_to_base64(image)
        body = {
            "model": model_id,
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": {"image_url": f"data:image/png;base64,{b64}"}},
            ],
            "resolution": resolution,
            "duration": duration,
            "seed": seed,
        }
        resp = run_async(http_request("POST", BYTEPLUS_API_BASE, headers, body))
        task_id = resp.get("id") or resp.get("data", {}).get("id")
        if not task_id:
            raise RuntimeError(f"No task_id in response: {resp}")

        t2v = BYOKey_Seedance2_Text2Video()
        return (t2v._poll_and_download(task_id, headers, poll_interval, max_wait),)
