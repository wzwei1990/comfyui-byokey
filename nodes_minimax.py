"""MiniMax (Hailuo) video generation nodes."""
import os
import time
import uuid

from .utils import download_bytes, http_request, run_async, tensor_to_base64

MINIMAX_API_BASE = "https://api.minimaxi.chat/v1"


class BYOKey_MiniMax_Text2Video:
    """MiniMax/Hailuo text-to-video generation.
    Output type: STRING (video file path).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "MiniMax API key"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["T2V-01-Director", "T2V-01", "video-01"], {"default": "T2V-01-Director"}),
                "resolution": (["1080p", "720p", "480p"], {"default": "1080p"}),
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
    DESCRIPTION = "MiniMax/Hailuo text-to-video generation."

    def execute(self, api_key, prompt, model="T2V-01-Director",
                resolution="1080p", duration=5, seed=0,
                poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {"model": model, "prompt": prompt}

        url = f"{MINIMAX_API_BASE}/video_generation"
        resp = run_async(http_request("POST", url, headers, body))
        task_id = resp.get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in MiniMax response: {resp}")

        return (self._poll_and_download(task_id, headers, poll_interval, max_wait),)

    def _poll_and_download(self, task_id, headers, poll_interval, max_wait):
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            url = f"{MINIMAX_API_BASE}/query/video_generation?task_id={task_id}"
            resp = run_async(http_request("GET", url, headers))
            status = resp.get("status", "").lower()
            if status in ("failed", "error"):
                raise RuntimeError(f"MiniMax task failed: {resp}")
            if status in ("success", "finished"):
                video_url = resp.get("file_id") or resp.get("video_url")
                if not video_url:
                    raise RuntimeError(f"No video URL in MiniMax result: {resp}")
                return self._save_video(video_url)
        raise RuntimeError(f"MiniMax timed out after {max_wait}s")

    def _save_video(self, url):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filename = f"minimax_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(output_dir, filename)
        video_bytes = run_async(download_bytes(url))
        with open(filepath, "wb") as f:
            f.write(video_bytes)
        return filepath


class BYOKey_MiniMax_Image2Video:
    """MiniMax/Hailuo image-to-video generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["I2V-01-Director", "I2V-01", "video-01"], {"default": "I2V-01-Director"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("Path to generated video file",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "MiniMax/Hailuo image-to-video generation."

    def execute(self, api_key, image, prompt, model="I2V-01-Director",
                seed=0, poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        b64 = tensor_to_base64(image)
        body = {"model": model, "prompt": prompt, "first_frame_image": f"data:image/png;base64,{b64}"}

        url = f"{MINIMAX_API_BASE}/video_generation"
        resp = run_async(http_request("POST", url, headers, body))
        task_id = resp.get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in MiniMax response: {resp}")

        t2v = BYOKey_MiniMax_Text2Video()
        return (t2v._poll_and_download(task_id, headers, poll_interval, max_wait),)
