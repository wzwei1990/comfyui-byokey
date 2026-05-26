"""Kling AI video generation nodes."""
import os
import time
import uuid

from .utils import download_bytes, http_request, run_async, tensor_to_base64

KLING_API_BASE = "https://api.klingai.com/v1"


class BYOKey_Kling_Text2Video:
    """Kling text-to-video generation.
    Output type: STRING (video file path).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Kling API key"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["kling-v2-master", "kling-v1-6", "kling-v1-5"], {"default": "kling-v2-master"}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "mode": (["std", "pro"], {"default": "std"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("Path to generated video file",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Kling AI text-to-video generation."

    def execute(self, api_key, prompt, model="kling-v2-master",
                duration="5", aspect_ratio="16:9", mode="std",
                negative_prompt="", seed=0, poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model_name": model, "prompt": prompt,
            "duration": duration, "aspect_ratio": aspect_ratio, "mode": mode,
        }
        if negative_prompt:
            body["negative_prompt"] = negative_prompt

        url = f"{KLING_API_BASE}/videos/text2video"
        resp = run_async(http_request("POST", url, headers, body))
        task_id = resp.get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in Kling response: {resp}")

        return (self._poll_and_download(task_id, headers, poll_interval, max_wait),)

    def _poll_and_download(self, task_id, headers, poll_interval, max_wait):
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            url = f"{KLING_API_BASE}/videos/text2video/{task_id}"
            resp = run_async(http_request("GET", url, headers))
            status = resp.get("data", {}).get("task_status", "").lower()
            if status in ("failed",):
                raise RuntimeError(f"Kling task failed: {resp}")
            if status in ("succeed",):
                videos = resp.get("data", {}).get("task_result", {}).get("videos", [])
                if not videos:
                    raise RuntimeError(f"No videos in Kling result: {resp}")
                return self._save_video(videos[0].get("url", ""))
        raise RuntimeError(f"Kling timed out after {max_wait}s")

    def _save_video(self, url):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filename = f"kling_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(output_dir, filename)
        video_bytes = run_async(download_bytes(url))
        with open(filepath, "wb") as f:
            f.write(video_bytes)
        return filepath


class BYOKey_Kling_Image2Video:
    """Kling image-to-video generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["kling-v2-master", "kling-v1-6", "kling-v1-5"], {"default": "kling-v2-master"}),
                "duration": (["5", "10"], {"default": "5"}),
                "mode": (["std", "pro"], {"default": "std"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("Path to generated video file",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Kling AI image-to-video generation."

    def execute(self, api_key, image, prompt, model="kling-v2-master",
                duration="5", mode="std", negative_prompt="", seed=0,
                poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        b64 = tensor_to_base64(image)
        body = {
            "model_name": model, "prompt": prompt,
            "duration": duration, "mode": mode,
            "image": f"data:image/png;base64,{b64}",
        }
        if negative_prompt:
            body["negative_prompt"] = negative_prompt

        url = f"{KLING_API_BASE}/videos/image2video"
        resp = run_async(http_request("POST", url, headers, body))
        task_id = resp.get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in Kling response: {resp}")

        t2v = BYOKey_Kling_Text2Video()
        return (t2v._poll_and_download(task_id, headers, poll_interval, max_wait),)
