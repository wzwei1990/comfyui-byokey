"""Runway video and image generation nodes."""
import os
import time
import uuid

from .utils import bytes_to_tensor, download_bytes, http_request, run_async, tensor_to_base64

RUNWAY_API_BASE = "https://api.dev.runwayml.com/v1"


class BYOKey_Runway_Image2Video:
    """Runway image-to-video (Gen4 Turbo). Output: STRING (video path)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Runway API key"}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["gen4_turbo", "gen3a_turbo"], {"default": "gen4_turbo"}),
                "duration": (["5", "10"], {"default": "5"}),
                "ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Runway image-to-video generation."

    def execute(self, api_key, image, prompt, model="gen4_turbo",
                duration="5", ratio="16:9", seed=0, poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "X-Runway-Version": "2024-11-06"}
        b64 = tensor_to_base64(image)
        body = {
            "model": model, "promptImage": f"data:image/png;base64,{b64}",
            "promptText": prompt, "duration": int(duration), "ratio": ratio,
        }
        if seed > 0:
            body["seed"] = seed
        resp = run_async(http_request("POST", f"{RUNWAY_API_BASE}/image_to_video", headers, body))
        task_id = resp.get("id")
        if not task_id:
            raise RuntimeError(f"No task id: {resp}")
        return (self._poll(task_id, headers, poll_interval, max_wait),)

    def _poll(self, task_id, headers, poll_interval, max_wait):
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            resp = run_async(http_request("GET", f"{RUNWAY_API_BASE}/tasks/{task_id}", headers))
            status = resp.get("status", "").lower()
            if status in ("failed",):
                raise RuntimeError(f"Runway task failed: {resp}")
            if status in ("succeeded",):
                output_url = resp.get("output", [None])[0]
                if not output_url:
                    raise RuntimeError(f"No output URL: {resp}")
                return self._save(output_url)
        raise RuntimeError(f"Runway timed out after {max_wait}s")

    def _save(self, url):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, f"runway_{uuid.uuid4().hex[:8]}.mp4")
        with open(filepath, "wb") as f:
            f.write(run_async(download_bytes(url)))
        return filepath


class BYOKey_Runway_Text2Image:
    """Runway text-to-image. Output: IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Runway text-to-image."

    def execute(self, api_key, prompt, width=1024, height=1024, seed=0):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "X-Runway-Version": "2024-11-06"}
        body = {"promptText": prompt, "width": width, "height": height}
        if seed > 0:
            body["seed"] = seed
        resp = run_async(http_request("POST", f"{RUNWAY_API_BASE}/text_to_image", headers, body))
        output = resp.get("output", [])
        if not output:
            raise RuntimeError(f"No output: {resp}")
        img_bytes = run_async(download_bytes(output[0]))
        return (bytes_to_tensor(img_bytes),)
