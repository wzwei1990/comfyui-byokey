"""Luma AI image and video generation nodes."""
import os
import time
import uuid

from .utils import bytes_to_tensor, download_bytes, http_request, run_async, tensor_to_base64

LUMA_API_BASE = "https://api.lumalabs.ai/dream-machine/v1"


class BYOKey_Luma_Text2Video:
    """Luma Dream Machine text-to-video. Output: STRING (video path)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["luma-1.6", "luma-1.5"], {"default": "luma-1.6"}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "loop": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Luma Dream Machine text-to-video."

    def execute(self, api_key, prompt, model="luma-1.6", aspect_ratio="16:9",
                loop=False, seed=0, poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {"prompt": prompt, "model": model, "aspect_ratio": aspect_ratio, "loop": loop}
        resp = run_async(http_request("POST", f"{LUMA_API_BASE}/generations", headers, body))
        gen_id = resp.get("id")
        if not gen_id:
            raise RuntimeError(f"No generation id: {resp}")
        return (self._poll(gen_id, headers, poll_interval, max_wait),)

    def _poll(self, gen_id, headers, poll_interval, max_wait):
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            resp = run_async(http_request("GET", f"{LUMA_API_BASE}/generations/{gen_id}", headers))
            state = resp.get("state", "").lower()
            if state in ("failed",):
                raise RuntimeError(f"Luma failed: {resp}")
            if state in ("completed",):
                video_url = resp.get("assets", {}).get("video")
                if not video_url:
                    raise RuntimeError(f"No video URL: {resp}")
                return self._save(video_url)
        raise RuntimeError(f"Luma timed out after {max_wait}s")

    def _save(self, url):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, f"luma_{uuid.uuid4().hex[:8]}.mp4")
        with open(filepath, "wb") as f:
            f.write(run_async(download_bytes(url)))
        return filepath


class BYOKey_Luma_Image2Video:
    """Luma Dream Machine image-to-video. Output: STRING (video path)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["luma-1.6", "luma-1.5"], {"default": "luma-1.6"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Luma Dream Machine image-to-video."

    def execute(self, api_key, image, prompt, model="luma-1.6",
                aspect_ratio="16:9", seed=0, poll_interval=10, max_wait=600):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        b64 = tensor_to_base64(image)
        body = {
            "prompt": prompt, "model": model, "aspect_ratio": aspect_ratio,
            "keyframes": {"frame0": {"type": "image", "url": f"data:image/png;base64,{b64}"}},
        }
        resp = run_async(http_request("POST", f"{LUMA_API_BASE}/generations", headers, body))
        gen_id = resp.get("id")
        if not gen_id:
            raise RuntimeError(f"No generation id: {resp}")
        t2v = BYOKey_Luma_Text2Video()
        return (t2v._poll(gen_id, headers, poll_interval, max_wait),)
