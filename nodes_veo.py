"""Google Veo video generation nodes."""
import os
import time
import uuid

from .utils import download_bytes, http_request, run_async, tensor_to_base64

VEO_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class BYOKey_Veo:
    """Google Veo video generation. Output: STRING (video path)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Google AI API key"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["veo-3.0-generate-preview", "veo-2.0-generate-001"], {"default": "veo-3.0-generate-preview"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration": (["5s", "8s"], {"default": "8s"}),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "poll_interval": ("INT", {"default": 15, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Google Veo video generation."

    def execute(self, api_key, prompt, model="veo-3.0-generate-preview",
                aspect_ratio="16:9", duration="8s", image=None, seed=0,
                poll_interval=15, max_wait=600):
        url = f"{VEO_API_BASE}/models/{model}:predictLongRunning?key={api_key}"
        headers = {"Content-Type": "application/json"}
        instance = {"prompt": prompt}
        if image is not None:
            b64 = tensor_to_base64(image)
            instance["image"] = {"bytesBase64Encoded": b64}
        body = {
            "instances": [instance],
            "parameters": {"aspectRatio": aspect_ratio, "duration": duration},
        }
        resp = run_async(http_request("POST", url, headers, body))
        op_name = resp.get("name")
        if not op_name:
            raise RuntimeError(f"No operation name: {resp}")

        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            poll_url = f"{VEO_API_BASE}/{op_name}?key={api_key}"
            resp = run_async(http_request("GET", poll_url, headers))
            if resp.get("done"):
                videos = resp.get("response", {}).get("predictions", [])
                if videos and videos[0].get("videoUri"):
                    return (self._save(videos[0]["videoUri"]),)
                raise RuntimeError(f"No video in response: {resp}")
            if resp.get("error"):
                raise RuntimeError(f"Veo failed: {resp['error']}")
        raise RuntimeError(f"Veo timed out after {max_wait}s")

    def _save(self, url):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, f"veo_{uuid.uuid4().hex[:8]}.mp4")
        with open(filepath, "wb") as f:
            f.write(run_async(download_bytes(url)))
        return filepath
