"""BFL Flux image generation nodes."""
import time

import torch

from .utils import bytes_to_tensor, download_bytes, http_request, run_async, tensor_to_base64

BFL_API_BASE = "https://api.bfl.ml/v1"


class BYOKey_Flux:
    """Flux Pro image generation via BFL API (submit+poll).
    Output type: IMAGE (compatible with official BFL Flux nodes).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "BFL API key"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": (["flux-pro-1.1-ultra", "flux-pro-1.1", "flux-pro", "flux-dev"], {"default": "flux-pro-1.1-ultra"}),
                "aspect_ratio": ("STRING", {"default": "16:9", "tooltip": "e.g. 16:9, 1:1, 4:3"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**63 - 1}),
                "raw": ("BOOLEAN", {"default": False, "tooltip": "Less processed, more natural look"}),
                "image_prompt": ("IMAGE",),
                "image_prompt_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 30}),
                "max_wait": ("INT", {"default": 120, "min": 30, "max": 600}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "BFL Flux Pro image generation."

    def execute(self, api_key, prompt, model="flux-pro-1.1-ultra",
                aspect_ratio="16:9", seed=0, raw=False,
                image_prompt=None, image_prompt_strength=0.1,
                poll_interval=3, max_wait=120):
        headers = {"X-Key": api_key, "Content-Type": "application/json"}
        body = {"prompt": prompt, "seed": seed, "aspect_ratio": aspect_ratio, "raw": raw}
        if image_prompt is not None:
            body["image_prompt"] = tensor_to_base64(image_prompt)
            body["image_prompt_strength"] = round(image_prompt_strength, 2)

        url = f"{BFL_API_BASE}/{model}/generate"
        resp = run_async(http_request("POST", url, headers, body))
        poll_url = resp.get("polling_url") or resp.get("id")
        if not poll_url:
            raise RuntimeError(f"No polling_url in BFL response: {resp}")
        if not poll_url.startswith("http"):
            poll_url = f"{BFL_API_BASE}/get_result?id={poll_url}"

        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            status_resp = run_async(http_request("GET", poll_url, headers))
            status = status_resp.get("status", "").lower()
            if status in ("ready", "completed"):
                sample_url = status_resp.get("result", {}).get("sample") or status_resp.get("result", {}).get("url")
                if not sample_url:
                    raise RuntimeError(f"No sample URL in BFL result: {status_resp}")
                img_bytes = run_async(download_bytes(sample_url))
                return (bytes_to_tensor(img_bytes),)
            if status in ("error", "moderated", "request_moderated", "content_moderated", "task_not_found"):
                raise RuntimeError(f"BFL generation failed: {status_resp}")

        raise RuntimeError(f"BFL generation timed out after {max_wait}s")
