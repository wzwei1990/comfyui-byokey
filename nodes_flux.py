"""BFL Flux image generation node (bring-your-own-key).

Faithfully mirrors the official ComfyUI ``FluxProUltraImageNode``
(``comfy_api_nodes/nodes_bfl.py`` + ``comfy_api_nodes/apis/bfl.py``) but calls the
Black Forest Labs API directly with your own key instead of going through the
ComfyOrg proxy. The official node POSTs to ``/proxy/bfl/flux-pro-1.1-ultra/generate``;
the proxy forwards the body verbatim to ``https://api.bfl.ml/v1/flux-pro-1.1-ultra/generate``,
so the official request/response models and endpoint path are the real BFL API.

Flow (matches official):
  1. POST <base_url>/flux-pro-1.1-ultra/generate  (header ``x-key``)
       -> {"id": ..., "polling_url": ..., "cost": ...}
  2. Poll ``polling_url`` (GET) until status == "Ready".
  3. Download ``result["sample"]`` and return it as an IMAGE tensor.
"""
import asyncio

from .utils import (
    bytes_to_tensor,
    download_bytes,
    http_request,
    run_async,
    tensor_to_base64,
    validate_prompt,
)

# Real BFL vendor host (default; overridable for relays). Mirrors the path the
# official ComfyOrg proxy forwards to: /proxy/bfl/<path> -> api.bfl.ml/v1/<path>.
BFL_API_BASE = "https://api.bfl.ml/v1"

# Official failure statuses from BFLStatus (apis/bfl.py). Compared case-insensitively.
_FAILED_STATUSES = {
    "task not found",
    "request moderated",
    "content moderated",
    "error",
}
# Official completion status (BFLStatus.ready).
_DONE_STATUSES = {"ready"}


def _parse_aspect_ratio(aspect_ratio: str) -> float:
    """Parse an 'X:Y' aspect ratio string into a float ratio."""
    try:
        w_str, h_str = aspect_ratio.split(":")
        w, h = float(w_str), float(h_str)
    except Exception as exc:
        raise ValueError(
            f"Invalid aspect ratio '{aspect_ratio}'. Expected format 'width:height', e.g. '16:9'."
        ) from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"Aspect ratio components must be positive, got '{aspect_ratio}'.")
    return w / h


def _validate_aspect_ratio(aspect_ratio: str) -> None:
    """Validate the aspect ratio is within BFL's 1:4 .. 4:1 bounds (official check)."""
    ratio = _parse_aspect_ratio(aspect_ratio)
    min_ratio = 1 / 4  # 1:4
    max_ratio = 4 / 1  # 4:1
    if ratio < min_ratio or ratio > max_ratio:
        raise ValueError(
            f"Aspect ratio '{aspect_ratio}' is out of range; must be between 1:4 and 4:1."
        )


class BYOKey_Flux:
    """Flux 1.1 [pro] Ultra image generation via the BFL API (submit + poll).

    Mirrors the official ``FluxProUltraImageNode``. Output type is IMAGE, matching
    the official node exactly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your Black Forest Labs API key (sent as the 'x-key' header).",
                }),
                "base_url": ("STRING", {
                    "default": BFL_API_BASE,
                    "tooltip": "BFL API base URL. Default is the real vendor host; "
                               "override for a relay/proxy.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt for the image generation",
                }),
                "prompt_upsampling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether to perform upsampling on the prompt. If active, "
                               "automatically modifies the prompt for more creative generation, "
                               "but results are nondeterministic (same seed will not produce "
                               "exactly the same result).",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "The random seed used for creating the noise.",
                }),
                "aspect_ratio": ("STRING", {
                    "default": "16:9",
                    "tooltip": "Aspect ratio of image; must be between 1:4 and 4:1.",
                }),
                "raw": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When True, generate less processed, more natural-looking images.",
                }),
            },
            "optional": {
                "image_prompt": ("IMAGE", {
                    "tooltip": "Optional reference image to guide generation.",
                }),
                "image_prompt_strength": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend between the prompt and the image prompt.",
                }),
                "poll_interval": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 60,
                    "tooltip": "Seconds between status polls.",
                }),
                "max_wait": ("INT", {
                    "default": 300,
                    "min": 30,
                    "max": 1800,
                    "tooltip": "Maximum seconds to wait for the image before timing out.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = (
        "Generates images using Flux Pro 1.1 Ultra via the BFL API based on prompt and "
        "resolution (bring your own key)."
    )

    def execute(
        self,
        api_key,
        base_url,
        prompt,
        prompt_upsampling=False,
        seed=0,
        aspect_ratio="16:9",
        raw=False,
        image_prompt=None,
        image_prompt_strength=0.1,
        poll_interval=3,
        max_wait=300,
    ):
        # Official validation: aspect ratio bounds, and prompt non-empty when no image_prompt.
        _validate_aspect_ratio(aspect_ratio)
        if image_prompt is None:
            validate_prompt(prompt)

        base = (base_url or BFL_API_BASE).rstrip("/")
        headers = {"x-key": api_key, "Content-Type": "application/json"}

        # Request body mirrors BFLFluxProUltraGenerateRequest field-for-field.
        body = {
            "prompt": prompt,
            "prompt_upsampling": prompt_upsampling,
            "seed": seed,
            "aspect_ratio": aspect_ratio,
            "safety_tolerance": 6,
            "output_format": "png",
            "raw": raw,
        }
        if image_prompt is not None:
            body["image_prompt"] = tensor_to_base64(image_prompt)
            body["image_prompt_strength"] = round(image_prompt_strength, 2)

        return (run_async(self._generate(base, headers, body, poll_interval, max_wait)),)

    async def _generate(self, base, headers, body, poll_interval, max_wait):
        # 1. Submit the generation job.
        submit_url = f"{base}/flux-pro-1.1-ultra/generate"
        initial = await http_request("POST", submit_url, headers, json_data=body)

        polling_url = initial.get("polling_url")
        if not polling_url:
            task_id = initial.get("id")
            if not task_id:
                raise RuntimeError(f"BFL submit response missing 'polling_url'/'id': {initial}")
            polling_url = f"{base}/get_result?id={task_id}"

        # 2. Poll until the task reaches a terminal status.
        elapsed = 0
        last = None
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            last = await http_request("GET", polling_url, headers)
            status = str(last.get("status") or "").lower()
            if status in _FAILED_STATUSES:
                raise RuntimeError(f"BFL generation failed. Status={last.get('status')}. Response={last}")
            if status in _DONE_STATUSES:
                result = last.get("result") or {}
                sample_url = result.get("sample")
                if not sample_url:
                    raise RuntimeError(f"BFL result missing 'sample' URL: {last}")
                # 3. Download the generated image -> IMAGE tensor (RGBA, matches official).
                img_bytes = await download_bytes(sample_url, headers=None)
                return bytes_to_tensor(img_bytes)

        raise RuntimeError(f"BFL generation timed out after {max_wait}s. Last response={last}")
