"""LTXV (Lightricks LTX-2) video generation nodes (bring-your-own-key).

Mirrors the official ComfyUI ``LtxvApiTextToVideo`` node (comfy_api_nodes/nodes_ltxv.py)
but calls the Lightricks LTX-2 API directly with the user's own API key instead of
routing through the ComfyOrg proxy.

The official node POSTs to ``/proxy/ltx/v1/text-to-video`` with ``as_binary=True``; the
proxy forwards the body verbatim to the real vendor, so the real endpoint is
``https://api.ltx.video/v1/text-to-video``. The request model
(``ExecuteTaskRequest``: prompt / model / duration / resolution / fps / generate_audio)
and the synchronous ``video/mp4`` binary response are taken directly from the official
source. No polling is needed — the MP4 bytes are returned in the response body.
"""

import aiohttp

from .utils import run_async, validate_prompt, video_from_bytes

# Real Lightricks LTX-2 vendor host (overridable for relays).
# Official node path is /proxy/ltx/v1/text-to-video -> real path is /v1/text-to-video.
LTXV_API_BASE = "https://api.ltx.video/v1"

# Display label -> vendor model id (mirrors official MODELS_MAP).
MODELS_MAP = {
    "LTX-2 (Pro)": "ltx-2-pro",
    "LTX-2 (Fast)": "ltx-2-fast",
}

MODEL_LABELS = list(MODELS_MAP.keys())
DURATION_OPTIONS = [6, 8, 10, 12, 14, 16, 18, 20]
RESOLUTION_OPTIONS = ["1920x1080", "2560x1440", "3840x2160"]
FPS_OPTIONS = [25, 50]


async def _post_video_bytes(url: str, headers: dict, body: dict, timeout: float) -> bytes:
    """POST a JSON body and return the raw binary (video/mp4) response."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers=headers,
            json=body,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            data = await resp.read()
            if resp.status >= 400:
                raise RuntimeError(
                    f"LTXV API error (HTTP {resp.status}): "
                    f"{data[:2000].decode('utf-8', 'replace')}"
                )
            return data


class BYOKey_LTXV_Text2Video:
    """LTXV (LTX-2) text-to-video. Output: VIDEO.

    Generates a professional-quality video from a text prompt with customizable
    duration, resolution and frame rate, optionally with AI-generated audio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Your Lightricks LTX API key (Bearer token)."}),
                "base_url": ("STRING", {"default": LTXV_API_BASE, "tooltip": "Vendor API base URL. Override for relays/proxies."}),
                "model": (MODEL_LABELS, {"default": MODEL_LABELS[0]}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": (DURATION_OPTIONS, {"default": 8}),
                "resolution": (RESOLUTION_OPTIONS, {"default": RESOLUTION_OPTIONS[0]}),
                "fps": (FPS_OPTIONS, {"default": 25}),
            },
            "optional": {
                "generate_audio": ("BOOLEAN", {"default": False, "tooltip": "When true, the generated video will include AI-generated audio matching the scene."}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600, "tooltip": "Maximum seconds to wait for the synchronous response."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "LTXV (Lightricks LTX-2) text-to-video. Professional-quality videos with customizable duration and resolution."

    def execute(
        self,
        api_key,
        base_url,
        model,
        prompt,
        duration,
        resolution,
        fps=25,
        generate_audio=False,
        max_wait=600,
    ):
        # Match official validation: validate_string(prompt, min_length=1, max_length=10000)
        prompt = validate_prompt(prompt, min_length=1)
        if len(prompt) > 10000:
            raise ValueError("Prompt must be at most 10,000 characters long.")

        if model not in MODELS_MAP:
            raise ValueError(f"Unknown model '{model}'. Choose one of: {MODEL_LABELS}")

        duration = int(duration)
        fps = int(fps)

        # Official constraint: durations over 10s require Fast model, 1920x1080, 25 fps.
        if duration > 10 and (model != "LTX-2 (Fast)" or resolution != "1920x1080" or fps != 25):
            raise ValueError(
                "Durations over 10s are only available for the Fast model at "
                "1920x1080 resolution and 25 FPS."
            )

        base = (base_url or LTXV_API_BASE).rstrip("/")
        url = f"{base}/text-to-video"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "video/mp4",
        }

        # Request body matches official ExecuteTaskRequest field-for-field.
        body = {
            "prompt": prompt,
            "model": MODELS_MAP[model],
            "duration": duration,
            "resolution": resolution,
            "fps": fps,
            "generate_audio": bool(generate_audio),
        }

        mp4_bytes = run_async(_post_video_bytes(url, headers, body, float(max_wait)))
        if not mp4_bytes:
            raise RuntimeError("LTXV returned an empty response body.")

        return (video_from_bytes(mp4_bytes),)
