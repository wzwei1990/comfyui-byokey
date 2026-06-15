"""Kling AI video generation nodes (Bring Your Own Key).

Faithfully mirrors the official ComfyUI Kling API nodes (``KlingTextToVideoNode``
and ``KlingImage2VideoNode``) but calls the Kling API host directly with a
user-supplied API key instead of going through the ComfyOrg proxy.

Ground truth (request/response models, endpoint paths, defaults, ranges):
- comfy_api_nodes/nodes_kling.py
- comfy_api_nodes/apis/__init__.py (KlingText2VideoRequest/Response,
  KlingImage2VideoRequest/Response, KlingVideoGen* enums, KlingTaskStatus)
- embedded-docs KlingTextToVideoNode / KlingImage2VideoNode

Submit returns ``data.task_id``; poll ``data.task_status`` (submitted /
processing / succeed / failed) until done; the result video is at
``data.task_result.videos[0].url``. Output type is VIDEO (plus video_id and
duration strings) exactly like the official nodes.
"""

from .utils import (
    download_video_output,
    http_request,
    poll_until,
    run_async,
    tensor_to_base64,
    validate_prompt,
)

# Default Kling API host. Overridable for relays/proxies. The official nodes hit
# ``/proxy/kling/v1/...`` on the ComfyOrg proxy, which forwards verbatim to the
# real Kling host (api.klingai.com) under ``/v1/...``.
KLING_API_BASE = "https://api.klingai.com"
KLING_API_VERSION = "v1"

PATH_TEXT_TO_VIDEO = f"/{KLING_API_VERSION}/videos/text2video"
PATH_IMAGE_TO_VIDEO = f"/{KLING_API_VERSION}/videos/image2video"

# Prompt length limits (official MAX_PROMPT_LENGTH_T2V / MAX_PROMPT_LENGTH_I2V).
MAX_PROMPT_LENGTH_T2V = 2500
MAX_PROMPT_LENGTH_I2V = 500

# Mode combo for text-to-video: label -> (mode, duration, model_name).
# Matches MODE_TEXT2VIDEO in the official node, in the same order. The official
# default is index 8 ("pro mode / 5s duration / kling-v2-5-turbo").
MODE_TEXT2VIDEO = {
    "standard mode / 5s duration / kling-v1-6": ("std", "5", "kling-v1-6"),
    "standard mode / 10s duration / kling-v1-6": ("std", "10", "kling-v1-6"),
    "pro mode / 5s duration / kling-v2-master": ("pro", "5", "kling-v2-master"),
    "pro mode / 10s duration / kling-v2-master": ("pro", "10", "kling-v2-master"),
    "standard mode / 5s duration / kling-v2-master": ("std", "5", "kling-v2-master"),
    "standard mode / 10s duration / kling-v2-master": ("std", "10", "kling-v2-master"),
    "pro mode / 5s duration / kling-v2-1-master": ("pro", "5", "kling-v2-1-master"),
    "pro mode / 10s duration / kling-v2-1-master": ("pro", "10", "kling-v2-1-master"),
    "pro mode / 5s duration / kling-v2-5-turbo": ("pro", "5", "kling-v2-5-turbo"),
    "pro mode / 10s duration / kling-v2-5-turbo": ("pro", "10", "kling-v2-5-turbo"),
}
_T2V_MODES = list(MODE_TEXT2VIDEO.keys())
_T2V_DEFAULT_MODE = _T2V_MODES[8]

# Image-to-video options (official KlingImage2VideoNode + embedded doc).
I2V_MODEL_NAMES = [
    "kling-v2-master",
    "kling-v2-1-master",
    "kling-v2-5-turbo",
    "kling-v2-1",
    "kling-v1-6",
    "kling-v1-5",
    "kling-v1-4",
    "kling-v1-0",
]
ASPECT_RATIOS = ["16:9", "9:16", "1:1"]
DURATIONS = ["5", "10"]
MODES = ["std", "pro"]

# KlingTaskStatus enum values.
DONE_STATUSES = ["succeed"]
FAILED_STATUSES = ["failed"]


def _build_headers(api_key: str) -> dict:
    """Bearer auth headers. The real Kling API expects a JWT bearer token; BYOK
    sends the user-supplied API key (or relay token) as the bearer credential."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _normalize_base(base_url: str) -> str:
    """Strip a trailing slash so path concatenation is clean."""
    return (base_url or KLING_API_BASE).rstrip("/")


def _status_of(resp: dict) -> str:
    data = resp.get("data") or {}
    return (data.get("task_status") or "").lower()


def _extract_first_video(resp: dict) -> dict:
    """Return the first video object from a succeeded Kling task response."""
    data = resp.get("data") or {}
    task_result = data.get("task_result") or {}
    videos = task_result.get("videos") or []
    if not videos:
        task_id = data.get("task_id")
        raise RuntimeError(
            f"Kling task {task_id} succeeded but no video data found in response: {resp}"
        )
    return videos[0]


async def _submit_and_poll(
    base: str,
    submit_path: str,
    body: dict,
    headers: dict,
    poll_interval: int,
    max_wait: int,
):
    """Submit a Kling generation task, poll until done, and download the video.

    Returns ``(VIDEO, video_id, duration)`` matching the official node outputs.
    """
    submit_resp = await http_request("POST", f"{base}{submit_path}", headers, body)
    code = submit_resp.get("code")
    data = submit_resp.get("data") or {}
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(
            f"Kling initial request failed. Code: {code}, "
            f"Message: {submit_resp.get('message')}, Response: {submit_resp}"
        )

    poll_url = f"{base}{submit_path}/{task_id}"

    async def poll_fn():
        return await http_request("GET", poll_url, headers)

    final = await poll_until(
        poll_fn,
        status_of=_status_of,
        done_statuses=DONE_STATUSES,
        failed_statuses=FAILED_STATUSES,
        interval=poll_interval,
        max_wait=max_wait,
        on_fail_message=f"Kling task {task_id} failed",
    )

    video = _extract_first_video(final)
    url = video.get("url")
    if not url:
        raise RuntimeError(f"Kling video result missing url: {final}")
    video_out = await download_video_output(url, headers=headers)
    return video_out, str(video.get("id") or ""), str(video.get("duration") or "")


class BYOKey_Kling_Text2Video:
    """Kling Text to Video (BYOK). Mirrors the official KlingTextToVideoNode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Kling API key (Bearer token)"}),
                "base_url": ("STRING", {"default": KLING_API_BASE, "tooltip": "Kling API host. Override for relays/proxies."}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Positive text prompt. Max 2500 characters."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Negative text prompt. Max 2500 characters."}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How closely the video follows the prompt."}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
                "mode": (_T2V_MODES, {"default": _T2V_DEFAULT_MODE, "tooltip": "Configuration: mode / duration / model_name."}),
            },
            "optional": {
                "poll_interval": ("INT", {"default": 10, "min": 1, "max": 120, "tooltip": "Seconds between status polls."}),
                "max_wait": ("INT", {"default": 600, "min": 30, "max": 3600, "tooltip": "Max seconds to wait for completion."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("output", "video_id", "duration")
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Kling AI text-to-video generation (bring your own key)."

    def execute(self, api_key, base_url, prompt, negative_prompt="", cfg_scale=1.0,
                aspect_ratio="16:9", mode=_T2V_DEFAULT_MODE,
                poll_interval=10, max_wait=600):
        validate_prompt(prompt, min_length=1)
        if len(prompt) > MAX_PROMPT_LENGTH_T2V:
            raise ValueError(f"Positive prompt is too long: {len(prompt)} characters (max {MAX_PROMPT_LENGTH_T2V}).")
        if negative_prompt and len(negative_prompt) > MAX_PROMPT_LENGTH_T2V:
            raise ValueError(f"Negative prompt is too long: {len(negative_prompt)} characters (max {MAX_PROMPT_LENGTH_T2V}).")

        model_mode, duration, model_name = MODE_TEXT2VIDEO[mode]
        base = _normalize_base(base_url)
        headers = _build_headers(api_key)

        body = {
            "model_name": model_name,
            "prompt": prompt,
            "duration": duration,
            "mode": model_mode,
            "cfg_scale": float(cfg_scale),
            "aspect_ratio": aspect_ratio,
        }
        if negative_prompt:
            body["negative_prompt"] = negative_prompt

        return run_async(
            _submit_and_poll(base, PATH_TEXT_TO_VIDEO, body, headers, poll_interval, max_wait)
        )


class BYOKey_Kling_Image2Video:
    """Kling Image (First Frame) to Video (BYOK). Mirrors KlingImage2VideoNode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Kling API key (Bearer token)"}),
                "base_url": ("STRING", {"default": KLING_API_BASE, "tooltip": "Kling API host. Override for relays/proxies."}),
                "start_frame": ("IMAGE", {"tooltip": "Reference image. At least 300x300px, aspect ratio 1:2.5 ~ 2.5:1."}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Positive text prompt. Max 500 characters."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Negative text prompt. Max 500 characters."}),
                "model_name": (I2V_MODEL_NAMES, {"default": "kling-v2-master"}),
                "cfg_scale": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How closely the video follows the prompt."}),
                "mode": (MODES, {"default": "std", "tooltip": "std = standard, pro = higher quality. kling-v2-5-turbo forces pro."}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
                "duration": (DURATIONS, {"default": "5"}),
            },
            "optional": {
                "end_frame": ("IMAGE", {"tooltip": "Optional end frame (image_tail). Same constraints as start_frame."}),
                "poll_interval": ("INT", {"default": 10, "min": 1, "max": 120, "tooltip": "Seconds between status polls."}),
                "max_wait": ("INT", {"default": 600, "min": 30, "max": 3600, "tooltip": "Max seconds to wait for completion."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("output", "video_id", "duration")
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Kling AI image-to-video generation (bring your own key)."

    def execute(self, api_key, base_url, start_frame, prompt, negative_prompt="",
                model_name="kling-v2-master", cfg_scale=0.8, mode="std",
                aspect_ratio="16:9", duration="5", end_frame=None,
                poll_interval=10, max_wait=600):
        validate_prompt(prompt, min_length=1)
        if len(prompt) > MAX_PROMPT_LENGTH_I2V:
            raise ValueError(f"Positive prompt is too long: {len(prompt)} characters (max {MAX_PROMPT_LENGTH_I2V}).")
        if negative_prompt and len(negative_prompt) > MAX_PROMPT_LENGTH_I2V:
            raise ValueError(f"Negative prompt is too long: {len(negative_prompt)} characters (max {MAX_PROMPT_LENGTH_I2V}).")

        # As of the official node: std mode is not supported for kling-v2-5-turbo,
        # so it is automatically switched to pro.
        if mode == "std" and model_name == "kling-v2-5-turbo":
            mode = "pro"

        base = _normalize_base(base_url)
        headers = _build_headers(api_key)

        # The Kling API expects a raw base64 image WITHOUT the data:image prefix.
        body = {
            "model_name": model_name,
            "image": tensor_to_base64(start_frame, "image/png"),
            "prompt": prompt,
            "cfg_scale": float(cfg_scale),
            "mode": mode,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if end_frame is not None:
            body["image_tail"] = tensor_to_base64(end_frame, "image/png")

        return run_async(
            _submit_and_poll(base, PATH_IMAGE_TO_VIDEO, body, headers, poll_interval, max_wait)
        )
