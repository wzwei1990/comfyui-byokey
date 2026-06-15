"""Wan (Alibaba DashScope / Tongyi Wanxiang) video generation nodes (bring-your-own-key).

These nodes mirror the official ComfyUI Wan API nodes (``WanTextToVideoApi`` /
``WanImageToVideoApi`` in ``comfy_api_nodes/nodes_wan.py``) but call the Alibaba
DashScope API DIRECTLY instead of going through the ComfyOrg proxy. The official
nodes hit ``/proxy/wan/api/v1/...`` which the proxy forwards verbatim to DashScope,
so the request/response models and endpoint paths used here ARE the real vendor API.

DashScope video synthesis is an asynchronous flow:
  1. POST ``/api/v1/services/aigc/video-generation/video-synthesis`` with the
     ``X-DashScope-Async: enable`` header -> returns ``output.task_id``.
  2. GET ``/api/v1/tasks/{task_id}`` (Authorization only) -> poll until
     ``output.task_status`` is ``SUCCEEDED`` (or ``FAILED``/``UNKNOWN``).
  3. On success, download ``output.video_url`` and wrap it as a ComfyUI VIDEO.

Output type is VIDEO (never a string path), matching the official nodes.
"""
import logging

from .utils import (
    download_video_output,
    http_request,
    poll_until,
    run_async,
    tensor_to_data_uri,
    validate_prompt,
)

logger = logging.getLogger("comfyui_byokey")

# Real DashScope host. Path segments mirror the official proxy path
# (``/proxy/wan`` -> ``https://dashscope.aliyuncs.com``). Overridable for relays
# (e.g. the international endpoint ``https://dashscope-intl.aliyuncs.com``).
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
SUBMIT_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
TASK_PATH = "/api/v1/tasks/{task_id}"

# Resolution / aspect-ratio combos for the size-driven text-to-video node, copied
# verbatim from the official WanTextToVideoApi schema. The "(WIDTHxHEIGHT)" tail is
# parsed out and sent to DashScope as the "WIDTH*HEIGHT" size string.
T2V_SIZE_OPTIONS = [
    "480p: 1:1 (624x624)",
    "480p: 16:9 (832x480)",
    "480p: 9:16 (480x832)",
    "720p: 1:1 (960x960)",
    "720p: 16:9 (1280x720)",
    "720p: 9:16 (720x1280)",
    "720p: 4:3 (1088x832)",
    "720p: 3:4 (832x1088)",
    "1080p: 1:1 (1440x1440)",
    "1080p: 16:9 (1920x1080)",
    "1080p: 9:16 (1080x1920)",
    "1080p: 4:3 (1632x1248)",
    "1080p: 3:4 (1248x1632)",
]

# Resolution labels for the image-to-video node (official WanImageToVideoApi).
I2V_RESOLUTION_OPTIONS = ["480P", "720P", "1080P"]

SHOT_TYPE_OPTIONS = ["single", "multi"]

# Task status values returned by DashScope under output.task_status.
DONE_STATUSES = ["SUCCEEDED"]
FAILED_STATUSES = ["FAILED", "UNKNOWN", "CANCELED"]


def _parse_size_from_label(size_label: str) -> str:
    """Extract ``WIDTH*HEIGHT`` from a label like ``"720p: 16:9 (1280x720)"``.

    DashScope expects the ``size`` parameter as ``"<width>*<height>"``.
    """
    start = size_label.rfind("(")
    end = size_label.rfind(")")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not parse resolution from size label: {size_label!r}")
    inner = size_label[start + 1:end]
    for sep in ("x", "X", "×"):  # x, X, ×
        if sep in inner:
            w, h = inner.split(sep, 1)
            return f"{w.strip()}*{h.strip()}"
    raise ValueError(f"Could not parse resolution from size label: {size_label!r}")


def _auth_headers(api_key: str, *, async_submit: bool = False) -> dict:
    """Build DashScope auth headers. The async header is required on submit only."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if async_submit:
        headers["X-DashScope-Async"] = "enable"
    return headers


async def _submit_and_wait(
    base_url: str,
    api_key: str,
    body: dict,
    poll_interval: int,
    max_wait: int,
):
    """Submit a DashScope video task, poll until done, return the VIDEO output."""
    base = base_url.rstrip("/")
    submit_url = f"{base}{SUBMIT_PATH}"

    submit_resp = await http_request(
        "POST", submit_url, _auth_headers(api_key, async_submit=True), json_data=body
    )
    output = submit_resp.get("output") or {}
    task_id = output.get("task_id")
    if not task_id:
        code = submit_resp.get("code")
        message = submit_resp.get("message")
        raise RuntimeError(
            f"DashScope did not return a task_id. code={code} message={message} response={submit_resp}"
        )

    task_url = f"{base}{TASK_PATH.format(task_id=task_id)}"
    poll_headers = _auth_headers(api_key)

    async def _poll():
        return await http_request("GET", task_url, poll_headers)

    final = await poll_until(
        _poll,
        status_of=lambda r: (r.get("output") or {}).get("task_status"),
        done_statuses=DONE_STATUSES,
        failed_statuses=FAILED_STATUSES,
        interval=poll_interval,
        max_wait=max_wait,
        on_fail_message="Wan video generation failed",
    )

    final_output = final.get("output") or {}
    video_url = final_output.get("video_url")
    if not video_url:
        raise RuntimeError(f"Wan task succeeded but no video_url was returned: {final}")

    # The download URL is authenticated on DashScope's CDN and does not need the
    # API key, so download without auth headers.
    return await download_video_output(video_url)


class BYOKey_Wan_Text2Video:
    """Wan text-to-video via Alibaba DashScope (bring-your-own-key).

    Mirrors the official ``WanTextToVideoApi`` node. Output: VIDEO.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "DashScope (Alibaba Model Studio) API key.",
                }),
                "model": (["wan2.5-t2v-preview", "wan2.6-t2v"], {
                    "default": "wan2.6-t2v",
                    "tooltip": "Model to use.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt describing the elements and visual features. "
                               "Supports English and Chinese.",
                }),
            },
            "optional": {
                "base_url": ("STRING", {
                    "default": DEFAULT_BASE_URL,
                    "tooltip": "DashScope API host. Use https://dashscope-intl.aliyuncs.com "
                               "for the international endpoint, or a relay base URL.",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Negative prompt describing what to avoid.",
                }),
                "size": (T2V_SIZE_OPTIONS, {
                    "default": "720p: 1:1 (960x960)",
                    "tooltip": "Video resolution and aspect ratio.",
                }),
                "duration": ("INT", {
                    "default": 5,
                    "min": 5,
                    "max": 15,
                    "step": 5,
                    "tooltip": "Video duration in seconds. A 15-second duration is "
                               "available only for the Wan 2.6 model.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "generate_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If no audio input is provided, generate audio automatically.",
                }),
                "prompt_extend": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to enhance the prompt with AI assistance.",
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether to add an AI-generated watermark to the result.",
                }),
                "shot_type": (SHOT_TYPE_OPTIONS, {
                    "default": "single",
                    "tooltip": "Single continuous shot or multiple shots with cuts. "
                               "Takes effect only when prompt_extend is True.",
                }),
                "poll_interval": ("INT", {"default": 6, "min": 2, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Wan text-to-video generation via Alibaba DashScope (bring your own key)."

    def execute(
        self,
        api_key,
        model,
        prompt,
        base_url=DEFAULT_BASE_URL,
        negative_prompt="",
        size="720p: 1:1 (960x960)",
        duration=5,
        seed=0,
        generate_audio=False,
        prompt_extend=True,
        watermark=False,
        shot_type="single",
        poll_interval=6,
        max_wait=600,
    ):
        validate_prompt(prompt, min_length=1)
        if "480p" in size and model == "wan2.6-t2v":
            raise ValueError("The Wan 2.6 model does not support 480p.")
        if duration == 15 and model == "wan2.5-t2v-preview":
            raise ValueError("A 15-second duration is supported only by the Wan 2.6 model.")

        size_value = _parse_size_from_label(size)

        input_field = {"prompt": prompt}
        if negative_prompt:
            input_field["negative_prompt"] = negative_prompt

        body = {
            "model": model,
            "input": input_field,
            "parameters": {
                "size": size_value,
                "duration": duration,
                "seed": seed,
                "audio": generate_audio,
                "prompt_extend": prompt_extend,
                "watermark": watermark,
                "shot_type": shot_type,
            },
        }

        video = run_async(
            _submit_and_wait(base_url, api_key, body, poll_interval, max_wait)
        )
        return (video,)


class BYOKey_Wan_Image2Video:
    """Wan image-to-video via Alibaba DashScope (bring-your-own-key).

    Mirrors the official ``WanImageToVideoApi`` node. The input image is the first
    frame; the output video's aspect ratio is derived from it. Output: VIDEO.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "DashScope (Alibaba Model Studio) API key.",
                }),
                "model": (["wan2.5-i2v-preview", "wan2.6-i2v"], {
                    "default": "wan2.6-i2v",
                    "tooltip": "Model to use.",
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image used as the first frame. Exactly one image is required.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt describing the elements and visual features. "
                               "Supports English and Chinese.",
                }),
            },
            "optional": {
                "base_url": ("STRING", {
                    "default": DEFAULT_BASE_URL,
                    "tooltip": "DashScope API host. Use https://dashscope-intl.aliyuncs.com "
                               "for the international endpoint, or a relay base URL.",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Negative prompt describing what to avoid.",
                }),
                "resolution": (I2V_RESOLUTION_OPTIONS, {
                    "default": "720P",
                    "tooltip": "Video resolution quality. The Wan 2.6 model does not support 480P.",
                }),
                "duration": ("INT", {
                    "default": 5,
                    "min": 5,
                    "max": 15,
                    "step": 5,
                    "tooltip": "Video duration in seconds. A 15-second duration is "
                               "supported only by the Wan 2.6 model.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "generate_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If no audio input is provided, generate audio automatically.",
                }),
                "prompt_extend": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to enhance the prompt with AI assistance.",
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether to add an AI-generated watermark to the result.",
                }),
                "shot_type": (SHOT_TYPE_OPTIONS, {
                    "default": "single",
                    "tooltip": "Single continuous shot or multiple shots with cuts. "
                               "Takes effect only when prompt_extend is True.",
                }),
                "poll_interval": ("INT", {"default": 6, "min": 2, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Wan image-to-video generation via Alibaba DashScope (bring your own key)."

    def execute(
        self,
        api_key,
        model,
        image,
        prompt,
        base_url=DEFAULT_BASE_URL,
        negative_prompt="",
        resolution="720P",
        duration=5,
        seed=0,
        generate_audio=False,
        prompt_extend=True,
        watermark=False,
        shot_type="single",
        poll_interval=6,
        max_wait=600,
    ):
        validate_prompt(prompt, min_length=1)
        if image.shape[0] != 1:
            raise ValueError(
                f"Exactly one input image is required, but got a batch of {image.shape[0]}."
            )
        if "480P" in resolution and model == "wan2.6-i2v":
            raise ValueError("The Wan 2.6 model does not support 480P.")
        if duration == 15 and model == "wan2.5-i2v-preview":
            raise ValueError("A 15-second duration is supported only by the Wan 2.6 model.")

        img_url = tensor_to_data_uri(image, mime_type="image/png")

        input_field = {"prompt": prompt, "img_url": img_url}
        if negative_prompt:
            input_field["negative_prompt"] = negative_prompt

        body = {
            "model": model,
            "input": input_field,
            "parameters": {
                "resolution": resolution,
                "duration": duration,
                "seed": seed,
                "audio": generate_audio,
                "prompt_extend": prompt_extend,
                "watermark": watermark,
                "shot_type": shot_type,
            },
        }

        video = run_async(
            _submit_and_wait(base_url, api_key, body, poll_interval, max_wait)
        )
        return (video,)
