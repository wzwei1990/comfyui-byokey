"""ByteDance Seedance 2.0 video-generation nodes (bring-your-own-key).

Faithfully mirrors the official ComfyUI ByteDance Seedance 2.0 nodes
(``comfy_api_nodes/nodes_bytedance.py`` + ``comfy_api_nodes/apis/bytedance.py``),
but calls the BytePlus ark API DIRECTLY using a user-supplied ``api_key`` /
``base_url`` instead of going through the ComfyOrg proxy.

The official nodes POST to ``/proxy/byteplus/api/v3/contents/generations/tasks``;
the proxy forwards the body verbatim to the real BytePlus ark endpoint, so the
official request/response models and the ``/api/v3/contents/generations/tasks``
path ARE the real vendor contract reproduced here.

* ``BYOKey_Seedance2_Text2Video``  -> ByteDance2TextToVideoNode
* ``BYOKey_Seedance2_Image2Video`` -> ByteDance2FirstLastFrameNode flow
                                      (first/last-frame image-to-video)

Both nodes output a ComfyUI ``VIDEO`` object (never a string path).
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

# Real BytePlus ark host. The official proxy path is
# /proxy/byteplus/api/v3/contents/generations/tasks ; stripping the proxy
# prefix leaves the real vendor path used below. Overridable for relays.
DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com"
TASKS_PATH = "/api/v3/contents/generations/tasks"

# Mirrors SEEDANCE_MODELS in the official node.
MODEL_IDS = {
    "Seedance 2.0": "dreamina-seedance-2-0-260128",
    "Seedance 2.0 Fast": "dreamina-seedance-2-0-fast-260128",
}

# Resolutions supported per model (official: Fast drops 1080p).
MODEL_RESOLUTIONS = {
    "Seedance 2.0": ["480p", "720p", "1080p"],
    "Seedance 2.0 Fast": ["480p", "720p"],
}

# Union of all supported resolutions for the COMBO widget (validated per-model
# against MODEL_RESOLUTIONS at execute time).
RESOLUTION_OPTIONS = ["480p", "720p", "1080p"]

# Aspect ratios (official _seedance2_text_inputs ratio combo).
RATIO_OPTIONS = ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"]

# Official TaskStatusResponse.status literals.
DONE_STATUSES = ["succeeded"]
FAILED_STATUSES = ["failed", "cancelled"]


def _auth_headers(api_key: str) -> dict:
    """Build BytePlus ark auth headers (Bearer token, JSON body)."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _build_request_body(
    model_id: str,
    content: list,
    *,
    resolution: str,
    ratio: str,
    duration: int,
    seed: int,
    watermark: bool,
    generate_audio: bool,
) -> dict:
    """Build a Seedance2TaskCreationRequest-equivalent JSON body.

    Field names/nesting match ``apis.bytedance.Seedance2TaskCreationRequest``.
    """
    return {
        "model": model_id,
        "content": content,
        "generate_audio": generate_audio,
        "resolution": resolution,
        "ratio": ratio,
        "duration": duration,
        "seed": seed,
        "watermark": watermark,
    }


def _extract_video_url(final: dict) -> str:
    """Extract the result video URL the way the official node does.

    Official: ``response.content.video_url`` from TaskStatusResponse.
    """
    error = final.get("error")
    if error:
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else error
        raise RuntimeError(f"Seedance task failed. Code: {code}, message: {message}")
    content = final.get("content") or {}
    if isinstance(content, dict):
        url = content.get("video_url")
        if url:
            return url
    raise RuntimeError(f"No video_url in completed Seedance response: {final}")


async def _submit_and_wait(
    base_url: str,
    headers: dict,
    body: dict,
    *,
    poll_interval: int,
    max_wait: int,
):
    """Submit a task, poll until done, then download the VIDEO output.

    Returns a ComfyUI VIDEO object (utils.download_video_output).
    """
    base = base_url.rstrip("/")
    submit_url = f"{base}{TASKS_PATH}"
    created = await http_request("POST", submit_url, headers, json_data=body)

    task_id = created.get("id") or (created.get("data") or {}).get("id")
    if not task_id:
        raise RuntimeError(f"No task id in Seedance creation response: {created}")

    status_url = f"{base}{TASKS_PATH}/{task_id}"

    async def _poll():
        return await http_request("GET", status_url, headers)

    final = await poll_until(
        _poll,
        status_of=lambda r: r.get("status"),
        done_statuses=DONE_STATUSES,
        failed_statuses=FAILED_STATUSES,
        interval=poll_interval,
        max_wait=max_wait,
        on_fail_message="Seedance generation failed",
    )

    video_url = _extract_video_url(final)
    # Download via the vendor host with auth headers; the VIDEO object is
    # constructed from raw bytes (utils.download_video_output).
    return await download_video_output(video_url, headers=headers)


class BYOKey_Seedance2_Text2Video:
    """ByteDance Seedance 2.0 text-to-video (mirrors ByteDance2TextToVideoNode).

    Output type: VIDEO (a ComfyUI video object), exactly like the official node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "tooltip": "BytePlus ark API key (Bearer token)."},
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": DEFAULT_BASE_URL,
                        "tooltip": "BytePlus ark host. Override for a relay/proxy.",
                    },
                ),
                "model": (
                    list(MODEL_IDS.keys()),
                    {
                        "default": "Seedance 2.0",
                        "tooltip": "Seedance 2.0 for max quality; Seedance 2.0 Fast for speed.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for video generation.",
                    },
                ),
                "resolution": (
                    RESOLUTION_OPTIONS,
                    {
                        "default": "720p",
                        "tooltip": "Output resolution. Seedance 2.0 Fast supports 480p/720p only.",
                    },
                ),
                "ratio": (
                    RATIO_OPTIONS,
                    {"default": "16:9", "tooltip": "Aspect ratio of the output video."},
                ),
                "duration": (
                    "INT",
                    {
                        "default": 7,
                        "min": 4,
                        "max": 15,
                        "step": 1,
                        "tooltip": "Duration of the output video in seconds (4-15).",
                    },
                ),
                "generate_audio": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable audio generation for the output video."},
                ),
            },
            "optional": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "Seed (results are non-deterministic regardless of seed).",
                    },
                ),
                "watermark": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Whether to add a watermark to the video."},
                ),
                "poll_interval": (
                    "INT",
                    {"default": 9, "min": 2, "max": 60, "tooltip": "Seconds between status polls."},
                ),
                "max_wait": (
                    "INT",
                    {"default": 600, "min": 60, "max": 3600, "tooltip": "Max seconds to wait."},
                ),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "ByteDance Seedance 2.0 text-to-video generation (bring your own key)."

    def execute(
        self,
        api_key,
        base_url,
        model,
        prompt,
        resolution,
        ratio,
        duration,
        generate_audio,
        seed=0,
        watermark=False,
        poll_interval=9,
        max_wait=600,
    ):
        validate_prompt(prompt, min_length=1)
        model_id = MODEL_IDS[model]

        allowed = MODEL_RESOLUTIONS[model]
        if resolution not in allowed:
            raise ValueError(
                f"Resolution {resolution!r} is not supported by {model!r}. "
                f"Supported: {', '.join(allowed)}."
            )

        headers = _auth_headers(api_key)
        content = [{"type": "text", "text": prompt}]
        body = _build_request_body(
            model_id,
            content,
            resolution=resolution,
            ratio=ratio,
            duration=duration,
            seed=seed,
            watermark=watermark,
            generate_audio=generate_audio,
        )

        video = run_async(
            _submit_and_wait(
                base_url,
                headers,
                body,
                poll_interval=poll_interval,
                max_wait=max_wait,
            )
        )
        return (video,)


class BYOKey_Seedance2_Image2Video:
    """ByteDance Seedance 2.0 first/last-frame image-to-video.

    Mirrors the official ByteDance2FirstLastFrameNode flow: a required first
    frame image plus an optional last frame image guide the generation. Frames
    are sent inline as data-URI image_url content with first_frame/last_frame
    roles. Output type: VIDEO.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "tooltip": "BytePlus ark API key (Bearer token)."},
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": DEFAULT_BASE_URL,
                        "tooltip": "BytePlus ark host. Override for a relay/proxy.",
                    },
                ),
                "model": (
                    list(MODEL_IDS.keys()),
                    {
                        "default": "Seedance 2.0",
                        "tooltip": "Seedance 2.0 for max quality; Seedance 2.0 Fast for speed.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for video generation.",
                    },
                ),
                "first_frame": (
                    "IMAGE",
                    {"tooltip": "First frame image for the video."},
                ),
                "resolution": (
                    RESOLUTION_OPTIONS,
                    {
                        "default": "720p",
                        "tooltip": "Output resolution. Seedance 2.0 Fast supports 480p/720p only.",
                    },
                ),
                "ratio": (
                    RATIO_OPTIONS,
                    {"default": "adaptive", "tooltip": "Aspect ratio of the output video."},
                ),
                "duration": (
                    "INT",
                    {
                        "default": 7,
                        "min": 4,
                        "max": 15,
                        "step": 1,
                        "tooltip": "Duration of the output video in seconds (4-15).",
                    },
                ),
                "generate_audio": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable audio generation for the output video."},
                ),
            },
            "optional": {
                "last_frame": (
                    "IMAGE",
                    {"tooltip": "Optional last frame image for the video."},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "Seed (results are non-deterministic regardless of seed).",
                    },
                ),
                "watermark": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Whether to add a watermark to the video."},
                ),
                "poll_interval": (
                    "INT",
                    {"default": 9, "min": 2, "max": 60, "tooltip": "Seconds between status polls."},
                ),
                "max_wait": (
                    "INT",
                    {"default": 600, "min": 60, "max": 3600, "tooltip": "Max seconds to wait."},
                ),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = (
        "ByteDance Seedance 2.0 first/last-frame image-to-video (bring your own key)."
    )

    def execute(
        self,
        api_key,
        base_url,
        model,
        prompt,
        first_frame,
        resolution,
        ratio,
        duration,
        generate_audio,
        last_frame=None,
        seed=0,
        watermark=False,
        poll_interval=9,
        max_wait=600,
    ):
        validate_prompt(prompt, min_length=1)
        model_id = MODEL_IDS[model]

        allowed = MODEL_RESOLUTIONS[model]
        if resolution not in allowed:
            raise ValueError(
                f"Resolution {resolution!r} is not supported by {model!r}. "
                f"Supported: {', '.join(allowed)}."
            )

        headers = _auth_headers(api_key)

        # content mirrors the official FLF flow: text + first_frame (+ last_frame).
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": tensor_to_data_uri(first_frame, "image/png")},
                "role": "first_frame",
            },
        ]
        if last_frame is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": tensor_to_data_uri(last_frame, "image/png")},
                    "role": "last_frame",
                }
            )

        body = _build_request_body(
            model_id,
            content,
            resolution=resolution,
            ratio=ratio,
            duration=duration,
            seed=seed,
            watermark=watermark,
            generate_audio=generate_audio,
        )

        video = run_async(
            _submit_and_wait(
                base_url,
                headers,
                body,
                poll_interval=poll_interval,
                max_wait=max_wait,
            )
        )
        return (video,)
