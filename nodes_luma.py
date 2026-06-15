"""Luma AI (Dream Machine) video generation nodes for the BYOKey package.

Bring-your-own-key versions of the official ComfyUI Luma video nodes
(``LumaVideoNode`` / ``LumaImageToVideoNode``). Each node exposes ``api_key`` and
``base_url`` text inputs and calls the Luma Dream Machine REST API DIRECTLY
(``https://api.lumalabs.ai/dream-machine/v1``) instead of going through the
ComfyOrg proxy.

The request bodies and response parsing mirror the official
``comfy_api_nodes.apis.luma`` models field-for-field:

  * submit:  POST {base_url}/generations
  * poll:    GET  {base_url}/generations/{id}   (state: queued|dreaming|completed|failed)
  * result:  assets.video  (an mp4 URL)

Outputs are ComfyUI ``VIDEO`` objects (never string paths), matching official.
"""
from __future__ import annotations

from .utils import (
    download_video_output,
    http_request,
    poll_until,
    run_async,
    tensor_to_data_uri,
    validate_prompt,
)

LUMA_API_BASE = "https://api.lumalabs.ai/dream-machine/v1"

# Video models (from comfy_api_nodes.apis.luma.LumaVideoModel).
LUMA_VIDEO_MODELS = ["ray-2", "ray-flash-2", "ray-1-6"]
# Aspect ratios (LumaAspectRatio), default 16:9.
LUMA_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"]
# Output resolutions (LumaVideoOutputResolution). 4k is only meaningful for i2v
# per docs, but the enum carries it; t2v doc lists 540p/720p/1080p.
LUMA_T2V_RESOLUTIONS = ["540p", "720p", "1080p"]
LUMA_I2V_RESOLUTIONS = ["540p", "720p", "1080p", "4k"]
# Output durations (LumaVideoModelOutputDuration).
LUMA_DURATIONS = ["5s", "9s"]
# Model id that ignores duration/resolution (official sets these to None).
LUMA_RAY_1_6 = "ray-1-6"


def _auth_headers(api_key: str) -> dict:
    """Build the Bearer-auth JSON headers the Luma API expects."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _build_concepts(luma_concepts) -> list | None:
    """Normalize an optional luma_concepts input into the API ``concepts`` list.

    Accepts the official ``LumaConceptChain`` (has ``create_api_model``), a plain
    list of concept-key strings, or a list of ``{"key": ...}`` dicts. Returns
    ``None`` when there is nothing to send.
    """
    if luma_concepts is None:
        return None
    # Official LumaConceptChain object.
    create = getattr(luma_concepts, "create_api_model", None)
    if callable(create):
        api_model = create()
        if not api_model:
            return None
        out = []
        for c in api_model:
            key = getattr(c, "key", None)
            if key is None and isinstance(c, dict):
                key = c.get("key")
            if key and key != "None":
                out.append({"key": key})
        return out or None
    # Plain list of strings or dicts.
    if isinstance(luma_concepts, (list, tuple)):
        out = []
        for c in luma_concepts:
            if isinstance(c, dict):
                key = c.get("key")
            else:
                key = c
            if key and key != "None":
                out.append({"key": key})
        return out or None
    return None


async def _submit_and_poll(base_url: str, headers: dict, body: dict,
                           poll_interval: int, max_wait: int):
    """Submit a generation, poll until completion, and return a VIDEO output.

    Mirrors the official flow: POST /generations, then poll GET /generations/{id}
    on ``state`` until ``completed`` (or ``failed``), then download ``assets.video``.
    """
    base = base_url.rstrip("/")
    submit = await http_request("POST", f"{base}/generations", headers, json_data=body)
    gen_id = submit.get("id")
    if not gen_id:
        raise RuntimeError(f"Luma did not return a generation id. Response={submit}")

    async def poll_fn():
        return await http_request("GET", f"{base}/generations/{gen_id}", headers)

    final = await poll_until(
        poll_fn,
        status_of=lambda r: r.get("state"),
        done_statuses=["completed"],
        failed_statuses=["failed"],
        interval=poll_interval,
        max_wait=max_wait,
        on_fail_message="Luma generation failed",
    )

    assets = final.get("assets") or {}
    video_url = assets.get("video")
    if not video_url:
        reason = final.get("failure_reason")
        raise RuntimeError(
            f"Luma generation completed without a video URL. "
            f"failure_reason={reason!r} Response={final}"
        )
    return await download_video_output(video_url, headers=headers)


class BYOKey_Luma_Text2Video:
    """Luma Dream Machine text-to-video (BYOKey). Mirrors official LumaVideoNode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Your Luma API key."}),
                "base_url": ("STRING", {"default": LUMA_API_BASE,
                                        "tooltip": "Luma API base URL (override for relays)."}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Prompt for the video generation."}),
                "model": (LUMA_VIDEO_MODELS, {"default": "ray-2",
                                              "tooltip": "The video generation model to use."}),
                "aspect_ratio": (LUMA_ASPECT_RATIOS, {"default": "16:9",
                                                      "tooltip": "Aspect ratio of the generated video."}),
                "resolution": (LUMA_T2V_RESOLUTIONS, {"default": "540p",
                                                      "tooltip": "Output resolution (ignored for ray-1-6)."}),
                "duration": (LUMA_DURATIONS, {"default": "5s",
                                              "tooltip": "Video duration (ignored for ray-1-6)."}),
                "loop": ("BOOLEAN", {"default": False, "tooltip": "Whether the video should loop."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": True,
                                 "tooltip": "Seed to determine if node should re-run; "
                                            "results are nondeterministic regardless of seed."}),
            },
            "optional": {
                "luma_concepts": ("LUMA_CONCEPTS", {
                    "tooltip": "Optional Camera Concepts to dictate camera motion "
                               "via the Luma Concepts node."}),
                "poll_interval": ("INT", {"default": 10, "min": 1, "max": 120,
                                          "tooltip": "Seconds between poll attempts."}),
                "max_wait": ("INT", {"default": 600, "min": 30, "max": 3600,
                                     "tooltip": "Maximum seconds to wait for completion."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Luma Dream Machine text-to-video. Generates a video from a text prompt."

    def execute(self, api_key, base_url, prompt, model, aspect_ratio, resolution,
                duration, loop, seed, luma_concepts=None, poll_interval=10, max_wait=600):
        # Official validates the prompt with min_length=3.
        validate_prompt(prompt, min_length=3)
        headers = _auth_headers(api_key)

        # For ray-1-6, duration and resolution are ignored by the API.
        send_duration = duration if model != LUMA_RAY_1_6 else None
        send_resolution = resolution if model != LUMA_RAY_1_6 else None

        # Mirrors LumaGenerationRequest: only include non-None fields.
        body = {"prompt": prompt, "model": model, "aspect_ratio": aspect_ratio, "loop": loop}
        if send_resolution is not None:
            body["resolution"] = send_resolution
        if send_duration is not None:
            body["duration"] = send_duration
        concepts = _build_concepts(luma_concepts)
        if concepts is not None:
            body["concepts"] = concepts

        video = run_async(
            _submit_and_poll(base_url, headers, body, poll_interval, max_wait)
        )
        return (video,)


class BYOKey_Luma_Image2Video:
    """Luma Dream Machine image-to-video (BYOKey). Mirrors official LumaImageToVideoNode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Your Luma API key."}),
                "base_url": ("STRING", {"default": LUMA_API_BASE,
                                        "tooltip": "Luma API base URL (override for relays)."}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Prompt for the video generation."}),
                "model": (LUMA_VIDEO_MODELS, {"default": "ray-2",
                                              "tooltip": "The video generation model to use."}),
                "resolution": (LUMA_I2V_RESOLUTIONS, {"default": "540p",
                                                      "tooltip": "Output resolution (ignored for ray-1-6)."}),
                "duration": (LUMA_DURATIONS, {"default": "5s",
                                              "tooltip": "Video duration (ignored for ray-1-6)."}),
                "loop": ("BOOLEAN", {"default": False, "tooltip": "Whether the video should loop."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": True,
                                 "tooltip": "Seed to determine if node should re-run; "
                                            "results are nondeterministic regardless of seed."}),
            },
            "optional": {
                "first_image": ("IMAGE", {"tooltip": "First frame of generated video."}),
                "last_image": ("IMAGE", {"tooltip": "Last frame of generated video."}),
                "luma_concepts": ("LUMA_CONCEPTS", {
                    "tooltip": "Optional Camera Concepts to dictate camera motion "
                               "via the Luma Concepts node."}),
                "poll_interval": ("INT", {"default": 10, "min": 1, "max": 120,
                                          "tooltip": "Seconds between poll attempts."}),
                "max_wait": ("INT", {"default": 600, "min": 30, "max": 3600,
                                     "tooltip": "Maximum seconds to wait for completion."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = ("Luma Dream Machine image-to-video. Generates a video from a prompt "
                   "and at least one of a first/last frame image.")

    def execute(self, api_key, base_url, prompt, model, resolution, duration, loop, seed,
                first_image=None, last_image=None, luma_concepts=None,
                poll_interval=10, max_wait=600):
        # Official requires at least one of first_image / last_image.
        if first_image is None and last_image is None:
            raise Exception("At least one of first_image and last_image requires an input.")

        headers = _auth_headers(api_key)

        # Build keyframes (frame0=first, frame1=last) using data URIs so the
        # vendor receives the images directly (no ComfyOrg upload step).
        keyframes: dict = {}
        if first_image is not None:
            keyframes["frame0"] = {
                "type": "image",
                "url": tensor_to_data_uri(first_image, mime_type="image/png"),
            }
        if last_image is not None:
            keyframes["frame1"] = {
                "type": "image",
                "url": tensor_to_data_uri(last_image, mime_type="image/png"),
            }

        # For ray-1-6, duration and resolution are ignored by the API.
        send_duration = duration if model != LUMA_RAY_1_6 else None
        send_resolution = resolution if model != LUMA_RAY_1_6 else None

        # Mirrors LumaGenerationRequest. aspect_ratio is ignored for i2v but the
        # API still requires it, so the official node hard-codes 16:9.
        body = {
            "prompt": prompt,
            "model": model,
            "aspect_ratio": "16:9",
            "loop": loop,
            "keyframes": keyframes,
        }
        if send_resolution is not None:
            body["resolution"] = send_resolution
        if send_duration is not None:
            body["duration"] = send_duration
        concepts = _build_concepts(luma_concepts)
        if concepts is not None:
            body["concepts"] = concepts

        video = run_async(
            _submit_and_poll(base_url, headers, body, poll_interval, max_wait)
        )
        return (video,)
