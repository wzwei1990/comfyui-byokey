"""Grok (xAI) image and video generation nodes (bring-your-own-key).

These nodes mirror the official ComfyUI Grok API nodes (GrokImageNode,
GrokVideoNode) but call the xAI API directly with a user-supplied API key
instead of going through the ComfyOrg proxy.

Ground truth: comfy_api_nodes/nodes_grok.py + comfy_api_nodes/apis/grok.py.
The official nodes POST to ``/proxy/xai/v1/...``; the proxy forwards the body
verbatim to xAI, so the official request/response models and endpoint paths
are the real xAI API contract. Real host: https://api.x.ai/v1, Bearer auth.
"""

import torch

from .utils import (
    bytes_to_tensor,
    download_bytes,
    download_video_output,
    http_request,
    poll_until,
    run_async,
    tensor_to_base64,
    validate_prompt,
)

GROK_API_BASE = "https://api.x.ai/v1"

# Model id label -> real xAI api id (see _GROK_VIDEO_MODEL_API_IDS in official node).
_GROK_VIDEO_MODEL_API_IDS = {
    "grok-imagine-video-1.5": "grok-imagine-video-1.5-preview",
}

_IMAGE_MODELS = [
    "grok-imagine-image-quality",
    "grok-imagine-image-pro",
    "grok-imagine-image",
]

_IMAGE_ASPECT_RATIOS = [
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "9:16",
    "16:9",
    "9:19.5",
    "19.5:9",
    "9:20",
    "20:9",
    "1:2",
    "2:1",
]

_IMAGE_RESOLUTIONS = ["1K", "2K"]

_VIDEO_MODELS = ["grok-imagine-video", "grok-imagine-video-1.5"]

_VIDEO_RESOLUTIONS = ["480p", "720p"]

_VIDEO_ASPECT_RATIOS = ["auto", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"]


def _auth_headers(api_key: str) -> dict:
    """Build the xAI Bearer auth headers."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _images_from_response(resp: dict) -> torch.Tensor:
    """Extract IMAGE tensor(s) from an xAI image-generations response.

    Matches official extraction: download each data item's url (the official
    request sends response_format="url"); also tolerate b64_json. Concatenate
    multiple images into a single [B,H,W,C] batch.
    """
    data = resp.get("data") or []
    tensors = []
    for item in data:
        if item.get("url"):
            tensors.append(bytes_to_tensor(run_async(download_bytes(item["url"]))))
        elif item.get("b64_json"):
            import base64

            tensors.append(bytes_to_tensor(base64.b64decode(item["b64_json"])))
    if not tensors:
        raise RuntimeError(f"No images in Grok response: {resp}")
    return torch.cat(tensors, dim=0)


class BYOKey_Grok_Image:
    """Grok (xAI) image generation. Output: IMAGE.

    Mirrors official GrokImageNode -> POST /v1/images/generations
    (ImageGenerationRequest / ImageGenerationResponse).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "base_url": ("STRING", {"default": GROK_API_BASE, "multiline": False}),
                "model": (_IMAGE_MODELS, {"default": _IMAGE_MODELS[0]}),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "The text prompt used to generate the image"},
                ),
                "aspect_ratio": (_IMAGE_ASPECT_RATIOS, {"default": "1:1"}),
                "number_of_images": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1, "tooltip": "Number of images to generate"},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "control_after_generate": True,
                        "tooltip": "Seed to determine if node should re-run; "
                        "actual results are nondeterministic regardless of seed.",
                    },
                ),
            },
            "optional": {
                "resolution": (_IMAGE_RESOLUTIONS, {"default": "1K"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Generate images using Grok (xAI) based on a text prompt."

    def execute(
        self,
        api_key,
        base_url,
        model,
        prompt,
        aspect_ratio,
        number_of_images,
        seed,
        resolution="1K",
    ):
        validate_prompt(prompt, min_length=1)
        base = (base_url or GROK_API_BASE).rstrip("/")
        url = f"{base}/images/generations"
        headers = _auth_headers(api_key)
        body = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "n": number_of_images,
            "seed": seed,
            "response_format": "url",
            "resolution": resolution.lower(),
        }
        resp = run_async(http_request("POST", url, headers, body))
        return (_images_from_response(resp),)


class BYOKey_Grok_Video:
    """Grok (xAI) video generation. Output: VIDEO.

    Mirrors official GrokVideoNode: submit POST /v1/videos/generations
    (VideoGenerationRequest -> request_id), then poll GET /v1/videos/{request_id}
    (VideoStatusResponse) until done, then download response.video.url.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "base_url": ("STRING", {"default": GROK_API_BASE, "multiline": False}),
                "model": (
                    _VIDEO_MODELS,
                    {
                        "default": _VIDEO_MODELS[0],
                        "tooltip": "grok-imagine-video-1.5 currently always requires an input image.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Text description of the desired video."},
                ),
                "resolution": (
                    _VIDEO_RESOLUTIONS,
                    {"default": "480p", "tooltip": "The resolution of the output video."},
                ),
                "aspect_ratio": (
                    _VIDEO_ASPECT_RATIOS,
                    {"default": "auto", "tooltip": "The aspect ratio of the output video."},
                ),
                "duration": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 15,
                        "step": 1,
                        "tooltip": "The duration of the output video in seconds.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "control_after_generate": True,
                        "tooltip": "Seed to determine if node should re-run; "
                        "actual results are nondeterministic regardless of seed.",
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional starting image for grok-imagine-video. "
                        "Required for grok-imagine-video-1.5."
                    },
                ),
                "poll_interval": ("INT", {"default": 10, "min": 1, "max": 120, "step": 1}),
                "max_wait": ("INT", {"default": 600, "min": 30, "max": 3600, "step": 10}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Generate a video from a prompt or an image using Grok (xAI)."

    def execute(
        self,
        api_key,
        base_url,
        model,
        prompt,
        resolution,
        aspect_ratio,
        duration,
        seed,
        image=None,
        poll_interval=10,
        max_wait=600,
    ):
        if image is None and model == "grok-imagine-video-1.5":
            raise ValueError(
                f"The '{model}' model requires an input image; connect one to the 'image' input."
            )
        image_url = None
        if image is not None:
            # Only one input image is supported.
            if len(image.shape) == 4 and image.shape[0] != 1:
                raise ValueError("Only one input image is supported.")
            image_url = {"url": f"data:image/png;base64,{tensor_to_base64(image)}"}
        validate_prompt(prompt, min_length=1)

        base = (base_url or GROK_API_BASE).rstrip("/")
        headers = _auth_headers(api_key)

        # Build VideoGenerationRequest; drop None fields the official model omits.
        body = {
            "model": _GROK_VIDEO_MODEL_API_IDS.get(model, model),
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": None if aspect_ratio == "auto" else aspect_ratio,
            "seed": seed,
        }
        if image_url is not None:
            body["image"] = image_url

        initial = run_async(http_request("POST", f"{base}/videos/generations", headers, body))
        request_id = initial.get("request_id") or initial.get("id")
        if not request_id:
            raise RuntimeError(f"No request_id in Grok video response: {initial}")

        poll_url = f"{base}/videos/{request_id}"

        async def _poll():
            return await http_request("GET", poll_url, headers)

        # Official status_extractor: r.status if not None else "complete".
        # Treat a missing status (video ready) as done; standard terminal states too.
        def _status_of(r):
            status = r.get("status")
            if status is None:
                return "complete"
            return status

        final = run_async(
            poll_until(
                _poll,
                status_of=_status_of,
                done_statuses=["complete", "completed", "succeeded", "success", "done", "ready"],
                failed_statuses=["failed", "error", "canceled", "cancelled", "rejected"],
                interval=poll_interval,
                max_wait=max_wait,
                on_fail_message="Grok video generation failed",
            )
        )

        video = final.get("video") or {}
        video_url = video.get("url") or final.get("url")
        if not video_url:
            raise RuntimeError(f"No video URL in Grok response: {final}")

        return (run_async(download_video_output(video_url, headers=headers)),)
