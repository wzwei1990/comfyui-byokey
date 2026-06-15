"""Vidu video generation nodes (bring-your-own-key).

Faithful re-implementations of the official ComfyUI Vidu API nodes
(``ViduTextToVideoNode`` / ``ViduImageToVideoNode``) that call the real Vidu
API directly with the user's own key instead of going through the ComfyOrg
proxy.

Ground truth: ``comfy_api_nodes/nodes_vidu.py`` + ``comfy_api_nodes/apis/vidu.py``.
The official nodes hit ``/proxy/vidu/<path>`` which forwards the body verbatim to
the real Vidu endpoints, so the official request/response models and endpoint
paths ARE the real Vidu API:

  POST   {base_url}/text2video                -> TaskCreationResponse
  POST   {base_url}/img2video                 -> TaskCreationResponse
  GET    {base_url}/tasks/{task_id}/creations -> TaskStatusResponse

Auth header: ``Authorization: Token <api_key>``.
Default host: ``https://api.vidu.com/ent/v2`` (overridable for relays).
"""

from .utils import (
    download_video_output,
    http_request,
    poll_until,
    run_async,
    tensor_to_data_uri,
    validate_prompt,
)

# Real Vidu endpoint host (matches the path the ComfyOrg proxy forwards to).
VIDU_API_BASE = "https://api.vidu.com/ent/v2"

# Endpoint paths (relative to base_url), mirroring the official proxy paths
# VIDU_TEXT_TO_VIDEO = "/proxy/vidu/text2video", etc.
VIDU_TEXT_TO_VIDEO = "/text2video"
VIDU_IMAGE_TO_VIDEO = "/img2video"
VIDU_GET_GENERATION_STATUS = "/tasks/%s/creations"

# Official model / option lists (verbatim from the official node schemas + docs).
VIDU_MODELS = ["viduq1"]
VIDU_ASPECT_RATIOS = ["16:9", "9:16", "1:1"]
VIDU_RESOLUTIONS = ["1080p"]
VIDU_MOVEMENT_AMPLITUDES = ["auto", "small", "medium", "large"]

# Official task states. ``poll_op``'s status_extractor reads ``state``; the
# official success path is reaching a state where ``creations`` is populated.
VIDU_DONE_STATES = ["success", "completed", "succeeded"]
VIDU_FAILED_STATES = ["failed", "error"]


def _auth_headers(api_key: str) -> dict:
    """Build the exact auth headers Vidu expects."""
    return {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }


def _build_payload(
    *,
    model: str,
    prompt: str,
    duration: int,
    seed: int,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    movement_amplitude: str | None = None,
    images: list[str] | None = None,
) -> dict:
    """Assemble a TaskCreationRequest body, omitting unset optional fields.

    Mirrors ``comfy_api_nodes.apis.vidu.TaskCreationRequest`` field-for-field
    (pydantic drops ``None`` optionals on serialization, so we do the same).
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "duration": duration,
        "seed": seed,
    }
    if aspect_ratio is not None:
        payload["aspect_ratio"] = aspect_ratio
    if resolution is not None:
        payload["resolution"] = resolution
    if movement_amplitude is not None:
        payload["movement_amplitude"] = movement_amplitude
    if images is not None:
        payload["images"] = images
    return payload


async def _execute_task(base_url: str, endpoint: str, headers: dict, payload: dict,
                        poll_interval: float, max_wait: float):
    """Submit a Vidu task and poll until a video result is available.

    Returns a ComfyUI VIDEO object built from the first creation's URL, matching
    the official ``execute_task`` + ``download_url_to_video_output`` flow.
    """
    base = base_url.rstrip("/")

    creation = await http_request("POST", f"{base}{endpoint}", headers, payload)
    if str(creation.get("state", "")).lower() == "failed":
        raise RuntimeError(f"Vidu request failed. Code: {creation.get('code')}")

    task_id = creation.get("task_id") or creation.get("id")
    if not task_id:
        raise RuntimeError(f"Vidu response did not contain a task_id: {creation}")

    status_url = f"{base}{VIDU_GET_GENERATION_STATUS % task_id}"

    async def poll_fn():
        return await http_request("GET", status_url, headers)

    final = await poll_until(
        poll_fn,
        status_of=lambda r: r.get("state"),
        done_statuses=VIDU_DONE_STATES,
        failed_statuses=VIDU_FAILED_STATES,
        interval=poll_interval,
        max_wait=max_wait,
        on_fail_message="Vidu generation failed",
    )

    creations = final.get("creations") or []
    if not creations or not creations[0].get("url"):
        raise RuntimeError(
            f"Vidu request does not contain results. "
            f"State: {final.get('state')}, Error Code: {final.get('err_code')}"
        )

    # Result URLs are presigned/public (valid ~1 hour) and need no auth header,
    # exactly like the official download_url_to_video_output.
    return await download_video_output(creations[0]["url"])


class BYOKey_Vidu_Text2Video:
    """Vidu text-to-video generation. Output: VIDEO."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "base_url": ("STRING", {"default": VIDU_API_BASE, "multiline": False}),
                "model": (VIDU_MODELS, {"default": VIDU_MODELS[0]}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "A textual description for video generation",
                }),
            },
            "optional": {
                "duration": ("INT", {
                    "default": 5, "min": 5, "max": 5, "step": 1,
                    "tooltip": "Duration of the output video in seconds (fixed at 5)",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2147483647, "step": 1,
                    "control_after_generate": True,
                    "tooltip": "Seed for video generation (0 for random)",
                }),
                "aspect_ratio": (VIDU_ASPECT_RATIOS, {
                    "default": VIDU_ASPECT_RATIOS[0],
                    "tooltip": "The aspect ratio of the output video",
                }),
                "resolution": (VIDU_RESOLUTIONS, {
                    "default": VIDU_RESOLUTIONS[0],
                    "tooltip": "Supported values may vary by model & duration",
                }),
                "movement_amplitude": (VIDU_MOVEMENT_AMPLITUDES, {
                    "default": VIDU_MOVEMENT_AMPLITUDES[0],
                    "tooltip": "The movement amplitude of objects in the frame",
                }),
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
                "max_wait": ("INT", {"default": 600, "min": 30, "max": 3600, "step": 1}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Vidu text-to-video generation (bring your own key)."

    def execute(self, api_key, base_url, model, prompt, duration=5, seed=0,
                aspect_ratio="16:9", resolution="1080p", movement_amplitude="auto",
                poll_interval=5, max_wait=600):
        # Official node: prompt is required and cannot be empty.
        prompt = validate_prompt(prompt, min_length=1)
        headers = _auth_headers(api_key)
        payload = _build_payload(
            model=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        video = run_async(_execute_task(
            base_url, VIDU_TEXT_TO_VIDEO, headers, payload, poll_interval, max_wait,
        ))
        return (video,)


class BYOKey_Vidu_Image2Video:
    """Vidu image-to-video generation. Output: VIDEO."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "base_url": ("STRING", {"default": VIDU_API_BASE, "multiline": False}),
                "model": (VIDU_MODELS, {"default": VIDU_MODELS[0]}),
                "image": ("IMAGE", {
                    "tooltip": "An image to be used as the start frame of the generated video",
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "A textual description for video generation",
                }),
                "duration": ("INT", {
                    "default": 5, "min": 5, "max": 5, "step": 1,
                    "tooltip": "Duration of the output video in seconds (fixed at 5)",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2147483647, "step": 1,
                    "control_after_generate": True,
                    "tooltip": "Seed for video generation (0 for random)",
                }),
                "resolution": (VIDU_RESOLUTIONS, {
                    "default": VIDU_RESOLUTIONS[0],
                    "tooltip": "Supported values may vary by model & duration",
                }),
                "movement_amplitude": (VIDU_MOVEMENT_AMPLITUDES, {
                    "default": VIDU_MOVEMENT_AMPLITUDES[0],
                    "tooltip": "The movement amplitude of objects in the frame",
                }),
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
                "max_wait": ("INT", {"default": 600, "min": 30, "max": 3600, "step": 1}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Vidu image-to-video generation (bring your own key)."

    def execute(self, api_key, base_url, model, image, prompt="", duration=5, seed=0,
                resolution="1080p", movement_amplitude="auto",
                poll_interval=5, max_wait=600):
        # Official node: only one input image is allowed.
        if image.shape[0] > 1:
            raise ValueError("Only one input image is allowed.")
        # prompt is optional for image-to-video (no validate_prompt).
        headers = _auth_headers(api_key)
        # Vidu's `images` field accepts a base64 data URI or an image URL.
        # Direct API (no ComfyOrg upload service) => inline the image as a data URI.
        image_data_uri = tensor_to_data_uri(image, "image/png")
        payload = _build_payload(
            model=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
            images=[image_data_uri],
        )
        video = run_async(_execute_task(
            base_url, VIDU_IMAGE_TO_VIDEO, headers, payload, poll_interval, max_wait,
        ))
        return (video,)
