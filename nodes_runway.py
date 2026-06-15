"""Runway video and image generation nodes (bring-your-own-key).

Faithful BYOKey reimplementation of the official ComfyUI Runway API nodes
(``RunwayImageToVideoNodeGen4`` and ``RunwayTextToImageNode``). Each node calls
the Runway developer API directly (https://api.dev.runwayml.com/v1) with the
user's own key instead of going through the ComfyOrg proxy. Request bodies,
endpoint paths, poll behaviour and response parsing mirror the official
``comfy_api_nodes.apis.runway`` models field-for-field:

  - image_to_video  ->  RunwayImageToVideoRequest  ->  VIDEO
  - text_to_image   ->  RunwayTextToImageRequest   ->  IMAGE

Submit returns a task id; we poll ``/tasks/{id}`` until status SUCCEEDED/FAILED
and read the result URL from ``output[0]``.
"""

from .utils import (
    bytes_to_tensor,
    download_bytes,
    download_video_output,
    http_request,
    poll_until,
    run_async,
    tensor_to_data_uri,
    validate_prompt,
)

# Real vendor host (overridable for relays). NOT the ComfyOrg proxy.
RUNWAY_API_BASE = "https://api.dev.runwayml.com/v1"

# Runway developer API version header value.
RUNWAY_API_VERSION = "2024-11-06"

# Task status strings returned by Runway's /tasks/{id} endpoint.
# (SUCCEEDED, RUNNING, FAILED, PENDING, CANCELLED, THROTTLED)
DONE_STATUSES = ["SUCCEEDED"]
FAILED_STATUSES = ["FAILED", "CANCELLED"]

# Aspect ratios supported by the Image-to-Video API, per model
# (matches RunwayGen4TurboAspectRatio / RunwayGen3aAspectRatio in the official source).
GEN4_TURBO_RATIOS = ["1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672"]
GEN3A_TURBO_RATIOS = ["768:1280", "1280:768"]
# Combined list for a single combo widget; model-specific validation happens at runtime.
I2V_RATIOS = GEN4_TURBO_RATIOS + GEN3A_TURBO_RATIOS

I2V_MODELS = ["gen4_turbo", "gen3a_turbo"]
I2V_DURATIONS = ["5", "10"]

# Aspect ratios supported by the Text-to-Image API
# (matches RunwayTextToImageAspectRatioEnum in the official source — these are the
# real API enum values the official node passes straight through to Runway).
T2I_RATIOS = [
    "1920:1080",
    "1080:1920",
    "1024:1024",
    "1360:768",
    "1080:1080",
    "1168:880",
    "1440:1080",
    "1080:1440",
    "1808:768",
    "2112:912",
]
T2I_MODEL = "gen4_image"

SEED_MAX = 4294967295


def _auth_headers(api_key: str) -> dict:
    """Build the Runway auth headers exactly as the vendor expects."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Runway-Version": RUNWAY_API_VERSION,
    }


class BYOKey_Runway_Image2Video:
    """Runway Image to Video (Gen4 Turbo / Gen3a Turbo). Output: VIDEO.

    Mirrors the official ``RunwayImageToVideoNodeGen4`` node: a text prompt plus a
    single start frame produce a video at the chosen duration and aspect ratio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Runway API key (sent as Bearer token)."}),
                "base_url": ("STRING", {"default": RUNWAY_API_BASE, "tooltip": "Runway API base URL. Override for relays."}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text prompt for the generation."}),
                "start_frame": ("IMAGE", {"tooltip": "Start frame to be used for the video."}),
                "model": (I2V_MODELS, {"default": "gen4_turbo"}),
                "duration": (I2V_DURATIONS, {"default": "5", "tooltip": "Video duration in seconds."}),
                "ratio": (I2V_RATIOS, {"default": "1280:720", "tooltip": "Aspect ratio. gen4_turbo and gen3a_turbo support different ratios."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX, "step": 1, "control_after_generate": True, "tooltip": "Random seed for generation."}),
            },
            "optional": {
                "poll_interval": ("INT", {"default": 10, "min": 1, "max": 120, "tooltip": "Seconds between status polls."}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600, "tooltip": "Max seconds to wait for the task."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Generate a video from a single starting frame using Runway's Gen4/Gen3a Turbo model."

    def execute(self, api_key, base_url, prompt, start_frame, model="gen4_turbo",
                duration="5", ratio="1280:720", seed=0, poll_interval=10, max_wait=600):
        validate_prompt(prompt, min_length=1)

        # Enforce the model-specific aspect-ratio support from the official source.
        allowed = GEN4_TURBO_RATIOS if model == "gen4_turbo" else GEN3A_TURBO_RATIOS
        if ratio not in allowed:
            raise ValueError(
                f"Aspect ratio {ratio!r} is not supported by model {model!r}. "
                f"Supported ratios: {allowed}."
            )

        base = base_url.rstrip("/")
        headers = _auth_headers(api_key)

        # The start frame is sent inline as a data URI. Runway's promptImage accepts
        # "A HTTPS URL or data URI containing an encoded image" as the uri field.
        data_uri = tensor_to_data_uri(start_frame, "image/png")

        # Body matches RunwayImageToVideoRequest:
        #   duration: int enum (5 | 10)
        #   model:    gen4_turbo | gen3a_turbo
        #   promptImage: [{uri, position}]  (RunwayPromptImageObject -> RunwayPromptImageDetailedObject)
        #   promptText, ratio, seed
        body = {
            "duration": int(duration),
            "model": model,
            "promptImage": [{"uri": data_uri, "position": "first"}],
            "promptText": prompt,
            "ratio": ratio,
            "seed": int(seed),
        }

        # Submit -> RunwayImageToVideoResponse { id }
        submit = run_async(http_request("POST", f"{base}/image_to_video", headers, body))
        task_id = submit.get("id")
        if not task_id:
            raise RuntimeError(f"Runway did not return a task id: {submit}")

        # Poll /tasks/{id} until SUCCEEDED/FAILED -> RunwayTaskStatusResponse.
        final = run_async(self._poll(base, headers, task_id, poll_interval, max_wait))

        output = final.get("output") or []
        if not output:
            raise RuntimeError(f"Runway task succeeded but no video output found: {final}")
        video_url = output[0]

        return (run_async(download_video_output(video_url)),)

    async def _poll(self, base, headers, task_id, poll_interval, max_wait):
        async def poll_fn():
            return await http_request("GET", f"{base}/tasks/{task_id}", headers)

        return await poll_until(
            poll_fn,
            status_of=lambda r: r.get("status"),
            done_statuses=DONE_STATUSES,
            failed_statuses=FAILED_STATUSES,
            interval=poll_interval,
            max_wait=max_wait,
            on_fail_message="Runway image-to-video task failed",
        )


class BYOKey_Runway_Text2Image:
    """Runway Text to Image (Gen4 Image). Output: IMAGE.

    Mirrors the official ``RunwayTextToImageNode``: a text prompt and an optional
    reference image produce a generated image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Runway API key (sent as Bearer token)."}),
                "base_url": ("STRING", {"default": RUNWAY_API_BASE, "tooltip": "Runway API base URL. Override for relays."}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text prompt for the generation."}),
                "ratio": (T2I_RATIOS, {"default": "1024:1024", "tooltip": "Aspect ratio for the generated image."}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Optional reference image to guide the generation."}),
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 120, "tooltip": "Seconds between status polls."}),
                "max_wait": ("INT", {"default": 300, "min": 30, "max": 3600, "tooltip": "Max seconds to wait for the task."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Generate an image from a text prompt using Runway's Gen 4 model, optionally guided by a reference image."

    def execute(self, api_key, base_url, prompt, ratio="1024:1024",
                reference_image=None, poll_interval=5, max_wait=300):
        validate_prompt(prompt, min_length=1)

        base = base_url.rstrip("/")
        headers = _auth_headers(api_key)

        # Body matches RunwayTextToImageRequest:
        #   model: gen4_image
        #   promptText, ratio
        #   referenceImages: [{uri}]  (ReferenceImage)
        body = {
            "model": T2I_MODEL,
            "promptText": prompt,
            "ratio": ratio,
        }
        if reference_image is not None:
            uri = tensor_to_data_uri(reference_image, "image/png")
            body["referenceImages"] = [{"uri": uri}]

        # Submit -> RunwayTextToImageResponse { id }
        submit = run_async(http_request("POST", f"{base}/text_to_image", headers, body))
        task_id = submit.get("id")
        if not task_id:
            raise RuntimeError(f"Runway did not return a task id: {submit}")

        # Poll /tasks/{id} until SUCCEEDED/FAILED -> RunwayTaskStatusResponse.
        final = run_async(self._poll(base, headers, task_id, poll_interval, max_wait))

        output = final.get("output") or []
        if not output:
            raise RuntimeError(f"Runway task succeeded but no image output found: {final}")
        image_url = output[0]

        img_bytes = run_async(download_bytes(image_url))
        return (bytes_to_tensor(img_bytes, "RGB"),)

    async def _poll(self, base, headers, task_id, poll_interval, max_wait):
        async def poll_fn():
            return await http_request("GET", f"{base}/tasks/{task_id}", headers)

        return await poll_until(
            poll_fn,
            status_of=lambda r: r.get("status"),
            done_statuses=DONE_STATUSES,
            failed_statuses=FAILED_STATUSES,
            interval=poll_interval,
            max_wait=max_wait,
            on_fail_message="Runway text-to-image task failed",
        )
