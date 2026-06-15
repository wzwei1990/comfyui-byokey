"""Google Veo video generation node (bring-your-own-key).

Mirrors the official ComfyUI ``VeoVideoGenerationNode`` / ``Veo3VideoGenerationNode``
request/response handling, but calls Google's Generative Language API directly with the
user's own API key instead of going through the ComfyOrg proxy.

The official nodes hit ``/proxy/veo/{model}/generate`` and ``/proxy/veo/{model}/poll``;
the proxy forwards the body verbatim to Google's ``predictLongRunning`` long-running
operation endpoint, so the request/response shape used here is the real vendor API:

    POST  {base_url}/v1beta/models/{model}:predictLongRunning?key={api_key}
    GET   {base_url}/v1beta/{operationName}?key={api_key}

Output type is VIDEO, matching the official nodes.
"""

import base64

from .utils import (
    download_video_output,
    http_request,
    run_async,
    tensor_to_base64,
    validate_prompt,
    video_from_bytes,
)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

# Maps the user-facing model label to the concrete Google model id, matching the
# official ``MODELS_MAP`` in comfy_api_nodes/nodes_veo2.py.
MODELS_MAP = {
    "veo-2.0-generate-001": "veo-2.0-generate-001",
    "veo-3.1-generate": "veo-3.1-generate-001",
    "veo-3.1-fast-generate": "veo-3.1-fast-generate-001",
    "veo-3.1-lite": "veo-3.1-lite-generate-001",
    "veo-3.0-generate-001": "veo-3.0-generate-001",
    "veo-3.0-fast-generate-001": "veo-3.0-fast-generate-001",
}

MODEL_OPTIONS = [
    "veo-2.0-generate-001",
    "veo-3.1-generate",
    "veo-3.1-fast-generate",
    "veo-3.1-lite",
    "veo-3.0-generate-001",
    "veo-3.0-fast-generate-001",
]


class BYOKey_Veo:
    """Google Veo video generation (Veo 2 / Veo 3), output VIDEO.

    Submits a long-running ``predictLongRunning`` operation to Google's Generative
    Language API and polls the operation until it reports ``done``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Google AI (Gemini API) key"}),
                "base_url": (
                    "STRING",
                    {
                        "default": DEFAULT_BASE_URL,
                        "tooltip": "Vendor API host. Default targets Google directly; override for relays.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Text description of the video"},
                ),
                "aspect_ratio": (
                    ["16:9", "9:16"],
                    {"default": "16:9", "tooltip": "Aspect ratio of the output video"},
                ),
            },
            "optional": {
                "model": (
                    MODEL_OPTIONS,
                    {"default": "veo-2.0-generate-001", "tooltip": "Veo model to use for video generation"},
                ),
                "resolution": (
                    ["720p", "1080p", "4k"],
                    {
                        "default": "720p",
                        "tooltip": "Output video resolution. Only applied for veo-3.1 models. "
                        "4K is not available for veo-3.1-lite or veo-3.0 models.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Negative text prompt to guide what to avoid in the video",
                    },
                ),
                "duration_seconds": (
                    "INT",
                    {
                        "default": 5,
                        "min": 4,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Duration of the output video in seconds (5-8 for Veo 2, 4-8 step 2 for Veo 3)",
                    },
                ),
                "enhance_prompt": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enhance the prompt with AI assistance. Forced True for Veo 3 models.",
                    },
                ),
                "person_generation": (
                    ["ALLOW", "BLOCK"],
                    {"default": "ALLOW", "tooltip": "Whether to allow generating people in the video"},
                ),
                "generate_audio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Generate audio for the video. Only supported by Veo 3 models.",
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFF, "tooltip": "Seed for video generation (0 for random)"},
                ),
                "image": ("IMAGE", {"tooltip": "Optional reference image to guide video generation"}),
                "poll_interval": (
                    "INT",
                    {"default": 9, "min": 2, "max": 60, "tooltip": "Seconds between operation status polls"},
                ),
                "max_wait": (
                    "INT",
                    {"default": 600, "min": 60, "max": 3600, "tooltip": "Maximum seconds to wait for the video"},
                ),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Generate videos from text prompts using Google's Veo 2 / Veo 3 API (bring your own key)."

    def execute(
        self,
        api_key,
        base_url,
        prompt,
        aspect_ratio="16:9",
        model="veo-2.0-generate-001",
        resolution="720p",
        negative_prompt="",
        duration_seconds=5,
        enhance_prompt=True,
        person_generation="ALLOW",
        generate_audio=False,
        seed=0,
        image=None,
        poll_interval=9,
        max_wait=600,
    ):
        prompt = validate_prompt(prompt)

        if resolution == "4k" and ("lite" in model or "3.0" in model):
            raise RuntimeError("4K resolution is not supported by the veo-3.1-lite or veo-3.0 models.")

        resolved_model = MODELS_MAP.get(model, model)
        is_veo2 = resolved_model.find("veo-2.0") != -1

        # --- build instance ---------------------------------------------------
        instance = {"prompt": prompt}
        if image is not None:
            image_base64 = tensor_to_base64(image, mime_type="image/png")
            if image_base64:
                instance["image"] = {"bytesBase64Encoded": image_base64, "mimeType": "image/png"}

        # --- build parameters (field names match VeoRequestParameters) --------
        parameters = {
            "aspectRatio": aspect_ratio,
            "personGeneration": person_generation,
            "durationSeconds": duration_seconds,
            "enhancePrompt": enhance_prompt,
        }
        if negative_prompt:
            parameters["negativePrompt"] = negative_prompt
        if seed > 0:
            parameters["seed"] = seed
        if not is_veo2:
            # Veo 3: generateAudio supported, enhancePrompt forced True.
            parameters["generateAudio"] = generate_audio
            parameters["enhancePrompt"] = True
            if "veo-3.1" in resolved_model:
                parameters["resolution"] = resolution

        body = {"instances": [instance], "parameters": parameters}

        base = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}

        # --- submit long-running operation ------------------------------------
        submit_url = f"{base}/v1beta/models/{resolved_model}:predictLongRunning?key={api_key}"
        submit_resp = run_async(http_request("POST", submit_url, headers, body, timeout=120.0))
        operation_name = submit_resp.get("name")
        if not operation_name:
            raise RuntimeError(f"Veo submit returned no operation name: {submit_resp}")

        # --- poll operation ----------------------------------------------------
        poll_url = f"{base}/v1beta/{operation_name}?key={api_key}"
        final = run_async(
            self._poll(poll_url, headers, interval=poll_interval, max_wait=max_wait)
        )

        # --- error / RAI handling (matches official extraction) ---------------
        error = final.get("error")
        if error:
            raise RuntimeError(
                f"Veo API error: {error.get('message')} (code: {error.get('code')})"
            )

        response = final.get("response") or {}
        # The actual prediction payload may be nested under generateVideoResponse.
        gen = response.get("generateVideoResponse") or response

        filtered_count = gen.get("raiMediaFilteredCount") or 0
        if filtered_count:
            reasons = gen.get("raiMediaFilteredReasons") or []
            reason_part = f": {reasons[0]}" if reasons else ""
            raise RuntimeError(
                f"Content blocked by Google's Responsible AI filters{reason_part} "
                f"({filtered_count} video{'s' if filtered_count != 1 else ''} filtered)."
            )

        videos = gen.get("videos") or gen.get("predictions") or []
        if not videos:
            raise RuntimeError(f"Video generation completed but no video was returned: {final}")

        video = videos[0]
        # Inline base64 video content.
        b64 = video.get("bytesBase64Encoded")
        if b64:
            return (video_from_bytes(base64.b64decode(b64)),)
        # URL-referenced video (Cloud Storage / signed URL).
        url = video.get("gcsUri") or video.get("videoUri") or video.get("uri")
        if url:
            dl_headers = None if "key=" in url or url.startswith("data:") else headers
            return (run_async(download_video_output(url, headers=dl_headers)),)

        raise RuntimeError(f"Video returned but no data or URL was provided: {video}")

    async def _poll(self, poll_url, headers, *, interval, max_wait):
        """Poll the long-running operation until ``done`` is true or timeout."""
        import asyncio

        elapsed = 0.0
        last = None
        while elapsed < max_wait:
            last = await http_request("GET", poll_url, headers, timeout=120.0)
            if last.get("done"):
                return last
            if last.get("error"):
                return last
            await asyncio.sleep(interval)
            elapsed += interval
        raise RuntimeError(f"Veo timed out after {max_wait}s. Last response={last}")
