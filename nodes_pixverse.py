"""PixVerse text-to-video node (bring-your-own-key).

Faithfully mirrors the official ComfyUI ``PixverseTextToVideoNode`` request /
response contract, but calls the real PixVerse OpenAPI directly instead of going
through the ComfyOrg proxy.

Ground truth:
  * Official node:    comfy_api_nodes/nodes_pixverse.py (PixverseTextToVideoNode)
  * Official models:  comfy_api_nodes/apis/pixverse.py
  * Vendor API host:  https://app-api.pixverse.ai/openapi/v2
      - submit:  POST /video/text/generate   (body == PixverseTextVideoRequest)
      - poll:    GET  /video/result/{video_id}
      - headers: API-KEY, Ai-trace-id (unique uuid per request), Content-Type
      - status (int): 1 successful, 5 generating, 6 deleted,
                      7 contents_moderation, 8 failed

The ComfyOrg proxy forwards the official request body verbatim to this vendor
API, so the official models ARE the real vendor request/response shapes.
"""
import uuid

from .utils import (
    download_video_output,
    http_request,
    poll_until,
    run_async,
    validate_prompt,
)

# Real PixVerse OpenAPI host (overridable for relays).
PIXVERSE_API_BASE = "https://app-api.pixverse.ai/openapi/v2"

# COMBO option lists (match official define_schema + embedded docs).
ASPECT_RATIOS = ["16:9", "4:3", "1:1", "3:4", "9:16"]
QUALITIES = ["360p", "540p", "720p", "1080p"]
DURATIONS = [5, 8]
MOTION_MODES = ["normal", "fast"]

# Vendor status codes (PixverseStatus enum, ints).
STATUS_SUCCESSFUL = 1
STATUS_GENERATING = 5
STATUS_DELETED = 6
STATUS_CONTENTS_MODERATION = 7
STATUS_FAILED = 8


class BYOKey_PixVerse_Text2Video:
    """PixVerse text-to-video. Output: VIDEO.

    Mirrors official ``PixverseTextToVideoNode``: submit a text prompt with
    aspect ratio / quality / duration / motion mode, poll the result endpoint,
    and download the finished mp4 as a ComfyUI VIDEO object.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Your PixVerse API key (sent as the API-KEY header)."}),
                "base_url": ("STRING", {"default": PIXVERSE_API_BASE, "tooltip": "PixVerse OpenAPI base URL; override for a relay."}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Prompt for the video generation."}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
                "quality": (QUALITIES, {"default": "540p"}),
                "duration_seconds": (DURATIONS, {"default": 5}),
                "motion_mode": (MOTION_MODES, {"default": "normal"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True, "tooltip": "Seed for video generation."}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "An optional text description of undesired elements."}),
                "model": ("STRING", {"default": "v3.5", "tooltip": "PixVerse model version (e.g. v3.5)."}),
                "template_id": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "Optional PixVerse template id (0 = none)."}),
                "poll_interval": ("INT", {"default": 10, "min": 1, "max": 60, "tooltip": "Seconds between status polls."}),
                "max_wait": ("INT", {"default": 600, "min": 30, "max": 3600, "tooltip": "Max seconds to wait for the video."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Generates videos based on prompt and output_size (PixVerse, bring your own key)."

    def execute(
        self,
        api_key,
        base_url,
        prompt,
        aspect_ratio,
        quality,
        duration_seconds,
        motion_mode,
        seed,
        negative_prompt="",
        model="v3.5",
        template_id=0,
        poll_interval=10,
        max_wait=600,
    ):
        validate_prompt(prompt, min_length=1)

        # Official constraints:
        # 1080p is limited to 5 seconds duration and only normal motion mode.
        # Only normal motion mode is supported for non-5-second durations.
        if quality == "1080p":
            motion_mode = "normal"
            duration_seconds = 5
        elif duration_seconds != 5:
            motion_mode = "normal"

        base = base_url.rstrip("/")

        # Request body == PixverseTextVideoRequest (only set optional fields when present).
        body = {
            "aspect_ratio": aspect_ratio,
            "quality": quality,
            "duration": int(duration_seconds),
            "model": model,
            "motion_mode": motion_mode,
            "prompt": prompt,
        }
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if seed:
            body["seed"] = int(seed)
        if template_id:
            body["template_id"] = int(template_id)

        video = run_async(self._generate(base, api_key, body, poll_interval, max_wait))
        return (video,)

    @staticmethod
    def _headers(api_key):
        # PixVerse OpenAPI auth: API-KEY plus a unique Ai-trace-id per request.
        return {
            "API-KEY": api_key,
            "Ai-trace-id": str(uuid.uuid4()),
            "Content-Type": "application/json",
        }

    async def _generate(self, base, api_key, body, poll_interval, max_wait):
        # Submit generation (response == PixverseVideoResponse: {ErrCode, ErrMsg, Resp:{video_id}}).
        submit = await http_request(
            "POST",
            f"{base}/video/text/generate",
            self._headers(api_key),
            json_data=body,
        )
        resp = submit.get("Resp") or {}
        video_id = resp.get("video_id")
        if not video_id:
            raise RuntimeError(
                f"PixVerse request failed: '{submit.get('ErrMsg')}' (ErrCode={submit.get('ErrCode')}). Response={submit}"
            )

        # Poll result (response == PixverseGenerationStatusResponse: {..., Resp:{status, url}}).
        async def poll_fn():
            return await http_request(
                "GET",
                f"{base}/video/result/{video_id}",
                self._headers(api_key),
            )

        final = await poll_until(
            poll_fn,
            status_of=lambda x: (x.get("Resp") or {}).get("status"),
            done_statuses=[str(STATUS_SUCCESSFUL)],
            failed_statuses=[
                str(STATUS_CONTENTS_MODERATION),
                str(STATUS_FAILED),
                str(STATUS_DELETED),
            ],
            interval=poll_interval,
            max_wait=max_wait,
            on_fail_message="PixVerse generation failed",
        )

        url = (final.get("Resp") or {}).get("url")
        if not url:
            raise RuntimeError(f"PixVerse returned no video URL. Response={final}")
        return await download_video_output(url)
