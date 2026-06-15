"""MiniMax (Hailuo) video generation nodes (bring-your-own-key).

Faithful BYOKey port of the official ComfyUI MiniMax video nodes
(``comfy_api_nodes/nodes_minimax.py`` + ``comfy_api_nodes/apis/minimax.py``)
that calls the real MiniMax vendor API directly instead of the ComfyOrg proxy.

The official nodes post to ``/proxy/minimax/...``; the proxy forwards the body
verbatim, so the official request/response models ARE the real MiniMax API
contract. Three-step flow (mirrors the official node exactly):

  1. POST  {base}/video_generation            -> {task_id, base_resp}
  2. GET   {base}/query/video_generation?task_id=...  (poll status Success/Fail)
     -> {status, file_id}
  3. GET   {base}/files/retrieve?file_id=...  -> {file:{download_url, backup_download_url}}
  4. download -> VIDEO

Output type matches official: VIDEO.
"""
import asyncio

import torch

from .utils import (
    download_video_output,
    http_request,
    run_async,
    tensor_to_data_uri,
    validate_prompt,
)

# Real MiniMax international API host. Chinese mirror is https://api.minimax.chat/v1;
# override base_url to switch.
MINIMAX_DEFAULT_BASE_URL = "https://api.minimaxi.chat/v1"

# Done / failed task statuses (Status6 enum in apis/minimax.py).
_DONE = ["Success"]
_FAILED = ["Fail"]

# Official node model combos (MinimaxTextToVideoNode / MinimaxImageToVideoNode).
T2V_MODELS = ["T2V-01", "T2V-01-Director"]
I2V_MODELS = ["I2V-01", "I2V-01-Director", "I2V-01-live"]


def _build_headers(api_key: str) -> dict:
    """MiniMax auth: Bearer token + JSON."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


async def _generate_mm_video(
    *,
    api_key: str,
    base_url: str,
    prompt_text: str,
    model: str,
    seed: int,
    image: torch.Tensor | None = None,
    prompt_optimizer: bool = True,
    poll_interval: int = 10,
    max_wait: int = 600,
):
    """Submit a MiniMax video task, poll to completion, then download -> VIDEO."""
    base = base_url.rstrip("/")
    headers = _build_headers(api_key)

    # Step 1: submit. Mirrors MinimaxVideoGenerationRequest field-for-field.
    first_frame = None
    if image is not None:
        # Official uploads the image and passes a URL; for a direct call we pass an
        # inline data URI, which the MiniMax API accepts for first_frame_image.
        first_frame = tensor_to_data_uri(image, "image/png")

    body = {
        "model": model,
        "prompt": prompt_text,
        "callback_url": None,
        "first_frame_image": first_frame,
        "subject_reference": None,
        "prompt_optimizer": prompt_optimizer,
    }
    if seed and seed > 0:
        body["seed"] = seed

    submit = await http_request("POST", f"{base}/video_generation", headers, body)
    task_id = submit.get("task_id")
    if not task_id:
        raise RuntimeError(f"MiniMax generation failed to start: {submit.get('base_resp', submit)}")

    # Step 2: poll query/video_generation until Success/Fail.
    elapsed = 0.0
    last = None
    file_id = None
    while elapsed < float(max_wait):
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        last = await http_request(
            "GET", f"{base}/query/video_generation?task_id={task_id}", headers,
        )
        status = str((last.get("status") or "").lower())
        if status in [s.lower() for s in _FAILED]:
            raise RuntimeError(f"MiniMax task failed. Status={status}. Response={last}")
        if status in [s.lower() for s in _DONE]:
            file_id = last.get("file_id")
            break

    if file_id is None:
        raise RuntimeError(f"MiniMax task did not complete in time. Last response={last}")

    # Step 3: retrieve the file -> download_url (+ backup_download_url).
    file_result = await http_request(
        "GET", f"{base}/files/retrieve?file_id={int(file_id)}", headers,
    )
    file_info = file_result.get("file") or {}
    download_url = file_info.get("download_url")
    if not download_url:
        raise RuntimeError(f"MiniMax returned no download URL. Response={file_result}")

    # Step 4: download -> VIDEO. MiniMax's primary download_url can expire;
    # the vendor provides backup_download_url for exactly this case (the official
    # node retries with it too), so use it when the primary fetch fails.
    try:
        return await download_video_output(download_url, headers=headers, timeout=600.0)
    except Exception:
        backup = file_info.get("backup_download_url")
        if not backup:
            raise
        return await download_video_output(backup, headers=headers, timeout=600.0)


class BYOKey_MiniMax_Text2Video:
    """MiniMax/Hailuo text-to-video. Output: VIDEO.

    Mirrors the official ``MinimaxTextToVideoNode``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "MiniMax API key (Bearer token)."}),
                "base_url": ("STRING", {
                    "default": MINIMAX_DEFAULT_BASE_URL,
                    "tooltip": "MiniMax API base URL. International default; use "
                               "https://api.minimax.chat/v1 for the Chinese mirror.",
                }),
                "prompt_text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Text prompt to guide the video generation (max 2000 chars).",
                }),
                "model": (T2V_MODELS, {
                    "default": "T2V-01",
                    "tooltip": "Model to use for video generation.",
                }),
            },
            "optional": {
                "prompt_optimizer": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true (default), the model optimizes the prompt automatically.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "Random seed; results are non-deterministic regardless of seed.",
                }),
                "poll_interval": ("INT", {"default": 10, "min": 3, "max": 60, "tooltip": "Seconds between status polls."}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600, "tooltip": "Max seconds to wait for completion."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Generate video from a text prompt using MiniMax (Hailuo)."

    def execute(self, api_key, base_url, prompt_text, model,
                prompt_optimizer=True, seed=0, poll_interval=10, max_wait=600):
        validate_prompt(prompt_text, min_length=1)
        if len(prompt_text) > 2000:
            raise ValueError("Prompt must be at most 2000 characters.")
        video = run_async(_generate_mm_video(
            api_key=api_key, base_url=base_url, prompt_text=prompt_text, model=model,
            seed=seed, image=None, prompt_optimizer=prompt_optimizer,
            poll_interval=poll_interval, max_wait=max_wait,
        ))
        return (video,)


class BYOKey_MiniMax_Image2Video:
    """MiniMax/Hailuo image-to-video. Output: VIDEO.

    Mirrors the official ``MinimaxImageToVideoNode``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "MiniMax API key (Bearer token)."}),
                "base_url": ("STRING", {"default": MINIMAX_DEFAULT_BASE_URL, "tooltip": "MiniMax API base URL."}),
                "image": ("IMAGE", {"tooltip": "Image to use as the first frame of the generated video."}),
                "prompt_text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Text prompt to guide the video generation (max 2000 chars).",
                }),
                "model": (I2V_MODELS, {
                    "default": "I2V-01",
                    "tooltip": "Model to use for video generation.",
                }),
            },
            "optional": {
                "prompt_optimizer": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                }),
                "poll_interval": ("INT", {"default": 10, "min": 3, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Generate video from an image + text prompt using MiniMax (Hailuo)."

    def execute(self, api_key, base_url, image, prompt_text, model,
                prompt_optimizer=True, seed=0, poll_interval=10, max_wait=600):
        if len(prompt_text) > 2000:
            raise ValueError("Prompt must be at most 2000 characters.")
        video = run_async(_generate_mm_video(
            api_key=api_key, base_url=base_url, prompt_text=prompt_text, model=model,
            seed=seed, image=image, prompt_optimizer=prompt_optimizer,
            poll_interval=poll_interval, max_wait=max_wait,
        ))
        return (video,)
