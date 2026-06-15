"""Stability AI image generation nodes (bring-your-own-key).

Faithfully mirrors the official ComfyUI Stability AI image nodes
(``StabilityStableImageUltraNode`` and ``StabilityStableImageSD_3_5Node``) but
calls the Stability AI v2beta REST API DIRECTLY using a user-supplied API key and
base URL -- no ComfyOrg proxy.

Ground truth:
  * comfy_api_nodes/nodes_stability.py
  * comfy_api_nodes/apis/stability.py
  * embedded-docs: StabilityStableImageUltraNode / StabilityStableImageSD_3_5Node

The official nodes POST to ``/proxy/stability/v2beta/<path>``; the proxy forwards
the multipart body verbatim to ``https://api.stability.ai/v2beta/<path>``, so the
official request/response models and endpoint paths are the real vendor API.

Endpoints used here (relative to ``<base_url>``):
  * Stable Image Ultra : POST /stable-image/generate/ultra
  * Stable Diffusion 3.5: POST /stable-image/generate/sd3

Both are synchronous (no submit+poll). Auth is ``Authorization: Bearer <key>``,
the body is ``multipart/form-data`` and ``Accept: application/json`` makes the API
return a JSON object whose ``image`` field is base64-encoded PNG bytes.
"""
import base64

import aiohttp
import numpy as np
import torch
from PIL import Image

from .utils import bytes_to_tensor, run_async, validate_prompt

# Real vendor host -- works out of the box, overridable for relays/proxies.
STABILITY_API_BASE = "https://api.stability.ai/v2beta"

# --------------------------------------------------------------------------- #
# Option lists (kept identical to the official enums / embedded docs)         #
# --------------------------------------------------------------------------- #

# StabilityAspectRatio enum, in the official declaration order.
STABILITY_ASPECT_RATIOS = [
    "1:1",
    "16:9",
    "9:16",
    "3:2",
    "2:3",
    "5:4",
    "4:5",
    "21:9",
    "9:21",
]

# get_stability_style_presets() -> "None" + StabilityStylePreset values.
STABILITY_STYLE_PRESETS = [
    "None",
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
]

# "Stable Image Ultra" plus the Stable Diffusion 3.5 model ids. Selecting the
# Ultra entry routes to the /ultra endpoint; any sd3.5-* entry routes to /sd3.
ULTRA_MODEL = "stable-image-ultra"
SD35_MODELS = [
    "sd3.5-large",
    "sd3.5-large-turbo",
    "sd3.5-medium",
]
STABILITY_MODELS = [ULTRA_MODEL, *SD35_MODELS]


class BYOKey_StabilityImage:
    """Stability AI image generation (Stable Image Ultra and Stable Diffusion 3.5).

    Output type: IMAGE (a ``[1, H, W, C]`` float32 tensor), identical to the
    official Stability AI image nodes.

    Selecting ``model = "stable-image-ultra"`` uses the Stable Image Ultra
    endpoint; selecting any ``sd3.5-*`` model uses the Stable Diffusion 3.5
    endpoint (with ``cfg_scale`` and explicit ``mode``).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "tooltip": "Stability AI API key (sent as 'Authorization: Bearer ...')."},
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": STABILITY_API_BASE,
                        "tooltip": "Stability v2beta API base URL. Override only for relays/proxies.",
                    },
                ),
                "model": (
                    STABILITY_MODELS,
                    {
                        "default": ULTRA_MODEL,
                        "tooltip": "'stable-image-ultra' uses Stable Image Ultra; 'sd3.5-*' uses Stable Diffusion 3.5.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "What you wish to see in the output image. A strong, descriptive prompt "
                        "that clearly defines elements, colors, and subjects will lead to better results. "
                        "To control the weight of a given word use the format `(word:weight)`, where `word` "
                        "is the word you'd like to control the weight of and `weight` is a value between 0 "
                        "and 1.",
                    },
                ),
                "aspect_ratio": (
                    STABILITY_ASPECT_RATIOS,
                    {"default": "1:1", "tooltip": "Aspect ratio of generated image. Ignored in image-to-image mode."},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4294967294,
                        "step": 1,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
            },
            "optional": {
                "style_preset": (
                    STABILITY_STYLE_PRESETS,
                    {"default": "None", "tooltip": "Optional desired style of generated image. 'None' applies no preset."},
                ),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Stable Diffusion 3.5 only: how strictly the diffusion process adheres to the prompt.",
                    },
                ),
                "image": (
                    "IMAGE",
                    {"tooltip": "Optional input image for image-to-image generation."},
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "A blurb of text describing what you do not wish to see in the output image.",
                    },
                ),
                "image_denoise": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Denoise of input image; 0.0 yields an image identical to input, 1.0 is as if no "
                        "image was provided. Only used when an image is provided.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Stability AI image generation (Stable Image Ultra / Stable Diffusion 3.5), bring-your-own-key."

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tensor_to_png_bytes(image: torch.Tensor) -> bytes:
        """Encode the first frame of a ComfyUI IMAGE tensor as PNG bytes."""
        from io import BytesIO

        frame = image[0] if image.dim() == 4 else image
        arr = (frame.cpu().numpy() * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(arr)
        if pil.mode not in ("RGB", "RGBA", "L"):
            pil = pil.convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------ #
    # Execution                                                          #
    # ------------------------------------------------------------------ #
    def execute(
        self,
        api_key,
        base_url,
        model,
        prompt,
        aspect_ratio,
        seed,
        style_preset="None",
        cfg_scale=4.0,
        image=None,
        negative_prompt="",
        image_denoise=0.5,
    ):
        # Official nodes call validate_string(prompt, strip_whitespace=False).
        validate_prompt(prompt, min_length=1)

        base = (base_url or STABILITY_API_BASE).rstrip("/")
        is_sd35 = model in SD35_MODELS
        path = "/stable-image/generate/sd3" if is_sd35 else "/stable-image/generate/ultra"
        url = f"{base}{path}"

        # Normalize optional fields exactly like the official nodes.
        neg = negative_prompt if negative_prompt else None
        style = None if (not style_preset or style_preset == "None") else style_preset

        image_bytes = None
        strength = image_denoise
        if image is not None:
            image_bytes = self._tensor_to_png_bytes(image)
        else:
            # When no image is provided, strength/denoise is dropped.
            strength = None

        # Build multipart fields shared by both endpoints.
        fields: list[tuple[str, str]] = [
            ("prompt", prompt),
            ("output_format", "png"),
            ("seed", str(int(seed))),
        ]
        if neg is not None:
            fields.append(("negative_prompt", neg))
        if style is not None:
            fields.append(("style_preset", style))

        if is_sd35:
            # Stable Diffusion 3.5 specifics.
            mode = "image-to-image" if image is not None else "text-to-image"
            fields.append(("model", model))
            fields.append(("mode", mode))
            fields.append(("cfg_scale", str(float(cfg_scale))))
            # aspect_ratio is ignored / omitted in image-to-image mode.
            if image is None:
                fields.append(("aspect_ratio", aspect_ratio))
        else:
            # Stable Image Ultra always accepts aspect_ratio.
            fields.append(("aspect_ratio", aspect_ratio))

        if strength is not None:
            fields.append(("strength", str(float(strength))))

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

        resp = run_async(self._post(url, headers, fields, image_bytes))

        finish_reason = resp.get("finish_reason")
        if finish_reason is not None and finish_reason != "SUCCESS":
            raise RuntimeError(f"Stability image generation failed: finish_reason={finish_reason}.")

        b64 = resp.get("image")
        if not b64:
            raise RuntimeError(f"No image in Stability response: keys={list(resp.keys())}")

        img_bytes = base64.b64decode(b64)
        # Match official bytesio_to_image_tensor (RGB). Use RGBA-safe util in RGB mode.
        return (bytes_to_tensor(img_bytes, mode="RGB"),)

    @staticmethod
    async def _post(url, headers, fields, image_bytes):
        """POST a multipart/form-data request and return the parsed JSON body."""
        form = aiohttp.FormData()
        for name, value in fields:
            form.add_field(name, value)
        if image_bytes is not None:
            form.add_field(
                "image",
                image_bytes,
                filename="image.png",
                content_type="image/png",
            )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                data=form,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                body = await resp.read()
                if resp.status >= 400:
                    raise RuntimeError(
                        f"Stability API error (HTTP {resp.status}): {body[:2000].decode('utf-8', 'replace')}"
                    )
                import json as _json

                try:
                    return _json.loads(body)
                except Exception as exc:  # pragma: no cover - defensive
                    raise RuntimeError(
                        f"Could not parse Stability JSON response (HTTP {resp.status}): "
                        f"{body[:500].decode('utf-8', 'replace')}"
                    ) from exc
