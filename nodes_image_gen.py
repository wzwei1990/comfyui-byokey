"""OpenAI GPT Image / DALL-E 3 image generation node (bring-your-own-key).

Faithfully mirrors the official ComfyUI ``OpenAIGPTImageNodeV2`` node (with a
DALL-E 3 compatibility path) but calls the OpenAI Images API directly using the
caller's own API key and base URL instead of routing through the ComfyOrg proxy.

The official nodes POST to ``/proxy/openai/images/generations`` and
``/proxy/openai/images/edits``; the proxy forwards the body verbatim to the real
OpenAI API, so those request/response models and endpoint paths are the real
vendor contract. We reproduce them field-for-field against ``api.openai.com/v1``.

Request body (``OpenAIImageGenerationRequest`` / ``OpenAIImageEditRequest``):
    model, prompt, quality, background, n, size, moderation
Generation is JSON; editing (when reference images are supplied) is multipart.
Response (``OpenAIImageGenerationResponse``): ``{"data": [{"b64_json"|"url", ...}]}``.
"""
import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from .utils import (
    bytes_to_tensor,
    download_bytes,
    http_request,
    run_async,
    validate_prompt,
)

# Model id list, matching the official OpenAIGPTImageNodeV2 DynamicCombo options.
GPT_IMAGE_MODELS = ["gpt-image-2", "gpt-image-1.5", "gpt-image-1"]

# Size options. gpt-image-2 supports the full set incl. "Custom"; the legacy
# gpt-image-1 / gpt-image-1.5 models only support the first four base sizes.
GPT_IMAGE_2_SIZES = [
    "auto",
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "2048x2048",
    "2048x1152",
    "1152x2048",
    "3840x2160",
    "2160x3840",
    "Custom",
]
LEGACY_GPT_IMAGE_SIZES = ["auto", "1024x1024", "1024x1536", "1536x1024"]

# DALL-E 3 sizes/quality/style (from the OpenAIDalle3 official node).
DALLE3_SIZES = ["1024x1024", "1024x1792", "1792x1024"]


def _max_total_pixels() -> int:
    # Matches official downscale_image_tensor(total_pixels=2048*2048) behaviour
    # used before uploading reference images / masks.
    return 2048 * 2048


def _downscale_tensor(image: torch.Tensor, total_pixels: int) -> torch.Tensor:
    """Downscale a [1,H,W,C] float tensor so H*W <= total_pixels (bilinear).

    Mirrors comfy_api_nodes.util.downscale_image_tensor without importing it.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    samples = image.movedim(-1, 1)  # [B,C,H,W]
    _, _, h, w = samples.shape
    if h * w <= total_pixels:
        return image
    scale = (total_pixels / float(h * w)) ** 0.5
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = torch.nn.functional.interpolate(
        samples, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    return resized.movedim(1, -1)


def _tensor_to_png_bytes(image: torch.Tensor) -> BytesIO:
    """Convert a [1,H,W,C] (or [H,W,C]) float tensor to a PNG BytesIO."""
    scaled = _downscale_tensor(image, _max_total_pixels()).squeeze()
    image_np = (scaled.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _validate_custom_size(custom_width: int, custom_height: int) -> str:
    """Validate the Custom resolution constraints (official OpenAIGPTImageNodeV2)."""
    if custom_width % 16 != 0 or custom_height % 16 != 0:
        raise ValueError(
            f"Custom width and height must be multiples of 16, got {custom_width}x{custom_height}"
        )
    if max(custom_width, custom_height) > 3840:
        raise ValueError(
            f"Custom resolution max edge must be <= 3840, got {custom_width}x{custom_height}"
        )
    ratio = max(custom_width, custom_height) / min(custom_width, custom_height)
    if ratio > 3:
        raise ValueError(
            f"Custom resolution aspect ratio must not exceed 3:1, got {custom_width}x{custom_height}"
        )
    total_pixels = custom_width * custom_height
    if not 655_360 <= total_pixels <= 8_294_400:
        raise ValueError(
            f"Custom resolution total pixels must be between 655,360 and 8,294,400, got {total_pixels}"
        )
    return f"{custom_width}x{custom_height}"


def _flatten_images(image: torch.Tensor) -> list[torch.Tensor]:
    """Split a batched IMAGE tensor into a list of single [1,H,W,C] tensors."""
    if image is None:
        return []
    if image.dim() == 4:
        return [image[i : i + 1] for i in range(image.shape[0])]
    return [image.unsqueeze(0)]


async def _response_to_image_tensor(resp: dict) -> torch.Tensor:
    """Parse an OpenAIImageGenerationResponse dict into a stacked IMAGE tensor.

    Mirrors the official validate_and_cast_response: decode b64_json or download
    url for each item, convert to RGBA float, resize stragglers to the first
    image's dimensions, then stack into a [N,H,W,C] batch.
    """
    data = resp.get("data") or []
    if not data:
        raise ValueError("No images returned from API endpoint")

    tensors: list[torch.Tensor] = []
    for item in data:
        b64 = item.get("b64_json")
        url = item.get("url")
        if b64:
            img_bytes = base64.b64decode(b64)
        elif url:
            img_bytes = await download_bytes(url)
        else:
            raise ValueError("Invalid image payload - neither URL nor base64 data present.")
        # bytes_to_tensor returns [1,H,W,C] RGBA; drop the batch dim to stack later.
        tensors.append(bytes_to_tensor(img_bytes, mode="RGBA").squeeze(0))

    # With size="auto" the API can return images that differ by a few pixels;
    # resize them all to the first image's dimensions so they can be stacked.
    ref_h, ref_w = tensors[0].shape[:2]
    for i, t in enumerate(tensors):
        if t.shape[:2] != (ref_h, ref_w):
            samples = t.unsqueeze(0).movedim(-1, 1)
            samples = torch.nn.functional.interpolate(
                samples, size=(ref_h, ref_w), mode="bilinear", align_corners=False
            )
            tensors[i] = samples.movedim(1, -1).squeeze(0)
    return torch.stack(tensors, dim=0)


class BYOKey_ImageGen:
    """Image generation via OpenAI's GPT Image / DALL-E 3 Images API.

    Drop-in equivalent of the official OpenAI GPT Image 2 node (same IMAGE
    output), but calls the OpenAI API directly with your own key. Supports
    text-to-image (``images/generations``) and image editing / inpainting
    (``images/edits``, multipart) with optional reference images and a mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Your OpenAI API key (sk-...)."}),
                "base_url": (
                    "STRING",
                    {
                        "default": "https://api.openai.com/v1",
                        "tooltip": "OpenAI-compatible API base URL. Override for relays/proxies.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Text prompt for GPT Image."},
                ),
                "model": (
                    GPT_IMAGE_MODELS + ["dall-e-3"],
                    {
                        "default": "gpt-image-2",
                        "tooltip": "The OpenAI image model to use.",
                    },
                ),
                "size": (
                    GPT_IMAGE_2_SIZES,
                    {
                        "default": "auto",
                        "tooltip": "Image size. Select 'Custom' to use custom width/height (gpt-image-2 only).",
                    },
                ),
                "quality": (
                    ["low", "medium", "high", "standard", "hd"],
                    {
                        "default": "low",
                        "tooltip": "Image quality. GPT Image: low/medium/high. DALL-E 3: standard/hd.",
                    },
                ),
                "background": (
                    ["auto", "opaque", "transparent"],
                    {
                        "default": "auto",
                        "tooltip": "Return image with or without background. 'transparent' is not supported by gpt-image-2.",
                    },
                ),
                "n": (
                    "INT",
                    {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "How many images to generate."},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "control_after_generate": True,
                        "tooltip": "not implemented yet in backend",
                    },
                ),
            },
            "optional": {
                "custom_width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1024,
                        "max": 3840,
                        "step": 16,
                        "tooltip": "Used only when size is 'Custom'. Multiple of 16 (gpt-image-2 only).",
                    },
                ),
                "custom_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1024,
                        "max": 3840,
                        "step": 16,
                        "tooltip": "Used only when size is 'Custom'. Multiple of 16 (gpt-image-2 only).",
                    },
                ),
                "style": (
                    ["natural", "vivid"],
                    {
                        "default": "natural",
                        "tooltip": "DALL-E 3 only. Vivid leans hyper-real/dramatic; natural is more natural.",
                    },
                ),
                "image": (
                    "IMAGE",
                    {"tooltip": "Optional reference image(s) for editing (gpt-image-* only)."},
                ),
                "mask": (
                    "MASK",
                    {"tooltip": "Optional inpaint mask (white areas replaced). Requires exactly one reference image."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Generate images via OpenAI GPT Image / DALL-E 3 with your own API key (no ComfyOrg proxy)."

    def execute(
        self,
        api_key,
        base_url,
        prompt,
        model="gpt-image-2",
        size="auto",
        quality="low",
        background="auto",
        n=1,
        seed=0,
        custom_width=1024,
        custom_height=1024,
        style="natural",
        image=None,
        mask=None,
    ):
        validate_prompt(prompt, min_length=1)

        if model == "dall-e-3":
            return self._run_dalle3(api_key, base_url, prompt, quality, style, size, seed, n)
        return self._run_gpt_image(
            api_key,
            base_url,
            prompt,
            model,
            size,
            quality,
            background,
            n,
            seed,
            custom_width,
            custom_height,
            image,
            mask,
        )

    # ----------------------------------------------------------------- #
    # GPT Image (gpt-image-2 / 1.5 / 1)                                  #
    # ----------------------------------------------------------------- #
    def _run_gpt_image(
        self,
        api_key,
        base_url,
        prompt,
        model,
        size,
        quality,
        background,
        n,
        seed,
        custom_width,
        custom_height,
        image,
        mask,
    ):
        # GPT Image quality is low/medium/high; coerce DALL-E-style values.
        if quality in ("standard", "hd"):
            quality = "high" if quality == "hd" else "medium"

        flat_images = _flatten_images(image)

        if mask is not None and not flat_images:
            raise ValueError("Cannot use a mask without an input image")

        # Resolve size / custom resolution.
        if size == "Custom":
            if model != "gpt-image-2":
                raise ValueError("Custom resolution is only supported by the gpt-image-2 model")
            size = _validate_custom_size(custom_width, custom_height)
        elif model in ("gpt-image-1", "gpt-image-1.5"):
            if size not in LEGACY_GPT_IMAGE_SIZES:
                raise ValueError(f"Resolution {size} is only supported by the gpt-image-2 model")

        if model == "gpt-image-2" and background == "transparent":
            raise ValueError("Transparent background is not supported for the gpt-image-2 model")

        headers = {"Authorization": f"Bearer {api_key}"}

        # Request body fields match OpenAIImageGenerationRequest / OpenAIImageEditRequest.
        body = {
            "model": model,
            "prompt": prompt,
            "quality": quality,
            "background": background,
            "n": int(n),
            "size": size,
            "moderation": "low",
        }

        if flat_images:
            resp = run_async(
                self._edit_request(base_url, headers, body, flat_images, mask)
            )
        else:
            json_headers = dict(headers)
            json_headers["Content-Type"] = "application/json"
            url = f"{base_url.rstrip('/')}/images/generations"
            resp = run_async(http_request("POST", url, json_headers, json_data=body))

        return (run_async(_response_to_image_tensor(resp)),)

    async def _edit_request(self, base_url, headers, body, flat_images, mask):
        """POST a multipart images/edits request and return the parsed JSON."""
        import aiohttp

        form = aiohttp.FormData()
        for key, value in body.items():
            if value is None:
                continue
            form.add_field(key, str(value))

        # Field name is "image" for a single reference, "image[]" for multiple,
        # matching the official OpenAIGPTImageNodeV2 multipart layout.
        single = len(flat_images) == 1
        for i, single_image in enumerate(flat_images):
            buf = _tensor_to_png_bytes(single_image)
            field = "image" if single else "image[]"
            form.add_field(
                field, buf.getvalue(), filename=f"image_{i}.png", content_type="image/png"
            )

        if mask is not None:
            if not single:
                raise ValueError("Cannot use a mask with multiple images")
            ref_image = flat_images[0]
            if mask.shape[1:] != ref_image.shape[1:-1]:
                raise ValueError("Mask and Image must be the same size")
            _, height, width = mask.shape
            rgba_mask = torch.zeros(height, width, 4, device="cpu")
            rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()
            mask_buf = _tensor_to_png_bytes(rgba_mask.unsqueeze(0))
            form.add_field(
                "mask", mask_buf.getvalue(), filename="mask.png", content_type="image/png"
            )

        url = f"{base_url.rstrip('/')}/images/edits"
        return await http_request("POST", url, headers, data=form)

    # ----------------------------------------------------------------- #
    # DALL-E 3                                                          #
    # ----------------------------------------------------------------- #
    def _run_dalle3(self, api_key, base_url, prompt, quality, style, size, seed, n):
        # DALL-E 3 quality is standard/hd; coerce GPT-Image-style values.
        if quality in ("low", "medium", "high"):
            quality = "hd" if quality == "high" else "standard"
        if quality not in ("standard", "hd"):
            quality = "standard"

        if size not in DALLE3_SIZES:
            size = "1024x1024"

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": "dall-e-3",
            "prompt": prompt,
            "quality": quality,
            "size": size,
            "style": style,
            "n": int(n),
        }
        url = f"{base_url.rstrip('/')}/images/generations"
        resp = run_async(http_request("POST", url, headers, json_data=body))
        return (run_async(_response_to_image_tensor(resp)),)
