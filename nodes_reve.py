"""Reve AI image generation nodes (bring-your-own-key).

Mirrors the official ComfyUI ``ReveImageCreateNode`` but calls the real Reve
vendor API directly instead of going through the ComfyOrg proxy. The official
node posts to ``/proxy/reve/v1/image/create``; the proxy forwards the body
verbatim to ``https://api.reve.com/v1/image/create``, so the official request
model (``ReveImageCreateRequest``) and endpoint path ARE the real vendor API.

Request body (field-for-field with ``comfy_api_nodes/apis/reve.py``):
    prompt:            str
    aspect_ratio:      str | None
    version:           str            (the model id, e.g. "reve-create@20250915")
    test_time_scaling: int            (1..15)
    postprocessing:    list[{process, upscale_factor?}] | None

The official node sends ``Accept: image/webp`` and reads the response as raw
binary image bytes (``bytesio_to_image_tensor``). We do the same and fall back
to parsing a base64 ``image`` field if a relay returns JSON instead.
"""
from .utils import bytes_to_tensor, download_bytes, http_request, run_async, validate_prompt

# Real Reve vendor host (overridable for relays). The official proxy path is
# "/proxy/reve/v1/image/create"; stripping "/proxy/reve" yields the real path.
REVE_API_BASE = "https://api.reve.com"

# Model versions and their selectable aspect ratios, taken verbatim from the
# official ReveImageCreateNode schema.
REVE_CREATE_MODELS = ["reve-create@20250915"]
REVE_CREATE_ASPECT_RATIOS = ["3:2", "16:9", "9:16", "2:3", "4:3", "3:4", "1:1"]


class BYOKey_Reve_Image:
    """Generate images from text descriptions using Reve. Output: IMAGE.

    Faithful BYOKey port of the official ``ReveImageCreateNode``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Reve API key (papi.xxxxx)."}),
                "base_url": ("STRING", {"default": REVE_API_BASE, "tooltip": "Reve API host. Override for relays."}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text description of the desired image. Maximum 2560 characters.",
                    },
                ),
                "model": (
                    REVE_CREATE_MODELS,
                    {"default": REVE_CREATE_MODELS[0], "tooltip": "Model version to use for generation."},
                ),
                "aspect_ratio": (
                    REVE_CREATE_ASPECT_RATIOS,
                    {"default": REVE_CREATE_ASPECT_RATIOS[0], "tooltip": "Aspect ratio of the output image."},
                ),
            },
            "optional": {
                "test_time_scaling": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 5,
                        "step": 1,
                        "tooltip": "Higher values produce better images but cost more credits.",
                    },
                ),
                "upscale": (
                    ["disabled", "enabled"],
                    {"default": "disabled", "tooltip": "Upscale the generated image. May add additional cost."},
                ),
                "upscale_factor": (
                    "INT",
                    {"default": 2, "min": 2, "max": 4, "step": 1, "tooltip": "Upscale factor (2x, 3x, or 4x)."},
                ),
                "remove_background": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Remove the background from the generated image. May add additional cost."},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "control_after_generate": True,
                        "tooltip": "Seed controls whether the node should re-run; "
                        "results are non-deterministic regardless of seed.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Generate images from text descriptions using Reve."

    @staticmethod
    def _build_postprocessing(upscale: str, upscale_factor: int, remove_background: bool):
        """Match official ``_build_postprocessing``: list of ops, or None."""
        ops = []
        if upscale == "enabled":
            ops.append({"process": "upscale", "upscale_factor": int(upscale_factor)})
        if remove_background:
            ops.append({"process": "remove_background"})
        return ops or None

    def execute(
        self,
        api_key,
        base_url,
        prompt,
        model=REVE_CREATE_MODELS[0],
        aspect_ratio=REVE_CREATE_ASPECT_RATIOS[0],
        test_time_scaling=1,
        upscale="disabled",
        upscale_factor=2,
        remove_background=False,
        seed=0,
    ):
        # Official node: validate_string(prompt, min_length=1, max_length=2560)
        validate_prompt(prompt, min_length=1)
        if len(prompt) > 2560:
            raise ValueError("Prompt must be at most 2560 characters.")

        host = (base_url or REVE_API_BASE).rstrip("/")
        url = f"{host}/v1/image/create"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "image/webp",
        }

        # Request body field-for-field with ReveImageCreateRequest.
        body = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "version": model,
            "test_time_scaling": int(test_time_scaling),
        }
        postprocessing = self._build_postprocessing(upscale, upscale_factor, remove_background)
        if postprocessing is not None:
            body["postprocessing"] = postprocessing

        resp = run_async(http_request("POST", url, headers, json_data=body))

        # The vendor returns raw image bytes when Accept: image/webp is set;
        # utils.http_request surfaces non-JSON bodies as {"_raw": <bytes>}.
        if isinstance(resp, dict) and "_raw" in resp:
            return (bytes_to_tensor(resp["_raw"], mode="RGB"),)

        # Fallback: a relay returned JSON (matching ReveImageResponse).
        if isinstance(resp, dict):
            if resp.get("content_violation"):
                raise RuntimeError("The generated image was flagged for content policy violation.")
            image_b64 = resp.get("image")
            if image_b64:
                # download_bytes decodes data: URIs; prefix bare base64 to reuse it.
                data = run_async(download_bytes(f"data:image/webp;base64,{image_b64}"))
                return (bytes_to_tensor(data, mode="RGB"),)

        raise RuntimeError(f"Unexpected Reve response (no image bytes): {resp}")
