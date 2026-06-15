"""Ideogram image generation node (bring-your-own-key).

Faithfully mirrors the official ComfyUI ``IdeogramV3`` API node, but calls the
Ideogram API DIRECTLY (host ``api.ideogram.ai``, ``Api-Key`` header) instead of
going through the ComfyOrg proxy. The official node POSTs to
``/proxy/ideogram/ideogram-v3/generate`` which the proxy forwards verbatim to
``https://api.ideogram.ai/v1/ideogram-v3/generate`` as ``multipart/form-data``;
the request/response shapes below are taken from the official request model
(``comfy_api_nodes/apis/ideogram.py`` :: ``IdeogramV3Request`` /
``IdeogramGenerateResponse``) and the official node source.

Output: IMAGE (torch tensor [N, H, W, C], RGB, float32 0-1).
"""
import torch

from .utils import (
    bytes_to_tensor,
    download_bytes,
    http_request,
    run_async,
    validate_prompt,
)

IDEOGRAM_API_BASE = "https://api.ideogram.ai"

# Maps the human-facing "W:H" labels (official combo options) to the WxH string
# the V3 API expects in the `aspect_ratio` form field.
V3_RATIO_MAP = {
    "1:1": "1x1",
    "1:3": "1x3",
    "3:1": "3x1",
    "1:2": "1x2",
    "2:1": "2x1",
    "9:16": "9x16",
    "16:9": "16x9",
    "10:16": "10x16",
    "16:10": "16x10",
    "2:3": "2x3",
    "3:2": "3x2",
    "3:4": "3x4",
    "4:3": "4x3",
    "4:5": "4x5",
    "5:4": "5x4",
}

# Official IdeogramV3 combo options (order preserved from the node / docs).
V3_ASPECT_RATIOS = list(V3_RATIO_MAP.keys())

V3_RESOLUTIONS = [
    "Auto",
    "512x1536",
    "576x1408",
    "576x1472",
    "576x1536",
    "640x1344",
    "640x1408",
    "640x1472",
    "640x1536",
    "704x1152",
    "704x1216",
    "704x1280",
    "704x1344",
    "704x1408",
    "704x1472",
    "736x1312",
    "768x1088",
    "768x1216",
    "768x1280",
    "768x1344",
    "800x1280",
    "832x960",
    "832x1024",
    "832x1088",
    "832x1152",
    "832x1216",
    "832x1248",
    "864x1152",
    "896x960",
    "896x1024",
    "896x1088",
    "896x1120",
    "896x1152",
    "960x832",
    "960x896",
    "960x1024",
    "960x1088",
    "1024x832",
    "1024x896",
    "1024x960",
    "1024x1024",
    "1088x768",
    "1088x832",
    "1088x896",
    "1088x960",
    "1120x896",
    "1152x704",
    "1152x832",
    "1152x864",
    "1152x896",
    "1216x704",
    "1216x768",
    "1216x832",
    "1248x832",
    "1280x704",
    "1280x768",
    "1280x800",
    "1312x736",
    "1344x640",
    "1344x704",
    "1344x768",
    "1408x576",
    "1408x640",
    "1408x704",
    "1472x576",
    "1472x640",
    "1472x704",
    "1536x512",
    "1536x576",
    "1536x640",
]

# V3 magic prompt / style / speed enums (from IdeogramV3Request + official node).
V3_MAGIC_PROMPT_OPTIONS = ["AUTO", "ON", "OFF"]
V3_STYLE_TYPES = ["AUTO", "GENERAL", "REALISTIC", "DESIGN", "FICTION"]
V3_RENDERING_SPEEDS = ["DEFAULT", "TURBO", "QUALITY"]


class BYOKey_Ideogram:
    """Ideogram V3 image generation (bring-your-own-key). Output: IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Your Ideogram API key (sent as the 'Api-Key' header)."}),
                "base_url": ("STRING", {"default": IDEOGRAM_API_BASE, "tooltip": "Ideogram API host. Override for a relay/proxy."}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Prompt for the image generation."}),
            },
            "optional": {
                "aspect_ratio": (
                    V3_ASPECT_RATIOS,
                    {"default": "1:1", "tooltip": "The aspect ratio for image generation. Ignored if resolution is not set to Auto."},
                ),
                "resolution": (
                    V3_RESOLUTIONS,
                    {"default": "Auto", "tooltip": "The resolution for image generation. If not set to Auto, this overrides the aspect_ratio setting."},
                ),
                "magic_prompt_option": (
                    V3_MAGIC_PROMPT_OPTIONS,
                    {"default": "AUTO", "tooltip": "Determine if MagicPrompt should be used in generation."},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1, "tooltip": "Random seed for generation (0 = unset)."}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "Number of images to generate (1-8)."}),
                "rendering_speed": (
                    V3_RENDERING_SPEEDS,
                    {"default": "DEFAULT", "tooltip": "Controls the trade-off between generation speed and quality."},
                ),
                "style_type": (
                    V3_STYLE_TYPES,
                    {"default": "AUTO", "tooltip": "The type of style to apply (V3)."},
                ),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text specifying what to avoid in the generation."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Generates images using the Ideogram V3 model (direct API, bring your own key)."

    @staticmethod
    def _build_url(base_url: str, path: str) -> str:
        base = (base_url or IDEOGRAM_API_BASE).rstrip("/")
        # The real Ideogram endpoint is /v1/ideogram-v3/generate. The official
        # node hits /proxy/ideogram/ideogram-v3/generate (proxy strips the /v1).
        # Support both bare hosts and relays that already include a suffix.
        if base.endswith("/ideogram-v3/generate"):
            return base
        if base.endswith("/v1"):
            return f"{base}/{path}"
        if "api.ideogram.ai" in base and "/v1" not in base:
            return f"{base}/v1/{path}"
        return f"{base}/{path}"

    def execute(
        self,
        api_key,
        base_url=IDEOGRAM_API_BASE,
        prompt="",
        aspect_ratio="1:1",
        resolution="Auto",
        magic_prompt_option="AUTO",
        seed=0,
        num_images=1,
        rendering_speed="DEFAULT",
        style_type="AUTO",
        negative_prompt="",
    ):
        prompt = validate_prompt(prompt, min_length=1)

        if rendering_speed == "BALANCED":  # backward-compat with older official option
            rendering_speed = "DEFAULT"

        # Build the multipart form. Ideogram's v3/generate endpoint expects
        # multipart/form-data; scalar fields are sent as plain string parts.
        # Mirrors IdeogramV3Request field-for-field (names + value formats).
        form: dict[str, str] = {
            "prompt": prompt,
            "rendering_speed": rendering_speed,
        }

        # Resolution overrides aspect_ratio (matches official node logic).
        if resolution != "Auto":
            form["resolution"] = resolution
        elif aspect_ratio != "1:1":
            v3_aspect = V3_RATIO_MAP.get(aspect_ratio)
            if v3_aspect:
                form["aspect_ratio"] = v3_aspect

        if magic_prompt_option != "AUTO":
            form["magic_prompt"] = magic_prompt_option
        if style_type and style_type != "AUTO":
            form["style_type"] = style_type
        if negative_prompt:
            form["negative_prompt"] = negative_prompt
        if seed != 0:
            form["seed"] = str(seed)
        if num_images > 1:
            form["num_images"] = str(num_images)

        headers = {"Api-Key": api_key}
        url = self._build_url(base_url, "ideogram-v3/generate")

        resp = run_async(self._generate(url, headers, form))

        data = resp.get("data") or []
        if not data:
            raise RuntimeError(f"No images were generated in the Ideogram response: {resp}")

        image_urls = [item.get("url") for item in data if item.get("url")]
        if not image_urls:
            raise RuntimeError("No image URLs were generated in the Ideogram response")

        return (run_async(self._download_images(image_urls)),)

    @staticmethod
    async def _generate(url: str, headers: dict, form: dict) -> dict:
        import aiohttp

        data = aiohttp.FormData()
        for key, value in form.items():
            data.add_field(key, value)
        return await http_request("POST", url, headers, data=data)

    @staticmethod
    async def _download_images(image_urls: list) -> torch.Tensor:
        tensors = []
        for image_url in image_urls:
            raw = await download_bytes(image_url)
            tensors.append(bytes_to_tensor(raw, mode="RGB"))
        return torch.cat(tensors, dim=0)
