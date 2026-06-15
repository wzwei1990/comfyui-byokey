"""Recraft text-to-image node (bring-your-own-key).

Faithfully mirrors the official ComfyUI ``RecraftTextToImageNode``
(``comfy_api_nodes/nodes_recraft.py`` + ``apis/recraft.py``) but calls the
Recraft API directly with the user's own API key instead of going through the
ComfyOrg proxy.

Official node posts to ``/proxy/recraft/image_generation``; the proxy forwards
the body verbatim to Recraft's real ``images/generations`` endpoint, so the
official request/response models and the request body here are the real Recraft
API contract.

Output: IMAGE (torch tensor, multiple results concatenated along batch dim).
"""
import torch

from .utils import (
    bytes_to_tensor,
    download_bytes,
    http_request,
    run_async,
    validate_prompt,
)

# Real Recraft API host (overridable via base_url for relays/proxies).
RECRAFT_API_BASE = "https://external.api.recraft.ai/v1"

# Recraft V3 style options (from apis/recraft.py RecraftStyleV3).
RECRAFT_STYLES = [
    "realistic_image",
    "digital_illustration",
    "vector_illustration",
    "logo_raster",
]

# Image sizes — official RecraftTextToImageNode combo (from embedded docs).
RECRAFT_SIZES = [
    "1024x1024",
    "1152x896",
    "896x1152",
    "1216x832",
    "832x1216",
    "1344x768",
    "768x1344",
    "1536x640",
    "640x1536",
]

# Substyles per style (from apis/recraft.py dict_recraft_substyles_v3).
# "None" means "no substyle" and is converted to null in the request body.
RECRAFT_SUBSTYLES = [
    "None",
    # realistic_image
    "b_and_w", "enterprise", "evening_light", "faded_nostalgia", "forest_life",
    "hard_flash", "hdr", "motion_blur", "mystic_naturalism", "natural_light",
    "natural_tones", "organic_calm", "real_life_glow", "retro_realism",
    "retro_snapshot", "studio_portrait", "urban_drama", "village_realism",
    "warm_folk",
    # digital_illustration
    "2d_art_poster", "2d_art_poster_2", "antiquarian", "bold_fantasy",
    "child_book", "child_books", "cover", "crosshatch", "digital_engraving",
    "engraving_color", "expressionism", "freehand_details", "grain", "grain_20",
    "graphic_intensity", "hand_drawn", "hand_drawn_outline", "handmade_3d",
    "hard_comics", "infantile_sketch", "long_shadow", "modern_folk",
    "multicolor", "neon_calm", "noir", "nostalgic_pastel", "outline_details",
    "pastel_gradient", "pastel_sketch", "pixel_art", "plastic", "pop_art",
    "pop_renaissance", "seamless", "street_art", "tablet_sketch", "urban_glow",
    "urban_sketching", "vanilla_dreams", "young_adult_book", "young_adult_book_2",
    # vector_illustration
    "bold_stroke", "chemistry", "colored_stencil", "contour_pop_art", "cosmics",
    "cutout", "depressive", "editorial", "emotional_flat", "engraving",
    "infographical", "line_art", "line_circuit", "linocut", "marker_outline",
    "mosaic", "naivector", "roundish_flat", "segmented_colors", "sharp_contrast",
    "thin", "vector_photo", "vivid_shapes",
    # logo_raster
    "emblem_graffiti", "emblem_pop_art", "emblem_punk", "emblem_stamp",
    "emblem_vintage",
]


class BYOKey_Recraft_Image:
    """Recraft text-to-image. Output: IMAGE.

    Mirrors the official ``RecraftTextToImageNode`` request model. Exposes the
    style/substyle/controls that the official node passes via connected helper
    nodes as direct widgets so the BYOKey node is self-contained.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "base_url": ("STRING", {"default": RECRAFT_API_BASE, "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Prompt for the image generation."}),
                "size": (RECRAFT_SIZES, {"default": "1024x1024", "tooltip": "The size of the generated image."}),
                "n": ("INT", {"default": 1, "min": 1, "max": 6, "tooltip": "The number of images to generate."}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed to determine if node should re-run; actual results are "
                               "nondeterministic regardless of seed.",
                }),
            },
            "optional": {
                "style": (RECRAFT_STYLES, {
                    "default": "realistic_image",
                    "tooltip": "Style for image generation. Defaults to realistic_image.",
                }),
                "substyle": (RECRAFT_SUBSTYLES, {
                    "default": "None",
                    "tooltip": "Optional substyle (must match the chosen style; 'None' for no substyle).",
                }),
                "negative_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "An optional text description of undesired elements on an image.",
                }),
                "style_id": ("STRING", {
                    "default": "",
                    "tooltip": "Optional UUID of a style from Recraft's Infinite Style Library "
                               "(do not use a Vector art style; it returns SVG).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Recraft text-to-image generation (bring your own key)."

    def execute(self, api_key, base_url, prompt, size, n, seed,
                style="realistic_image", substyle="None",
                negative_prompt="", style_id=""):
        # Official: validate_string(prompt, strip_whitespace=False, min_length=1, max_length=1000)
        validate_prompt(prompt, min_length=1)
        if len(prompt) > 1000:
            raise ValueError("Prompt must be at most 1000 characters long.")

        base = (base_url or RECRAFT_API_BASE).rstrip("/")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Normalize "None" substyle to null, like RecraftStyle does.
        substyle_val = None if not substyle or substyle == "None" else substyle
        style_id_val = style_id or None
        # When a style_id is given, Recraft ignores style/substyle.
        style_val = None if style_id_val else (style or "realistic_image")

        # Mirror RecraftImageGenerationRequest field-for-field.
        body = {
            "prompt": prompt,
            "model": "recraftv3",
            "size": size,
            "n": n,
            "style": style_val,
            "substyle": substyle_val,
            "negative_prompt": negative_prompt if negative_prompt else None,
            "style_id": style_id_val,
        }
        body = {k: v for k, v in body.items() if v is not None}

        resp = run_async(http_request(
            "POST", f"{base}/images/generations", headers, body,
        ))

        # Mirror RecraftImageGenerationResponse extraction: prefer .data,
        # fall back to single .image; concat all results along batch dim.
        entries = resp.get("data")
        if not entries:
            single = resp.get("image")
            entries = [single] if single else []
        if not entries:
            raise RuntimeError(f"Recraft returned no image data: {resp}")

        images = []
        for entry in entries:
            url = entry.get("url") if isinstance(entry, dict) else None
            if not url:
                raise RuntimeError(f"Recraft result missing url: {entry}")
            img_bytes = run_async(download_bytes(url, headers=headers))
            tensor = bytes_to_tensor(img_bytes, mode="RGB")
            if tensor.ndim < 4:
                tensor = tensor.unsqueeze(0)
            images.append(tensor)

        return (torch.cat(images, dim=0),)
