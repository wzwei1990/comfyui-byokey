"""Google Gemini API nodes (LLM text + image generation), bring-your-own-key.

These nodes call the Google Gemini Developer API directly
(``generativelanguage.googleapis.com/v1beta``) using your own API key, with NO
ComfyOrg proxy. The request/response shapes mirror the official ComfyUI Gemini
nodes (``comfy_api_nodes/nodes_gemini.py`` + ``comfy_api_nodes/apis/gemini.py``):
``generateContent`` with ``contents`` -> ``parts`` (text + ``inline_data``),
``generationConfig`` (temperature, topP, maxOutputTokens, seed, thinkingConfig),
``systemInstruction`` and, for image models, ``responseModalities`` + ``imageConfig``.

Output types match the official nodes:
  - ``BYOKey_Gemini``      -> ("STRING",)              (GeminiNode)
  - ``BYOKey_GeminiImage`` -> ("IMAGE", "STRING")      (GeminiImageNode / GeminiImage2Node / GeminiNanoBanana2)

Reference media is sent inline as base64 (there is no ComfyAPI storage to upload
to when bypassing the proxy), exactly like the official inline-overflow path.
"""
import base64

import torch

from .utils import bytes_to_tensor, http_request, run_async, tensor_to_base64, validate_prompt

# Default Gemini Developer API host. Override base_url for third-party relays.
GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# System prompt used by the official image nodes to coax the model into always
# producing an image (GEMINI_IMAGE_SYS_PROMPT in nodes_gemini.py).
GEMINI_IMAGE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of "
    "format, intent, or abstraction—as literal visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, "
    "you must creatively invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or conversational requests."
)


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _build_url(base_url: str, model: str, api_key: str) -> str:
    """Build the generateContent endpoint URL with the API key as a query param."""
    base = (base_url or GEMINI_DEFAULT_BASE_URL).rstrip("/")
    return f"{base}/models/{model}:generateContent?key={api_key}"


def _image_parts(images: torch.Tensor) -> list[dict]:
    """Turn a [B,H,W,C] IMAGE tensor into Gemini inline_data parts (one per frame)."""
    parts: list[dict] = []
    if images is None:
        return parts
    for i in range(images.shape[0]):
        b64 = tensor_to_base64(images[i].unsqueeze(0), mime_type="image/png")
        parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
    return parts


def _candidate_parts(response: dict) -> list[dict]:
    """Collect parts across all candidates, raising on a blocked prompt.

    Mirrors official get_parts_by_type(): if there are no candidates and the
    promptFeedback carries a blockReason, surface it as an error.
    """
    candidates = response.get("candidates") or []
    if not candidates:
        feedback = response.get("promptFeedback") or {}
        block_reason = feedback.get("blockReason")
        if block_reason:
            raise RuntimeError(
                f"Gemini API blocked the request. Reason: {block_reason} "
                f"({feedback.get('blockReasonMessage', '')})"
            )
        raise RuntimeError(f"Gemini API returned no response candidates: {response}")

    parts: list[dict] = []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            parts.append(part)
    return parts


def _text_from_response(response: dict) -> str:
    """Concatenate every text part across candidates (official get_text_from_response)."""
    texts = [p["text"] for p in _candidate_parts(response) if p.get("text")]
    return "\n".join(texts)


def _system_instruction(system_prompt: str) -> dict | None:
    """Build the systemInstruction object the way the official nodes do."""
    if not system_prompt:
        return None
    return {"parts": [{"text": system_prompt}]}


# --------------------------------------------------------------------------- #
# BYOKey_Gemini — text generation (GeminiNode)                                #
# --------------------------------------------------------------------------- #
class BYOKey_Gemini:
    """Google Gemini text generation with optional multimodal (vision) context.

    Output type: STRING (compatible with the official GeminiNode).
    """

    # Matches the official GeminiNode `model` combo (en.md) plus the directly
    # callable current public model ids on the Gemini Developer API.
    MODELS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Google AI Studio (Gemini) API key."}),
                "base_url": (
                    "STRING",
                    {
                        "default": GEMINI_DEFAULT_BASE_URL,
                        "tooltip": "API base URL. Change only for third-party relays.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text input to the model. Include detailed instructions, "
                        "questions, or context for generating a response.",
                    },
                ),
                "model": (cls.MODELS, {"default": "gemini-2.5-flash"}),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Best-effort deterministic seed. By default a random seed is used.",
                    },
                ),
            },
            "optional": {
                "images": (
                    "IMAGE",
                    {"tooltip": "Optional image(s) as context. Use the Batch Images node for multiple."},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Foundational instructions that dictate the model's behavior.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Controls randomness. Lower is more deterministic, higher more creative.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling threshold."},
                ),
                "max_output_tokens": (
                    "INT",
                    {
                        "default": 32768,
                        "min": 16,
                        "max": 65536,
                        "tooltip": "Maximum tokens to generate, including internal thinking.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/text"
    DESCRIPTION = (
        "Generate text responses with Google's Gemini models. Provide a prompt and, optionally, "
        "image(s) as multimodal context. Calls the Gemini Developer API directly with your own key."
    )

    def execute(
        self,
        api_key,
        base_url,
        prompt,
        model,
        seed=42,
        images=None,
        system_prompt="",
        temperature=1.0,
        top_p=0.95,
        max_output_tokens=32768,
    ):
        validate_prompt(prompt, min_length=1)

        parts: list[dict] = [{"text": prompt}]
        parts.extend(_image_parts(images))

        generation_config: dict = {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_output_tokens,
        }
        if seed and seed > 0:
            generation_config["seed"] = seed

        data: dict = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        system_instruction = _system_instruction(system_prompt)
        if system_instruction is not None:
            data["systemInstruction"] = system_instruction

        url = _build_url(base_url, model, api_key)
        headers = {"Content-Type": "application/json"}

        response = run_async(http_request("POST", url, headers, data))
        output_text = _text_from_response(response)
        return (output_text or "Empty response from Gemini model...",)


# --------------------------------------------------------------------------- #
# BYOKey_GeminiImage — image generation/editing (GeminiImage* / NanoBanana)   #
# --------------------------------------------------------------------------- #
class BYOKey_GeminiImage:
    """Google Gemini image generation / editing (Nano Banana family).

    Output types: (IMAGE, STRING) — the generated image(s) plus any text, exactly
    like the official GeminiImageNode / GeminiImage2Node / GeminiNanoBanana2.
    """

    # Model labels -> Gemini Developer API model ids. The labels mirror the
    # official combos (GeminiImageModel enum + GeminiImage2 / NanoBanana2 lists).
    MODEL_MAP = {
        "gemini-2.5-flash-image": "gemini-2.5-flash-image",
        "gemini-2.5-flash-image-preview": "gemini-2.5-flash-image-preview",
        "gemini-2.0-flash-exp-image-generation": "gemini-2.0-flash-exp-image-generation",
        "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",
        "Nano Banana 2 (Gemini 3.1 Flash Image)": "gemini-3.1-flash-image-preview",
    }

    ASPECT_RATIOS = ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
    RESOLUTIONS = ["1K", "2K", "4K"]
    RESPONSE_MODALITIES = ["IMAGE+TEXT", "IMAGE"]
    THINKING_LEVELS = ["auto", "MINIMAL", "HIGH"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Google AI Studio (Gemini) API key."}),
                "base_url": (
                    "STRING",
                    {
                        "default": GEMINI_DEFAULT_BASE_URL,
                        "tooltip": "API base URL. Change only for third-party relays.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt describing the image to generate or the edits to apply.",
                    },
                ),
                "model": (list(cls.MODEL_MAP.keys()), {"default": "gemini-2.5-flash-image"}),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Best-effort deterministic seed. By default a random seed is used.",
                    },
                ),
            },
            "optional": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Optional reference image(s). Use the Batch Images node for multiple "
                        "(up to 14 for the Nano Banana Pro / 2 models).",
                    },
                ),
                "aspect_ratio": (
                    cls.ASPECT_RATIOS,
                    {
                        "default": "auto",
                        "tooltip": "'auto' matches the input image's aspect ratio, otherwise generates that ratio.",
                    },
                ),
                "resolution": (
                    cls.RESOLUTIONS,
                    {
                        "default": "1K",
                        "tooltip": "Target output resolution (only used by Nano Banana Pro / 2 models). "
                        "For 2K/4K the native Gemini upscaler is used.",
                    },
                ),
                "response_modalities": (
                    cls.RESPONSE_MODALITIES,
                    {
                        "default": "IMAGE+TEXT",
                        "tooltip": "'IMAGE' for image-only output, 'IMAGE+TEXT' to also return a text response.",
                    },
                ),
                "thinking_level": (
                    cls.THINKING_LEVELS,
                    {
                        "default": "auto",
                        "tooltip": "Depth of the model's reasoning (Nano Banana 2 only). "
                        "'auto' leaves it unset.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": GEMINI_IMAGE_SYS_PROMPT,
                        "tooltip": "Foundational instructions that dictate the model's behavior.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = (
        "Generate or edit images with Google's Gemini (Nano Banana) models. Returns the generated "
        "image(s) and any accompanying text. Calls the Gemini Developer API directly with your own key."
    )

    def _images_from_response(self, response: dict) -> torch.Tensor:
        """Decode every inline image part into a batched IMAGE tensor (official get_image_from_response)."""
        image_tensors: list[torch.Tensor] = []
        for part in _candidate_parts(response):
            inline = part.get("inlineData") or part.get("inline_data")
            if not inline or not inline.get("data"):
                continue
            mime = inline.get("mimeType") or inline.get("mime_type") or ""
            if mime and not mime.startswith("image/"):
                continue
            img_bytes = base64.b64decode(inline["data"])
            image_tensors.append(bytes_to_tensor(img_bytes, mode="RGBA"))

        if not image_tensors:
            model_message = _text_from_response(response).strip()
            if model_message:
                raise RuntimeError(f"Gemini did not generate an image. Model response: {model_message}")
            raise RuntimeError(
                "Gemini did not generate an image. Try rephrasing your prompt or setting "
                "response_modalities to 'IMAGE+TEXT' to see the model's reasoning."
            )
        return torch.cat(image_tensors, dim=0)

    def execute(
        self,
        api_key,
        base_url,
        prompt,
        model,
        seed=42,
        images=None,
        aspect_ratio="auto",
        resolution="1K",
        response_modalities="IMAGE+TEXT",
        thinking_level="auto",
        system_prompt=GEMINI_IMAGE_SYS_PROMPT,
    ):
        validate_prompt(prompt, min_length=1)

        model_id = self.MODEL_MAP.get(model, model)

        if images is not None and images.shape[0] > 14:
            raise ValueError("The current maximum number of supported images is 14.")

        parts: list[dict] = [{"text": prompt}]
        parts.extend(_image_parts(images))

        # imageConfig mirrors GeminiImageConfig: aspectRatio + imageSize (resolution).
        image_config: dict = {"imageSize": resolution}
        if aspect_ratio and aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio

        generation_config: dict = {
            "responseModalities": (["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]),
            "imageConfig": image_config,
        }
        if seed and seed > 0:
            generation_config["seed"] = seed
        if thinking_level and thinking_level != "auto":
            generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}

        data: dict = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        system_instruction = _system_instruction(system_prompt)
        if system_instruction is not None:
            data["systemInstruction"] = system_instruction

        url = _build_url(base_url, model_id, api_key)
        headers = {"Content-Type": "application/json"}

        response = run_async(http_request("POST", url, headers, data))
        image = self._images_from_response(response)
        text = _text_from_response(response)
        return (image, text)
