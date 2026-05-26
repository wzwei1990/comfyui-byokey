"""Google Gemini API nodes (LLM + Image Generation)."""
import base64

import torch

from .utils import bytes_to_tensor, http_request, run_async, tensor_to_base64


class BYOKey_Gemini:
    """Text generation via Google Gemini API. Supports vision when images are connected.
    Output type: STRING (compatible with official GeminiNode).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Google AI API key"}),
                "base_url": ("STRING", {"default": "https://generativelanguage.googleapis.com/v1beta", "tooltip": "API base URL, change for third-party proxies"}),
                "model": ("STRING", {"default": "gemini-2.5-flash-preview-05-20"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "images": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 8192, "min": 1, "max": 65536}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey"
    DESCRIPTION = "Google Gemini text generation with optional vision."

    def execute(self, api_key, base_url, model, prompt,
                system_prompt="", images=None, temperature=1.0, max_tokens=8192, seed=0):
        url = f"{base_url.rstrip('/')}/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        parts = []
        if images is not None:
            for i in range(images.shape[0]):
                b64 = tensor_to_base64(images[i].unsqueeze(0))
                parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
        parts.append({"text": prompt})

        data = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system_prompt:
            data["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        resp = run_async(http_request("POST", url, headers, data))
        candidates = resp.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates in Gemini response: {resp}")
        text_parts = []
        for part in candidates[0].get("content", {}).get("parts", []):
            if "text" in part:
                text_parts.append(part["text"])
        return ("\n".join(text_parts) or "Empty response from Gemini.",)


class BYOKey_GeminiImage:
    """Image generation via Google Gemini (Imagen / Nano Banana).
    Output type: IMAGE (compatible with official GeminiImage nodes).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": "https://generativelanguage.googleapis.com/v1beta", "tooltip": "API base URL"}),
                "model": ("STRING", {"default": "gemini-2.0-flash-exp-image-generation"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey"
    DESCRIPTION = "Google Gemini image generation."

    def execute(self, api_key, base_url, model, prompt, image=None, seed=0):
        url = f"{base_url.rstrip('/')}/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        parts = []
        if image is not None:
            b64 = tensor_to_base64(image)
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
        parts.append({"text": prompt})

        data = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
        }
        resp = run_async(http_request("POST", url, headers, data))
        candidates = resp.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates in Gemini response: {resp}")

        for part in candidates[0].get("content", {}).get("parts", []):
            if "inlineData" in part:
                img_bytes = base64.b64decode(part["inlineData"]["data"])
                return (bytes_to_tensor(img_bytes),)

        raise RuntimeError("No image returned from Gemini")

