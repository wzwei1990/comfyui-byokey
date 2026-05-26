"""Anthropic Claude Messages API node."""
from .utils import http_request, run_async, tensor_to_base64


class BYOKey_Claude:
    """Text generation via Anthropic Claude Messages API.
    Drop-in replacement for the official ClaudeNode — same output type (STRING).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Anthropic API key"}),
                "base_url": ("STRING", {"default": "https://api.anthropic.com", "tooltip": "API base URL, change for third-party proxies"}),
                "model": ("STRING", {"default": "claude-sonnet-4-20250514"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "images": ("IMAGE",),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 128000}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey"
    DESCRIPTION = "Anthropic Claude text generation. Supports vision when images are connected."

    def execute(self, api_key, base_url, model, prompt,
                system_prompt="", images=None, max_tokens=4096, temperature=1.0, seed=0):
        url = f"{base_url.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        content = []
        if images is not None:
            for i in range(images.shape[0]):
                b64 = tensor_to_base64(images[i].unsqueeze(0))
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": b64,
                }})
        content.append({"type": "text", "text": prompt})

        data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if system_prompt:
            data["system"] = system_prompt

        resp = run_async(http_request("POST", url, headers, data))
        text_parts = [block["text"] for block in resp.get("content", []) if block.get("type") == "text"]
        return ("\n".join(text_parts) or "Empty response from Claude.",)
