"""LLM Chat and Vision nodes (OpenAI Chat Completions compatible)."""
from .utils import http_request, run_async, tensor_to_base64


class BYOKey_LLM:
    """Text generation via OpenAI-compatible Chat Completions API.
    Drop-in replacement for OpenAI ChatGPT node — same output type (STRING).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Your API key"}),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "model": ("STRING", {"default": "gpt-4o"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "images": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 128000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey"
    DESCRIPTION = "LLM text generation (OpenAI-compatible). Supports vision when images are connected."

    def execute(self, api_key, base_url, model, prompt,
                system_prompt="", images=None, temperature=0.7, max_tokens=4096, seed=0):
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if images is not None:
            content = [{"type": "text", "text": prompt}]
            for i in range(images.shape[0]):
                b64 = tensor_to_base64(images[i].unsqueeze(0))
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        data = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        resp = run_async(http_request("POST", url, headers, data))
        text = resp["choices"][0]["message"]["content"]
        return (text,)
