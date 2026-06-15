"""Additional BYOKey providers sharing one module file.

  - BYOKey_ByteDance_LLM   : ByteDance Seed / Doubao LLM (Volcengine Ark,
                             OpenAI-compatible chat) -> STRING
  - BYOKey_WaveSpeed_Upscale : WaveSpeed image upscale (submit+poll) -> IMAGE
  - BYOKey_Bria_Image      : Bria text-to-image -> IMAGE

All three expose ``api_key`` + ``base_url`` so they can be pointed at the real
vendor host or a relay, consistent with the rest of the package.
"""
import asyncio

import torch

from .utils import (
    bytes_to_tensor,
    download_bytes,
    http_request,
    poll_until,
    run_async,
    tensor_to_data_uri,
    validate_prompt,
)

# Default vendor hosts (overridable via base_url for relays/mirrors).
ARK_DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
WAVESPEED_DEFAULT_BASE_URL = "https://api.wavespeed.ai/api/v3"
BRIA_DEFAULT_BASE_URL = "https://engine.prod.bria-api.com/v1"


class BYOKey_ByteDance_LLM:
    """ByteDance Seed / Doubao LLM via the Volcengine Ark API. Output: STRING.

    The Ark inference API is OpenAI-compatible (``{base}/chat/completions``),
    so this node mirrors the standard OpenAI chat request/response, including
    optional multimodal (vision) input sent as inline base64 data URIs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Volcengine Ark API key (Bearer token)."}),
                "base_url": ("STRING", {
                    "default": ARK_DEFAULT_BASE_URL,
                    "tooltip": "Ark API base URL. International default; override for relays.",
                }),
                "model": ("STRING", {
                    "default": "doubao-1-5-pro-32k-250115",
                    "tooltip": "Ark model/endpoint id (e.g. doubao-1-5-pro-32k-250115, deepseek-v3.1).",
                }),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text input to the model."}),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Foundational instructions that dictate the model's behavior.",
                }),
                "images": ("IMAGE", {"tooltip": "Optional reference image(s) for vision-capable models."}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 128000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/text"
    DESCRIPTION = "Generate text with ByteDance Seed / Doubao via the Volcengine Ark API."

    def execute(self, api_key, base_url, model, prompt,
                system_prompt="", images=None, max_tokens=4096, temperature=0.7, seed=0):
        validate_prompt(prompt, min_length=1)
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if images is not None:
            content = [{"type": "text", "text": prompt}]
            for i in range(images.shape[0]):
                content.append({"type": "image_url", "image_url": {"url": tensor_to_data_uri(images[i].unsqueeze(0))}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        data = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if seed and seed > 0:
            data["seed"] = seed
        resp = run_async(http_request("POST", url, headers, data))
        return (resp["choices"][0]["message"]["content"],)


class BYOKey_WaveSpeed_Upscale:
    """WaveSpeed image upscale (AuraSR). Output: IMAGE.

    Mirrors the official submit+poll flow: POST ``{base}/{model}`` -> data.id,
    poll ``{base}/predictions/{id}/result`` until data.status is completed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "WaveSpeed API key (Bearer token)."}),
                "base_url": ("STRING", {
                    "default": WAVESPEED_DEFAULT_BASE_URL,
                    "tooltip": "WaveSpeed API base URL. Override for relays.",
                }),
                "image": ("IMAGE", {"tooltip": "Image to upscale."}),
            },
            "optional": {
                "model": ("STRING", {"default": "wavespeed-ai/aura-sr", "tooltip": "WaveSpeed model id."}),
                "scale": ("INT", {"default": 4, "min": 2, "max": 8, "tooltip": "Upscaling factor."}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 30}),
                "max_wait": ("INT", {"default": 120, "min": 30, "max": 600}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Upscale an image with WaveSpeed (AuraSR)."

    def execute(self, api_key, base_url, image, model="wavespeed-ai/aura-sr",
                scale=4, poll_interval=3, max_wait=120):
        base = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {"input": {"image": tensor_to_data_uri(image), "upscaling_factor": scale}}

        submit = run_async(http_request("POST", f"{base}/{model}", headers, body))
        request_id = (submit.get("data") or {}).get("id") or submit.get("id")
        if not request_id:
            raise RuntimeError(f"WaveSpeed returned no request id: {submit}")

        result = run_async(poll_until(
            lambda: http_request("GET", f"{base}/predictions/{request_id}/result", headers),
            status_of=lambda r: str((r.get("data") or {}).get("status", "")),
            done_statuses=["completed", "succeeded"],
            failed_statuses=["failed", "error"],
            interval=poll_interval, max_wait=max_wait,
            on_fail_message="WaveSpeed upscale failed",
        ))
        outputs = (result.get("data") or {}).get("outputs") or []
        if not outputs:
            raise RuntimeError(f"WaveSpeed completed with no output: {result}")
        img_bytes = run_async(download_bytes(outputs[0]))
        return (bytes_to_tensor(img_bytes),)


class BYOKey_Bria_Image:
    """Bria text-to-image. Output: IMAGE.

    Calls Bria's ``{base}/text-to-image/{model}`` endpoint (header ``api_token``)
    and concatenates all returned image URLs along the batch dimension.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Bria API key (sent as the api_token header)."}),
                "base_url": ("STRING", {
                    "default": BRIA_DEFAULT_BASE_URL,
                    "tooltip": "Bria API base URL. Override for relays.",
                }),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text description of the desired image."}),
            },
            "optional": {
                "model": ("STRING", {"default": "bria-2.3", "tooltip": "Bria model id (e.g. bria-2.3)."}),
                "num_results": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "Number of images to generate."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/image"
    DESCRIPTION = "Generate images from text with Bria."

    def execute(self, api_key, base_url, prompt, model="bria-2.3", num_results=1, seed=0):
        validate_prompt(prompt, min_length=1)
        base = base_url.rstrip("/")
        headers = {"api_token": api_key, "Content-Type": "application/json"}
        body = {"prompt": prompt, "num_results": num_results}
        if seed and seed > 0:
            body["seed"] = seed

        resp = run_async(http_request("POST", f"{base}/text-to-image/{model}", headers, body))
        results = resp.get("result") or []
        if not results:
            raise RuntimeError(f"Bria returned no results: {resp}")

        images = []
        for item in results:
            url = (item.get("urls") or [None])[0] or item.get("url")
            if url:
                images.append(bytes_to_tensor(run_async(download_bytes(url))))
        if not images:
            raise RuntimeError("Bria returned no image URLs")
        return (torch.cat(images, dim=0),)
