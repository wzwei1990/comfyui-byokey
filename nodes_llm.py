"""LLM chat/vision node (OpenAI Chat Completions compatible).

Mirrors the official ComfyUI ``OpenAIChatNode`` (comfy_api_nodes/nodes_openai.py)
as closely as the "bring your own key" design allows, while calling the vendor API
DIRECTLY (no ComfyOrg proxy).

The official node targets OpenAI's Responses API (``/v1/responses``) through the
ComfyOrg proxy. For a BYOK node we instead use the universally available
OpenAI-compatible Chat Completions endpoint (``/v1/chat/completions``) so the same
node works out of the box against OpenAI and against relays / DeepSeek / Qwen / etc.
via the ``base_url`` override. Inputs (model list, prompt, optional image context,
validation) and the STRING output match the official node; request/response handling
follows the standard Chat Completions schema that those backends implement.
"""
from .utils import http_request, run_async, tensor_to_data_uri, validate_prompt

# Exact model id list from the official OpenAIChatNode (SupportedOpenAIModel enum),
# default "gpt-5" as in the official node / docs. Kept as an editable combo so the
# same node can target relay/DeepSeek/Qwen model ids the official list does not cover.
OPENAI_CHAT_MODELS = [
    "gpt-5.5-pro",
    "gpt-5.5",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o4-mini",
    "o3",
    "o1-pro",
    "o1",
]


class BYOKey_LLM:
    """Generate text responses from an OpenAI-compatible chat model.

    Drop-in equivalent of the official OpenAI ChatGPT node: same model choices,
    optional image context, and a single STRING (``output_text``) output. Uses the
    standard ``/chat/completions`` endpoint so it also works against OpenAI-compatible
    relays by overriding ``base_url``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your API key (sent as 'Authorization: Bearer <key>').",
                }),
                "base_url": ("STRING", {
                    "default": "https://api.openai.com/v1",
                    "tooltip": "API base URL. Override for OpenAI-compatible relays "
                               "(e.g. DeepSeek, Qwen). '/chat/completions' is appended.",
                }),
                "model": ("STRING", {
                    "default": "gpt-5",
                    "tooltip": "The model used to generate the response. "
                               "Common OpenAI ids: " + ", ".join(OPENAI_CHAT_MODELS),
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text inputs to the model, used to generate a response.",
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Optional image(s) to use as context for the model. "
                               "To include multiple images, use the Batch Images node.",
                }),
                "instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System instructions for the model on how to generate "
                               "the response (sent as the 'system' message).",
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Controls randomness in the response (0.0-2.0).",
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Controls diversity via nucleus sampling (0.0-1.0).",
                }),
                "max_tokens": ("INT", {
                    "default": 4096,
                    "min": 16,
                    "max": 16384,
                    "tooltip": "Upper bound for the number of tokens generated for a response.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "control_after_generate": True,
                    "tooltip": "Sampling seed (best-effort; not honored by all backends).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_text",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/text"
    DESCRIPTION = "Generate text responses from an OpenAI-compatible chat model. Supports image context when images are connected."

    @staticmethod
    def _build_messages(prompt, instructions, images):
        """Build the Chat Completions ``messages`` array (with optional vision)."""
        messages = []
        if instructions and instructions.strip():
            messages.append({"role": "system", "content": instructions})

        if images is not None:
            content = [{"type": "text", "text": prompt}]
            for i in range(images.shape[0]):
                data_uri = tensor_to_data_uri(images[i].unsqueeze(0), mime_type="image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri, "detail": "auto"},
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _extract_text(resp):
        """Extract assistant text from a Chat Completions response, matching the
        official node's 'return text, else a sentinel string' behaviour."""
        choices = resp.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices returned from API. Response={resp}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            # Some backends return content as a list of typed parts.
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in (None, "text", "output_text")
            ]
            content = "".join(parts)
        if content is None or content == "":
            return "No text output found in response"
        return str(content)

    def execute(self, api_key, base_url, model, prompt,
                images=None, instructions="", temperature=1.0, top_p=1.0,
                max_tokens=4096, seed=0):
        # Official node validates the prompt via validate_string(strip_whitespace=False).
        validate_prompt(prompt, min_length=1, strip=False)

        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": self._build_messages(prompt, instructions, images),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if seed:
            data["seed"] = seed

        resp = run_async(http_request("POST", url, headers, json_data=data))
        return (self._extract_text(resp),)
