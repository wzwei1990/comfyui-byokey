"""Anthropic Claude (Messages API) node for BYOKey.

Bring-your-own-key node that calls the Anthropic Messages API directly
(``POST {base_url}/v1/messages`` with the ``x-api-key`` + ``anthropic-version``
headers) instead of going through the ComfyOrg proxy.

Modeled field-for-field on the official ComfyUI ``ClaudeNode``
(``comfy_api_nodes/nodes_anthropic.py`` + ``comfy_api_nodes/apis/anthropic.py``):
same model labels/ids, same max_tokens / temperature / reasoning_effort
handling, the same request body shape, the same text-block extraction, and the
same STRING output. The only intentional difference is that images are sent as
base64 ``source`` blocks (the official node uploads them and uses URL sources;
the underlying Anthropic API natively supports base64 sources for direct calls).
"""
from .utils import http_request, run_async, tensor_to_base64, validate_prompt

# Model label -> Anthropic model id (mirrors CLAUDE_MODELS in the official node).
CLAUDE_MODELS = {
    "Opus 4.7": "claude-opus-4-7",
    "Opus 4.6": "claude-opus-4-6",
    "Sonnet 4.6": "claude-sonnet-4-6",
    "Sonnet 4.5": "claude-sonnet-4-5-20250929",
    "Haiku 4.5": "claude-haiku-4-5-20251001",
}

CLAUDE_MAX_IMAGES = 20

# Models that do not support extended thinking.
_THINKING_UNSUPPORTED = {"Haiku 4.5"}
# Models that use the newer "adaptive" thinking mode (Anthropic picks the budget
# from the output_config.effort hint). Others use the explicit budget API.
_ADAPTIVE_THINKING_MODELS = {"Opus 4.7", "Opus 4.6", "Sonnet 4.6"}

# Budget mode (e.g. Sonnet 4.5): effort -> reasoning budget in tokens (< max_tokens).
_REASONING_BUDGET = {
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}
_REASONING_EFFORTS = ["off", "low", "medium", "high"]


class BYOKey_Claude:
    """Generate text responses from an Anthropic Claude model.

    Drop-in replacement for the official ``ClaudeNode`` (same STRING output and
    same model/parameter handling), but calls the Anthropic API directly with
    your own key.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Anthropic API key (sent as the x-api-key header).",
                }),
                "base_url": ("STRING", {
                    "default": "https://api.anthropic.com",
                    "tooltip": "API base URL. Default targets Anthropic directly; "
                               "override for a compatible relay/proxy.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text input to the model.",
                }),
                "model": (list(CLAUDE_MODELS.keys()), {
                    "default": "Sonnet 4.6",
                    "tooltip": "The Claude model used to generate the response.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "control_after_generate": True,
                    "tooltip": "Seed controls whether the node should re-run; "
                               "results are non-deterministic regardless of seed.",
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": f"Optional image(s) for multimodal context. Up to {CLAUDE_MAX_IMAGES} images.",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Foundational instructions that dictate the model's behavior.",
                }),
                "max_tokens": ("INT", {
                    "default": 32768,
                    "min": 4096,
                    "max": 64000,
                    "tooltip": "Maximum number of tokens to generate "
                               "(includes reasoning tokens when enabled).",
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Controls randomness. 0.0 is deterministic, 1.0 is most random. "
                               "Ignored for Opus 4.7 and any model when reasoning_effort is set.",
                }),
                "reasoning_effort": (_REASONING_EFFORTS, {
                    "default": "off",
                    "tooltip": "Extended thinking effort. 'off' disables reasoning. "
                               "Ignored for models that do not support thinking (e.g. Haiku 4.5).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/text"
    DESCRIPTION = (
        "Generate text responses with Anthropic's Claude models. Provide a text "
        "prompt and optionally one or more images for multimodal context."
    )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_text_from_response(response: dict) -> str:
        """Join text blocks from the response; thinking blocks are dropped."""
        blocks = response.get("content") or []
        return "\n".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
        )

    # ------------------------------------------------------------------ #
    # Execution                                                          #
    # ------------------------------------------------------------------ #
    def execute(self, api_key, base_url, prompt, model, seed,
                images=None, system_prompt="", max_tokens=32768,
                temperature=1.0, reasoning_effort="off"):
        validate_prompt(prompt, min_length=1)

        model_label = model
        model_id = CLAUDE_MODELS.get(model_label, model_label)

        thinking_enabled = (
            reasoning_effort not in ("off", None)
            and model_label not in _THINKING_UNSUPPORTED
        )

        # Anthropic requires temperature unset (defaults to 1.0) when thinking is
        # enabled; Opus 4.7 also rejects a user-supplied temperature.
        if thinking_enabled or model_label == "Opus 4.7":
            send_temperature = None
        else:
            send_temperature = temperature

        thinking_cfg = None
        output_cfg = None
        if thinking_enabled:
            if model_label in _ADAPTIVE_THINKING_MODELS:
                # Adaptive mode - Anthropic chooses the budget from the effort hint.
                thinking_cfg = {"type": "adaptive"}
                output_cfg = {"effort": reasoning_effort}
            else:
                # Budget mode. Leave at least 1024 tokens for the actual response.
                budget = _REASONING_BUDGET[reasoning_effort]
                budget = min(budget, max(1024, max_tokens - 1024))
                thinking_cfg = {"type": "enabled", "budget_tokens": budget}

        # Build the user message content: image blocks first, then the text block.
        content = []
        if images is not None:
            n = images.shape[0]
            if n > CLAUDE_MAX_IMAGES:
                raise ValueError(
                    f"Up to {CLAUDE_MAX_IMAGES} images are supported per request (got {n})."
                )
            for i in range(n):
                b64 = tensor_to_base64(images[i].unsqueeze(0), "image/png")
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                })
        content.append({"type": "text", "text": prompt})

        # Request body mirrors AnthropicMessagesRequest. Omit unset optionals.
        data = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system_prompt:
            data["system"] = system_prompt
        if send_temperature is not None:
            data["temperature"] = send_temperature
        if thinking_cfg is not None:
            data["thinking"] = thinking_cfg
        if output_cfg is not None:
            data["output_config"] = output_cfg

        url = f"{base_url.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        response = run_async(http_request("POST", url, headers, data))
        text = self._get_text_from_response(response)
        return (text or "Empty response from Claude model.",)
