"""BYOKey OpenRouter LLM node.

A "bring your own key" port of the official ComfyUI ``OpenRouterLLMNode`` that
talks to the OpenRouter chat-completions API DIRECTLY (no ComfyOrg proxy).

The official node routes through ``/proxy/openrouter/api/v1/chat/completions``,
which forwards the body verbatim to ``https://openrouter.ai/api/v1/chat/completions``.
This node therefore reproduces the official request body (model / messages /
seed / reasoning / web_search_options / stream) and response parsing
(``choices[0].message.content`` with error/refusal handling) field-for-field,
while exposing ``api_key`` + ``base_url`` text inputs and sending images as
inline base64 data URIs (OpenRouter accepts data URIs for ``image_url``).

Output type matches official: STRING.
"""

from .utils import http_request, run_async, tensor_to_data_uri, validate_prompt

# Real vendor host (overridable for relays/mirrors).
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# Curated model list mirroring the official node's DynamicCombo options.
# Each entry is the exact OpenRouter model slug sent in the request body.
OPENROUTER_MODELS = [
    "anthropic/claude-opus-4.7",
    "openai/gpt-5.5-pro",
    "openai/gpt-5.5",
    "google/gemini-3.5-flash",
    "x-ai/grok-4.20",
    "x-ai/grok-4.3",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.6-max-preview",
    "qwen/qwen3.6-plus",
    "qwen/qwen3.6-flash",
    "mistralai/mistral-large-2512",
    "mistralai/mistral-medium-3-5",
    "z-ai/glm-4.6",
    "z-ai/glm-5",
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k2-thinking",
    "perplexity/sonar-pro",
    "perplexity/sonar-reasoning-pro",
    "perplexity/sonar-deep-research",
]

# Reasoning efforts; "off" disables reasoning entirely (matches official node).
REASONING_EFFORTS = ["off", "low", "medium", "high"]

# Web search context sizes for Perplexity Sonar models (matches official node).
SEARCH_CONTEXT_SIZES = ["off", "low", "medium", "high"]


class BYOKey_OpenRouter:
    """OpenRouter LLM (curated popular models via a single API key). Output: STRING."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "tooltip": "OpenRouter API key (sent as Bearer token)."},
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": OPENROUTER_API_BASE,
                        "tooltip": "OpenRouter API base URL. Override only for relays/mirrors.",
                    },
                ),
                "model": (
                    OPENROUTER_MODELS,
                    {
                        "default": OPENROUTER_MODELS[0],
                        "tooltip": "The OpenRouter model used to generate the response.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Text input to the model."},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "control_after_generate": True,
                        "tooltip": "Seed for sampling. Set to 0 to omit. Most models treat this as a hint only.",
                    },
                ),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Foundational instructions that dictate the model's behavior.",
                    },
                ),
                "images": (
                    "IMAGE",
                    {"tooltip": "Optional reference image(s) for vision-capable models. Sent as inline data URIs."},
                ),
                "reasoning_effort": (
                    REASONING_EFFORTS,
                    {
                        "default": "off",
                        "tooltip": "Reasoning effort for reasoning-capable models. 'off' disables reasoning entirely.",
                    },
                ),
                "search_context_size": (
                    SEARCH_CONTEXT_SIZES,
                    {
                        "default": "off",
                        "tooltip": (
                            "Web search context size (Perplexity Sonar models). Larger = more grounded "
                            "but slower/pricier. 'off' omits web search options."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/text"
    DESCRIPTION = (
        "Generate text responses through OpenRouter. Routes to a curated set of popular models "
        "from Anthropic, OpenAI, Google, xAI, DeepSeek, Qwen, Mistral, Z.AI (GLM), Moonshot "
        "(Kimi), and Perplexity Sonar. Bring your own OpenRouter API key. Output: STRING."
    )

    # ------------------------------------------------------------------ #
    # Request building (mirrors comfy_api_nodes/apis/openrouter.py)       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_messages(system_prompt, prompt, images):
        """Build the OpenRouter ``messages`` array.

        Mirrors the official node: an optional system message, then a single user
        message whose content is either a plain string (no media) or a list of
        content blocks (media blocks first, then the text block).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        media_blocks = []
        if images is not None:
            for i in range(images.shape[0]):
                data_uri = tensor_to_data_uri(images[i].unsqueeze(0), "image/png")
                media_blocks.append(
                    {"type": "image_url", "image_url": {"url": data_uri}}
                )

        if media_blocks:
            content = list(media_blocks)
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        return messages

    @classmethod
    def _build_request(cls, model, system_prompt, prompt, seed, reasoning_effort,
                       search_context_size, images):
        """Build the OpenRouterChatRequest body field-for-field."""
        body = {
            "model": model,
            "messages": cls._build_messages(system_prompt, prompt, images),
            "stream": False,
        }
        # seed: omit (null) when 0 — matches `seed if seed > 0 else None`.
        if seed and seed > 0:
            body["seed"] = seed
        # reasoning: {effort, exclude=True} when not "off" — internal reasoning,
        # trace excluded from the response (matches official node).
        if reasoning_effort and reasoning_effort != "off":
            body["reasoning"] = {"effort": reasoning_effort, "exclude": True}
        # web_search_options: {search_context_size} when enabled.
        if search_context_size and search_context_size != "off":
            body["web_search_options"] = {"search_context_size": search_context_size}
        return body

    # ------------------------------------------------------------------ #
    # Response parsing (mirrors official _extract_text)                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_text(response):
        error = response.get("error")
        if error:
            code = error.get("code")
            code = code if code is not None else "unknown"
            message = error.get("message") or "no message"
            raise ValueError(f"OpenRouter error ({code}): {message}")
        choices = response.get("choices")
        if not choices:
            raise ValueError("Empty response from OpenRouter (no choices).")
        message = choices[0].get("message")
        if not message:
            raise ValueError("Empty response from OpenRouter (no message).")
        refusal = message.get("refusal")
        if refusal:
            raise ValueError(f"Model refused to respond: {refusal}")
        return message.get("content") or ""

    # ------------------------------------------------------------------ #
    # Execution                                                           #
    # ------------------------------------------------------------------ #
    def execute(self, api_key, base_url, model, prompt, seed,
                system_prompt="", images=None,
                reasoning_effort="off", search_context_size="off"):
        validate_prompt(prompt, min_length=1)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = self._build_request(
            model, system_prompt, prompt, seed,
            reasoning_effort, search_context_size, images,
        )
        url = f"{base_url.rstrip('/')}/chat/completions"
        response = run_async(http_request("POST", url, headers, body))
        return (self._extract_text(response),)
