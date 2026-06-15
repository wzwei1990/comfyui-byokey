"""ElevenLabs audio generation nodes (bring-your-own-key).

Faithful re-implementations of the official ComfyUI ElevenLabs API nodes
(``ElevenLabsTextToSpeech`` / ``ElevenLabsTextToSoundEffects``) that call the
real ElevenLabs REST API directly with the user's own ``xi-api-key`` instead of
routing through the ComfyOrg proxy.

The official nodes POST to ``/proxy/elevenlabs/v1/<path>`` and the proxy forwards
the body verbatim to ``https://api.elevenlabs.io/v1/<path>``, so the official
request/response shapes are the real vendor API shapes. Both nodes return a
ComfyUI ``AUDIO`` object (not a file path).
"""

from .utils import audio_from_bytes, http_request, run_async, validate_prompt

# Real ElevenLabs API host. Overridable for relays/proxies.
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"

# Output formats supported by the official nodes.
ELEVENLABS_OUTPUT_FORMATS = ["mp3_44100_192", "opus_48000_192"]

# Text normalization modes (official apply_text_normalization combo).
ELEVENLABS_TEXT_NORMALIZATION = ["auto", "on", "off"]


def _auth_headers(api_key: str) -> dict:
    """Build the ElevenLabs auth headers exactly as the vendor expects."""
    return {"xi-api-key": api_key, "Content-Type": "application/json"}


def _base_url(base_url: str) -> str:
    """Normalize the base URL, stripping a trailing slash."""
    return (base_url or ELEVENLABS_API_BASE).rstrip("/")


class BYOKey_ElevenLabs_TTS:
    """ElevenLabs Text to Speech.

    Mirrors the official ``ElevenLabsTextToSpeech`` node: POSTs to
    ``/text-to-speech/{voice_id}`` with the ``TextToSpeechRequest`` body and an
    ``output_format`` query param, then decodes the returned audio bytes into a
    ComfyUI ``AUDIO`` object.
    """

    # Model -> which voice_settings fields the model actually supports.
    # eleven_multilingual_v2 exposes the full set; eleven_v3 only speed +
    # similarity_boost (matching the official DynamicCombo options/doc).
    MODELS = ["eleven_multilingual_v2", "eleven_v3"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Your ElevenLabs API key (xi-api-key)."}),
                "base_url": ("STRING", {
                    "default": ELEVENLABS_API_BASE,
                    "tooltip": "ElevenLabs API base URL. Change only for relays/proxies.",
                }),
                "voice": ("STRING", {
                    "default": "21m00Tcm4TlvDq8ikWAM",
                    "tooltip": "ElevenLabs voice ID to use for speech synthesis.",
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The text to convert to speech.",
                }),
            },
            "optional": {
                "model": (cls.MODELS, {
                    "default": "eleven_multilingual_v2",
                    "tooltip": "Model to use for text-to-speech.",
                }),
                "stability": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Voice stability. Lower values give broader emotional range, "
                               "higher values produce more consistent but potentially monotonous speech.",
                }),
                "similarity_boost": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Similarity boost. Higher values make the voice more similar to the original.",
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.7, "max": 1.3, "step": 0.01,
                    "tooltip": "Speech speed. 1.0 is normal, <1.0 slower, >1.0 faster.",
                }),
                "style": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01,
                    "tooltip": "Style exaggeration (eleven_multilingual_v2 only). Higher values increase "
                               "stylistic expression but may reduce stability.",
                }),
                "use_speaker_boost": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Boost similarity to the original speaker voice (eleven_multilingual_v2 only).",
                }),
                "apply_text_normalization": (ELEVENLABS_TEXT_NORMALIZATION, {
                    "default": "auto",
                    "tooltip": "Text normalization mode. 'auto' lets the system decide, "
                               "'on' always applies normalization, 'off' skips it.",
                }),
                "language_code": ("STRING", {
                    "default": "",
                    "tooltip": "ISO-639-1 or ISO-639-3 language code (e.g. 'en', 'es', 'fra'). "
                               "Leave empty for automatic detection.",
                }),
                "seed": ("INT", {
                    "default": 1, "min": 0, "max": 2147483647,
                    "tooltip": "Seed for reproducibility (determinism not guaranteed).",
                }),
                "output_format": (ELEVENLABS_OUTPUT_FORMATS, {
                    "default": "mp3_44100_192",
                    "tooltip": "Audio output format.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/audio"
    DESCRIPTION = "ElevenLabs text-to-speech. Converts text to spoken audio (AUDIO output)."

    def execute(self, api_key, base_url, voice, text,
                model="eleven_multilingual_v2", stability=0.5, similarity_boost=0.75,
                speed=1.0, style=0.0, use_speaker_boost=False,
                apply_text_normalization="auto", language_code="", seed=1,
                output_format="mp3_44100_192"):
        validate_prompt(text, min_length=1)

        # voice_settings only carries the fields the chosen model supports,
        # matching the official DynamicCombo options (style + use_speaker_boost
        # are eleven_multilingual_v2 only).
        voice_settings = {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "speed": speed,
        }
        if model == "eleven_multilingual_v2":
            voice_settings["style"] = style
            voice_settings["use_speaker_boost"] = use_speaker_boost

        # TextToSpeechRequest body (seed is required by the official model).
        body = {
            "text": text,
            "model_id": model,
            "voice_settings": voice_settings,
            "seed": seed,
            "apply_text_normalization": apply_text_normalization,
        }
        if language_code.strip():
            body["language_code"] = language_code

        url = f"{_base_url(base_url)}/text-to-speech/{voice}"
        params = f"output_format={output_format}"
        full_url = f"{url}?{params}"

        async def _call():
            result = await http_request(
                "POST", full_url, _auth_headers(api_key), json_data=body, timeout=300.0
            )
            # http_request returns parsed JSON for JSON responses, or
            # {"_raw": bytes, ...} for binary audio responses.
            if isinstance(result, dict) and "_raw" in result:
                return result["_raw"]
            raise RuntimeError(f"Expected audio bytes from ElevenLabs TTS, got JSON: {result}")

        audio_bytes = run_async(_call())
        return (audio_from_bytes(audio_bytes),)


class BYOKey_ElevenLabs_SFX:
    """ElevenLabs Text to Sound Effects.

    Mirrors the official ``ElevenLabsTextToSoundEffects`` node: POSTs to
    ``/sound-generation`` with the ``TextToSoundEffectsRequest`` body and an
    ``output_format`` query param, then decodes the returned audio bytes into a
    ComfyUI ``AUDIO`` object.
    """

    MODELS = ["eleven_sfx_v2"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Your ElevenLabs API key (xi-api-key)."}),
                "base_url": ("STRING", {
                    "default": ELEVENLABS_API_BASE,
                    "tooltip": "ElevenLabs API base URL. Change only for relays/proxies.",
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text description of the sound effect to generate.",
                }),
                "model": (cls.MODELS, {
                    "default": "eleven_sfx_v2",
                    "tooltip": "Model to use for sound effect generation.",
                }),
                "output_format": (ELEVENLABS_OUTPUT_FORMATS, {
                    "default": "mp3_44100_192",
                    "tooltip": "Audio output format.",
                }),
            },
            "optional": {
                "duration": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": 30.0, "step": 0.1,
                    "tooltip": "Duration of generated sound in seconds.",
                }),
                "loop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Create a smoothly looping sound effect.",
                }),
                "prompt_influence": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How closely generation follows the prompt. "
                               "Higher values make the sound follow the text more closely.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/audio"
    DESCRIPTION = "ElevenLabs sound-effect generation from a text description (AUDIO output)."

    def execute(self, api_key, base_url, text, model="eleven_sfx_v2",
                output_format="mp3_44100_192", duration=5.0, loop=False,
                prompt_influence=0.3):
        validate_prompt(text, min_length=1)

        # TextToSoundEffectsRequest body. The official request does not carry a
        # model_id field; duration_seconds and prompt_influence are required,
        # loop is optional.
        body = {
            "text": text,
            "duration_seconds": duration,
            "prompt_influence": prompt_influence,
            "loop": loop,
        }

        url = f"{_base_url(base_url)}/sound-generation?output_format={output_format}"

        async def _call():
            result = await http_request(
                "POST", url, _auth_headers(api_key), json_data=body, timeout=300.0
            )
            if isinstance(result, dict) and "_raw" in result:
                return result["_raw"]
            raise RuntimeError(f"Expected audio bytes from ElevenLabs SFX, got JSON: {result}")

        audio_bytes = run_async(_call())
        return (audio_from_bytes(audio_bytes),)
