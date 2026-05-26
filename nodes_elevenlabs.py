"""ElevenLabs audio generation nodes."""
import base64
import os
import uuid

from .utils import http_request, run_async

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"


class BYOKey_ElevenLabs_TTS:
    """ElevenLabs text-to-speech. Output: STRING (audio file path)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "voice_id": ("STRING", {"default": "21m00Tcm4TlvDq8ikWAM", "tooltip": "ElevenLabs voice ID"}),
            },
            "optional": {
                "model": (["eleven_multilingual_v2", "eleven_turbo_v2_5", "eleven_monolingual_v1"], {"default": "eleven_multilingual_v2"}),
                "stability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "similarity_boost": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("Path to generated audio file",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/audio"
    DESCRIPTION = "ElevenLabs text-to-speech."

    def execute(self, api_key, text, voice_id, model="eleven_multilingual_v2",
                stability=0.5, similarity_boost=0.75):
        import aiohttp

        async def _call():
            url = f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}"
            headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
            body = {"text": text, "model_id": model,
                    "voice_settings": {"stability": stability, "similarity_boost": similarity_boost}}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=body,
                                       timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status >= 400:
                        raise RuntimeError(f"ElevenLabs error (HTTP {resp.status}): {await resp.text()}")
                    return await resp.read()

        audio_bytes = run_async(_call())
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, f"elevenlabs_{uuid.uuid4().hex[:8]}.mp3")
        with open(filepath, "wb") as f:
            f.write(audio_bytes)
        return (filepath,)


class BYOKey_ElevenLabs_SFX:
    """ElevenLabs sound effects generation. Output: STRING (audio path)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "Description of the sound effect"}),
            },
            "optional": {
                "duration_seconds": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 22.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/audio"
    DESCRIPTION = "ElevenLabs sound effect generation."

    def execute(self, api_key, text, duration_seconds=3.0):
        import aiohttp

        async def _call():
            url = f"{ELEVENLABS_API_BASE}/sound-generation"
            headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
            body = {"text": text, "duration_seconds": duration_seconds}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=body,
                                       timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status >= 400:
                        raise RuntimeError(f"ElevenLabs SFX error: {await resp.text()}")
                    return await resp.read()

        audio_bytes = run_async(_call())
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, f"elevenlabs_sfx_{uuid.uuid4().hex[:8]}.mp3")
        with open(filepath, "wb") as f:
            f.write(audio_bytes)
        return (filepath,)
