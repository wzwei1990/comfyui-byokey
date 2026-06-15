"""ComfyUI BYOKey — Bring Your Own Key API nodes.

Lightweight custom nodes that call AI provider APIs DIRECTLY with your own keys,
bypassing the ComfyOrg proxy. Every node exposes ``api_key`` + ``base_url`` text
inputs so it works out of the box against the real vendor host AND against any
third-party relay. Request/response shapes and output types mirror the official
ComfyUI API nodes (see comfy_api_nodes/ + Comfy-Org/embedded-docs).

Output types per node match the official counterparts:
  - text  -> STRING
  - image -> IMAGE (torch tensor)
  - video -> VIDEO (comfy_api.latest.InputImpl.VideoFromFile)
  - audio -> AUDIO ({"waveform", "sample_rate"})
"""

from .nodes_llm import BYOKey_LLM
from .nodes_claude import BYOKey_Claude
from .nodes_gemini import BYOKey_Gemini, BYOKey_GeminiImage
from .nodes_openrouter import BYOKey_OpenRouter
from .nodes_extra import BYOKey_ByteDance_LLM

from .nodes_image_gen import BYOKey_ImageGen
from .nodes_flux import BYOKey_Flux
from .nodes_stability import BYOKey_StabilityImage
from .nodes_grok import BYOKey_Grok_Image
from .nodes_ideogram import BYOKey_Ideogram
from .nodes_recraft import BYOKey_Recraft_Image
from .nodes_reve import BYOKey_Reve_Image
from .nodes_extra import BYOKey_Bria_Image, BYOKey_WaveSpeed_Upscale
from .nodes_runway import BYOKey_Runway_Text2Image

from .nodes_upscale import BYOKey_Magnific

from .nodes_video_gen import BYOKey_VideoGen
from .nodes_kling import BYOKey_Kling_Text2Video, BYOKey_Kling_Image2Video
from .nodes_minimax import BYOKey_MiniMax_Text2Video, BYOKey_MiniMax_Image2Video
from .nodes_seedance import BYOKey_Seedance2_Text2Video, BYOKey_Seedance2_Image2Video
from .nodes_runway import BYOKey_Runway_Image2Video
from .nodes_luma import BYOKey_Luma_Text2Video, BYOKey_Luma_Image2Video
from .nodes_pixverse import BYOKey_PixVerse_Text2Video
from .nodes_vidu import BYOKey_Vidu_Text2Video, BYOKey_Vidu_Image2Video
from .nodes_veo import BYOKey_Veo
from .nodes_wan import BYOKey_Wan_Text2Video, BYOKey_Wan_Image2Video
from .nodes_ltxv import BYOKey_LTXV_Text2Video
from .nodes_grok import BYOKey_Grok_Video

from .nodes_elevenlabs import BYOKey_ElevenLabs_TTS, BYOKey_ElevenLabs_SFX

NODE_CLASS_MAPPINGS = {
    # ---- Text / LLM ----
    "BYOKey_LLM": BYOKey_LLM,
    "BYOKey_Claude": BYOKey_Claude,
    "BYOKey_Gemini": BYOKey_Gemini,
    "BYOKey_OpenRouter": BYOKey_OpenRouter,
    "BYOKey_ByteDance_LLM": BYOKey_ByteDance_LLM,
    # ---- Image generation ----
    "BYOKey_ImageGen": BYOKey_ImageGen,
    "BYOKey_GeminiImage": BYOKey_GeminiImage,
    "BYOKey_Flux": BYOKey_Flux,
    "BYOKey_StabilityImage": BYOKey_StabilityImage,
    "BYOKey_Grok_Image": BYOKey_Grok_Image,
    "BYOKey_Ideogram": BYOKey_Ideogram,
    "BYOKey_Recraft_Image": BYOKey_Recraft_Image,
    "BYOKey_Reve_Image": BYOKey_Reve_Image,
    "BYOKey_Bria_Image": BYOKey_Bria_Image,
    "BYOKey_Runway_Text2Image": BYOKey_Runway_Text2Image,
    # ---- Image upscale ----
    "BYOKey_Magnific": BYOKey_Magnific,
    "BYOKey_WaveSpeed_Upscale": BYOKey_WaveSpeed_Upscale,
    # ---- Video generation ----
    "BYOKey_VideoGen": BYOKey_VideoGen,
    "BYOKey_Kling_Text2Video": BYOKey_Kling_Text2Video,
    "BYOKey_Kling_Image2Video": BYOKey_Kling_Image2Video,
    "BYOKey_MiniMax_Text2Video": BYOKey_MiniMax_Text2Video,
    "BYOKey_MiniMax_Image2Video": BYOKey_MiniMax_Image2Video,
    "BYOKey_Seedance2_Text2Video": BYOKey_Seedance2_Text2Video,
    "BYOKey_Seedance2_Image2Video": BYOKey_Seedance2_Image2Video,
    "BYOKey_Runway_Image2Video": BYOKey_Runway_Image2Video,
    "BYOKey_Luma_Text2Video": BYOKey_Luma_Text2Video,
    "BYOKey_Luma_Image2Video": BYOKey_Luma_Image2Video,
    "BYOKey_PixVerse_Text2Video": BYOKey_PixVerse_Text2Video,
    "BYOKey_Vidu_Text2Video": BYOKey_Vidu_Text2Video,
    "BYOKey_Vidu_Image2Video": BYOKey_Vidu_Image2Video,
    "BYOKey_Veo": BYOKey_Veo,
    "BYOKey_Wan_Text2Video": BYOKey_Wan_Text2Video,
    "BYOKey_Wan_Image2Video": BYOKey_Wan_Image2Video,
    "BYOKey_LTXV_Text2Video": BYOKey_LTXV_Text2Video,
    "BYOKey_Grok_Video": BYOKey_Grok_Video,
    # ---- Audio ----
    "BYOKey_ElevenLabs_TTS": BYOKey_ElevenLabs_TTS,
    "BYOKey_ElevenLabs_SFX": BYOKey_ElevenLabs_SFX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BYOKey_LLM": "BYOKey LLM (OpenAI Compatible)",
    "BYOKey_Claude": "BYOKey Claude (Anthropic)",
    "BYOKey_Gemini": "BYOKey Gemini (Google)",
    "BYOKey_OpenRouter": "BYOKey OpenRouter LLM",
    "BYOKey_ByteDance_LLM": "BYOKey ByteDance Seed LLM",
    "BYOKey_ImageGen": "BYOKey GPT Image (OpenAI)",
    "BYOKey_GeminiImage": "BYOKey Gemini Image (Google)",
    "BYOKey_Flux": "BYOKey Flux (BFL)",
    "BYOKey_StabilityImage": "BYOKey Stability AI Image",
    "BYOKey_Grok_Image": "BYOKey Grok Image (xAI)",
    "BYOKey_Ideogram": "BYOKey Ideogram Image",
    "BYOKey_Recraft_Image": "BYOKey Recraft Image",
    "BYOKey_Reve_Image": "BYOKey Reve Image",
    "BYOKey_Bria_Image": "BYOKey Bria Image",
    "BYOKey_Runway_Text2Image": "BYOKey Runway Text to Image",
    "BYOKey_Magnific": "BYOKey Magnific Upscale",
    "BYOKey_WaveSpeed_Upscale": "BYOKey WaveSpeed Upscale",
    "BYOKey_VideoGen": "BYOKey Video Gen (Generic)",
    "BYOKey_Kling_Text2Video": "BYOKey Kling Text to Video",
    "BYOKey_Kling_Image2Video": "BYOKey Kling Image to Video",
    "BYOKey_MiniMax_Text2Video": "BYOKey MiniMax Text to Video",
    "BYOKey_MiniMax_Image2Video": "BYOKey MiniMax Image to Video",
    "BYOKey_Seedance2_Text2Video": "BYOKey Seedance 2.0 Text to Video",
    "BYOKey_Seedance2_Image2Video": "BYOKey Seedance 2.0 Image to Video",
    "BYOKey_Runway_Image2Video": "BYOKey Runway Image to Video",
    "BYOKey_Luma_Text2Video": "BYOKey Luma Text to Video",
    "BYOKey_Luma_Image2Video": "BYOKey Luma Image to Video",
    "BYOKey_PixVerse_Text2Video": "BYOKey PixVerse Text to Video",
    "BYOKey_Vidu_Text2Video": "BYOKey Vidu Text to Video",
    "BYOKey_Vidu_Image2Video": "BYOKey Vidu Image to Video",
    "BYOKey_Veo": "BYOKey Google Veo Video",
    "BYOKey_Wan_Text2Video": "BYOKey Wan Text to Video",
    "BYOKey_Wan_Image2Video": "BYOKey Wan Image to Video",
    "BYOKey_LTXV_Text2Video": "BYOKey LTXV Text to Video",
    "BYOKey_Grok_Video": "BYOKey Grok Video (xAI)",
    "BYOKey_ElevenLabs_TTS": "BYOKey ElevenLabs TTS",
    "BYOKey_ElevenLabs_SFX": "BYOKey ElevenLabs Sound Effects",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
