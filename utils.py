"""Utility functions for BYOKey custom nodes.

Provides ComfyUI-native output construction (IMAGE/VIDEO/AUDIO), tensor<->base64
conversion, async HTTP, and input validation so that BYOKey nodes produce the
exact same output types as the official ComfyUI API nodes.
"""
import asyncio
import base64
import logging
from io import BytesIO

import aiohttp
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("comfyui_byokey")


# --------------------------------------------------------------------------- #
# Image conversion                                                            #
# --------------------------------------------------------------------------- #
def tensor_to_base64(image_tensor: torch.Tensor, mime_type: str = "image/png") -> str:
    """Convert a [B,H,W,C] or [H,W,C] float tensor to a base64-encoded string."""
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]
    image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    if img.mode == "RGBA" and mime_type in ("image/jpeg", "image/jpg"):
        img = img.convert("RGB")
    buf = BytesIO()
    fmt = mime_type.split("/")[-1].upper()
    if fmt == "JPG":
        fmt = "JPEG"
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def tensor_to_data_uri(image_tensor: torch.Tensor, mime_type: str = "image/png") -> str:
    """Convert an image tensor to a ``data:<mime>;base64,...`` URI."""
    return f"data:{mime_type};base64,{tensor_to_base64(image_tensor, mime_type)}"


def bytes_to_tensor(image_bytes: bytes, mode: str = "RGBA") -> torch.Tensor:
    """Convert raw image bytes to a [1,H,W,C] float32 tensor."""
    img = Image.open(BytesIO(image_bytes)).convert(mode)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Video output (ComfyUI VIDEO type)                                           #
# --------------------------------------------------------------------------- #
def video_from_bytes(video_bytes: bytes):
    """Wrap raw video bytes (e.g. mp4) in a ComfyUI VIDEO output object.

    Returns a ``comfy_api.latest.InputImpl.VideoFromFile`` so the node can declare
    ``RETURN_TYPES = ("VIDEO",)`` exactly like the official API nodes.
    """
    from comfy_api.latest import InputImpl
    return InputImpl.VideoFromFile(BytesIO(video_bytes))


async def download_video_output(url: str, headers: dict | None = None, timeout: float = 600.0):
    """Download a video from a URL and return it as a ComfyUI VIDEO output object."""
    data = await download_bytes(url, headers=headers, timeout=timeout)
    return video_from_bytes(data)


# --------------------------------------------------------------------------- #
# Audio output (ComfyUI AUDIO type)                                           #
# --------------------------------------------------------------------------- #
def _f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert integer PCM tensors to float32 in the range [-1, 1]."""
    if wav.dtype.is_floating_point:
        return wav
    if wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    if wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def audio_from_bytes(audio_bytes: bytes) -> dict:
    """Decode audio bytes (mp3/wav/etc) into a ComfyUI AUDIO dict.

    Returns ``{"waveform": [1, C, T] float32, "sample_rate": int}`` so the node can
    declare ``RETURN_TYPES = ("AUDIO",)`` exactly like the official API nodes.
    """
    import av

    with av.open(BytesIO(audio_bytes)) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in response.")
        stream = af.streams.audio[0]
        sample_rate = int(stream.codec_context.sample_rate)
        n_channels = stream.channels or 1

        frames: list[torch.Tensor] = []
        for frame in af.decode(streams=stream.index):
            arr = frame.to_ndarray()  # [C, T] | [T, C] | [T]
            buf = torch.from_numpy(arr)
            if buf.ndim == 1:
                buf = buf.unsqueeze(0)
            elif buf.shape[0] != n_channels and buf.shape[-1] == n_channels:
                buf = buf.transpose(0, 1).contiguous()
            elif buf.shape[0] != n_channels:
                buf = buf.reshape(-1, n_channels).t().contiguous()
            frames.append(buf)

    if not frames:
        raise ValueError("Decoded zero audio frames.")

    wav = _f32_pcm(torch.cat(frames, dim=1))  # [C, T]
    return {"waveform": wav.unsqueeze(0).contiguous(), "sample_rate": sample_rate}


async def download_audio_output(url: str, headers: dict | None = None, timeout: float = 300.0) -> dict:
    """Download audio from a URL and return it as a ComfyUI AUDIO dict."""
    data = await download_bytes(url, headers=headers, timeout=timeout)
    return audio_from_bytes(data)


def audio_to_bytes(audio: dict, container: str = "mp4", codec: str = "aac") -> bytes:
    """Encode a ComfyUI AUDIO dict ({"waveform","sample_rate"}) into container bytes."""
    import av

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.ndim == 3:
        waveform = waveform[0]  # [C, T]
    samples = waveform.cpu().numpy()
    n_channels = samples.shape[0]

    out = BytesIO()
    with av.open(out, mode="w", format=container) as oc:
        ostream = oc.add_stream(codec, rate=sample_rate)
        layout = "mono" if n_channels == 1 else "stereo"
        frame = av.AudioFrame.from_ndarray(
            (samples * (2 ** 15)).astype(np.int16), format="s16", layout=layout
        )
        frame.sample_rate = sample_rate
        for packet in ostream.encode(frame):
            oc.mux(packet)
        for packet in ostream.encode(None):
            oc.mux(packet)
    return out.getvalue()


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def validate_prompt(prompt: str, *, min_length: int = 1, strip: bool = True) -> str:
    """Validate a prompt string the way official nodes do; returns the prompt."""
    check = prompt.strip() if strip else prompt
    if len(check) < min_length:
        raise ValueError(
            f"Prompt must be at least {min_length} character(s) long after trimming whitespace."
        )
    return prompt


# --------------------------------------------------------------------------- #
# Async + HTTP                                                                #
# --------------------------------------------------------------------------- #
def run_async(coro):
    """Run an async coroutine from a sync ComfyUI node execution context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


async def http_request(
    method: str,
    url: str,
    headers: dict,
    json_data: dict | None = None,
    *,
    data=None,
    timeout: float = 300.0,
) -> dict:
    """Make an HTTP request and return parsed JSON.

    Pass ``json_data`` for a JSON body or ``data`` for a raw/multipart body
    (e.g. ``aiohttp.FormData``).
    """
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method, url, headers=headers, json=json_data, data=data,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            body = await resp.read()
            if resp.status >= 400:
                raise RuntimeError(f"API error (HTTP {resp.status}): {body[:2000].decode('utf-8', 'replace')}")
            try:
                import json as _json
                return _json.loads(body)
            except Exception:
                return {"_raw": body, "_content_type": resp.headers.get("Content-Type", "")}


async def download_bytes(url: str, headers: dict | None = None, timeout: float = 120.0) -> bytes:
    """Download raw bytes from a URL (handles ``data:`` URIs too)."""
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url, headers=headers or {},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Download failed (HTTP {resp.status}) for {url[:200]}")
            return await resp.read()


async def poll_until(
    poll_fn,
    *,
    status_of,
    done_statuses,
    failed_statuses,
    interval: float = 10.0,
    max_wait: float = 600.0,
    on_fail_message: str = "Generation failed",
):
    """Generic submit+poll loop.

    ``poll_fn`` is an async callable returning the latest task JSON; ``status_of`` maps
    that JSON to a lowercase status string. Returns the final JSON on success.
    """
    elapsed = 0.0
    last = None
    done = {s.lower() for s in done_statuses}
    failed = {s.lower() for s in failed_statuses}
    while elapsed < max_wait:
        await asyncio.sleep(interval)
        elapsed += interval
        last = await poll_fn()
        status = str(status_of(last) or "").lower()
        if status in failed:
            raise RuntimeError(f"{on_fail_message}. Status={status}. Response={last}")
        if status in done:
            return last
    raise RuntimeError(f"Timed out after {max_wait:.0f}s. Last response={last}")
