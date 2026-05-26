"""Utility functions for BYOKey custom nodes."""
import asyncio
import base64
import logging
from io import BytesIO

import aiohttp
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def tensor_to_base64(image_tensor: torch.Tensor, mime_type: str = "image/png") -> str:
    """Convert [B,H,W,C] or [H,W,C] tensor to base64 string."""
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]
    image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    buf = BytesIO()
    fmt = mime_type.split("/")[-1].upper()
    if fmt == "JPG":
        fmt = "JPEG"
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def bytes_to_tensor(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes to [1,H,W,C] float32 tensor (RGBA)."""
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def run_async(coro):
    """Run an async coroutine from sync context (ComfyUI node execution)."""
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
    timeout: float = 300.0,
) -> dict:
    """Make an HTTP request and return parsed JSON."""
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method, url, headers=headers, json=json_data,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"API error (HTTP {resp.status}): {text[:2000]}")
            return await resp.json()


async def download_bytes(url: str, headers: dict | None = None, timeout: float = 120.0) -> bytes:
    """Download raw bytes from a URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url, headers=headers or {},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Download failed (HTTP {resp.status})")
            return await resp.read()
