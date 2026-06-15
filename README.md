# ComfyUI BYOKey — Bring Your Own Key

**直接用自己的 API Key 调用 AI 服务，支持自定义中转站地址，输入/输出类型与官方节点 100% 兼容。**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Nodes](https://img.shields.io/badge/Nodes-37-blue)](#node-list)

ComfyUI 官方 API 节点强制所有请求走 `api.comfy.org` 代理：用户无法使用自己的 API Key，也无法指定第三方中转站。

本项目提供 **37 个轻量替代节点**，覆盖所有主流官方 API 提供商。每个节点都有 `api_key` 和 `base_url` 输入框：

- 填自己的 Key，直连官方 API
- 改 `base_url`，走第三方中转站（对国内用户尤其重要）
- **输出类型与官方节点完全一致**（`VIDEO` / `AUDIO` / `IMAGE` / `STRING`），可直接替换官方工作流中的节点

## 与 comfy-api-liberation 的区别

| | comfy-api-liberation | **BYOKey（本项目）** |
|---|---|---|
| 实现方式 | Monkey-patch 官方节点的请求 | 独立替代节点 |
| 自定义 URL | ❌ 不支持（硬编码官方地址） | ✅ 每个节点都有 `base_url` |
| 中转站支持 | 无 | **完整支持** |
| 稳定性 | 依赖官方内部实现，升级易失效 | 独立实现，不受官方更新影响 |

## 输出类型对齐官方

本项目的请求/响应格式以 **官方 `comfy_api_nodes/apis/*.py` 的 pydantic 模型 + `Comfy-Org/embedded-docs`** 为基准（官方代理 `api.comfy.org` 对 body 做透明转发，所以官方模型即真实厂商 API 契约）。

| 类型 | 官方 | BYOKey |
|------|------|--------|
| 视频 | `VIDEO` | ✅ `VIDEO`（`comfy_api.latest.InputImpl.VideoFromFile`） |
| 音频 | `AUDIO` | ✅ `AUDIO`（`{"waveform","sample_rate"}`，PyAV 解码） |
| 图像 | `IMAGE` | ✅ `IMAGE`（torch tensor，BHWC） |
| 文本 | `STRING` | ✅ `STRING` |

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/MeteorAndy/comfyui-byokey.git
```

重启 ComfyUI，节点出现在 `api/byokey/...` 分类下。

## Node List

### 文本 / LLM → `STRING`

| 节点 | 提供商 | Vision |
|------|--------|--------|
| BYOKey LLM (OpenAI Compatible) | OpenAI / DeepSeek / 通义千问等 | ✅ |
| BYOKey Claude (Anthropic) | Anthropic | ✅ |
| BYOKey Gemini (Google) | Google | ✅ |
| BYOKey OpenRouter LLM | OpenRouter（200+ 模型） | ✅ |
| BYOKey ByteDance Seed LLM | ByteDance / Doubao（火山方舟） | ✅ |

### 图像生成 → `IMAGE`

| 节点 | 提供商 |
|------|--------|
| BYOKey GPT Image (OpenAI) | OpenAI GPT Image 2 / DALL·E 3 |
| BYOKey Gemini Image (Google) | Google Gemini / Nano Banana |
| BYOKey Flux (BFL) | Black Forest Labs Flux Pro |
| BYOKey Stability AI Image | Stability AI (SD3.5 / Ultra) |
| BYOKey Grok Image (xAI) | xAI |
| BYOKey Ideogram Image | Ideogram V3 |
| BYOKey Recraft Image | Recraft V3 |
| BYOKey Reve Image | Reve |
| BYOKey Bria Image | Bria |
| BYOKey Runway Text to Image | Runway Gen4 Image |

### 图像超分 → `IMAGE`

| 节点 | 提供商 |
|------|--------|
| BYOKey Magnific Upscale | Magnific AI |
| BYOKey WaveSpeed Upscale | WaveSpeed (AuraSR) |

### 视频生成 → `VIDEO`

| 节点 | 提供商 |
|------|--------|
| BYOKey Seedance 2.0 Text/Image to Video | ByteDance |
| BYOKey Kling Text/Image to Video | Kling |
| BYOKey MiniMax Text/Image to Video | MiniMax / Hailuo |
| BYOKey Runway Image to Video | Runway Gen4 Turbo |
| BYOKey Luma Text/Image to Video | Luma Dream Machine |
| BYOKey Google Veo Video | Google Veo 3 |
| BYOKey Wan Text/Image to Video | Wan |
| BYOKey PixVerse Text to Video | PixVerse |
| BYOKey Vidu Text/Image to Video | Vidu |
| BYOKey LTXV Text to Video | LTX Video |
| BYOKey Grok Video (xAI) | xAI |
| BYOKey Video Gen (Generic) | 任意 submit+poll API |

### 音频 → `AUDIO`

| 节点 | 提供商 |
|------|--------|
| BYOKey ElevenLabs TTS | ElevenLabs |
| BYOKey ElevenLabs Sound Effects | ElevenLabs |

## 使用示例

### 用中转站访问 OpenAI

`base_url` 填你的中转站地址即可（适用于所有节点）：

```
api_key:  sk-xxxx
base_url: https://your-proxy.com/v1
model:    gpt-4o
```

### 用 DeepSeek（OpenAI 兼容）

```
api_key:  sk-your-deepseek-key
base_url: https://api.deepseek.com/v1
model:    deepseek-chat
```

### 用通义千问（OpenAI 兼容）

```
api_key:  sk-xxxx
base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
model:    qwen-max
```

### GPT Image 2 生图

```
api_key:  sk-xxxx
base_url: https://api.openai.com/v1
model:    gpt-image-2
size:     1536x1024
quality:  high
```

## 依赖

仅需 ComfyUI 自带的依赖（`torch`、`aiohttp`、`PIL`、`numpy`、`av`），无额外安装。

## 验证与已知限制

- 请求/响应格式、模型 id、参数范围均以官方 `comfy_api_nodes/` 源码与 `Comfy-Org/embedded-docs` 为基准对齐。
- 直连模式下，需图片输入的节点把图像作为 **inline base64 data URI** 发送（官方走代理时上传到 ComfyAPI 存储）；MiniMax 等 API 原生接受该格式。
- 部分厂商（如 Bria）官方仅提供 edit 类节点，本项目按其公开 text-to-image endpoint 实现。

## License

MIT
