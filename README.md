# ComfyUI BYOKey - Bring Your Own Key

**直接用自己的 API Key 调用 AI 服务，支持自定义中转站地址。**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-blue)](https://github.com/comfyanonymous/ComfyUI)

ComfyUI 官方 API 节点强制所有请求走 `api.comfy.org` 代理，用户无法使用自己的 API Key，也无法指定第三方中转站。

本项目提供 **38 个轻量替代节点**，覆盖所有官方 API 提供商，每个节点都有 `api_key` 和 `base_url` 输入框：

- 填自己的 Key，直连官方 API
- 改 `base_url`，走第三方中转站（对国内用户尤其重要）
- 输入输出类型与官方节点完全兼容，可直接替换

## 与 comfy-api-liberation 的区别

| | comfy-api-liberation | BYOKey (本项目) |
|---|---|---|
| 方式 | Monkey-patch 官方节点请求 | 独立替代节点 |
| 自定义 URL | 不支持（硬编码官方地址） | 每个节点都有 `base_url` |
| 中转站支持 | 无 | 完整支持 |
| 稳定性 | 依赖官方内部实现，升级可能失效 | 独立实现，不受官方更新影响 |

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/MeteorAndy/comfyui-byokey.git
```

重启 ComfyUI，节点出现在 `api/byokey` 分类下。

## 节点列表

### LLM 文本生成 (输出 STRING)

| 节点 | 提供商 | 支持 Vision |
|------|--------|-------------|
| BYOKey LLM (OpenAI Compatible) | OpenAI / DeepSeek / 通义千问等 | Yes |
| BYOKey Claude (Anthropic) | Anthropic | Yes |
| BYOKey Gemini (Google) | Google | Yes |
| BYOKey OpenRouter LLM | OpenRouter (200+ models) | Yes |
| BYOKey ByteDance Seed LLM | ByteDance / Doubao | Yes |

### 图像生成 (输出 IMAGE)

| 节点 | 提供商 |
|------|--------|
| BYOKey GPT Image (OpenAI) | OpenAI GPT Image 2 / DALL-E 3 |
| BYOKey Gemini Image (Google) | Google Gemini |
| BYOKey Flux (BFL) | Black Forest Labs |
| BYOKey Stability AI Image | Stability AI (SD3.5, Ultra) |
| BYOKey Grok Image (xAI) | xAI |
| BYOKey Ideogram Image | Ideogram |
| BYOKey Recraft Image | Recraft |
| BYOKey Reve Image | Reve AI |
| BYOKey Bria Image | Bria AI |
| BYOKey Runway Text to Image | Runway |

### 图像超分 (输出 IMAGE)

| 节点 | 提供商 |
|------|--------|
| BYOKey Magnific Upscale | Magnific AI |
| BYOKey WaveSpeed Upscale | WaveSpeed |

### 视频生成 (输出 STRING 文件路径)

| 节点 | 提供商 |
|------|--------|
| BYOKey Seedance 2.0 Text/Image to Video | ByteDance |
| BYOKey Kling Text/Image to Video | Kling AI |
| BYOKey MiniMax Text/Image to Video | MiniMax / Hailuo |
| BYOKey Runway Image to Video | Runway (Gen4 Turbo) |
| BYOKey Luma Text/Image to Video | Luma Dream Machine |
| BYOKey Google Veo Video | Google Veo 3 |
| BYOKey Wan Text/Image to Video | Wan |
| BYOKey PixVerse Text to Video | PixVerse |
| BYOKey Vidu Text/Image to Video | Vidu |
| BYOKey LTXV Text to Video | LTX Video |
| BYOKey Grok Video (xAI) | xAI |
| BYOKey Video Gen (Generic) | 任意 submit+poll API |

### 音频 (输出 STRING 文件路径)

| 节点 | 提供商 |
|------|--------|
| BYOKey ElevenLabs TTS | ElevenLabs |
| BYOKey ElevenLabs Sound Effects | ElevenLabs |

## 使用示例

### 使用中转站访问 OpenAI

在 `base_url` 填入你的中转站地址即可：

```
api_key: sk-your-key
base_url: https://your-proxy.com/v1
model: gpt-4o
```

### 使用 DeepSeek

```
api_key: sk-your-deepseek-key
base_url: https://api.deepseek.com/v1
model: deepseek-chat
```

## 依赖

仅需 ComfyUI 自带的依赖（torch, aiohttp, PIL, numpy），无额外安装。

## License

MIT
