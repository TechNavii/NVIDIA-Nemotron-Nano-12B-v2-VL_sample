NVIDIA Nemotron Nano VL 12B v2 FP8 — Local GUI

This repo provides a lightweight GUI to chat with NVIDIA’s vision‑language model “Nemotron Nano VL 12B v2 FP8” using an OpenAI‑compatible API. The recommended local serving path is vLLM’s OpenAI server, which works with the FP8 checkpoint from Hugging Face.

What you get
- Streamlit GUI to upload an image and prompt the model
- Works with any OpenAI‑compatible endpoint (local vLLM, NIM, or NVIDIA cloud if exposed as OpenAI API)
- Simple configuration via environment variables or onscreen settings
- Supports images, PDFs, and videos (video pre-processed into frames by default)
- Remote URL ingestion for image/PDF/video; optional experimental video_url pass-through
  - When using the NVIDIA Integrate endpoint, video pass-through encodes the video as a base64 data URL (e.g., `data:video/mp4;base64,...`) to match NVIDIA’s sample.
  - Some sites (e.g., Pexels pages) are HTML landing pages; use a direct media URL (ending in .mp4/.jpg/.pdf) or enable "Send as video_url (experimental)" to pass the remote URL to the backend without downloading.
- Multiple images per request supported (uploads capped by `MAX_IMAGES`)
- Reasoning toggle: Enable (/think), Disable (/no_think), or Default
- Backend selector: Remote API or Local vLLM (NVIDIA GPU)

Quick Start (Single Script)
- One command (installs deps, starts server, launches GUI):
  - `bash scripts/run.sh`  (on macOS auto-switches to GUI-only)
- Server only (Linux + NVIDIA GPU):
  - `bash scripts/run.sh server`
- GUI only (point to a remote server via envs):
  - `OPENAI_API_BASE=http://<server>:8000/v1 OPENAI_API_KEY=EMPTY bash scripts/run.sh gui`
Notes
- You can run the script from any directory; it resolves the repo root automatically.
- On macOS, the script defaults to GUI-only since vLLM/CUDA server is not supported.

Prerequisites
- NVIDIA GPU and CUDA drivers installed
- Python 3.10+
- Enough VRAM for the model (the model card indicates H100 SXM 80GB as the reference target; smaller GPUs may OOM)

Serve the Model (vLLM, local)
1) Create a Python env (optional but recommended)
   - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`
   - Windows (PowerShell): `python -m venv .venv; .venv\Scripts\Activate.ps1`

2) Install vLLM and model dependencies (from the model card):
   pip install causal_conv1d "transformers>4.53,<4.54" torch timm "mamba-ssm==2.2.5" accelerate open_clip_torch numpy pillow vllm

3) Launch vLLM’s OpenAI server with the Nemotron VL FP8 checkpoint:
   python -m vllm.entrypoints.openai.api_server \
     --model nvidia/Nemotron-Nano-VL-12B-V2-FP8 \
     --trust-remote-code \
     --quantization modelopt

   Notes
   - Default address is http://127.0.0.1:8000/v1
   - If your GPU/driver stack lacks FP8 support, the server may fail or OOM.
   - You may add flags like `--gpu-memory-utilization 0.9` or `--max-model-len 8192` depending on your GPU.

Run the GUI
1) In another terminal (same env is fine), install GUI requirements:
   pip install -r requirements.txt

2) Start the app:
   streamlit run app.py

3) Open the displayed local URL. Configure as needed in the sidebar:
   - API Base: `http://127.0.0.1:8000/v1` (for local vLLM)
   - API Key: any non-empty string (vLLM doesn’t verify it; `EMPTY` is fine)
   - Model ID: `nvidia/Nemotron-Nano-VL-12B-V2-FP8`

Using NVIDIA NIM or Cloud Endpoints (optional)
- If you serve this model via NVIDIA NIM or another OpenAI‑compatible gateway, set the GUI’s API Base to that endpoint and provide the real API key. The request format uses OpenAI Chat Completions with multimodal content (image + text), which is what most modern VLM endpoints accept.
  - Default API Base: `https://integrate.api.nvidia.com/v1/chat/completions` (editable in the sidebar).
  - For NVIDIA Integrate, use model `nvidia/nemotron-nano-12b-v2-vl`.
  - The GUI also accepts a base ending in `/v1` and will call `/chat/completions` under it.
  - Some hosted endpoints do not expose `/models`. Use the sidebar “Test API” button; it falls back to a minimal `/chat/completions` ping.

How it works
- The GUI uses the OpenAI Python SDK and calls `chat.completions.create` with a `messages` payload containing both text and the uploaded image (as base64 data URL). The server (vLLM/NIM) must support multimodal inputs for the selected model.

Troubleshooting
- 404/405 from the server: Ensure you’re using the OpenAI entrypoint (`.../v1`) and that the server is actually running with that model.
- Out of memory: Try smaller `max_tokens`, lower `gpu-memory-utilization`, or move to a larger GPU (H100 is the reference target per model card).
- Long load times on first request: Model warming can take time; subsequent requests are faster.
- Video not working: By default, the GUI samples frames and sends them as images. If your server supports video directly, enable "Send as video_url (experimental)" under the Video tab. If it fails, disable it to fall back to frames.
- Reasoning + video_url: The model template does not allow explicit reasoning with `video_url`. If reasoning is enabled, the GUI automatically falls back to frames instead of `video_url`.
- PDFs too long: The GUI limits rendered pages. Adjust `MAX_PDF_PAGES` env or summarize per page.

Security & Licensing
- Your use of the model is subject to the NVIDIA Open Model License. See the model card for details on allowed usage.
 - No secrets committed: This repo contains no API keys or PII. Use environment variables for credentials (e.g., `OPENAI_API_KEY`). A `.gitignore` is included to prevent committing `.env`, virtualenvs, and logs.
Advanced Config (env vars)
- `MAX_IMAGE_DIM` (default 1024): Longest side for uploaded/derived images.
- `MAX_IMAGES` (default 8): Max number of images accepted in Image mode.
- `MAX_PDF_PAGES` (default 3): Max PDF pages to render as images.
- `MAX_VIDEO_FRAMES` (default 8): Max frames sampled from a video.
- `DOWNLOAD_MAX_MB` (default 200): Max size for remote URL downloads.
- Reasoning toggle is controlled in the sidebar (no env by default). Enabling adds `/think`; disabling adds `/no_think` to the prompt per the model’s chat template.

Capabilities
- Images: Upload or provide remote URL.
- PDFs: Upload or remote URL; pages rendered to images and sent as multimodal inputs.
- Videos: Upload or remote URL. Default behavior is to extract a small set of frames (uniformly) and send as images. Experimental option to send `video_url` directly if your server supports it.
- Backends
- Remote API (default): Connects to an OpenAI‑compatible `/v1` endpoint (NIM/NVCF/vLLM). Provide API Base and key.
- Local vLLM (NVIDIA GPU): Same request format but default base is `http://127.0.0.1:8000/v1`. Requires Linux + NVIDIA GPUs.
