import base64
import io
import os
import time
import tempfile
from typing import List, Optional, Tuple

import streamlit as st
from PIL import Image
import requests
import numpy as np
import imageio.v3 as iio
from urllib.parse import urlparse

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


MAX_IMAGE_DIM = int(os.environ.get("MAX_IMAGE_DIM", "1024"))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "3"))
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "8"))
DOWNLOAD_MAX_MB = int(os.environ.get("DOWNLOAD_MAX_MB", "200"))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "8"))


def b64_image(file_bytes: bytes, mime: str) -> str:
    enc = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{enc}"


def pil_to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    out = io.BytesIO()
    save_kwargs = {}
    if fmt.upper() == "JPEG":
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    img.save(out, format=fmt, **save_kwargs)
    mime = f"image/{fmt.lower()}"
    return b64_image(out.getvalue(), mime)


def resize_image(img: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    w, h = img.size
    scale = min(1.0, float(max_dim) / float(max(w, h)))
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def download_to_temp(url: str, max_mb: int = DOWNLOAD_MAX_MB) -> Tuple[str, Optional[str]]:
    """Download URL to a temporary file with size cap. Returns (path, content_type)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Accept": "*/*",
        "Referer": url,
    }
    try:
        head = requests.head(url, allow_redirects=True, timeout=10, headers=headers)
        ct = head.headers.get("content-type")
        cl = head.headers.get("content-length")
        if cl and int(cl) > max_mb * 1024 * 1024:
            raise RuntimeError(f"Remote file too large: {int(cl)/(1024*1024):.1f}MB > {max_mb}MB")
        headers["Range"] = "bytes=0-"  # allow partial responses
    except Exception:
        ct = None

    with requests.get(url, stream=True, timeout=30, headers=headers) as r:
        r.raise_for_status()
        total = 0
        suffix = None
        if ct:
            if "pdf" in ct:
                suffix = ".pdf"
            elif "mp4" in ct or "video" in ct:
                suffix = ".mp4"
            elif "png" in ct:
                suffix = ".png"
            elif "jpeg" in ct or "jpg" in ct:
                suffix = ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_mb * 1024 * 1024:
                    f.close()
                    os.unlink(f.name)
                    raise RuntimeError(f"Download exceeded {max_mb}MB limit")
                f.write(chunk)
            path = f.name
    return path, ct


def detect_media_type(path: str, ct: Optional[str], url: str) -> str:
    """Best-effort media type detection: returns 'pdf' | 'video' | 'image' | 'unknown'."""
    kind = (ct or "").lower()
    try:
        ext = os.path.splitext(urlparse(url).path)[1].lower()
    except Exception:
        ext = os.path.splitext(url)[1].lower()

    if "pdf" in kind or ext == ".pdf":
        return "pdf"
    if "video" in kind or ext in (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"):
        return "video"
    if "image" in kind or ext in (".png", ".jpg", ".jpeg", ".webp"):
        return "image"

    # Try reading as video
    try:
        meta = iio.immeta(path, plugin="ffmpeg")
        if meta and (meta.get("fps") or meta.get("duration")):
            return "video"
    except Exception:
        pass

    # Try reading as image
    try:
        with Image.open(path) as im:
            im.verify()
        return "image"
    except Exception:
        pass
    return "unknown"


def extract_pdf_images(file_bytes: bytes, max_pages: int = MAX_PDF_PAGES) -> List[Image.Image]:
    if fitz is None:
        raise RuntimeError("PyMuPDF (pymupdf) not installed. Install via requirements.txt")
    images: List[Image.Image] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        pages = min(len(doc), max_pages)
        for i in range(pages):
            page = doc.load_page(i)
            # 196 dpi balances quality and size
            mat = fitz.Matrix(196 / 72.0, 196 / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(resize_image(img))
    return images


def extract_video_frames(path: str, max_frames: int = MAX_VIDEO_FRAMES) -> List[Image.Image]:
    """Uniformly sample up to max_frames frames using imageio."""
    try:
        meta = iio.immeta(path, plugin="ffmpeg")
        duration = float(meta.get("duration", 0)) or None
        fps = float(meta.get("fps", 0)) or None
    except Exception:
        duration = None
        fps = None

    frames: List[Image.Image] = []
    try:
        total_frames = None
        if fps and duration:
            total_frames = int(fps * duration)
        # Read iteratively and subsample
        reader = iio.imiter(path, plugin="ffmpeg")
        if total_frames and total_frames > max_frames:
            stride = max(total_frames // max_frames, 1)
        else:
            stride = 1
        idx = 0
        for frame in reader:
            if idx % stride == 0:
                # convert ndarray (H,W,C) uint8 to PIL
                if isinstance(frame, np.ndarray):
                    img = Image.fromarray(frame)
                else:
                    img = Image.fromarray(np.asarray(frame))
                frames.append(resize_image(img))
                if len(frames) >= max_frames:
                    break
            idx += 1
    except Exception as e:
        raise RuntimeError(f"Failed to read video: {e}")
    if not frames:
        raise RuntimeError("No frames extracted from video")
    return frames


def create_client(base_url: str, api_key: str):
    if OpenAI is None:
        raise RuntimeError(
            "openai package is not installed. Run `pip install -r requirements.txt`."
        )
    # vLLM/NIM often accept any non-empty key for local use
    key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    base = base_url or os.environ.get("OPENAI_API_BASE") or "http://127.0.0.1:8000/v1"
    return OpenAI(base_url=base, api_key=key)


def is_chat_endpoint(url: str) -> bool:
    try:
        path = urlparse(url).path.rstrip("/")
    except Exception:
        return url.rstrip("/").endswith("/chat/completions") or url.rstrip("/").endswith("/responses")
    return path.endswith("/chat/completions") or path.endswith("/responses")


def is_nvidia_integrate(url: str) -> bool:
    try:
        host = urlparse(url).netloc
    except Exception:
        host = ""
    return "integrate.api.nvidia.com" in host


def send_request(
    client,
    model: str,
    user_text: str,
    image_data_urls: Optional[List[str]] = None,
    video_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    reasoning_tag: Optional[str] = None,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 512,
    chat_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content_items = []
    if reasoning_tag:
        content_items.append({"type": "text", "text": reasoning_tag})
    if user_text:
        content_items.append({"type": "text", "text": user_text})
    if image_data_urls:
        for url in image_data_urls:
            content_items.append({"type": "image_url", "image_url": {"url": url}})
    if video_url:
        # Experimental: some servers/models may accept video_url directly.
        content_items.append({"type": "video_url", "video_url": {"url": video_url}})

    messages.append({"role": "user", "content": content_items or user_text})

    start = time.time()
    if chat_url:
        headers = {
            "Authorization": f"Bearer {api_key or ''}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        r = requests.post(chat_url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        resp = r.json()
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    latency = time.time() - start
    return resp, latency


def main():
    st.set_page_config(page_title="Nemotron-Nano-VL GUI", page_icon="🖼️")
    st.title("NVIDIA Nemotron Nano VL 12B v2 FP8 — GUI")

    with st.sidebar:
        st.header("Backend")
        backend = st.selectbox(
            "Mode",
            [
                "Remote API (OpenAI-compatible)",
                "Local vLLM (NVIDIA GPU)",
            ],
            index=0,
        )

        st.header("Server Settings")
        if backend == "Local vLLM (NVIDIA GPU)":
            default_base = "http://127.0.0.1:8000/v1"
        else:
            default_base = os.environ.get(
                "OPENAI_API_BASE", "https://integrate.api.nvidia.com/v1/chat/completions"
            )

        # Initialize session state for input defaults
        if "api_base" not in st.session_state:
            st.session_state["api_base"] = default_base
        if "api_key" not in st.session_state:
            st.session_state["api_key"] = os.environ.get("OPENAI_API_KEY", "")
        if "model_id" not in st.session_state:
            st.session_state["model_id"] = os.environ.get(
                "MODEL_ID", "nvidia/Nemotron-Nano-VL-12B-V2-FP8"
            )

        api_base_input = st.text_input(
            "API Base", key="api_base", help="For Remote API/Local vLLM modes"
        )
        api_key = st.text_input("API Key", key="api_key", type="password")
        model_id = st.text_input(
            "Model ID",
            key="model_id",
            help=(
                "Backend's model name. Examples: "
                "vLLM: nvidia/Nemotron-Nano-VL-12B-V2-FP8 • NVIDIA Integrate: nvidia/nemotron-nano-12b-v2-vl"
            ),
        )

        # Sanitize API base for use; do not overwrite the textbox value
        api_base = (api_base_input or "").strip().rstrip("/")
        if api_base and "://" not in api_base:
            # Assume https when scheme missing
            api_base = "https://" + api_base
        elif not api_base:
            api_base = default_base.rstrip("/")
        # Determine if user provided full chat endpoint (robust to trailing slash/query)
        is_full_chat = is_chat_endpoint(api_base)
        # Ensure '/v1' is present for OpenAI-compatible APIs when not a full chat endpoint
        if not is_full_chat and "/v1" not in api_base:
            api_base = api_base + "/v1"
        chat_url = api_base if is_full_chat else None
        using_integrate = is_nvidia_integrate(api_base)

        # Preset buttons removed per request

        def ping_api(base: str, key: str, model: str) -> str:
            url_base = base.rstrip("/")
            try:
                if is_chat_endpoint(url_base):
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                    }
                    r2 = requests.post(
                        url_base,
                        headers={
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        },
                        json=payload,
                        timeout=10,
                    )
                    return f"{url_base} → {r2.status_code} {r2.reason}"
                else:
                    r = requests.get(
                        url_base + "/models",
                        headers={"Authorization": f"Bearer {key}"} if key else None,
                        timeout=5,
                    )
                    if r.status_code == 200:
                        return f"{url_base}/models → {r.status_code} {r.reason}"
                    # Fallback to chat
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                    }
                    r2 = requests.post(
                        url_base.rstrip("/") + "/chat/completions",
                        headers={
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        },
                        json=payload,
                        timeout=10,
                    )
                    return f"{url_base.rstrip('/')}/chat/completions → {r2.status_code} {r2.reason}"
            except Exception as e:
                return f"ERR {e.__class__.__name__}: {e}"

        # Determine effective model id based on backend
        effective_model_id = model_id
        if using_integrate and model_id.strip().lower() != "nvidia/nemotron-nano-12b-v2-vl":
            effective_model_id = "nvidia/nemotron-nano-12b-v2-vl"
            st.caption("Note: Using NVIDIA Integrate. Overriding model id to 'nvidia/nemotron-nano-12b-v2-vl'.")

        if st.button("Test API"):
            st.info(ping_api(api_base, api_key, effective_model_id))
        # Display the exact endpoint that will be used for generation
        effective_endpoint = chat_url or (api_base.rstrip("/") + "/chat/completions")
        st.caption(f"Using Endpoint: {effective_endpoint}")

        st.header("Generation Settings")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.2, 0.05)
        top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.01)
        max_tokens = st.number_input("Max Tokens", min_value=16, max_value=32768, value=512, step=16)
        reasoning_mode = st.selectbox(
            "Reasoning",
            ["Default", "Enable (/think)", "Disable (/no_think)"],
            index=0,
            help="Enable adds /think to the prompt. Disable adds /no_think. Default sends no flag.",
        )

        st.markdown(
            "Tips\n- Use vLLM OpenAI server for local inference.\n- For vLLM, any non-empty API key is typically accepted.\n- PDFs and videos are pre-processed into images by default."
        )
    st.subheader("Input")
    input_mode = st.selectbox("Input Type", ["Image", "PDF", "Video", "URL"], index=0)

    uploaded = None
    remote_url = None
    image_list: List[Image.Image] = []
    video_pass_url: Optional[str] = None
    image_urls: Optional[List[str]] = None

    if input_mode == "Image":
        uploaded = st.file_uploader(
            "Upload image(s) (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        ) 
    elif input_mode == "PDF":
        uploaded = st.file_uploader("Upload PDF", type=["pdf"]) 
        max_pages = st.slider("Max pages", 1, 12, min(MAX_PDF_PAGES, 12))
    elif input_mode == "Video":
        uploaded = st.file_uploader("Upload video (MP4/AVI/MOV)", type=["mp4", "mov", "avi", "mkv"]) 
        max_frames = st.slider("Frames to sample", 2, 24, min(MAX_VIDEO_FRAMES, 12))
        pass_through = st.checkbox("Send as video_url (experimental)", value=False)
        remote_url = st.text_input("Or remote video URL (http/https)")
        # If reasoning is explicitly enabled, avoid video_url to prevent server errors.
        if 'reasoning_mode' in locals() and reasoning_mode.startswith("Enable") and pass_through:
            st.warning("Reasoning is not supported with video_url. Falling back to frame extraction.")
            pass_through = False
    elif input_mode == "URL":
        remote_url = st.text_input("Remote media URL (image/pdf/video)")
        url_pass_through = st.checkbox(
            "Pass URL to backend (video_url, no download)", value=False,
            help="If enabled, the URL is sent directly to the backend as a video_url. Useful when sites block downloads.")

    user_prompt = st.text_area("Prompt", placeholder="Ask about the content…")
    system_prompt = st.text_area("System Prompt (optional)", placeholder="You are a helpful vision assistant.")

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("Generate", type="primary")
    with col2:
        clear_btn = st.button("Clear")

    if clear_btn:
        st.experimental_rerun()

    previews: List[Image.Image] = []
    b64_urls: List[str] = []

    try:
        if input_mode == "Image" and uploaded:
            files = uploaded[:MAX_IMAGES]
            imgs: List[Image.Image] = []
            for f in files:
                try:
                    data = f.read()
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    imgs.append(resize_image(img))
                except Exception as e:
                    st.warning(f"Skipping one image due to error: {e}")
            previews = imgs
            b64_urls = [pil_to_data_url(im, fmt="JPEG") for im in imgs]

        elif input_mode == "PDF" and uploaded is not None:
            file_bytes = uploaded.read()
            imgs = extract_pdf_images(file_bytes, max_pages=max_pages)
            previews = imgs
            b64_urls = [pil_to_data_url(im, fmt="JPEG") for im in imgs]

        elif input_mode == "Video":
            # Prefer remote URL if provided
            if remote_url:
                if pass_through:
                    # Pass remote URL directly; backend should fetch it
                    video_pass_url = remote_url
                else:
                    path, ct = download_to_temp(remote_url)
                    frames = extract_video_frames(path, max_frames=max_frames)
                    previews = frames
                    b64_urls = [pil_to_data_url(im, fmt="JPEG") for im in frames]
            elif uploaded is not None:
                # Save to temp for frame extraction if needed
                suffix = os.path.splitext(uploaded.name)[1]
                raw = uploaded.read()
                if pass_through:
                    # Build base64 data URL for video
                    ext = (suffix or "").lower()
                    if ext in (".mp4", ".m4v"):
                        mime = "video/mp4"
                    elif ext == ".mov":
                        mime = "video/mov"
                    elif ext == ".webm":
                        mime = "video/webm"
                    elif ext == ".avi":
                        mime = "video/avi"
                    else:
                        mime = "video/mp4"
                    video_pass_url = b64_image(raw, mime)
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(raw)
                        tmp_path = tmp.name
                    frames = extract_video_frames(tmp_path, max_frames=max_frames)
                    previews = frames
                    b64_urls = [pil_to_data_url(im, fmt="JPEG") for im in frames]

        elif input_mode == "URL" and remote_url:
            if url_pass_through:
                # Do not download; pass URL directly as video_url
                video_pass_url = remote_url
            else:
                # Download and robustly auto-detect
                path, ct = download_to_temp(remote_url)
                mtype = detect_media_type(path, ct, remote_url)
                if mtype == "pdf":
                    with open(path, "rb") as f:
                        imgs = extract_pdf_images(f.read(), max_pages=MAX_PDF_PAGES)
                    previews = imgs
                    b64_urls = [pil_to_data_url(im, fmt="JPEG") for im in imgs]
                elif mtype == "video":
                    # For URL mode, always extract frames; pass-through exists as a checkbox
                    frames = extract_video_frames(path, max_frames=MAX_VIDEO_FRAMES)
                    previews = frames
                    b64_urls = [pil_to_data_url(im, fmt="JPEG") for im in frames]
                elif mtype == "image":
                    img0 = Image.open(path).convert("RGB")
                    img0 = resize_image(img0)
                    previews = [img0]
                    b64_urls = [pil_to_data_url(img0, fmt="JPEG")]
                else:
                    # Fall back to passing URL directly to backend as video_url
                    video_pass_url = remote_url
                    st.info("Media type unknown — sending URL directly to backend as video_url.")
    except Exception as e:
        st.error(f"Input processing error: {e}")

    if previews:
        if len(previews) == 1:
            caption_arg = "Preview"
        else:
            caption_arg = [f"Page {i+1}" for i in range(len(previews))]
        st.image(previews, caption=caption_arg, use_container_width=True)

    if run_btn:
        if not user_prompt and not (b64_urls or video_pass_url):
            st.warning("Please provide a prompt and some content (image/PDF/video).")
        else:
            try:
                client = None if chat_url else create_client(api_base, api_key)
                with st.spinner("Generating…"):
                    attempt_video_url = video_pass_url is not None
                    # Build final system prompt with reasoning control
                    effective_system = (system_prompt or "").strip()
                    if reasoning_mode.startswith("Enable"):
                        effective_system = ("/think\n" + effective_system).strip()
                    elif reasoning_mode.startswith("Disable"):
                        effective_system = ("/no_think\n" + effective_system).strip()
                    # Enforce /no_think for videos
                    if attempt_video_url or (b64_urls and input_mode == "Video"):
                        if not effective_system.startswith("/no_think"):
                            effective_system = ("/no_think\n" + effective_system).strip()

                    # Use backend-specific model id
                    eff_model = effective_model_id
                    resp, latency = send_request(
                        client,
                        model=eff_model,
                        user_text=user_prompt,
                        image_data_urls=b64_urls or None,
                        video_url=video_pass_url if attempt_video_url else None,
                        system_prompt=effective_system if effective_system else None,
                        reasoning_tag=None,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=int(max_tokens),
                        chat_url=chat_url,
                        api_key=api_key,
                    )
                # Extract text for both OpenAI SDK objects and plain dicts
                text = ""
                usage = None
                if hasattr(resp, "choices"):
                    text = resp.choices[0].message.content if resp.choices else ""
                    usage = getattr(resp, "usage", None)
                elif isinstance(resp, dict):
                    choices = resp.get("choices")
                    if choices:
                        text = choices[0].get("message", {}).get("content", "")
                    usage = resp.get("usage")

                st.subheader("Response")
                st.write(text)

                if usage:
                    try:
                        if isinstance(usage, dict):
                            pt = usage.get("prompt_tokens")
                            ct = usage.get("completion_tokens")
                        else:
                            pt = getattr(usage, "prompt_tokens", None)
                            ct = getattr(usage, "completion_tokens", None)
                    except Exception:
                        pt = ct = None
                    if pt is not None or ct is not None:
                        st.caption(
                            f"Latency: {latency:.2f}s • Prompt tokens: {pt if pt is not None else '?'} • Completion tokens: {ct if ct is not None else '?'}"
                        )
                    else:
                        st.caption(f"Latency: {latency:.2f}s")
                else:
                    st.caption(f"Latency: {latency:.2f}s")

                with st.expander("Raw response"):
                    # openai>=1.0 returns pydantic-like objects
                    try:
                        st.json(resp.model_dump())
                    except Exception:
                        st.write(resp)
            except Exception as e:
                detail = str(e)
                # Try to extract status and body from OpenAI API exceptions
                try:
                    resp_obj = getattr(e, "response", None)
                    if resp_obj is not None:
                        status = getattr(resp_obj, "status_code", None)
                        text = getattr(resp_obj, "text", None)
                        if status or text:
                            detail = f"status={status} body={text}"
                except Exception:
                    pass
                st.error(f"Connection error: {detail}")
                if video_pass_url:
                    st.info(
                        "Video pass-through may not be supported by your server. Disable 'Send as video_url' to use frames extraction instead."
                    )
                else:
                    st.info(
                        "If you are using vLLM locally, make sure the server is running at the configured API Base and the model is loaded."
                    )


if __name__ == "__main__":
    main()
