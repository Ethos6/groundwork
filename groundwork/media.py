"""Loading images, video frames and documents referenced by message parts."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".py", ".js", ".ts", ".html", ".xml", ".yaml", ".yml", ".log"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def resolve(path: str, root: str | os.PathLike = ".") -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def load_image(part_or_path, root: str | os.PathLike = "."):
    """Return a PIL image from a part ({"path"} or {"url": data URL}) or a path."""
    from PIL import Image

    if isinstance(part_or_path, dict):
        if part_or_path.get("url"):
            url = part_or_path["url"]
            if not url.startswith("data:"):
                raise ValueError("Only data: URLs can be decoded locally; pass remote URLs to an API model instead.")
            b64 = url.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        part_or_path = part_or_path["path"]
    return Image.open(resolve(part_or_path, root)).convert("RGB")


def video_frames(path: str, n: int = 8, root: str | os.PathLike = "."):
    """Sample ``n`` evenly spaced frames from a video as PIL images (needs PyAV or OpenCV)."""
    full = str(resolve(path, root))
    try:
        import av  # type: ignore

        with av.open(full) as container:
            frames = [f.to_image() for f in container.decode(video=0)]
    except ImportError:
        try:
            import cv2  # type: ignore
            from PIL import Image
        except ImportError as e:
            raise ImportError("Reading video needs PyAV (`pip install av`) or OpenCV.") from e
        cap = cv2.VideoCapture(full)
        frames = []
        ok, frame = cap.read()
        while ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            ok, frame = cap.read()
        cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {full}")
    if len(frames) <= n:
        return frames
    step = (len(frames) - 1) / (n - 1) if n > 1 else 0
    return [frames[round(i * step)] for i in range(n)]


def read_document(path: str, root: str | os.PathLike = ".", max_chars: int = 20000) -> str:
    full = resolve(path, root)
    suffix = full.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = full.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as e:
            raise ImportError("Reading PDF files needs `pip install pypdf`.") from e
        reader = PdfReader(str(full))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    elif suffix == ".docx":
        try:
            import docx  # type: ignore
        except ImportError as e:
            raise ImportError("Reading .docx files needs `pip install python-docx`.") from e
        text = "\n".join(p.text for p in docx.Document(str(full)).paragraphs)
    elif suffix in IMAGE_SUFFIXES:
        return f"[{full.name} is an image; it is attached to the conversation as an image instead of text]"
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated, {len(text) - max_chars} more characters]"
    return text


def image_to_data_url(img, max_side: int = 1536) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
