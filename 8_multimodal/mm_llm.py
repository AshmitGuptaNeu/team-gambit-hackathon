"""The one local LLM every input type shares: an open-weight vision-language model served by Ollama.

Reuses OllamaProvider from 2_summarization/llm_providers.py (the live notes use the same one), and adds images.
Images go in as raw bytes; they are downscaled here so a 12-megapixel phone photo doesn't cost minutes.
"""
import base64
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "2_summarization"))
from llm_providers import LLMError, OllamaProvider  # noqa: E402

import mm_settings as S  # noqa: E402

_providers = {}


def provider(model=None):
    """One OllamaProvider per model, created on first use (it checks that Ollama runs and has the model)."""
    model = model or S.VISION_MODEL
    if model not in _providers:
        _providers[model] = OllamaProvider(model)
    return _providers[model]


def prepare_image(data, max_side=None):
    """Image bytes (any format Pillow reads) -> base64 JPEG, longest side at most max_side pixels."""
    from PIL import Image, ImageOps
    max_side = max_side or S.IMAGE_MAX_SIDE
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))   # phone photos: honour the rotation flag
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return base64.b64encode(out.getvalue()).decode()


def ask(prompt, system=None, images=(), schema=None, max_tokens=2000, model=None, temperature=0):
    """One call. images: list of raw image bytes. schema: a JSON schema -> returns a dict; else returns text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    user = {"role": "user", "content": prompt}
    if images:
        user["images"] = [prepare_image(i) for i in images]
    messages.append(user)
    text = provider(model).chat(messages, schema=schema, max_tokens=max_tokens, temperature=temperature)
    if schema is None:
        return text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"reply was not valid JSON: {e}: {text[:200]}") from e
