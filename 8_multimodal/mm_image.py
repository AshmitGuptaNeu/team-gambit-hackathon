"""Image input: describe a photo (whiteboard, slide, diagram, a scene) and answer questions about it.

Written for a deaf or hard-of-hearing student: the description reads out everything written in the picture
(the professor's whiteboard is often the only thing they can follow), then the layout, then what it shows.
"""
import mm_llm

DESCRIBE_SYSTEM = (
    "You describe images for a deaf or hard-of-hearing university student who uses your description in place of "
    "listening to the lecturer. Be accurate and concrete. Never invent text you cannot read: write [unreadable] "
    "instead. If the image is a whiteboard, slide or code, transcribe ALL text, formulas and code exactly "
    "(formulas in plain-text math, code in a fenced block).")

DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["whiteboard", "slide", "code", "diagram", "document", "photo", "other"]},
        "summary": {"type": "string", "description": "one or two sentences: what the image shows"},
        "text_in_image": {"type": "string", "description": "every word, formula and line of code visible, in reading order; empty if none"},
        "details": {"type": "array", "items": {"type": "string"}, "description": "important visual details, top to bottom"},
    },
    "required": ["kind", "summary", "text_in_image", "details"],
}

ASK_SYSTEM = (
    "You answer questions about an image for a deaf or hard-of-hearing university student. Answer only from what "
    "is visible in the image. If the image does not show the answer, say exactly: \"The image doesn't show that.\" "
    "and say what it does show instead. Keep answers short and concrete; quote any text from the image exactly.")


def describe(image_bytes):
    """-> {"kind", "summary", "text_in_image", "details"}"""
    return mm_llm.ask("Describe this image.", system=DESCRIBE_SYSTEM, images=[image_bytes], schema=DESCRIBE_SCHEMA,
                      max_tokens=2000)


def ask(image_bytes, question, history=()):
    """A question about the image -> answer text. history: earlier (question, answer) pairs about the same image."""
    prompt = "".join(f"Earlier question: {q}\nYour answer: {a}\n\n" for q, a in history) + f"Question: {question}"
    return mm_llm.ask(prompt, system=ASK_SYSTEM, images=[image_bytes], max_tokens=1000)
