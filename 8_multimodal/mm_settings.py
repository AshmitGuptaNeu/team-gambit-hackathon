"""Every setting the multimodal app uses, in one place. All models are open-weight and run on this machine.

Named mm_settings (not settings) so it never clashes with the other components' settings.py on sys.path.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# ---- The shared vision-language model (Ollama) ----
# gemma3:12b: Google Gemma 3, open weights, reads text + images, 128K context. ~8 GB at Q4 (fits a 16 GB+ laptop).
VISION_MODEL = os.environ.get("MM_VISION_MODEL", "gemma3:12b")
# Code (diagram code) goes to a code-specialised model: gemma3 wrote invalid Mermaid on 3 of 5 test requests.
# qwen2.5-coder:7b: Alibaba Qwen2.5-Coder, open weights (Apache 2.0), ~4.7 GB. Text only, so a whiteboard photo is
# first read by the vision model, then drawn by this one.
CODE_MODEL = os.environ.get("MM_CODE_MODEL", "qwen2.5-coder:7b")
IMAGE_MAX_SIDE = 1280        # longer photo sides are downscaled to this before they go to the model

# ---- Documents (text / PDF): answer only from the document, say so when it isn't there ----
DOC_CHUNK_WORDS = 120        # words per chunk (a paragraph or two; bge-small reads up to 512 tokens)
DOC_CHUNK_OVERLAP = 30       # words repeated between neighbouring chunks, so a sentence isn't cut in half
DOC_TOP_K = 5                # chunks given to the model
# Retrieval guard (before the model is asked at all): only for questions clearly off the document's topic.
# On the demo syllabus, bge-small cosines of answerable (0.54-0.81) and unanswerable (0.55-0.65) questions
# overlap, so a high cutoff refuses real answers; "capital of France" scores 0.38. BM25 is unreliable on short
# documents (a word in half the chunks gets IDF ~0), so it only adds evidence, never refuses on its own.
DOC_MIN_COSINE = 0.45        # bge-small cosine similarity of the best chunk
DOC_MIN_BM25 = 1.0           # ...and the best chunk's BM25 score is below this too
# Quote check (after the model answers): each supporting quote must really be in the cited chunk.
DOC_QUOTE_MATCH = 0.85       # fuzzy match ratio a quote needs against the chunk text
