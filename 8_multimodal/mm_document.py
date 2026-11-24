"""Text input: answer questions about a document (syllabus, assignment, lecture notes; PDF, .txt or .md), and say
so when the answer is NOT in it.

    document -> pages -> overlapping ~120-word chunks (each keeps its page number)
    question -> hybrid retrieval (BM25 + bge-small vectors, fused with RRF: the same parts as 4_qa/)
             -> guard 1, retrieval: best chunk too far from the question on BOTH signals -> "not in the document"
             -> the local LLM answers ONLY from the top chunks, as JSON: found / answer / supporting quotes
             -> guard 2, quote check: every quote must really be in the chunk it cites (fuzzy match);
                an answer with no quote that checks out is turned into "not in the document"
             -> answer + page citations

Two guards because each misses cases: retrieval alone can't tell "related words, but no answer" (the syllabus
mentions exams but not the exam room), and a model alone can confidently answer from general knowledge.
"""
import difflib
import io
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for folder in ["4_qa", "4_qa/knowledge", "4_qa/retrieval"]:
    sys.path.insert(0, os.path.join(REPO, folder))
import mm_llm  # noqa: E402
import mm_settings as S  # noqa: E402
from retrieval_common import rrf  # noqa: E402
from text import tokenize  # noqa: E402

NOT_FOUND = "That isn't in this document."

ANSWER_SYSTEM = (
    "You answer a deaf or hard-of-hearing student's question about ONE document, using ONLY the numbered excerpts "
    "given. Never use outside knowledge, never guess.\n"
    "Step 1, quotes: copy WORD FOR WORD the sentences from the excerpts that answer the question, each with its "
    "excerpt number. A sentence that answers it indirectly counts (\"There is no final exam\" answers \"Is there a "
    "final exam?\"; \"Late days cannot be used for the final project\" answers \"Can I use late days on the "
    "project?\"). If no sentence answers it, give no quotes.\n"
    "Step 2, answer: a short, direct answer built only from those quotes.\n"
    "Step 3, found: true if the quotes answer the question, false if there are none or they only mention related "
    "words without giving the answer.")

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {            # in this order: the model writes the evidence before it decides
        "quotes": {"type": "array", "items": {
            "type": "object",
            "properties": {"excerpt": {"type": "integer"}, "quote": {"type": "string"}},
            "required": ["excerpt", "quote"]}},
        "answer": {"type": "string"},
        "found": {"type": "boolean"},
    },
    "required": ["quotes", "answer", "found"],
}


@dataclass
class Chunk:
    id: int
    page: int
    text: str


@dataclass
class Answer:
    question: str
    found: bool
    answer: str
    citations: list = field(default_factory=list)   # [{"page", "quote"}]
    reason: str = ""                                 # why "not found": retrieval / model / quote_check / judge
    retrieved: list = field(default_factory=list)    # [{"chunk", "page", "score", "cosine", "bm25"}] for the UI/debug
    directness: str = ""                             # found answers: "direct" or "partial" (judge)

    def to_dict(self):
        return dict(self.__dict__)


# ---------------- loading ----------------

def read_pages(data, filename):
    """bytes -> [(page_number, text)]. PDF: one entry per page. Text/Markdown: form-feeds or '## Page N' split pages,
    else the whole file is page 1."""
    if filename.lower().endswith(".pdf") or data[:5] == b"%PDF-":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [(i, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages, 1)]
    text = data.decode("utf-8", errors="replace")
    parts = text.split("\f") if "\f" in text else [text]
    return [(i, p.strip()) for i, p in enumerate(parts, 1)]


def chunk_pages(pages, words=None, overlap=None):
    """Overlapping word windows inside each page (a chunk never spans two pages, so its citation is exact)."""
    words, overlap = words or S.DOC_CHUNK_WORDS, overlap or S.DOC_CHUNK_OVERLAP
    chunks = []
    for page, text in pages:
        toks = text.split()
        start = 0
        while start < len(toks):
            piece = " ".join(toks[start:start + words])
            chunks.append(Chunk(len(chunks), page, piece))
            if start + words >= len(toks):
                break
            start += words - overlap
    return chunks


# ---------------- the document index ----------------

_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        import qa_settings
        from embed import Embedder
        _embedder = Embedder(qa_settings.EMBEDDING)
    return _embedder


class Document:
    def __init__(self, name, pages, embed=None):
        from rank_bm25 import BM25Okapi
        self.name, self.pages = name, pages
        self.chunks = chunk_pages(pages)
        if not self.chunks:
            raise ValueError(f"{name}: no text found (a scanned PDF needs OCR: send it as an image instead)")
        self.embed = embed or embedder()
        self.vectors = self.embed.passages([c.text for c in self.chunks])
        self.bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in self.chunks])

    @classmethod
    def from_bytes(cls, data, filename, embed=None):
        return cls(filename, read_pages(data, filename), embed=embed)

    def retrieve(self, question, k=None):
        """-> [{"chunk", "page", "text", "score", "cosine", "bm25"}] best first (RRF of BM25 and vector ranks)."""
        k = k or S.DOC_TOP_K
        bm = self.bm25.get_scores(tokenize(question) or ["_"])
        cos = self.vectors @ self.embed.query(question)
        fused = rrf([(1.0, list(np.argsort(-bm, kind="stable"))), (1.0, list(np.argsort(-cos, kind="stable")))])
        return [{"chunk": int(i), "page": self.chunks[i].page, "text": self.chunks[i].text, "score": round(s, 5),
                 "cosine": round(float(cos[i]), 3), "bm25": round(float(bm[i]), 2)} for i, s in fused[:k]]

    def ask(self, question, k=None):
        hits = self.retrieve(question, k)
        public = [{key: h[key] for key in ("chunk", "page", "score", "cosine", "bm25")} for h in hits]
        best_cos, best_bm = max(h["cosine"] for h in hits), max(h["bm25"] for h in hits)
        # Guard 1: nothing in the document is close to the question, by meaning OR by words.
        if best_cos < S.DOC_MIN_COSINE and best_bm < S.DOC_MIN_BM25:
            return Answer(question, False, NOT_FOUND, reason="retrieval", retrieved=public)
        excerpts = "\n\n".join(f"[{n}] (page {h['page']})\n{h['text']}" for n, h in enumerate(hits, 1))
        out = mm_llm.ask(f"Document: {self.name}\n\nExcerpts:\n{excerpts}\n\nQuestion: {question}",
                         system=ANSWER_SYSTEM, schema=ANSWER_SCHEMA, max_tokens=800)
        if not out.get("found"):
            return Answer(question, False, NOT_FOUND, reason="model", retrieved=public)
        # Guard 2: keep only quotes that really are in the excerpt they cite.
        citations = []
        for q in out.get("quotes") or []:
            n = q.get("excerpt")
            if isinstance(n, int) and 1 <= n <= len(hits) and quote_in(q.get("quote", ""), hits[n - 1]["text"]):
                citations.append({"page": hits[n - 1]["page"], "quote": q["quote"].strip()})
        if not citations:
            return Answer(question, False, NOT_FOUND, reason="quote_check", retrieved=public)
        # Guard 3: do the checked quotes answer THIS question, or only something close to it?
        verdict = judge(question, [c["quote"] for c in citations])
        answer = out.get("answer", "").strip()
        if verdict == "no":
            return Answer(question, False, NOT_FOUND, reason="judge", retrieved=public)
        if verdict == "partial":
            closest = "; ".join(f'"{c["quote"]}" (page {c["page"]})' for c in citations)
            answer = f"The document doesn't say exactly that. The closest it gets: {closest}"
        return Answer(question, True, answer, citations, retrieved=public, directness=verdict)


JUDGE_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["direct", "partial", "no"]}},
                "required": ["verdict"]}


def judge(question, quotes):
    """Do these quotes answer the question? direct = yes, as asked; partial = only related or needs a guess
    (asked for a room, the text says "in class"; asked about one homework, the text gives all homework);
    no = they don't answer it."""
    prompt = ("Question: " + question + "\nQuotes from the document:\n" + "\n".join(f"- {q}" for q in quotes) +
              "\n\nDo the quotes give the specific thing the question asks for?\n"
              "direct: yes. Different words, a general rule that covers the case, or a yes/no that follows from "
              "the text all count (\"AI coding assistants are allowed\" answers a question about ChatGPT; "
              "\"register with the Disability Resource Center\" answers how to get captioning; \"one page of notes "
              "is allowed\" answers what to bring).\n"
              "partial: the quotes are about the topic but the specific detail asked for is missing (asked for a "
              "room number, the text only says \"in class\"; asked about one homework's points, the text only "
              "gives all homework together).\n"
              "no: the quotes don't address the question.")
    try:
        return mm_llm.ask(prompt, schema=JUDGE_SCHEMA, max_tokens=50)["verdict"]
    except Exception:
        return "direct"   # the judge is a second opinion: if it fails, keep the checked answer


def _norm(s):
    """Lowercase, punctuation to spaces (apostrophes kept, so "can't" stays one negation word)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", s.lower().replace("\u2019", "'"))).strip()


NEGATIONS = {"not", "no", "never", "cannot", "can't", "cant", "don't", "dont", "doesn't", "doesnt", "won't", "isn't",
             "aren't", "without", "nor", "none", "n't", "only"}


def _negations(words):
    """The negations in a word list, with every spelling of "not" (cannot, can't, don't, ...) counted as "not"."""
    out = []
    for w in words:
        if w in ("not", "cannot", "cant", "dont", "doesnt") or w.endswith("n't"):
            out.append("not")
        elif w in NEGATIONS:
            out.append(w)
    return sorted(out)


def quote_in(quote, text, threshold=None):
    """Is `quote` (near-)verbatim in `text`? Exact after normalising, else the best fuzzy window >= threshold.
    A fuzzy match must keep the same negation words: "may be used" vs "cannot be used" is ~90% the same
    characters and the opposite meaning."""
    threshold = threshold or S.DOC_QUOTE_MATCH
    q, t = _norm(quote), _norm(text)
    if len(q) < 8:
        return False      # "yes", "the": too short to prove anything
    if q in t:
        return True
    qw, tw = q.split(), t.split()
    n = len(qw)
    for i in range(max(1, len(tw) - n + 1)):
        window = tw[i:i + n]
        if (difflib.SequenceMatcher(None, q, " ".join(window)).ratio() >= threshold
                and _negations(qw) == _negations(window)):
            return True
    return False
