"""Offline tests for document Q&A: a fake embedder and a fake LLM, so no model downloads and no Ollama.

    .venv/bin/python -m pytest 8_multimodal/tests
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import mm_document as D  # noqa: E402

PAGES = [(1, "Office hours are Tuesdays 2 to 3 PM in room H 210. " + "filler " * 10),
         (2, "Each student has 3 late days. Late days cannot be used for the final project."),
         (3, "The midterm is on October 21 in class.")]


class FakeEmbed:
    """Bag-of-words vectors: enough to rank, and a question with no shared words scores 0."""
    vocab = {}

    def _vec(self, text):
        v = np.zeros(64, dtype=np.float32)
        for w in D.tokenize(text):
            v[self.vocab.setdefault(w, len(self.vocab) % 64)] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def query(self, text):
        return self._vec(text)


@pytest.fixture
def doc(monkeypatch):
    monkeypatch.setattr(D.S, "DOC_MIN_COSINE", 0.05)   # the fake embedder's cosines are far lower than bge's
    return D.Document("syllabus.pdf", PAGES, embed=FakeEmbed())


def fake_llm(answer=None, verdict="direct"):
    calls = []

    def ask(prompt, system=None, schema=None, **kw):
        calls.append(prompt)
        if schema is D.JUDGE_SCHEMA:
            return {"verdict": verdict}
        return answer
    return ask, calls


def test_chunks_keep_their_page_and_never_span_pages():
    chunks = D.chunk_pages([(1, "a " * 250), (2, "b c")], words=120, overlap=30)
    assert [c.page for c in chunks] == [1, 1, 1, 2]
    assert chunks[1].text.split()[0] == "a" and len(chunks[0].text.split()) == 120


def test_quote_check():
    text = "Late days cannot be used for the final project."
    assert D.quote_in("late days cannot be used for the final project", text)
    assert D.quote_in("Late days can not be used for the final project.", text)   # small differences are fine
    assert not D.quote_in("Late days can be used for anything", text)
    assert not D.quote_in("Late", text)                                            # too short to prove anything
    assert not D.quote_in("Late days can be used for the final project.", text)    # negation flipped
    assert not D.quote_in("Late days may be used for the final project", text)


def test_off_topic_question_is_refused_without_asking_the_model(doc, monkeypatch):
    ask, calls = fake_llm()
    monkeypatch.setattr(D.mm_llm, "ask", ask)
    a = doc.ask("capital of France?")
    assert not a.found and a.answer == D.NOT_FOUND and a.reason == "retrieval" and calls == []


def test_answer_with_a_real_quote_cites_its_page(doc, monkeypatch):
    ask, _ = fake_llm({"quotes": [{"excerpt": 1, "quote": "Late days cannot be used for the final project"}],
                       "answer": "No.", "found": True})
    monkeypatch.setattr(D.mm_llm, "ask", ask)
    a = doc.ask("Can I use late days on the final project?")
    assert a.found and a.answer == "No." and a.citations[0]["page"] == 2 and a.directness == "direct"


def test_invented_quote_turns_into_not_found(doc, monkeypatch):
    ask, _ = fake_llm({"quotes": [{"excerpt": 1, "quote": "Late days may be used for the final project"}],
                       "answer": "Yes.", "found": True})
    monkeypatch.setattr(D.mm_llm, "ask", ask)
    a = doc.ask("Can I use late days on the final project?")
    assert not a.found and a.reason == "quote_check"


def test_partial_answer_is_flagged_with_the_closest_quote(doc, monkeypatch):
    ask, _ = fake_llm({"quotes": [{"excerpt": 1, "quote": "The midterm is on October 21 in class."}],
                       "answer": "In class.", "found": True}, verdict="partial")
    monkeypatch.setattr(D.mm_llm, "ask", ask)
    a = doc.ask("What room is the midterm in?")
    assert a.found and a.directness == "partial" and "doesn't say exactly" in a.answer and "page 3" in a.answer


def test_model_says_not_found(doc, monkeypatch):
    ask, _ = fake_llm({"quotes": [], "answer": "", "found": False})
    monkeypatch.setattr(D.mm_llm, "ask", ask)
    assert doc.ask("What textbook do we need for the midterm?").reason == "model"


def test_text_files_split_pages_on_form_feeds():
    assert D.read_pages(b"one\fsecond page", "notes.txt") == [(1, "one"), (2, "second page")]
