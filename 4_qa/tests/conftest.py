"""Shared test setup for Feature 3.

    .venv/bin/python -m pytest 4_qa/tests                     # everything that can run here
    .venv/bin/python -m pytest 4_qa/tests -m "not model"      # fast: no embedding model, no AMI data

Markers: model (loads bge-small, ~130 MB download once), pg (needs LIVEMINUTES_TEST_DB_URL, a Postgres
with pgvector; tables are created and DROPPED, so never point it at a real database), ami (needs the AMI data).
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for folder in ["4_qa", "4_qa/knowledge", "4_qa/retrieval", "4_qa/retrieval/bm25", "4_qa/retrieval/vector",
               "4_qa/retrieval/two_stage", "5_data/ami", "6_evaluation/qa"]:
    sys.path.insert(0, os.path.join(REPO, folder))

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
TEST_DB_URL = os.environ.get("LIVEMINUTES_TEST_DB_URL", "")


def pytest_configure(config):
    for name, text in [("model", "loads the embedding model"), ("pg", "needs LIVEMINUTES_TEST_DB_URL"),
                       ("ami", "needs the AMI data in 5_data/raw/ami")]:
        config.addinivalue_line("markers", f"{name}: {text}")


@pytest.fixture(scope="session")
def scenarios():
    from transcript import MeetingInput, parse_transcript
    with open(os.path.join(FIXTURES, "scenarios.json")) as f:
        data = json.load(f)
    meetings = [MeetingInput(meeting_id=m["meeting_id"], title=m["title"],
                             started_at=datetime.fromisoformat(m["started_at"]),
                             utterances=parse_transcript(m["transcript"]), source={"kind": "fixture"})
                for m in data["meetings"]]
    return {"now": datetime.fromisoformat(data["now"]), "meetings": meetings, "questions": data["questions"]}


@pytest.fixture(scope="session")
def embedder():
    import qa_settings
    from embed import Embedder
    return Embedder(qa_settings.EMBEDDING)


class FakeEmbedder:
    """Deterministic bag-of-words vectors: no model, but similar words -> similar vectors. For store tests."""
    dim = 64
    version = "fake"

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in text.lower().split():
            v[hash(w.strip(".,?!")) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v + 1 / np.sqrt(self.dim)

    def passages(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def query(self, text):
        return self._vec(text)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture(scope="session")
def pg_url():
    if not TEST_DB_URL:
        pytest.skip("set LIVEMINUTES_TEST_DB_URL to a throwaway Postgres with pgvector")
    return TEST_DB_URL
