"""The AMI benchmark's gold answers are built correctly. Needs the AMI data (skipped otherwise)."""
import os

import pytest

pytestmark = pytest.mark.ami

ami_settings = pytest.importorskip("ami_settings")
if not os.path.isdir(ami_settings.AMI_DIR):
    pytest.skip("AMI data not downloaded (run 5_data/ami/fetch_data.py)", allow_module_level=True)

import ami_qa  # noqa: E402
import fetch_data  # noqa: E402
import qa_settings as S  # noqa: E402
from chunking import chunk_meeting  # noqa: E402

ZIP = os.path.join(ami_settings.RAW_DIR, fetch_data.load_manifest()["file"])
M = "ES2003a"


@pytest.fixture(scope="module")
def meeting():
    return ami_qa.load_meeting(M, ami_qa.series_dates([M])[M])


def test_meeting_reads_with_ids(meeting):
    assert len(meeting.utterances) > 100
    assert all(u.uid and u.uid.startswith(M) for u in meeting.utterances)
    assert set(meeting.attendees) <= {"Project Manager", "Marketing Expert", "Industrial Designer",
                                      "User Interface Designer"}


def test_every_query_links_to_real_utterances(meeting):
    queries = ami_qa.load_queries([M], ZIP)
    uids = {u.uid for u in meeting.utterances}
    assert queries and {q.section for q in queries} <= set(ami_qa.SECTIONS)
    linked = [q for q in queries if set(q.gold_uids) & uids]
    assert len(linked) / len(queries) > 0.95


def test_gold_chunks_contain_the_linked_lines(meeting):
    chunks = chunk_meeting(meeting, **S.CHUNKING)
    for q in ami_qa.load_queries([M], ZIP):
        for cid in ami_qa.gold_chunks(q, {M: chunks}):
            c = next(c for c in chunks if c.chunk_id == cid)
            assert set(q.gold_uids) & set(c.utterance_uids)


def test_series_dates_are_a_to_d_consecutive():
    d = ami_qa.series_dates(["ES2003a", "ES2003b", "IS1008a"])
    assert (d["ES2003b"] - d["ES2003a"]).days == 1 and d["IS1008a"] > d["ES2003b"]


def test_ids_in_href():
    assert ami_qa.ids_in("x.xml#id(a.b.1)") == ["a.b.1"]
    assert ami_qa.ids_in("x.xml#id(a)..id(b)") == ["a", "b"]
