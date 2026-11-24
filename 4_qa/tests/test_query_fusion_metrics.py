"""Question understanding (dates, people, recency), RRF fusion, retrieval metrics. No model needed."""
from datetime import datetime, timezone

import pytest

from qa_metrics import percentile, score_query
from query import parse_query
from retrieval_common import rrf
from two_stage import context_terms, meeting_order

NOW = datetime(2026, 11, 18, 15, 0, tzinfo=timezone.utc)   # a Wednesday
PEOPLE = ["Priya Shah", "Marcus Lee", "Dana Kim", "Project Manager", "Marketing Expert"]


def d(y, m, day):
    return datetime(y, m, day, tzinfo=timezone.utc)


@pytest.mark.parametrize("q,start,end", [
    ("what did we say today", d(2026, 11, 18), d(2026, 11, 19)),
    ("what happened yesterday?", d(2026, 11, 17), d(2026, 11, 18)),
    ("anything this week", d(2026, 11, 16), d(2026, 11, 19)),
    ("what did we decide last week", d(2026, 11, 9), d(2026, 11, 16)),
    ("in the last 3 days", d(2026, 11, 15), d(2026, 11, 19)),
    ("over the past two weeks", d(2026, 11, 4), d(2026, 11, 19)),
    ("two weeks ago", d(2026, 11, 2), d(2026, 11, 9)),
    ("5 days ago", d(2026, 11, 13), d(2026, 11, 14)),
    ("this month", d(2026, 11, 1), d(2026, 11, 19)),
    ("last month", d(2026, 10, 1), d(2026, 11, 1)),
    ("on Monday", d(2026, 11, 16), d(2026, 11, 17)),
    ("last wednesday", d(2026, 11, 11), d(2026, 11, 12)),     # today is Wednesday: a week ago, not today
    ("in October", d(2026, 10, 1), d(2026, 11, 1)),
    ("back in December", d(2025, 12, 1), d(2026, 1, 1)),      # December is in the future: last year's
])
def test_date_phrases(q, start, end):
    p = parse_query(q, NOW)
    assert (p.date_from, p.date_to) == (start, end)


def test_dates_use_the_askers_timezone():
    from zoneinfo import ZoneInfo
    now = datetime(2026, 11, 18, 1, 0, tzinfo=ZoneInfo("America/New_York"))   # 06:00 UTC
    p = parse_query("yesterday", now)
    assert p.date_from == datetime(2026, 11, 17, tzinfo=ZoneInfo("America/New_York"))


def test_no_date_and_text_kept():
    p = parse_query("What price did we agree on?", NOW, PEOPLE)
    assert p.date_from is None and not p.latest and p.people == [] and p.text == "What price did we agree on?"


def test_date_phrase_removed_from_search_text():
    assert parse_query("What did we decide last week about pricing?", NOW).text == "What did we decide about pricing?"


@pytest.mark.parametrize("q,latest", [
    ("Last time we discussed the launch, what date?", True),
    ("in the previous meeting", True),
    ("what was decided in the last meeting with Sam", True),
    ("the most recent sync", True),
    ("the last item on the agenda", False),
])
def test_recency(q, latest):
    assert parse_query(q, NOW).latest is latest


@pytest.mark.parametrize("q,people", [
    ("What did Dana promise?", ["Dana Kim"]),
    ("what did priya and marcus agree", ["Priya Shah", "Marcus Lee"]),
    ("Priya's update", ["Priya Shah"]),
    ("Dana Kim said", ["Dana Kim"]),
    ("the project manager said", ["Project Manager"]),        # role names: only in full
    ("the marketing budget", []),                             # ... never by their first word
    ("a danish customer", []),                                # whole words only
])
def test_people(q, people):
    assert parse_query(q, NOW, PEOPLE).people == people


def test_naive_now_is_rejected():
    with pytest.raises(ValueError):
        parse_query("x", datetime(2026, 1, 1))


# ---- fusion
def test_rrf_weights_and_ties():
    out = rrf([(1.0, ["a", "b"]), (1.0, ["b", "a"])], k=60)
    assert [x for x, _ in out] == ["a", "b"] and out[0][1] == pytest.approx(out[1][1])   # tie -> by id
    out = rrf([(1.0, ["a", "b"]), (2.0, ["b"])], k=60)
    assert out[0][0] == "b"
    assert rrf([(0.0, ["a"])], k=60) == [("a", 0.0)]


def test_meeting_order_and_context_terms():
    assert meeting_order(["m2#0003", "m1#0000", "m2#0001"]) == ["m2", "m1"]
    terms = context_terms(["[00:10] Priya: The Stripe webhook bug again", "[00:20] Leo: webhook handler"],
                          "who fixes that", 5, 12)
    assert terms[0] == "webhook" and "stripe" in terms and "priya" not in terms
    assert context_terms(None, "q", 3, 12) == []


# ---- metrics
def test_score_query_values():
    s = score_query(["x", "g1", "y", "g2"], ["g1", "g2", "g3"], ["m2", "m1"], "m1")
    assert s["hit@1"] == 0 and s["hit@3"] == 1 and s["recall@5"] == pytest.approx(2 / 3)
    assert s["mrr"] == 0.5 and s["meeting@1"] == 0
    ideal = 1 + 1 / 1.5849625007211562 + 0.5
    assert s["ndcg@10"] == pytest.approx((1 / 1.5849625007211562 + 1 / 2.321928094887362) / ideal)
    perfect = score_query(["g"], ["g"], ["m"], "m")
    assert perfect["ndcg@10"] == 1 and perfect["mrr"] == 1 and perfect["meeting@1"] == 1
    assert score_query([], ["g"], [], "m")["mrr"] == 0


def test_score_query_needs_gold():
    with pytest.raises(ValueError):
        score_query(["a"], [], ["m"], "m")


def test_percentile():
    assert percentile([5, 1, 3, 2, 4], 50) == 3 and percentile([1, 2], 95) == 2 and percentile([], 50) is None
