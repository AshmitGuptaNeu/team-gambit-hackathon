"""Architecture 2: pick the right past meetings first, then the right moments inside them.

    question (+ live context)
      -> understand: date range, people, "last time"                      (query.py)
      -> candidates: meetings that pass the date/people filters            (relaxed if nothing passes)
      -> stage 1: rank candidate meetings by RRF of
             best chunk   (where each meeting's best chunk lands in a hybrid search over all candidates)
             card vector / card BM25 (the meeting card)
             recency      (only for "last time" questions)
      -> stage 2: hybrid search (BM25 + vector) inside the top meetings
      -> final: RRF of stage 2 + global safety net + decision/action items (+ the same with live context)

The best-chunk signal leads stage 1 because, on AMI dev, meetings ranked by their best chunk put the right
meeting in the top 3 for 58% of questions vs 32-34% for the cards alone (see docs/QA.md).
"""
import re
import time
from datetime import datetime, timezone

from bm25 import BM25Cache
from query import parse_query
from retrieval_common import as_results, rrf
from text import content_words

PREFIX_RE = re.compile(r"^\[[\d:]+\]\s*[^:]+:\s*")


def meeting_order(chunk_ids):
    """Meetings in the order their first chunk appears in a ranked chunk list."""
    seen = {}
    for cid in chunk_ids:
        seen.setdefault(cid.split("#")[0], None)
    return list(seen)


def context_terms(context, question, n, lines):
    """Key words from the last `lines` live-transcript lines that the question does not already contain."""
    if not context:
        return []
    recent = (context.splitlines() if isinstance(context, str) else list(context))[-lines:]
    asked = set(content_words(question))
    counts = {}
    for i, line in enumerate(recent):
        body = PREFIX_RE.sub("", line.strip())           # drop "[mm:ss] Name: "
        for w in content_words(body):
            if w not in asked:
                counts[w] = counts.get(w, 0) + 1 + i / len(recent)   # later lines count a bit more
    return [w for w, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:n]]


class TwoStageRetriever:
    name = "two_stage"

    def __init__(self, store, embedder, cfg):
        self.store, self.embedder, self.cfg = store, embedder, cfg
        self.bm25 = BM25Cache(store)
        self.last_debug = {}
        self._people = (None, [])

    def people(self):
        version = self.store.version
        if self._people[0] != version:
            names = {}
            for m in self.store.meetings():
                for a in m.attendees:
                    names.setdefault(a, None)
            self._people = (version, list(names))
        return self._people[1]

    # ---- pieces
    def hybrid(self, text, qvec, meeting_ids):
        d = self.cfg["depth"]
        lexical = [c for c, _ in self.bm25.search("chunk", text, d, meeting_ids)]
        dense = [c for c, _ in self.store.search("chunk", qvec, d, meeting_ids)]
        return [c for c, _ in rrf([(1.0, lexical), (1.0, dense)], self.cfg["rrf_k"])]

    def candidates(self, parsed, meeting_ids, exclude):
        meetings = [m for m in self.store.meetings()
                    if (meeting_ids is None or m.meeting_id in meeting_ids) and m.meeting_id not in exclude]
        notes = []
        if parsed.date_from:
            dated = [m for m in meetings if parsed.date_from <= m.started_at.astimezone(parsed.date_from.tzinfo) < parsed.date_to]
            if dated:
                meetings = dated
            else:
                notes.append("no meeting in the date range: date filter dropped")
        if parsed.people:
            want = set(parsed.people)
            with_all = [m for m in meetings if want <= set(m.attendees)]
            with_any = [m for m in meetings if want & set(m.attendees)]
            if with_all:
                meetings = with_all
            elif with_any:
                meetings, _ = with_any, notes.append("nobody's meeting has all those people: kept meetings with any")
            else:
                notes.append("no meeting with those people: people filter dropped")
        return meetings, notes

    # ---- the retriever contract
    def retrieve(self, question, k=10, now=None, context=None, meeting_ids=None, exclude_meeting_ids=(),
                 with_neighbors=False):
        t0 = time.perf_counter()
        cfg, w = self.cfg, self.cfg["meeting_weights"]
        now = now or datetime.now(timezone.utc)
        parsed = parse_query(question, now, self.people())
        pool, notes = self.candidates(parsed, meeting_ids, set(exclude_meeting_ids))
        if not pool:
            self.last_debug = {"parsed": parsed, "notes": notes + ["no meetings indexed"]}
            return []
        pool_ids = [m.meeting_id for m in pool]
        text = parsed.text

        qvec = self.embedder.query(text)
        extra = context_terms(context, text, cfg["context_terms"], cfg["context_lines"])
        ctx_text = f"{text} {' '.join(extra)}" if extra else None
        ctx_vec = self.embedder.query(ctx_text) if ctx_text else None

        # global search over every candidate meeting
        global_ids = self.hybrid(text, qvec, pool_ids)
        global_ctx = self.hybrid(ctx_text, ctx_vec, pool_ids) if ctx_text else []

        # stage 1: which meetings
        lists = [(w["best_chunk"], meeting_order(global_ids))]
        if global_ctx:
            lists.append((w["best_chunk"] * cfg["context_weight"], meeting_order(global_ctx)))
        if w["card_vector"]:
            lists.append((w["card_vector"], [m for m, _ in self.store.search("card", qvec, len(pool_ids), pool_ids)]))
        if w["card_bm25"]:
            lists.append((w["card_bm25"], [m for m, _ in self.bm25.search("card", text, len(pool_ids), pool_ids)]))
        if parsed.latest:
            lists.append((w.get("recency", 1.0), [m.meeting_id for m in sorted(pool, key=lambda m: m.started_at, reverse=True)]))
        ranked_meetings = [m for m, _ in rrf(lists, cfg["rrf_k"])]
        top = ranked_meetings[:cfg["top_meetings"]]

        # stage 2: which moments, inside the chosen meetings
        stage2 = self.hybrid(text, qvec, top)
        item_chunks = (meeting_order_items(self.store.search("item", qvec, cfg["depth"], top), self.store, top)
                       if cfg["item_weight"] else [])
        boost = cfg["stage2_weight_latest"] if parsed.latest else cfg["stage2_weight"]
        final = [(boost, stage2), (cfg["global_weight"], global_ids), (cfg["item_weight"], item_chunks)]
        if ctx_text:
            final.append((cfg["context_weight"], self.hybrid(ctx_text, ctx_vec, top)))
        scored = rrf(final, cfg["rrf_k"])
        results = as_results(self.store, scored, k)
        if with_neighbors:
            for r in results:
                r["before"], r["after"] = self.neighbor(r["chunk_id"], -1), self.neighbor(r["chunk_id"], 1)
        self.last_debug = {"parsed": parsed, "notes": notes, "candidates": pool_ids, "meetings": ranked_meetings[:5],
                           "context_terms": extra, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        return results

    def neighbor(self, chunk_id, step):
        meeting, ord_ = chunk_id.rsplit("#", 1)
        try:
            return self.store.chunk(f"{meeting}#{int(ord_) + step:04d}").text
        except KeyError:
            return None


def meeting_order_items(item_hits, store, meetings):
    """Item search hits -> the chunks they were said in, best first, each chunk once."""
    by_id = {i.item_id: i for i in store.items(meetings)}
    seen = {}
    for item_id, _ in item_hits:
        if item_id in by_id:
            seen.setdefault(by_id[item_id].chunk_id, None)
    return list(seen)
