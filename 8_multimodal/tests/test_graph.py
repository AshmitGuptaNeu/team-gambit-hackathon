"""mm_graph: the decisions / tasks / deadlines graph over the captions (no models)."""
import app as A
import mm_diagram
import mm_graph as G
from test_app import client  # noqa: F401  (the fixture)

CAPS = """[00:02] Priya: Okay so we decided to go with Postgres for the database.
[00:09] Priya: Sam, can you write the API endpoints by Friday?
[00:15] Sam: Sure. I'll also set up the CI pipeline next week.
[00:21] Dana: I'll draft the design doc for the observer pattern.
[00:30] Remember homework five is due on Friday at midnight.
[00:41] The midterm covers chapters one to four."""


def test_build_finds_items_owners_and_dates():
    g = G.build(CAPS)
    assert [(n.kind, n.owner, n.when) for n in g.items] == [
        ("decision", None, None), ("action", "Sam", "by Friday"), ("action", "Sam", "next week"),
        ("action", "Dana", None), ("deadline", None, "on Friday")]
    assert g.people == ["Priya", "Sam", "Dana"]
    assert ("action:1", "OWNED_BY", "Sam") in g.edges() and ("action:1", "SAID_BY", "Priya") in g.edges()
    assert G.build("").items == [] and G.build("no timestamps here").items == []


def test_answer_quotes_lines_and_falls_through_when_unsure():
    g = G.build(CAPS)
    a = G.answer("What is Sam doing?", g)
    assert a["found"] and a["answer"].count("**Task**") == 2 and "Dana" not in a["answer"]
    assert a["citations"][0]["quote"] == "[00:09] Priya: Sam, can you write the API endpoints by Friday?"
    assert "Postgres" in G.answer("what did we decide?", g)["answer"]
    assert "homework five" in G.answer("when is homework five due?", g)["answer"]
    assert "Dana" in G.answer("who owns the design doc?", g)["answer"]
    assert G.answer("who owns the frontend?", g) is None      # nothing on it: the captions are searched instead
    assert G.answer("what's the midterm about?", g) is None   # not a graph question


def test_mermaid_is_valid():
    code = G.to_mermaid(G.build(CAPS))
    assert "p1 -->|owns| a1" in code and "subgraph deadlines" in code
    assert G.to_mermaid(G.Graph()).startswith("flowchart LR")
    assert mm_diagram.check(code)["ok"]        # the official Mermaid parser (needs node + mermaid_check/)


async def test_routes(client):  # noqa: F811
    r = await (await client.post("/api/ask", json={"text": "who's doing what?", "captions": CAPS})).json()
    assert r["kind"] == "graph" and r["answer"].count("**Task**") == 3
    r = await (await client.post("/api/graph", json={"captions": CAPS})).json()
    assert r["kind"] == "diagram" and r["diagram_type"] == "knowledge graph" and r["people"] == ["Priya", "Sam", "Dana"]
    assert r["notes"].startswith("1 decision, 3 tasks, 1 deadline, 3 people")
    assert A.mm_graph is G
