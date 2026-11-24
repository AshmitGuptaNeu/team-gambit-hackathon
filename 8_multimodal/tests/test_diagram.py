"""Diagram code: clean / lint / the local Mermaid parser, and the repair loop (LLM faked)."""
import shutil

import pytest

import mm_diagram as M

needs_node = pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")


def test_clean_strips_fences():
    assert M.clean("```mermaid\nflowchart TD\n  A --> B\n```") == "flowchart TD\n  A --> B"
    assert M.clean("  flowchart TD\n  A --> B  ") == "flowchart TD\n  A --> B"


@pytest.mark.parametrize("kind", M.TYPES)
def test_every_example_passes_lint(kind):
    assert M.lint(M.EXAMPLES[kind], kind) is None


def test_lint_catches_what_the_parser_lets_through():
    assert "first line" in M.lint("graph TD\nA-->B\nB-->C", "flowchart")
    assert "single line" in M.lint("sequenceDiagram participant A participant B A->>B: hi", "sequenceDiagram")
    assert "arrows" in M.lint("mindmap\n  root((x))\n    A --> B", "mindmap")
    assert "gantt task" in M.lint("timeline\n    title t\n    Sam :sam1, 2026-11-20, 2026-11-27", "timeline")
    assert "dateFormat" in M.lint("gantt\n  title t\n  section A\n  x :a, 2026-01-01, 1d", "gantt")
    assert "note for" in M.lint("classDiagram\n  class A\n  note right of A\n  end note", "classDiagram")


@needs_node
@pytest.mark.parametrize("kind", M.TYPES)
def test_every_example_parses(kind):
    assert M.check(M.EXAMPLES[kind])["ok"], kind


@needs_node
def test_parser_rejects_broken_code_and_accepts_generics():
    assert not M.check("classDiagram\n  class A {\n    - x List<B\n")["ok"]
    assert M.check("classDiagram\n  class A {\n    - xs: List<B>\n  }")["ok"]   # needs jsdom's Element


@needs_node
def test_repair_loop_feeds_the_error_back(monkeypatch):
    replies = iter([{"diagram_type": "flowchart", "title": "T", "notes": "n"},
                    "flowchart TD\n  A[\"start\"] --> B{\"ok?\"\n  B --> C",                 # broken
                    "flowchart TD\n  A[\"start\"] --> B{\"ok?\"}\n  B -->|\"yes\"| C[\"done\"]"])
    prompts = []

    def fake_ask(prompt, **kw):
        prompts.append(prompt)
        return next(replies)
    monkeypatch.setattr(M.mm_llm, "ask", fake_ask)
    r = M.generate("draw it", today="2026-11-20")
    assert r["valid"] and r["tries"] == 2 and len(r["errors"]) == 1
    assert "It was rejected" in prompts[2] and "Your previous code" in prompts[2]


def test_a_cut_off_reply_is_retried_shorter_not_a_crash(monkeypatch):
    def replies():
        yield {"diagram_type": "flowchart", "title": "T", "notes": "n"}
        raise M.mm_llm.LLMError("reply cut off at max_tokens=2000: raise LLM['max_tokens'] in settings.py")
    calls, gen = [], replies()
    first = next(gen)

    def fake_ask(prompt, **kw):
        calls.append(prompt)
        if len(calls) == 1:
            return first
        if len(calls) == 2:
            next(gen)
        return "flowchart TD\n  A[\"start\"] --> B[\"next\"]\n  B --> C[\"done\"]"
    monkeypatch.setattr(M.mm_llm, "ask", fake_ask)
    monkeypatch.setattr(M, "check", lambda code: {"ok": True})
    r = M.generate("draw it", today="2026-11-20")
    assert r["valid"] and r["tries"] == 2 and "cut off" in r["errors"][0]
    assert "shorter diagram" in calls[2] and "Your previous code" not in calls[2]


def test_other_llm_errors_still_raise(monkeypatch):
    def fake_ask(prompt, **kw):
        if "schema" in kw:
            return {"diagram_type": "flowchart", "title": "T", "notes": "n"}
        raise M.mm_llm.LLMError("Ollama request failed: refused")
    monkeypatch.setattr(M.mm_llm, "ask", fake_ask)
    with pytest.raises(M.mm_llm.LLMError):
        M.generate("draw it", today="2026-11-20")
