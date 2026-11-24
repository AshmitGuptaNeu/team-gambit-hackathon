"""Diagram-code check: typed requests, requests about a meeting transcript, and a whiteboard photo.
Valid = passes our lint and the official Mermaid parser. Read the printed code to judge whether it is RIGHT.

    .venv/bin/python 8_multimodal/eval_diagram.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mm_diagram as M  # noqa: E402

TRANSCRIPT = """[00:05] Priya: Okay, for the CS 3500 final project, let's split it up.
[00:12] Sam: I'll do the model and the Observer event system, done by November 27.
[00:20] Alex: Then I'll build the caption view on top, that takes about a week after Sam is done.
[00:31] Priya: I'll write the JUnit tests in parallel starting November 23, five days.
[00:40] Priya: And we all rehearse the presentation on December 1, presentation is December 2."""

CASES = [
    ("typed flowchart", dict(request="Make a flowchart of the late-day policy: 3 late days, each adds 24 hours, not "
                             "for the final project, after that minus 10 percent per day, not accepted after 3 days")),
    ("typed sequence", dict(request="Sequence diagram: student uploads a lecture recording, the app transcribes it with "
                            "Whisper, indexes it, then the student asks a question and gets an answer with a timestamp")),
    ("meeting -> plan", dict(request="Turn our project plan into a timeline with owners", transcript=TRANSCRIPT)),
    ("meeting -> mindmap", dict(request="Who does what? Make a mind map of the tasks by person", transcript=TRANSCRIPT)),
    ("whiteboard -> UML", dict(request="Redraw this whiteboard as a clean UML class diagram",
                               image=open(os.path.join(HERE, "demo/whiteboard_observer.jpg"), "rb").read())),
]


def main():
    valid = 0
    for name, kw in CASES:
        t = time.time()
        r = M.generate(today="2026-11-20", **kw)
        valid += r["valid"]
        print(f"=== {name}: valid={r['valid']} tries={r['tries']} type={r.get('diagram_type')} {time.time() - t:.1f}s")
        print("title:", r["title"]); print(r["code"])
        for e in r["errors"]:
            print("  rejected:", e.splitlines()[0][:150])
    print(f"\nvalid: {valid}/{len(CASES)}")


if __name__ == "__main__":
    main()
