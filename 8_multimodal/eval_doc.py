"""Document Q&A check on the demo syllabus: does it answer what is in the document (with the right page)
and say "not in the document" for what isn't?

    .venv/bin/python 8_multimodal/eval_doc.py [--show]
Questions: 8_multimodal/demo/syllabus_questions.json (written by Claude, reviewed by us; 10 in, 10 not in).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mm_document  # noqa: E402


def main():
    show = "--show" in sys.argv
    doc = mm_document.Document.from_bytes(open(os.path.join(HERE, "demo/syllabus_cs3500.pdf"), "rb").read(),
                                          "syllabus_cs3500.pdf")
    qs = json.load(open(os.path.join(HERE, "demo/syllabus_questions.json")))
    tp = fp = tn = fn = page_ok = flagged = direct_ok = 0
    times = []
    for item in qs:
        t = time.monotonic()
        a = doc.ask(item["q"])
        times.append(time.monotonic() - t)
        pages = {c["page"] for c in a.citations}
        if item["in_doc"]:
            if a.found:
                tp += 1
                page_ok += item["page"] in pages
                direct_ok += a.directness == "direct"
            else:
                fn += 1
        else:
            # Not in the document: right = "not found", or an answer clearly flagged as only partial.
            flagged += a.found and a.directness == "partial"
            fp += a.found and a.directness != "partial"
            tn += not a.found
        ok = (a.found and a.directness == "direct") if item["in_doc"] else (not a.found or a.directness == "partial")
        print(f"{'OK ' if ok else 'BAD'} {item['q']:<58} found={a.found!s:<5} {a.directness or a.reason} pages={sorted(pages)} "
              f"best_cos={max(r['cosine'] for r in a.retrieved):.2f} best_bm25={max(r['bm25'] for r in a.retrieved):.1f}"
              + (f"\n      -> {a.answer}" if show or not ok else ""))
    n_in = tp + fn
    print(f"\nin the document: answered directly {direct_ok}/{n_in}  (answered at all {tp}, right page cited {page_ok}/{tp})")
    print(f"not in the document: said so {tn}, flagged as only partial {flagged}, answered anyway {fp}  "
          f"(of {tn + flagged + fp})")
    print(f"seconds per question: median {sorted(times)[len(times) // 2]:.1f}, max {max(times):.1f}")


if __name__ == "__main__":
    main()
