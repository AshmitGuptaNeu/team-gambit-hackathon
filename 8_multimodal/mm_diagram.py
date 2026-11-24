"""Code input: write diagram code (Mermaid) from a prompt, so a meeting's plan becomes a flowchart, timeline,
UML diagram or mind map the student can see, edit and reuse.

    request ("flowchart of the signup process")      ─┐
    + optional context: the meeting transcript         ├─> local LLM writes Mermaid code (JSON: type, title, code)
    + optional image: a whiteboard photo to redraw    ─┘        │
                                                                ▼
                                    mermaid_check/check.mjs: the official Mermaid parser, run locally (Node)
                                                                │ error? the parser's message goes back to the
                                                                │ model, which fixes its code (up to MAX_TRIES)
                                                                ▼
                                    valid code -> rendered in the page (mermaid.js, served locally)

Why diagrams: in a meeting or lecture people rarely need code that runs, but they do need to plan. A deaf student
who can't follow a fast spoken walkthrough of "first this, then if that..." gets the same plan as a picture.
"""
import datetime as dt
import json
import os
import re
import subprocess

import mm_llm
import mm_settings as S

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(HERE, "mermaid_check", "check.mjs")
MAX_TRIES = 3
WRITE_TOKENS = 2000      # a ~20-node diagram is 300-800 tokens; more than this means the model is looping
SHORT_NODES = 15         # the node limit asked for after a cut-off

TYPES = ["flowchart", "sequenceDiagram", "classDiagram", "stateDiagram-v2", "erDiagram", "gantt", "mindmap",
         "timeline"]

# One example per type, on a topic far from any course or meeting (a coffee shop) so the model copies the SYNTAX,
# not the content: with course-themed examples it pasted "Midterm" into a plan where nobody mentioned one.
EXAMPLES = {
    "flowchart": """flowchart TD
    A["Customer orders"] --> B{"Is the drink in stock?"}
    B -->|"yes"| C["Make the drink"]
    B -->|"no"| D["Offer another drink"]
    D --> E{"Accepted?"}
    E -->|"yes"| C
    E -->|"no"| F["Refund"]
    C --> G["Serve"]""",
    "sequenceDiagram": """sequenceDiagram
    participant C as Customer
    participant K as Kiosk
    participant B as Barista
    C->>K: Orders a latte
    K->>B: Sends the order
    B-->>C: Hands over the latte""",
    "classDiagram": """classDiagram
    class Drink {
        <<interface>>
        +price() double
    }
    class Latte {
        -size: String
        +price() double
    }
    class Order {
        -drinks: List~Drink~
        +total() double
    }
    Drink <|.. Latte
    Order o-- "1..*" Drink
    note for Order "total() adds up every drink's price"
""".strip(),
    "stateDiagram-v2": """stateDiagram-v2
    [*] --> Ordered
    Ordered --> Brewing: barista starts
    Brewing --> Ready: done
    Ready --> [*]""",
    "erDiagram": """erDiagram
    CUSTOMER ||--o{ ORDER : places
    ORDER {
        int id
        date placed_on
    }""",
    "gantt": """gantt
    title Opening the shop
    dateFormat YYYY-MM-DD
    section Maria
    Buy the espresso machine :maria1, 2030-03-01, 2030-03-05
    section Joe
    Train the staff :joe1, after maria1, 7d
    section Everyone
    Opening day :milestone, open1, 2030-03-20, 0d""",
    "mindmap": """mindmap
  root((Opening the shop))
    Maria
      Espresso machine
      Suppliers
    Joe
      Staff training""",
    "timeline": """timeline
    title Shop history
    2028-05-01 : First shop opens
    2029-09-15 : Second shop opens
    2030-03-20 : Online orders start""",
}

RULES = {
    "flowchart": "Quote every label: A[\"text\"], B{\"question?\"}, edge labels -->|\"yes\"|. { } is ONLY for a "
                 "yes/no question, and every { } node has a \"yes\" and a \"no\" arrow out of it; actions and "
                 "results use [ ]. Start each step from the request, not from the example.",
    "sequenceDiagram": "One participant line per actor or system named in the request (people AND software parts), "
                       "then one message per line, in order.",
    "classDiagram": "Members go INSIDE class Name { ... } braces, never on their own line. An interface is a class "
                    "with <<interface>> as its first line inside the braces. A class that implements an interface: "
                    "Interface <|.. Class. Notes: note for ClassName \"text\" (never note right of / end note).",
    "stateDiagram-v2": "One transition per line: State --> State: event.",
    "erDiagram": "Entities in CAPITALS, relationships like A ||--o{ B : verb.",
    "gantt": "One section per owner. Every task line is  Task name :id, start, end-or-duration  where start is a "
             "YYYY-MM-DD date or \"after id\" and duration is like 5d. A one-day event is :milestone, id, date, 0d. \"Done by DATE\" is the END "
             "date; when no start is given, the task starts today.",
    "mindmap": "Indentation only: no arrows, no brackets except root((...)).",
    "timeline": "Only  date : event  lines after the title. No arrows, no brackets, no owners.",
}

CHOOSE_SYSTEM = """You choose the best Mermaid diagram type for a request from a deaf or hard-of-hearing student, who
uses the diagram in place of a spoken explanation. Types:
flowchart: steps and decisions, a process, a policy with conditions.
sequenceDiagram: who sends what to whom, in order (people and software parts).
classDiagram: classes, interfaces, UML, object-oriented design.
stateDiagram-v2: the states something goes through.
erDiagram: database tables and their links.
gantt: a plan or schedule with owners, start dates and durations ("timeline with owners", "project plan").
mindmap: topics and subtopics, "who does what" without dates.
timeline: dated events only, no owners, no durations.
Give a short title and, in notes, one or two sentences on what the diagram will show."""

CHOOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "diagram_type": {"type": "string", "enum": TYPES},
        "title": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["diagram_type", "title", "notes"],
}

WRITE_SYSTEM = """You write Mermaid (mermaid.js v11) {type} code. Today is {today}.
Reply with ONLY the code: no ``` fences, no explanation, no comments, no style lines.
Write it in exactly this syntax (the example's topic is unrelated: use ONLY names, owners, dates and steps from the
request, transcript or whiteboard text; never copy the example's content):
{example}

Rules: the first line is {first}. {rules} Never put a " inside a label. At most ~20 nodes."""

# Messages from the Mermaid parser that don't tell a small model what to change.
HINTS = {
    "annotations": "a member or <<interface>> line is outside a class's { } braces, or a note uses the wrong syntax",
}


def clean(code):
    """Model output -> bare Mermaid code: drop ``` fences and surrounding blank lines."""
    code = code.strip()
    m = re.match(r"^```(?:mermaid)?\s*\n(.*?)\n```$", code, re.S)
    return (m.group(1) if m else code).strip()


def lint(code, diagram_type):
    """Mistakes the Mermaid parser lets through, or rejects with a message a small model can't act on."""
    lines = [l for l in code.splitlines() if l.strip()]
    first = lines[0].strip() if lines else ""
    if not first.startswith(diagram_type):
        return f"the first line must be {diagram_type!r}, got {first[:60]!r}"
    if len(lines) < 3:
        return "write one statement per line (the code came out on a single line)"
    body = "\n".join(lines[1:])
    if diagram_type in ("mindmap", "timeline") and re.search(r"-->|--|\[|\]", body):
        return f"{diagram_type} code has arrows or [brackets]: write it like the {diagram_type} example"
    if diagram_type == "timeline" and re.search(r":\s*\w+\s*,", body):
        return "timeline lines are  date : event  only (that looks like gantt task syntax)"
    if diagram_type == "gantt" and "dateFormat" not in code:
        return "gantt code needs a dateFormat YYYY-MM-DD line"
    if diagram_type == "classDiagram" and re.search(r"^\s*note (right|left) of|^\s*end note", body, re.M):
        return 'class diagram notes are written  note for ClassName "text"  (no note right of / end note)'
    if re.search(r"^\s*(style|classDef)\s", code, re.M):
        return "remove the style lines"
    return None


def check(code):
    """Parse with the official Mermaid parser (Node, local). -> {"ok": bool, "type"|"error": str}"""
    try:
        r = subprocess.run(["node", CHECKER], input=code, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return {"ok": False, "error": "node is not installed: the diagram could not be checked"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "the Mermaid parser timed out"}
    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if not lines:
        return {"ok": False, "error": f"checker failed: {(r.stderr or r.stdout)[-300:]}"}
    out = json.loads(lines[-1])
    for key, hint in HINTS.items():
        if not out.get("ok") and key in out.get("error", ""):
            out["error"] += f" (likely cause: {hint})"
    return out


def generate(request, transcript=None, image=None, today=None):
    """-> {"diagram_type", "title", "code", "notes", "valid", "tries", "errors", "whiteboard"}

    request: what to draw. transcript: meeting/lecture text to draw from ("a timeline of what we decided").
    image: whiteboard/slide photo bytes to redraw as clean diagram code.

    Two steps: choose the type (small JSON reply), then write the code for that type as plain text, shown only
    that type's example and rules. Code inside a JSON string lost its newlines; one prompt with all eight examples
    mixed their syntax."""
    today = today or dt.date.today().isoformat()
    parts, whiteboard = [], None
    if image is not None:   # the code model can't see: the vision model reads the whiteboard first
        import mm_image
        whiteboard = mm_image.describe(image)
        parts.append("Whiteboard / slide content (read from a photo):\n" + whiteboard["text_in_image"]
                     + "\nDetails: " + "; ".join(whiteboard.get("details", [])))
    if transcript:
        parts.append("Meeting / lecture transcript to draw from:\n" + transcript.strip()[-12000:])
    parts.append("Request: " + request.strip())
    prompt = "\n\n".join(parts)

    plan = mm_llm.ask(prompt, system=CHOOSE_SYSTEM, schema=CHOOSE_SCHEMA, max_tokens=300, model=S.CODE_MODEL)
    kind = plan.get("diagram_type") if plan.get("diagram_type") in TYPES else "flowchart"
    system = (WRITE_SYSTEM.replace("{type}", kind).replace("{today}", today).replace("{example}", EXAMPLES[kind])
              .replace("{first}", "flowchart TD" if kind == "flowchart" else kind).replace("{rules}", RULES[kind]))
    errors, code, result = [], "", {"ok": False}
    for attempt in range(1, MAX_TRIES + 1):
        prompt_now = prompt
        if errors and code:   # show the model its last code and what was wrong with it
            prompt_now += (f"\n\nYour previous code:\n{code}\n\nIt was rejected: {errors[-1]}\n\n"
                           "Write the whole diagram again, corrected.")
        elif errors:          # the last reply was cut off: there is no code to show
            prompt_now += f"\n\nYour previous answer was rejected: {errors[-1]}"
        try:
            code = clean(mm_llm.ask(prompt_now, system=system, max_tokens=WRITE_TOKENS, model=S.CODE_MODEL,
                                    temperature=0 if attempt == 1 else 0.4))
        except mm_llm.LLMError as e:
            if "cut off" not in str(e):
                raise
            # A small model sometimes loops (the same lines again and again) on a long transcript.
            code = ""
            errors.append(f"it was too long and got cut off after {WRITE_TOKENS} tokens (probably repeating lines). "
                          f"Write a shorter diagram: at most {SHORT_NODES} nodes, each line written once.")
            continue
        problem = lint(code, kind)
        result = {"ok": False, "error": problem} if problem else check(code)
        if result.get("ok"):
            break
        errors.append(result.get("error", "unknown parse error"))
    return {"diagram_type": kind, "title": plan.get("title", ""), "notes": plan.get("notes", ""), "code": code,
            "valid": bool(result.get("ok")), "tries": attempt, "errors": errors, "whiteboard": whiteboard}
