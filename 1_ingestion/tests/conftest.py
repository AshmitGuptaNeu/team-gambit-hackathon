"""pytest setup for 1_ingestion's tests: import the bridge's modules (bridge, ui, ...) from 1_ingestion/."""

import os
import sys
from pathlib import Path

# The rate-limit tests check Groq's budgets (settings.py), which only exist with STT_BACKEND=groq.
# The local backend (the default) has no limits; local_models/tests covers its server.
os.environ.setdefault("STT_BACKEND", "groq")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
