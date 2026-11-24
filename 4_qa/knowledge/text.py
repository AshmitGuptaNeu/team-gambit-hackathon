"""Shared text helpers: the tokenizer BM25 and topic keywords use.

Lowercase, words and numbers only, a fixed English stopword list (written here so nothing needs an NLTK
data download), then Porter stemming (nltk.stem needs no download). Fillers common in speech ("um", "yeah")
are stopwords too, since transcripts are full of them.
"""
import re
from functools import lru_cache

from nltk.stem.porter import PorterStemmer

WORD_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because been before being below between
both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from
further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his
how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor
not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd
they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were
weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you
you'd you'll you're you've your yours yourself yourselves
um uh uhm hmm mm mhm mm-hmm yeah yep yes okay ok oh ah right well like just so really actually basically gonna
wanna kinda sort thing things stuff got get go going know mean think say said also one two maybe
""".split())

_stem = PorterStemmer().stem


@lru_cache(maxsize=200_000)
def stem(word):
    return _stem(word)


def words(text):
    """Lowercased word tokens, stopwords kept (used for word counts)."""
    return WORD_RE.findall(text.lower())


def tokenize(text):
    """BM25 tokens: lowercased, stopwords dropped, stemmed."""
    return [stem(w) for w in words(text) if w not in STOPWORDS and len(w) > 1]


def content_words(text):
    """Unstemmed content words (for human-readable topic keywords)."""
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 2 and not w.isdigit()]


FILLER_RE = re.compile(r"\b(?:mm-hmm|uh-huh|um+|uh+|uhm+|erm*|hmm+|mm+|mhm)\b[,.]?\s*", re.I)
REPEAT_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.I)


def clean_speech(text):
    """Readable text from spoken transcript: drop fillers ("um", "uh"), collapse stutters ("the the"),
    tidy the punctuation left behind (" ,." -> "."). Used for card summaries, never for the cited text."""
    t = FILLER_RE.sub("", text)
    t = REPEAT_RE.sub(r"\1", t)
    t = re.sub(r"\s+([,.?!])", r"\1", t)
    t = re.sub(r"([,.?!])(?:\s*[,.])+", r"\1", t)
    t = re.sub(r"^[\s,.\-]+", "", " ".join(t.split()))
    return t[:1].upper() + t[1:] if t else t
