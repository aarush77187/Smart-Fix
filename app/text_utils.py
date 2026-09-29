"""
A real dense-embedding model would treat "pair" and "connect", or
"earbuds" and "headphones", as near-identical. BM25/TF-IDF (our local,
no-internet stand-in -- see matcher.py and cache.py) cannot: they only
match shared literal tokens. This is a small, honest mitigation, not a
general solution -- a hand-picked synonym map for the vocabulary patterns
actually observed in this domain's complaints, applied before tokenizing
in both the deeplink matcher and the semantic cache.
"""

import re

_SYNONYM_GROUPS = [
    {"pair", "connect", "connecting", "link", "linking"},
    {"earbuds", "headphones", "earphones", "buds"},
    {"blank", "black"},
    {"crack", "cracked", "broken", "shattered"},
    {"freeze", "freezing", "frozen", "hang", "hangs", "hanging", "stuck"},
    {"slow", "sluggish", "laggy", "lag"},
    {"wont", "won't", "doesnt", "doesn't", "cant", "can't"},
    {"flicker", "flickers", "flickering", "flashing", "flashes"},
    {"dim", "dimmed", "dark", "darker"},
]

_CANONICAL: dict[str, str] = {}
for group in _SYNONYM_GROUPS:
    canonical = sorted(group)[0]
    for word in group:
        _CANONICAL[word] = canonical


def normalize_synonyms(text: str) -> str:
    def _replace(match: re.Match) -> str:
        word = match.group(0).lower()
        return _CANONICAL.get(word, word)

    return re.sub(r"[a-zA-Z']+", _replace, text)
