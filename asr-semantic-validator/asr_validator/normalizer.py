"""
Normalization utilities for ASR/reference text.
- lowercasing, punctuation normalization, number normalization (simple),
- custom dictionary substitutions (e.g., product names)
"""
import re
from typing import Dict

DEFAULT_DICT = {
    # domain-specific normalization, example:
    "八零后": "80后"
}

def normalize(text: str, custom_dict: Dict[str,str]=None) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    # basic punctuation removal (configurable)
    t = re.sub(r"[“”\"'，,。.!？?；;():\-]", " ", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    # simple number normalization (e.g., remove commas)
    t = re.sub(r"([0-9]+),([0-9]{3})", r"\1\2", t)
    # dictionary substitutions
    d = dict(DEFAULT_DICT)
    if custom_dict:
        d.update(custom_dict)
    for k,v in d.items():
        t = t.replace(k, v)
    return t
