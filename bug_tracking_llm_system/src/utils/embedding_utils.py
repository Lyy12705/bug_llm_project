from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def term_frequency(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine_text_similarity(left: str, right: str) -> float:
    left_counts = term_frequency(left)
    right_counts = term_frequency(right)
    if not left_counts or not right_counts:
        return 0.0
    common = set(left_counts) & set(right_counts)
    numerator = sum(left_counts[token] * right_counts[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return float(len(left_tokens & right_tokens) / len(left_tokens | right_tokens))


def weighted_similarity(parts: Iterable[tuple[str, str, float]]) -> float:
    total_weight = 0.0
    score = 0.0
    for left, right, weight in parts:
        if weight <= 0 or not str(left).strip() or not str(right).strip():
            continue
        score += cosine_text_similarity(left, right) * weight
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return float(score / total_weight)
