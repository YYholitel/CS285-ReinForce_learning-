from __future__ import annotations

import re
from typing import Optional

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
XML_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?")
BOXED_START_RE = re.compile(r"\\boxed\s*\{")
LATEX_FRAC_RE = re.compile(r"\\(?:d?frac|tfrac)\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}")
LATEX_SIGNED_FRAC_RE = re.compile(r"([+-]?)\s*\\(?:d?frac|tfrac)\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}")
LATEX_MIXED_FRAC_RE = re.compile(r"([-+]?\d+)\s*\\(?:d?frac|tfrac)\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}")
SIMPLE_MIXED_FRAC_RE = re.compile(r"([-+]?\d+)\s+([-+]?\d+)\s*/\s*([-+]?\d+)")
TEXT_WRAPPER_RE = re.compile(r"\\(?:text|mathrm)\s*\{(.*)\}", re.DOTALL)
PLAIN_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


def strip_think_blocks(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text)
    return text.replace("</think>", "").strip()


def is_strict_xml_answer(text: str) -> bool:
    cleaned = strip_think_blocks(text)
    return bool(re.fullmatch(r"\s*<answer>\s*.*?\s*</answer>\s*", cleaned, flags=re.DOTALL | re.IGNORECASE))


def extract_xml_answer_content(text: str) -> Optional[str]:
    cleaned = strip_think_blocks(text)
    m = XML_ANSWER_RE.search(cleaned)
    if not m:
        return None
    return m.group(1).strip()


def parse_number(text: str) -> Optional[float]:
    t = text.strip()
    if not t:
        return None
    t = t.replace("\\$", "")
    t = t.replace("$", "")
    t = t.replace(",", "")
    t = t.replace("\\left", "").replace("\\right", "").strip()
    # Strip one layer of braces around a single value.
    if t.startswith("{") and t.endswith("}") and len(t) >= 2:
        inner = t[1:-1].strip()
        if inner:
            t = inner
    text_wrap = TEXT_WRAPPER_RE.fullmatch(t)
    if text_wrap:
        return parse_number(text_wrap.group(1))
    signed_frac = LATEX_SIGNED_FRAC_RE.fullmatch(t)
    if signed_frac:
        sign = -1.0 if signed_frac.group(1) == "-" else 1.0
        num = float(signed_frac.group(2))
        den = float(signed_frac.group(3))
        if abs(den) < 1e-12:
            return None
        return sign * (num / den)
    mixed_frac = LATEX_MIXED_FRAC_RE.fullmatch(t)
    if mixed_frac:
        whole = float(mixed_frac.group(1))
        num = float(mixed_frac.group(2))
        den = float(mixed_frac.group(3))
        if abs(den) < 1e-12:
            return None
        frac = abs(num / den)
        return whole - frac if whole < 0 else whole + frac
    frac_full = LATEX_FRAC_RE.fullmatch(t)
    if frac_full:
        num = float(frac_full.group(1))
        den = float(frac_full.group(2))
        if abs(den) < 1e-12:
            return None
        return num / den
    mixed_simple = SIMPLE_MIXED_FRAC_RE.fullmatch(t)
    if mixed_simple:
        whole = float(mixed_simple.group(1))
        num = float(mixed_simple.group(2))
        den = float(mixed_simple.group(3))
        if abs(den) < 1e-12:
            return None
        frac = abs(num / den)
        return whole - frac if whole < 0 else whole + frac
    # Support simple fractional forms like "3/4" or "-7/2".
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", t):
        num_s, den_s = [x.strip() for x in t.split("/", 1)]
        try:
            num = float(num_s)
            den = float(den_s)
        except ValueError:
            return None
        if abs(den) < 1e-12:
            return None
        return num / den
    if not PLAIN_NUMBER_RE.fullmatch(t):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def extract_number_from_xml_answer(text: str) -> Optional[float]:
    content = extract_xml_answer_content(text)
    if content is None:
        return None
    direct = parse_number(content)
    if direct is not None:
        return direct
    nums = NUMBER_RE.findall(content)
    if not nums:
        return None
    return parse_number(nums[-1])


def extract_last_number(text: str) -> Optional[float]:
    cleaned = strip_think_blocks(text)
    nums = NUMBER_RE.findall(cleaned)
    if not nums:
        return None
    return parse_number(nums[-1])


def _find_matching_closing_brace(text: str, opening_brace_idx: int) -> Optional[int]:
    depth = 0
    for i in range(opening_brace_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _extract_last_boxed_span(text: str) -> Optional[tuple[int, int, str]]:
    cleaned = strip_think_blocks(text)
    starts = list(BOXED_START_RE.finditer(cleaned))
    for m in reversed(starts):
        start = m.start()
        open_idx = cleaned.find("{", m.start())
        if open_idx < 0:
            continue
        close_idx = _find_matching_closing_brace(cleaned, open_idx)
        if close_idx is None:
            continue
        content = cleaned[open_idx + 1 : close_idx].strip()
        return start, close_idx, content
    return None


def extract_last_boxed_content(text: str) -> Optional[str]:
    span = _extract_last_boxed_span(text)
    if span is None:
        return None
    return span[2]


def extract_number_from_boxed_answer(text: str) -> Optional[float]:
    content = extract_last_boxed_content(text)
    if content is None:
        return None
    # Intentionally strict: avoid spuriously mapping symbolic answers
    # (intervals, expressions, sets, letters) to a single scalar.
    return parse_number(content)


def is_strict_boxed_answer(text: str) -> bool:
    cleaned = strip_think_blocks(text).strip()
    span = _extract_last_boxed_span(cleaned)
    if span is None:
        return False
    start, end, _ = span
    if cleaned[:start].strip():
        return False
    if cleaned[end + 1 :].strip():
        return False
    # Must contain exactly one boxed segment when strict.
    return len(list(BOXED_START_RE.finditer(cleaned))) == 1
