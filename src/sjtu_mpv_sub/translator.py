"""Time-aligned PPT OCR context builder, subtitle text sanitizer, and hardened LLM translator."""

from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any

import httpx
from openai import AsyncOpenAI

from sjtu_mpv_sub.models import AsrSentence, BilingualCue, PptSlice

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_INVALID_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')
_CTRL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_WORD_CHAR_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_EN_WORD_RE = re.compile(r"[A-Za-z]{2,}")

_META_COT_RE = re.compile(
    r"(?:（|\()\s*(?:注|Note|译者注)\s*[:：]"
    r"|PPT\s*OCR|OCR\s*显示|结合上下文|讲师说的是|此处可能为口误|无意义填充词"
    r"|等等[，,]\s*(?:PPT|这里|这看起来)",
    re.IGNORECASE,
)
_EDITORIAL_BRACKET_RE = re.compile(
    r"^\[.*?(?:background\s*noise|unrelated\s*conversation|unrelated|背景噪音|无关对话|噪音/无关).*?\]$",
    re.IGNORECASE,
)
_RAW_MARKUP_RE = re.compile(
    r"\*\*|__|`|\\\[|\\\]|\\\(|\\\)"
    r"|(?<![0-9.,])\$\$?(?![0-9])"
    r"|\\(?:frac|dfrac|sqrt|boldsymbol|mathbf|tilde|bar|hat|vec|dot|lambda|omega|delta|beta|alpha|gamma|theta|phi|pi|chi|text|mathrm)\b"
)

_LATEX_STYLE_UNWRAP_RE = re.compile(
    r"\\(?:text|mathrm|mathbf|boldsymbol|mathit|operatorname|mathcal|mathbb|mathsf|rm|bf)\s*\{([^{}]*)\}"
)
_LATEX_FRAC_RE = re.compile(r"\\(?:d?frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_LATEX_FRAC_BARE_RE = re.compile(r"\\(?:d?frac)\s+([A-Za-z0-9])\s+([A-Za-z0-9]+)")
_LATEX_SQRT_OPT_RE = re.compile(r"\\sqrt\s*\[([^\]]+)\]\s*\{([^{}]*)\}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_LATEX_SQRT_BARE_RE = re.compile(r"\\sqrt\s+([A-Za-z0-9])")

_ACCENT_COMBINING = {
    "tilde": "\u0303",
    "widetilde": "\u0303",
    "bar": "\u0304",
    "overline": "\u0304",
    "hat": "\u0302",
    "widehat": "\u0302",
    "vec": "\u20d7",
    "dot": "\u0307",
    "ddot": "\u0308",
}
_LATEX_ACCENT_RE = re.compile(
    r"\\(tilde|widetilde|bar|overline|hat|widehat|vec|dot|ddot)\s*(?:\{([^{}]+)\}|\s+([A-Za-zα-ωΑ-Ω]))"
)

_LATEX_COMMANDS: dict[str, str] = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "π",
    "rho": "ρ",
    "varrho": "ρ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "times": "×",
    "cdot": "·",
    "div": "÷",
    "pm": "±",
    "mp": "∓",
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "neq": "≠",
    "ne": "≠",
    "approx": "≈",
    "equiv": "≡",
    "propto": "∝",
    "sim": "~",
    "simeq": "≃",
    "cong": "≅",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "Rightarrow": "⇒",
    "Leftarrow": "⇐",
    "leftrightarrow": "↔",
    "Leftrightarrow": "⇔",
    "mapsto": "↦",
    "infty": "∞",
    "partial": "∂",
    "nabla": "∇",
    "int": "∫",
    "iint": "∬",
    "oint": "∮",
    "sum": "∑",
    "prod": "∏",
    "in": "∈",
    "notin": "∉",
    "subset": "⊂",
    "subseteq": "⊆",
    "supset": "⊃",
    "supseteq": "⊇",
    "cup": "∪",
    "cap": "∩",
    "forall": "∀",
    "exists": "∃",
    "neg": "¬",
    "land": "∧",
    "lor": "∨",
    "perp": "⊥",
    "parallel": "∥",
    "angle": "∠",
    "degree": "°",
    "ell": "ℓ",
    "hbar": "ℏ",
    "dots": "...",
    "cdots": "...",
    "ldots": "...",
    "vdots": "⋮",
    "Re": "Re",
    "Im": "Im",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "cot": "cot",
    "sec": "sec",
    "csc": "csc",
    "arcsin": "arcsin",
    "arccos": "arccos",
    "arctan": "arctan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "exp": "exp",
    "ln": "ln",
    "log": "log",
    "lg": "lg",
    "det": "det",
    "dim": "dim",
    "ker": "ker",
    "rank": "rank",
    "tr": "tr",
    "lim": "lim",
    "max": "max",
    "min": "min",
    "sup": "sup",
    "inf": "inf",
    "arg": "arg",
    "deg": "deg",
    "gcd": "gcd",
    "mod": "mod",
    "diag": "diag",
    "span": "span",
    "boldsymbol": "",
    "mathbf": "",
    "mathrm": "",
    "mathit": "",
    "text": "",
    "left": "",
    "right": "",
    "big": "",
    "Big": "",
    "bigg": "",
    "Bigg": "",
    "quad": " ",
    "qquad": " ",
    "displaystyle": "",
    "textstyle": "",
}
_LATEX_CMD_RE = re.compile(r"\\([A-Za-z]+)")

_SUP_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "n": "ⁿ",
    "i": "ⁱ",
}
_SUB_MAP = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
}


def _simple_token(expr: str) -> bool:
    trimmed = expr.strip()
    return bool(re.fullmatch(r"(?:[A-Za-zα-ωΑ-Ω]|[0-9]+|d[A-Za-z])", trimmed))


def _wrap_atom(expr: str) -> str:
    trimmed = expr.strip()
    if not trimmed:
        return ""
    return trimmed if _simple_token(trimmed) else f"({trimmed})"


def _convert_script(prefix: str, body: str, table: dict[str, str]) -> str:
    clean = body.strip()
    if clean and all(ch in table for ch in clean):
        return "".join(table[ch] for ch in clean)
    if len(clean) == 1:
        return f"{prefix}{clean}"
    return f"{prefix}({clean})"


def _clean_latex_expr(expr: str, *, inside_math: bool) -> str:
    s = expr
    # Temporarily preserve escaped set braces so LaTeX group cleanup keeps literal sets.
    s = s.replace(r"\{", "\x01LBRACE\x01").replace(r"\}", "\x01RBRACE\x01")
    # Collapse doubled backslashes before alphabetic commands (e.g. \\lambda -> \lambda).
    s = re.sub(r"\\+(?=[A-Za-z])", r"\\", s)
    s = re.sub(r"\\([,:;! ])", " ", s)

    for _ in range(4):
        updated = _LATEX_STYLE_UNWRAP_RE.sub(r"\1", s)
        if updated == s:
            break
        s = updated

    def _replace_accent(m: re.Match[str]) -> str:
        mark = _ACCENT_COMBINING[m.group(1)]
        target = (m.group(2) or m.group(3) or "").strip()
        if not target:
            return ""
        if len(target) == 1:
            return target + mark
        return target[0] + mark + target[1:]

    s = _LATEX_ACCENT_RE.sub(_replace_accent, s)

    def _replace_cmd(m: re.Match[str]) -> str:
        cmd = m.group(1)
        if cmd in _LATEX_COMMANDS:
            return _LATEX_COMMANDS[cmd]
        return m.group(0)

    s = _LATEX_CMD_RE.sub(_replace_cmd, s)

    for _ in range(3):
        prev = s
        s = _LATEX_FRAC_RE.sub(
            lambda m: f"{_wrap_atom(m.group(1))}/{_wrap_atom(m.group(2))}",
            s,
        )
        s = _LATEX_FRAC_BARE_RE.sub(lambda m: f"{m.group(1)}/{m.group(2)}", s)
        s = _LATEX_SQRT_OPT_RE.sub(
            lambda m: f"{_convert_script('', m.group(1), _SUP_MAP)}√{_wrap_atom(m.group(2))}",
            s,
        )
        s = _LATEX_SQRT_RE.sub(lambda m: f"√{_wrap_atom(m.group(1))}", s)
        s = _LATEX_SQRT_BARE_RE.sub(lambda m: f"√{m.group(1)}", s)
        if s == prev:
            break

    s = re.sub(r"\^\{([^{}]+)\}", lambda m: _convert_script("^", m.group(1), _SUP_MAP), s)
    s = re.sub(r"\^([0-9ni])", lambda m: _SUP_MAP[m.group(1)], s)
    s = re.sub(r"_\{([^{}]+)\}", lambda m: _convert_script("_", m.group(1), _SUB_MAP), s)
    s = re.sub(r"_([0-9ijkmn])", lambda m: _SUB_MAP[m.group(1)], s)

    if inside_math:
        s = s.replace("{", "").replace("}", "")
    s = s.replace("\x01LBRACE\x01", "{").replace("\x01RBRACE\x01", "}")
    return s


def sanitize_subtitle_text(text: str, *, is_zh: bool = False) -> str:
    """Deterministically strip control chars, Markdown, and LaTeX into clean Unicode text."""
    _ = is_zh
    if not text:
        return ""

    s = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = _CTRL_CHARS_RE.sub("", s)

    # Strip Markdown emphasis and inline code spans without harming multiplication signs.
    s = re.sub(r"\*\*([^*]+?)\*\*", r"\1", s)
    s = re.sub(r"__([^_]+?)__", r"\1", s)
    s = re.sub(r"`([^`]+?)`", r"\1", s)
    s = re.sub(r"(?<!\w)\*([^\s*][^*]*?[^\s*]|[^\s*])\*(?!\w)", r"\1", s)
    s = s.replace("`", "")

    # Strip math display/inline delimiters and convert delimited LaTeX expressions first.
    s = s.replace(r"\(", "$").replace(r"\)", "$").replace(r"\[", "$").replace(r"\]", "$")
    s = re.sub(r"\$\$(.+?)\$\$", lambda m: _clean_latex_expr(m.group(1), inside_math=True), s)
    s = re.sub(r"\$([^$\n]+?)\$", lambda m: _clean_latex_expr(m.group(1), inside_math=True), s)
    # Remove any unmatched math '$' (not currency like '$100').
    s = re.sub(r"\$(?![0-9])|(?<=[A-Za-z0-9)}\].\\])\$", "", s)

    if "\\" in s or "^" in s or "_" in s:
        s = _clean_latex_expr(s, inside_math=False)

    return " ".join(s.split())


def collect_ppt_window(
    ppt_slices: list[PptSlice],
    chunk_start_sec: float,
    chunk_end_sec: float,
    max_chars: int = 1500,
) -> tuple[list[str], str]:
    """Collect active/in-window PPT keywords and formatted OCR context for a chunk."""
    if not ppt_slices:
        return [], ""

    ordered = sorted(ppt_slices, key=lambda s: s.seek_sec)
    active_before: PptSlice | None = None
    window_slices: list[PptSlice] = []

    for sl in ordered:
        if sl.seek_sec <= chunk_start_sec:
            active_before = sl
        elif sl.seek_sec <= chunk_end_sec + 20.0:
            window_slices.append(sl)

    selected: list[PptSlice] = []
    if active_before is not None:
        selected.append(active_before)
    selected.extend(window_slices)

    if not selected:
        return [], ""

    keywords_seen: list[str] = []
    keywords_set: set[str] = set()
    slide_blocks: list[str] = []

    for sl in selected:
        for kw in sl.keywords:
            cleaned_kw = kw.strip()
            if cleaned_kw and cleaned_kw not in keywords_set:
                keywords_set.add(cleaned_kw)
                keywords_seen.append(cleaned_kw)
        ocr = " ".join(sl.ocr_text.split())
        if ocr:
            slide_blocks.append(f"[{int(sl.seek_sec)}s] {ocr}")

    parts: list[str] = []
    if keywords_seen:
        parts.append("PPT Keywords: " + ", ".join(keywords_seen))
    if slide_blocks:
        parts.append("PPT OCR:\n" + "\n".join(slide_blocks))

    combined = "\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[: max_chars - 3].rstrip() + "..."
    return keywords_seen, combined


def build_ppt_context(
    ppt_slices: list[PptSlice],
    chunk_start_sec: float,
    chunk_end_sec: float,
    max_chars: int = 1500,
) -> str:
    """Collect active and in-window PPT slide OCR text and keywords for a chunk."""
    _, context = collect_ppt_window(ppt_slices, chunk_start_sec, chunk_end_sec, max_chars=max_chars)
    return context


def parse_llm_items(raw_content: str) -> dict[int, tuple[str, str]]:
    """Parse JSON items mapping index -> (en, zh), repairing unescaped LaTeX backslashes if needed."""
    text = (raw_content or "").strip()
    if text.startswith("```"):
        text = _CODE_FENCE_RE.sub("", text).strip()

    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        repaired = _INVALID_JSON_ESCAPE_RE.sub(r"\\\\", text)
        data = json.loads(repaired)

    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("LLM JSON missing 'items' array")

    mapped: dict[int, tuple[str, str]] = {}
    for entry in data["items"]:
        if not isinstance(entry, dict) or "i" not in entry:
            continue
        try:
            idx = int(entry["i"])
        except (TypeError, ValueError):
            continue
        en_val = str(entry.get("en", "")).strip()
        zh_val = str(entry.get("zh", "")).strip()
        mapped[idx] = (en_val, zh_val)
    return mapped


def is_chinese_spoken(text: str) -> bool:
    """Return True when the recognized utterance is primarily spoken in Chinese."""
    cjk_count = len(_CJK_CHAR_RE.findall(text))
    if cjk_count == 0:
        return False
    en_word_count = len(_EN_WORD_RE.findall(text))
    if en_word_count <= 1:
        return True
    return cjk_count > en_word_count


def _check_en_alignment(src_text: str, en_text: str) -> str | None:
    """Detect English hallucination, severe collapse, or index-shift drift."""
    ascii_cnt = len(_ASCII_LETTER_RE.findall(src_text))
    cjk_cnt = len(_CJK_CHAR_RE.findall(src_text))
    if cjk_cnt > 0 and ascii_cnt < 0.7 * max(1, ascii_cnt + cjk_cnt):
        return None

    src_tokens = [w.lower() for w in _EN_WORD_RE.findall(src_text)]
    en_tokens = [w.lower() for w in _EN_WORD_RE.findall(en_text)]

    if cjk_cnt == 0:
        if len(src_tokens) <= 4 and len(en_tokens) >= max(10, len(src_tokens) * 3 + 4):
            return "hallucination"
        if len(src_tokens) >= 10 and len(en_tokens) <= max(2, len(src_tokens) // 4):
            return "collapse"

    if len(src_tokens) < 5:
        return None
    if not en_tokens:
        return "soft_drift"

    src_set = set(src_tokens)
    en_set = set(en_tokens)
    overlap = len(src_set & en_set) / max(1, min(len(src_set), len(en_set)))
    seq_ratio = difflib.SequenceMatcher(None, src_tokens, en_tokens).ratio()

    if len(src_tokens) >= 8 and overlap < 0.15 and seq_ratio < 0.25:
        return "severe_drift"
    if overlap < 0.25 and seq_ratio < 0.35:
        return "soft_drift"
    return None


def validate_translated_batch(
    batch: list[AsrSentence],
    raw_mapped: dict[int, tuple[str, str]],
) -> list[str]:
    """Validate 1-to-1 alignment, cleanliness, and completeness for a translated batch."""
    reasons: list[str] = []
    expected = set(range(len(batch)))
    if set(raw_mapped.keys()) != expected:
        missing = sorted(expected - set(raw_mapped.keys()))
        extra = sorted(set(raw_mapped.keys()) - expected)
        reasons.append(f"index_mismatch(missing={missing}, extra={extra})")
        return reasons

    drifted_indices: list[int] = []
    for idx, sent in enumerate(batch):
        en_raw, zh_raw = raw_mapped[idx]
        if sent.text.lstrip().startswith("["):
            continue

        zh_only = is_chinese_spoken(sent.text)
        if zh_only:
            if not zh_raw.strip():
                reasons.append(f"empty@{idx}")
                continue
            if _CTRL_CHARS_RE.search(zh_raw):
                reasons.append(f"control_char@{idx}")
            if _META_COT_RE.search(zh_raw):
                reasons.append(f"meta_or_cot_leak@{idx}")
            src_len = max(20, len(sent.text.strip()))
            if len(zh_raw.strip()) > max(120, int(src_len * 2.5)):
                reasons.append(f"length_explosion@{idx}")
            if len(_WORD_CHAR_RE.findall(sent.text)) >= 2 and len(_WORD_CHAR_RE.findall(zh_raw)) == 0:
                reasons.append(f"punctuation_stub@{idx}")
            if _EDITORIAL_BRACKET_RE.match(zh_raw.strip()):
                reasons.append(f"editorial_placeholder@{idx}")
            if _RAW_MARKUP_RE.search(zh_raw):
                reasons.append(f"markup_or_latex@{idx}")
            continue

        if not en_raw.strip() or not zh_raw.strip():
            reasons.append(f"empty@{idx}")
            continue

        if _CTRL_CHARS_RE.search(en_raw) or _CTRL_CHARS_RE.search(zh_raw):
            reasons.append(f"control_char@{idx}")

        combined = f"{en_raw}\n{zh_raw}"
        if _META_COT_RE.search(combined):
            reasons.append(f"meta_or_cot_leak@{idx}")

        src_len = max(20, len(sent.text.strip()))
        en_len = max(20, len(en_raw.strip()))
        if len(zh_raw.strip()) > max(120, int(src_len * 2.5)) and len(zh_raw.strip()) > max(120, int(en_len * 2.5)):
            reasons.append(f"length_explosion@{idx}")

        if len(_WORD_CHAR_RE.findall(sent.text)) >= 2 and len(_WORD_CHAR_RE.findall(zh_raw)) == 0:
            reasons.append(f"punctuation_stub@{idx}")

        if (
            len(_EN_WORD_RE.findall(sent.text)) >= 4
            and len(_CJK_CHAR_RE.findall(zh_raw)) == 0
            and zh_raw.strip().lower() == en_raw.strip().lower()
        ):
            reasons.append(f"untranslated_zh@{idx}")

        if _EDITORIAL_BRACKET_RE.match(en_raw.strip()) or _EDITORIAL_BRACKET_RE.match(zh_raw.strip()):
            reasons.append(f"editorial_placeholder@{idx}")

        if _RAW_MARKUP_RE.search(en_raw) or _RAW_MARKUP_RE.search(zh_raw):
            reasons.append(f"markup_or_latex@{idx}")

        align_issue = _check_en_alignment(sent.text, en_raw)
        if align_issue in ("hallucination", "collapse", "severe_drift"):
            reasons.append(f"en_{align_issue}@{idx}")
        elif align_issue == "soft_drift":
            drifted_indices.append(idx)

    if len(drifted_indices) >= 2 or (
        len(batch) == 1 and len(drifted_indices) == 1 and len(_EN_WORD_RE.findall(batch[0].text)) >= 8
    ):
        reasons.append(f"index_drift@{drifted_indices}")

    return reasons


def _has_fatal_item_defect(sent: AsrSentence, en_raw: str, zh_raw: str) -> bool:
    """Check whether a single fallback item contains unusable CoT/placeholder/stub output."""
    zh_only = is_chinese_spoken(sent.text)
    if not zh_raw.strip() or (not zh_only and not en_raw.strip()):
        return True
    target_text = zh_raw if zh_only else f"{en_raw}\n{zh_raw}"
    if _META_COT_RE.search(target_text):
        return True
    src_len = max(20, len(sent.text.strip()))
    if len(zh_raw.strip()) > max(120, int(src_len * 2.5)):
        return True
    if len(_WORD_CHAR_RE.findall(sent.text)) >= 2 and len(_WORD_CHAR_RE.findall(zh_raw)) == 0:
        return True
    if not sent.text.lstrip().startswith("["):
        if _EDITORIAL_BRACKET_RE.match(zh_raw.strip()) or (not zh_only and _EDITORIAL_BRACKET_RE.match(en_raw.strip())):
            return True
    if not zh_only and _check_en_alignment(sent.text, en_raw) in ("hallucination", "collapse", "severe_drift"):
        return True
    return False


class LLMTranslator:
    """Async batch ASR refiner and bilingual translator via OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "qwen3.8-flash",
        proxy: str = "",
        enable_thinking: bool = False,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.enable_thinking = enable_thinking
        self._owned_http_client: httpx.AsyncClient | None = None
        if client is not None:
            self._client = client
        else:
            http_kwargs: dict[str, Any] = {"trust_env": False, "timeout": 90.0}
            if proxy.strip():
                http_kwargs["proxy"] = proxy.strip()
            self._owned_http_client = httpx.AsyncClient(**http_kwargs)
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=self._owned_http_client,
                timeout=90.0,
            )

    async def close(self) -> None:
        if self._owned_http_client is not None:
            await self._owned_http_client.aclose()
            self._owned_http_client = None

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are an expert bilingual academic lecture subtitle refiner and translator for Shanghai Jiao Tong "
            "University (SJTU) courses.\n"
            "Strict output rules:\n"
            "1. Preserve EVERY input index 0..N-1 strictly 1-to-1. NEVER merge, split, shift, skip, or reorder items.\n"
            "2. Keep each segment locally self-contained within its own time slot: NEVER move a phrase from segment i+1 "
            "into segment i in a way that leaves segment i+1 as bare punctuation or an empty stub.\n"
            "3. 'en': if the segment is spoken primarily in Chinese (including Chinese with occasional English "
            "technical terms), set 'en' to \"\" (empty string) and do NOT translate Chinese speech into English. "
            "Only when the segment is spoken in English (or English-dominant), output faithful, cleaned spoken English "
            "matching what the lecturer said aloud (fix ASR typos, homophones such as 'Sir.' -> 'So,', punctuation, "
            "and technical terms; keep spoken English natural without converting spoken math words into LaTeX or "
            "symbolic formulas).\n"
            "4. 'zh': for Chinese-spoken segments, refine ASR typos/homophones and punctuation into clean Simplified "
            "Chinese; for English-spoken segments, provide natural, concise, accurate Simplified Chinese translation "
            "for fast lecture watching. Express formulas and symbols using plain text and readable Unicode symbols "
            "ONLY (e.g. λ², √(4L/C), cos(ωt + φ), A = P J P⁻¹, dx/dt).\n"
            "5. STRICTLY FORBIDDEN in both 'en' and 'zh': any Markdown (**, *, __, `), any LaTeX ($, $$, \\frac, "
            "\\sqrt, \\lambda, backslashes), any thinking/reasoning commentary ('等等', 'PPT OCR', '讲师说的是'), any "
            "translator notes ('（注：...）', '(Note: ...)'), and any editorial placeholders such as "
            "'[Background noise/Unrelated conversation]' or '[背景噪音/无关对话]'. Faithfully refine and translate all "
            "audible speech.\n"
            "6. Return strict JSON ONLY: "
            '{"items": [{"i": <int>, "en": "<string>", "zh": "<string>"}]}'
        )

    async def refine_and_translate(
        self,
        sentences: list[AsrSentence],
        *,
        course_name: str = "",
        glossary: list[str] | None = None,
        extra_prompt: str = "",
        ppt_context: str = "",
        batch_size: int = 12,
    ) -> list[BilingualCue]:
        """Refine ASR typos with context and produce aligned English+Chinese cues."""
        if not sentences:
            return []

        step = max(1, batch_size)
        cues: list[BilingualCue] = []

        for start in range(0, len(sentences), step):
            batch = sentences[start : start + step]
            prev_context = [s.text for s in sentences[max(0, start - 3) : start]]
            batch_cues = await self._translate_batch(
                batch,
                prev_context=prev_context,
                course_name=course_name,
                glossary=glossary or [],
                extra_prompt=extra_prompt,
                ppt_context=ppt_context,
            )
            cues.extend(batch_cues)

        return cues

    async def _request_batch(
        self,
        batch: list[AsrSentence],
        *,
        prev_context: list[str],
        course_name: str,
        glossary: list[str],
        extra_prompt: str,
        ppt_context: str,
        correction_hint: str = "",
    ) -> dict[int, tuple[str, str]]:
        payload: dict[str, Any] = {
            "course_name": course_name,
            "glossary": glossary,
            "extra_instructions": extra_prompt.strip(),
            "ppt_context": ppt_context,
            "previous_context_read_only": prev_context,
            "segments": [{"i": idx, "text": s.text} for idx, s in enumerate(batch)],
        }
        if correction_hint:
            payload["validation_retry_instruction"] = correction_hint

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            extra_body={"enable_thinking": self.enable_thinking},
        )
        content = resp.choices[0].message.content or ""
        return parse_llm_items(content)

    async def _translate_batch(
        self,
        batch: list[AsrSentence],
        *,
        prev_context: list[str],
        course_name: str,
        glossary: list[str],
        extra_prompt: str,
        ppt_context: str,
        allow_split: bool = True,
    ) -> list[BilingualCue]:
        last_mapped: dict[int, tuple[str, str]] = {}
        last_reasons: list[str] = []
        correction_hint = ""

        for attempt in range(2):
            try:
                mapped = await self._request_batch(
                    batch,
                    prev_context=prev_context,
                    course_name=course_name,
                    glossary=glossary,
                    extra_prompt=extra_prompt,
                    ppt_context=ppt_context,
                    correction_hint=correction_hint,
                )
                last_mapped = mapped
                reasons = validate_translated_batch(batch, mapped)
                if not reasons:
                    return self._finalize_cues(batch, mapped)
                last_reasons = reasons
                correction_hint = (
                    f"Previous response failed validation ({', '.join(reasons[:5])}). "
                    f"Output ALL indices 0..{len(batch) - 1} strictly 1-to-1; for Chinese-spoken segments set 'en' to "
                    "\"\" and output refined 'zh' only; for English-spoken segments keep English faithful to spoken "
                    "words and translate into concise Simplified Chinese with Unicode symbols only; "
                    "forbid LaTeX, Markdown, CoT, translator notes, or [Background noise] tags."
                )
                logger.warning(
                    "LLM translation batch (size=%d) validation failed on attempt %d: %s",
                    len(batch),
                    attempt + 1,
                    ", ".join(reasons[:6]),
                )
            except Exception as exc:
                last_reasons = [str(exc)]
                correction_hint = f"Previous response could not be parsed ({exc}). Return valid JSON matching schema."
                logger.warning("LLM translation batch (size=%d) attempt %d failed: %s", len(batch), attempt + 1, exc)

        if allow_split and len(batch) > 1:
            mid = len(batch) // 2
            logger.warning(
                "Bisecting translation batch of %d items (%d + %d) due to: %s",
                len(batch),
                mid,
                len(batch) - mid,
                ", ".join(last_reasons[:4]),
            )
            left = await self._translate_batch(
                batch[:mid],
                prev_context=prev_context,
                course_name=course_name,
                glossary=glossary,
                extra_prompt=extra_prompt,
                ppt_context=ppt_context,
                allow_split=True,
            )
            right_ctx = [s.text for s in batch[max(0, mid - 3) : mid]]
            right = await self._translate_batch(
                batch[mid:],
                prev_context=right_ctx,
                course_name=course_name,
                glossary=glossary,
                extra_prompt=extra_prompt,
                ppt_context=ppt_context,
                allow_split=True,
            )
            return left + right

        # Single-item fallback: retry once without noisy PPT OCR before sanitizing best output.
        if len(batch) == 1 and ppt_context:
            try:
                clean_mapped = await self._request_batch(
                    batch,
                    prev_context=prev_context,
                    course_name=course_name,
                    glossary=glossary,
                    extra_prompt=extra_prompt,
                    ppt_context="",
                    correction_hint=(
                        "Translate this single segment directly and concisely without commentary, notes, "
                        "Markdown, or LaTeX."
                    ),
                )
                if not validate_translated_batch(batch, clean_mapped):
                    return self._finalize_cues(batch, clean_mapped)
                if 0 in clean_mapped and not _has_fatal_item_defect(batch[0], *clean_mapped[0]):
                    last_mapped = clean_mapped
            except Exception as exc:
                logger.warning("Single-item clean fallback failed: %s", exc)

        return self._finalize_cues(batch, last_mapped, strict_fallback=True)

    @staticmethod
    def _finalize_cues(
        batch: list[AsrSentence],
        mapped: dict[int, tuple[str, str]],
        *,
        strict_fallback: bool = False,
    ) -> list[BilingualCue]:
        result: list[BilingualCue] = []
        for idx, sent in enumerate(batch):
            raw_en, raw_zh = mapped.get(idx, ("", ""))
            clean_zh_src = sanitize_subtitle_text(sent.text, is_zh=True)
            if is_chinese_spoken(sent.text):
                en_out = ""
                if strict_fallback and _has_fatal_item_defect(sent, raw_en, raw_zh):
                    zh_out = clean_zh_src
                else:
                    zh_out = sanitize_subtitle_text(raw_zh, is_zh=True) or clean_zh_src
            else:
                clean_src = sanitize_subtitle_text(sent.text, is_zh=False)
                if strict_fallback and _has_fatal_item_defect(sent, raw_en, raw_zh):
                    en_out = clean_src
                    zh_out = clean_zh_src
                else:
                    en_out = sanitize_subtitle_text(raw_en, is_zh=False) or clean_src
                    zh_out = sanitize_subtitle_text(raw_zh, is_zh=True) or clean_zh_src
            result.append(
                BilingualCue(
                    start_ms=sent.start_ms,
                    end_ms=sent.end_ms,
                    en=en_out,
                    zh=zh_out,
                )
            )
        return result
