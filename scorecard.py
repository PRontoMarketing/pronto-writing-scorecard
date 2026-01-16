
"""
PRonto Writing Scorecard (MVP + clarity improvements)

- Accepts extracted plain text
- Computes metrics:
  Readability Indices, General Document Statistics,
  Sentence Structure Analysis, Complexity & Style Metrics, Word Usage
- Produces section scores + overall score + improvement suggestions

Notes:
- Passive voice detection is heuristic (good enough for trend + coaching).
- "Frequency bands" are within-document, not a global English frequency list.
"""
from __future__ import annotations

import re
import math
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_LETTER_RE = re.compile(r"[A-Za-z]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

_STOPWORDS = {
    "the","a","an","and","or","but","if","to","of","in","on","for","with","as","at","by","from",
    "is","are","was","were","be","been","being","it","this","that","these","those",
    "we","you","i","they","he","she","them","us","our","your","their","my","me",
    "not","no","yes","do","does","did","will","would","can","could","should","may","might","must",
    "have","has","had","having",
}

_HEDGING = {
    "maybe","perhaps","possibly","likely","unlikely","arguably","generally","usually","often",
    "somewhat","relatively","apparently","seems","seem","appears","appear","might","could","may",
    "tends","tend","suggests","suggest","around","roughly","approximately",
}

_SUBORDINATORS = {
    "although","because","since","while","whereas","if","when","unless","though","after","before",
    "until","once","rather","whether","even","so","than",
}

_BE_VERBS = {"am","is","are","was","were","be","been","being"}

_IRREG_PART = {
    "known","given","seen","done","made","taken","built","found","kept","held","said","shown","told",
    "written","driven","broken","chosen","grown","thrown","caught","bought","brought","thought",
    "paid","sent","spent","gone","won","understood","set","put","read","left","felt","cut","hit",
}


# ----------------- text splitting -----------------
def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def split_paragraphs(text: str) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def split_sentences(text: str) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    # Avoid splitting on a few common abbreviations
    text = re.sub(r"\b(e\.g|i\.e|Mr|Mrs|Ms|Dr|Prof)\.\s", lambda m: m.group(0).replace(".", "∯"), text)
    parts = _SENT_SPLIT_RE.split(text)
    sents = []
    for p in parts:
        p = p.replace("∯", ".").strip()
        if p:
            sents.append(p)
    return sents

def words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


# ----------------- syllables + readability -----------------
def count_syllables(word: str) -> int:
    """Heuristic syllable counter (good enough for scoring trends)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w2 = re.sub(r"e$", "", w)  # drop trailing silent e
    groups = re.findall(r"[aeiouy]+", w2)
    syl = len(groups)
    if re.search(r"[^aeiouy]le$", w) and not re.search(r"[aeiouy]{2}le$", w):
        syl += 1
    return max(1, syl)

def doc_stats(text: str) -> Dict[str, float]:
    ps = split_paragraphs(text)
    ss = split_sentences(text)
    ws = words(text)
    letters = sum(1 for c in text if _LETTER_RE.match(c))
    syllables = sum(count_syllables(w) for w in ws)
    reading_time_min = (len(ws) / 200.0) if ws else 0.0
    return {
        "total_words": float(len(ws)),
        "total_sentences": float(len(ss)),
        "total_paragraphs": float(len(ps)),
        "total_letters": float(letters),
        "total_syllables": float(syllables),
        "reading_time_minutes": float(reading_time_min),
    }

def readability_indices(text: str) -> Dict[str, float]:
    st = doc_stats(text)
    w = st["total_words"] or 1.0
    s = st["total_sentences"] or 1.0
    syll = st["total_syllables"] or 1.0
    letters = st["total_letters"] or 1.0

    wps = w / s
    spw = syll / w

    fre = 206.835 - 1.015 * wps - 84.6 * spw                 # Flesch Reading Ease
    fkg = 0.39 * wps + 11.8 * spw - 15.59                     # Flesch–Kincaid Grade

    complex_words = 0
    for word in words(text):
        if count_syllables(word) >= 3:
            complex_words += 1
    complex_ratio = complex_words / w

    fog = 0.4 * (wps + 100.0 * complex_ratio)                 # Gunning Fog

    if s < 30:
        smog = 1.0430 * math.sqrt(complex_words * (30.0 / max(1.0, s))) + 3.1291
    else:
        smog = 1.0430 * math.sqrt(complex_words * (30.0 / s)) + 3.1291

    L = (letters / w) * 100.0
    S = (s / w) * 100.0
    cli = 0.0588 * L - 0.296 * S - 15.8                       # Coleman–Liau

    return {
        "flesch_reading_ease": float(fre),
        "flesch_kincaid_grade": float(fkg),
        "gunning_fog": float(fog),
        "smog": float(smog),
        "coleman_liau": float(cli),
        "complex_words": float(complex_words),
    }


# ----------------- sentence structure -----------------
def sentence_structure(text: str) -> Dict[str, float]:
    ss = split_sentences(text)
    if not ss:
        return {k: 0.0 for k in [
            "mean_words_per_sentence","min_sentence_length","max_sentence_length","median_words_per_sentence",
            "mode_words_per_sentence","stdev_words_per_sentence",
            "len_0_5","len_6_10","len_11_15","len_16_20","len_21_25","len_26_30","len_31_plus",
            "share_len_31_plus"
        ]}
    lens = [len(words(s)) for s in ss]
    bins = {"len_0_5":0,"len_6_10":0,"len_11_15":0,"len_16_20":0,"len_21_25":0,"len_26_30":0,"len_31_plus":0}
    for L in lens:
        if L <= 5: bins["len_0_5"] += 1
        elif L <= 10: bins["len_6_10"] += 1
        elif L <= 15: bins["len_11_15"] += 1
        elif L <= 20: bins["len_16_20"] += 1
        elif L <= 25: bins["len_21_25"] += 1
        elif L <= 30: bins["len_26_30"] += 1
        else: bins["len_31_plus"] += 1

    mean = statistics.mean(lens)
    med = statistics.median(lens)
    try:
        mode = statistics.mode(lens)
    except statistics.StatisticsError:
        mode = int(round(mean))
    stdev = statistics.pstdev(lens) if len(lens) > 1 else 0.0

    share_31 = bins["len_31_plus"] / max(1, len(lens))
    return {
        "mean_words_per_sentence": float(mean),
        "min_sentence_length": float(min(lens)),
        "max_sentence_length": float(max(lens)),
        "median_words_per_sentence": float(med),
        "mode_words_per_sentence": float(mode),
        "stdev_words_per_sentence": float(stdev),
        **{k: float(v) for k, v in bins.items()},
        "share_len_31_plus": float(share_31),
    }


# ----------------- style + usage -----------------
def _is_probably_passive(sentence: str) -> bool:
    toks = [t.lower() for t in _WORD_RE.findall(sentence)]
    if len(toks) < 3:
        return False
    for i, t in enumerate(toks[:-1]):
        if t in _BE_VERBS:
            window = toks[i+1:i+8]
            for w in window:
                if w.endswith("ed") or w in _IRREG_PART:
                    return True
    return False

def complexity_style(text: str) -> Dict[str, float]:
    ws = words(text)
    ss = split_sentences(text)
    if not ws:
        return {
            "avg_letters_per_word": 0.0,
            "avg_syllables_per_word": 0.0,
            "passive_sentences": 0.0,
            "active_sentences": 0.0,
            "passive_ratio": 0.0,
        }

    letters = [len(re.findall(r"[A-Za-z]", w)) for w in ws]
    syll = [count_syllables(w) for w in ws]
    avg_letters = statistics.mean(letters) if letters else 0.0
    avg_syll = statistics.mean(syll) if syll else 0.0

    passive = sum(1 for s in ss if _is_probably_passive(s))
    active = max(0, len(ss) - passive)
    ratio = (passive / len(ss)) if ss else 0.0

    return {
        "avg_letters_per_word": float(avg_letters),
        "avg_syllables_per_word": float(avg_syll),
        "passive_sentences": float(passive),
        "active_sentences": float(active),
        "passive_ratio": float(ratio),
    }

def word_usage(text: str) -> Dict[str, float]:
    ws = [w.lower() for w in words(text)]
    if not ws:
        return {
            "count_we": 0.0, "count_you": 0.0, "ratio_we_to_you": 0.0,
            "hedging_words": 0.0, "subordinate_count": 0.0,
            "high_frequency_words": 0.0, "medium_frequency_words": 0.0, "low_frequency_words": 0.0,
        }
    count_we = sum(1 for w in ws if w == "we")
    count_you = sum(1 for w in ws if w in {"you","your","you'll","youre"})
    ratio = (count_we / count_you) if count_you else float("inf") if count_we else 0.0
    hedging = sum(1 for w in ws if w in _HEDGING)
    sub = sum(1 for w in ws if w in _SUBORDINATORS)

    freq: Dict[str, int] = {}
    for w in ws:
        if w in _STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    n = len(items)
    if n == 0:
        high = med = low = 0
    else:
        high_cut = max(1, int(round(n * 0.2)))
        med_cut = max(high_cut + 1, int(round(n * 0.6)))
        high = len(items[:high_cut])
        med = len(items[high_cut:med_cut])
        low = len(items[med_cut:])

    return {
        "count_we": float(count_we),
        "count_you": float(count_you),
        "ratio_we_to_you": float(ratio),
        "hedging_words": float(hedging),
        "subordinate_count": float(sub),
        "high_frequency_words": float(high),
        "medium_frequency_words": float(med),
        "low_frequency_words": float(low),
    }


# ----------------- interpretation helpers (human-friendly) -----------------
def flesch_band(fre: float) -> str:
    # common interpretation bands (kept simple)
    if fre >= 90: return "Very easy"
    if fre >= 80: return "Easy"
    if fre >= 70: return "Fairly easy"
    if fre >= 60: return "Standard"
    if fre >= 50: return "Fairly difficult"
    if fre >= 30: return "Difficult"
    return "Very difficult"

def grade_level_label(x: float) -> str:
    # For Fog / FK grade etc: lower is easier; round to 1dp but give plain range label
    if x <= 6: return "Very easy (approx. primary / early secondary)"
    if x <= 10: return "Easy–standard (approx. secondary)"
    if x <= 13: return "Harder (college entry level)"
    if x <= 16: return "Hard (undergraduate)"
    return "Very hard (specialist / academic)"

def interpretation_bundle(read: Dict[str, float], sent: Dict[str, float], style: Dict[str, float], stats: Dict[str, float]) -> Dict[str, str]:
    fre = read["flesch_reading_ease"]
    fog = read["gunning_fog"]
    fk = read["flesch_kincaid_grade"]
    long_share = sent.get("share_len_31_plus", 0.0)

    return {
        "flesch_reading_ease_band": flesch_band(fre),
        "flesch_reading_ease_what_good_looks_like": "Aim for ~55–75 for most marketing/B2B content. 70+ is generally very accessible; 30–50 reads more like dense reports.",
        "gunning_fog_simple": f"Estimated education level needed to understand the text on first read. Target ~8–12 for most client-facing writing. (Current: {fog:.1f})",
        "gunning_fog_band": grade_level_label(fog),
        "flesch_kincaid_band": grade_level_label(fk),
        "sentence_length_simple": f"Average sentence length is {sent.get('mean_words_per_sentence',0):.1f} words. Target ~14–20 for clarity in most marketing copy.",
        "very_long_sentences_simple": f"{int(sent.get('len_31_plus',0))} sentence(s) are 31+ words ({long_share*100:.0f}% of all sentences). Try to keep this under ~10%.",
        "passive_voice_simple": f"Approx. {style.get('passive_ratio',0)*100:.0f}% of sentences look passive. Under ~10–15% is a good benchmark for punchy, clear writing.",
        "reading_time_simple": f"Estimated reading time is {stats.get('reading_time_minutes',0):.1f} minutes at ~200 words/min.",
    }


# ----------------- scoring helpers -----------------
def _score_range(value: float, target_lo: float, target_hi: float, hard_lo: float, hard_hi: float) -> float:
    """Score 0..100 where best band is [target_lo, target_hi]. Outside [hard_lo, hard_hi] -> 0."""
    if value <= hard_lo or value >= hard_hi:
        return 0.0
    if target_lo <= value <= target_hi:
        return 100.0
    if value < target_lo:
        return 100.0 * (value - hard_lo) / (target_lo - hard_lo)
    return 100.0 * (hard_hi - value) / (hard_hi - target_hi)

def _score_low_is_good(value: float, target: float, hard: float) -> float:
    if value <= target:
        return 100.0
    if value >= hard:
        return 0.0
    return 100.0 * (hard - value) / (hard - target)


@dataclass
class ScorecardResult:
    readability_score: float
    sentence_structure_score: float
    style_score: float
    word_usage_score: float
    overall_score: float

    readability: Dict[str, float]
    general_stats: Dict[str, float]
    sentence_structure: Dict[str, float]
    style: Dict[str, float]
    word_usage: Dict[str, float]

    interpretations: Dict[str, str]
    improvements: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def score_text(text: str) -> ScorecardResult:
    text = _clean_text(text)

    stats = doc_stats(text)
    read = readability_indices(text)
    sent = sentence_structure(text)
    style = complexity_style(text)
    usage = word_usage(text)

    # --- scoring defaults for client-facing marketing/B2B copy ---
    # Readability scoring: emphasise Flesch Reading Ease (it’s the most intuitive for users).
    s_fre = _score_range(read["flesch_reading_ease"], target_lo=55, target_hi=75, hard_lo=10, hard_hi=95)
    s_fk  = _score_range(read["flesch_kincaid_grade"], target_lo=6, target_hi=10, hard_lo=2, hard_hi=18)
    s_fog = _score_range(read["gunning_fog"], target_lo=8, target_hi=12, hard_lo=4, hard_hi=22)
    s_smog= _score_range(read["smog"], target_lo=8, target_hi=12, hard_lo=4, hard_hi=22)
    s_cli = _score_range(read["coleman_liau"], target_lo=6, target_hi=11, hard_lo=2, hard_hi=20)

    readability_score = (
        0.45 * s_fre +
        0.1375 * s_fk +
        0.1375 * s_fog +
        0.1375 * s_smog +
        0.1375 * s_cli
    )

    # Sentence structure: avoid "all or nothing" zeros; allow more headroom on longer-form writing.
    s_mean = _score_range(sent["mean_words_per_sentence"], target_lo=14, target_hi=20, hard_lo=6, hard_hi=45)
    long_share = sent.get("share_len_31_plus", 0.0)
    s_long = _score_low_is_good(long_share, target=0.10, hard=0.60)  # keep 31+ under ~10%, but don't hard-zero until it's extreme
    sentence_structure_score = statistics.mean([s_mean, s_long])

    # Style: passive low, syllables per word moderate, letters per word moderate
    s_passive = _score_low_is_good(style["passive_ratio"], target=0.12, hard=0.40)
    s_syllpw = _score_range(style["avg_syllables_per_word"], target_lo=1.3, target_hi=1.6, hard_lo=1.0, hard_hi=2.3)
    s_ltrpw  = _score_range(style["avg_letters_per_word"], target_lo=4.2, target_hi=5.2, hard_lo=3.2, hard_hi=7.0)
    style_score = statistics.mean([s_passive, s_syllpw, s_ltrpw])

    # Word usage: more "you" than "we" (client focus), less hedging, less clause density
    ratio = usage["ratio_we_to_you"]
    if math.isinf(ratio):
        s_ratio = 10.0
    else:
        if ratio <= 0.7:
            s_ratio = 100.0
        elif ratio <= 1.2:
            s_ratio = 100.0 - (ratio - 0.7) / (1.2 - 0.7) * 30.0
        elif ratio <= 2.0:
            s_ratio = 70.0 - (ratio - 1.2) / (2.0 - 1.2) * 50.0
        else:
            s_ratio = 20.0

    w = stats["total_words"] or 1.0
    hedge_rate = usage["hedging_words"] / w * 1000.0
    s_hedge = _score_low_is_good(hedge_rate, target=6.0, hard=25.0)

    sub_rate = usage["subordinate_count"] / w * 1000.0
    s_sub = _score_low_is_good(sub_rate, target=10.0, hard=35.0)

    word_usage_score = statistics.mean([s_ratio, s_hedge, s_sub])

    # Overall (weighted): prioritise readability and sentence clarity
    overall = (
        0.40 * readability_score +
        0.25 * sentence_structure_score +
        0.20 * style_score +
        0.15 * word_usage_score
    )

    interp = interpretation_bundle(read, sent, style, stats)
    improvements = build_improvements(stats, read, sent, style, usage)

    return ScorecardResult(
        readability_score=float(readability_score),
        sentence_structure_score=float(sentence_structure_score),
        style_score=float(style_score),
        word_usage_score=float(word_usage_score),
        overall_score=float(overall),
        readability=read,
        general_stats=stats,
        sentence_structure=sent,
        style=style,
        word_usage=usage,
        interpretations=interp,
        improvements=improvements,
    )


def build_improvements(stats, read, sent, style, usage) -> List[str]:
    tips: List[Tuple[float, str]] = []

    fre = read["flesch_reading_ease"]
    fog = read["gunning_fog"]
    mean_len = sent.get("mean_words_per_sentence", 0.0)
    long_n = int(sent.get("len_31_plus", 0))
    long_share = sent.get("share_len_31_plus", 0.0)
    passive_pct = style.get("passive_ratio", 0.0) * 100.0

    # Prioritise clarity cues
    if mean_len > 22:
        tips.append((95, f"Sentence length is high (avg {mean_len:.1f} words). Aim for ~14–20 for clearer marketing copy."))
    if long_n >= 1:
        tips.append((90, f"There are {long_n} very long sentence(s) (31+ words) — about {long_share*100:.0f}% of all sentences. Split them into one idea per sentence."))
    if fre < 50:
        tips.append((92, f"Flesch Reading Ease is low ({fre:.1f} = {flesch_band(fre)}). Shorten sentences and swap jargon for plainer alternatives."))
    if fog > 14:
        tips.append((88, f"Gunning Fog is high ({fog:.1f}). This usually means long sentences + lots of 3+ syllable words. Tighten phrasing and simplify word choice."))
    if passive_pct > 18 and stats["total_sentences"] >= 5:
        tips.append((75, f"Passive voice looks a bit high (~{passive_pct:.0f}%). Use more direct, active constructions where possible."))

    # Reader focus + confidence
    w = stats["total_words"] or 1.0
    hedge_rate = usage["hedging_words"] / w * 1000.0
    if hedge_rate > 12:
        tips.append((70, "Lots of hedging words detected (e.g. might/could/perhaps). Replace with confident, specific claims when you have evidence."))

    ratio = usage["ratio_we_to_you"]
    if not math.isinf(ratio) and ratio > 1.2:
        tips.append((60, "Too much 'we' vs 'you'. Reframe lines around reader outcomes (time saved, clarity, confidence, revenue, risk reduced)."))

    # Always include one general “how to improve”
    tips.append((50, "Quick win: split long sentences, cut filler words, and keep paragraphs to 2–4 lines for easier scanning."))

    tips_sorted = sorted(tips, key=lambda x: -x[0])
    out, seen = [], set()
    for _, t in tips_sorted:
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= 6:
            break
    return out
