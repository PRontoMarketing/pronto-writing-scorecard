
"""
PRonto Writing Scorecard (MVP)

- Accepts extracted plain text
- Computes metrics inspired by the user's reference (TheWriter):
  Readability Indices, General Document Statistics,
  Sentence Structure Analysis, Complexity & Style Metrics, Word Usage
- Produces section scores + overall score + improvement suggestions

Note: Some metrics (passive voice, word frequency bands) are heuristic.
"""
from __future__ import annotations

import re
import math
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_LETTER_RE = re.compile(r"[A-Za-z]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

# Simple stopword list (kept short on purpose)
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

# A tiny irregular participle list to improve passive detection
_IRREG_PART = {
    "known","given","seen","done","made","taken","built","found","kept","held","said","shown","told",
    "written","driven","broken","chosen","grown","thrown","caught","bought","brought","thought",
    "paid","sent","spent","gone","won","understood","set","put","read","left","felt","cut","hit",
}

def _clean_text(text: str) -> str:
    # normalise whitespace
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
    # Avoid splitting on abbreviations a bit
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

def count_syllables(word: str) -> int:
    """Heuristic syllable counter (good enough for scoring trends)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    # special cases
    if len(w) <= 3:
        return 1
    # remove trailing silent e
    w2 = re.sub(r"e$", "", w)
    # count vowel groups
    groups = re.findall(r"[aeiouy]+", w2)
    syl = len(groups)
    # adjust for -le endings (e.g., "table")
    if re.search(r"[^aeiouy]le$", w) and not re.search(r"[aeiouy]{2}le$", w):
        syl += 1
    return max(1, syl)

def doc_stats(text: str) -> Dict[str, float]:
    ps = split_paragraphs(text)
    ss = split_sentences(text)
    ws = words(text)
    letters = sum(1 for c in text if _LETTER_RE.match(c))
    syllables = sum(count_syllables(w) for w in ws)
    # reading time: 200 wpm typical
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

    # Flesch Reading Ease
    fre = 206.835 - 1.015 * wps - 84.6 * spw

    # Flesch-Kincaid Grade
    fkg = 0.39 * wps + 11.8 * spw - 15.59

    # Complex words (>=3 syllables)
    complex_words = 0
    for word in words(text):
        if count_syllables(word) >= 3:
            complex_words += 1
    complex_ratio = complex_words / w

    # Gunning Fog
    fog = 0.4 * (wps + 100.0 * complex_ratio)

    # SMOG
    # Use standard formula; handle small sentence counts by scaling.
    if s < 30:
        smog = 1.0430 * math.sqrt(complex_words * (30.0 / max(1.0, s))) + 3.1291
    else:
        smog = 1.0430 * math.sqrt(complex_words * (30.0 / s)) + 3.1291

    # Coleman-Liau
    L = (letters / w) * 100.0
    S = (s / w) * 100.0
    cli = 0.0588 * L - 0.296 * S - 15.8

    return {
        "flesch_reading_ease": float(fre),
        "flesch_kincaid_grade": float(fkg),
        "gunning_fog": float(fog),
        "smog": float(smog),
        "coleman_liau": float(cli),
        "complex_words": float(complex_words),
    }

def sentence_structure(text: str) -> Dict[str, float]:
    ss = split_sentences(text)
    if not ss:
        return {k: 0.0 for k in [
            "mean_words_per_sentence","min_sentence_length","max_sentence_length","median_words_per_sentence",
            "mode_words_per_sentence","stdev_words_per_sentence",
            "len_0_5","len_6_10","len_11_15","len_16_20","len_21_25","len_26_30","len_31_plus"
        ]}
    lens = [len(words(s)) for s in ss]
    # bins
    bins = {
        "len_0_5": 0, "len_6_10": 0, "len_11_15": 0, "len_16_20": 0, "len_21_25": 0, "len_26_30": 0, "len_31_plus": 0
    }
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
        # multimodal -> pick rounded mean
        mode = int(round(mean))

    stdev = statistics.pstdev(lens) if len(lens) > 1 else 0.0
    return {
        "mean_words_per_sentence": float(mean),
        "min_sentence_length": float(min(lens)),
        "max_sentence_length": float(max(lens)),
        "median_words_per_sentence": float(med),
        "mode_words_per_sentence": float(mode),
        "stdev_words_per_sentence": float(stdev),
        **{k: float(v) for k, v in bins.items()}
    }

def _is_probably_passive(sentence: str) -> bool:
    """
    Heuristic passive detector:
    - be-verb + participle (word ending in ed OR in small irregular list)
    - optionally "by" later, but not required
    """
    toks = [t.lower() for t in _WORD_RE.findall(sentence)]
    if len(toks) < 3:
        return False
    # find be-verb positions
    for i, t in enumerate(toks[:-1]):
        if t in _BE_VERBS:
            window = toks[i+1:i+8]  # next few tokens
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

    passive = 0
    for s in ss:
        if _is_probably_passive(s):
            passive += 1
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
    count_you = sum(1 for w in ws if w == "you" or w == "your" or w == "you'll" or w == "youre")
    ratio = (count_we / count_you) if count_you else float("inf") if count_we else 0.0

    hedging = sum(1 for w in ws if w in _HEDGING)

    # subordinate conjunctions (rough proxy for clause density)
    sub = sum(1 for w in ws if w in _SUBORDINATORS)

    # frequency bands (within-document, excluding stopwords)
    freq: Dict[str, int] = {}
    for w in ws:
        if w in _STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    if not freq:
        return {
            "count_we": float(count_we), "count_you": float(count_you), "ratio_we_to_you": float(ratio),
            "hedging_words": float(hedging), "subordinate_count": float(sub),
            "high_frequency_words": 0.0, "medium_frequency_words": 0.0, "low_frequency_words": 0.0,
        }
    # sort by descending count
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    n = len(items)
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

def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))

def _score_range(value: float, target_lo: float, target_hi: float, hard_lo: float, hard_hi: float) -> float:
    """
    Score 0..100 where the best band is [target_lo, target_hi].
    Below hard_lo or above hard_hi -> 0.
    Linear between.
    """
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

def _score_high_is_good(value: float, target: float, hard: float) -> float:
    if value >= target:
        return 100.0
    if value <= hard:
        return 0.0
    return 100.0 * (value - hard) / (target - hard)

@dataclass
class ScorecardResult:
    # section scores
    readability_score: float
    sentence_structure_score: float
    style_score: float
    word_usage_score: float
    overall_score: float

    # raw metrics
    readability: Dict[str, float]
    general_stats: Dict[str, float]
    sentence_structure: Dict[str, float]
    style: Dict[str, float]
    word_usage: Dict[str, float]

    # improvement suggestions
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

    # --- scoring rules (MVP defaults for client-facing marketing copy) ---
    # Readability:
    s_fre = _score_range(read["flesch_reading_ease"], target_lo=55, target_hi=75, hard_lo=10, hard_hi=95)
    s_fk  = _score_range(read["flesch_kincaid_grade"], target_lo=6, target_hi=10, hard_lo=2, hard_hi=18)
    s_fog = _score_range(read["gunning_fog"], target_lo=8, target_hi=12, hard_lo=4, hard_hi=20)
    s_smog= _score_range(read["smog"], target_lo=8, target_hi=12, hard_lo=4, hard_hi=20)
    s_cli = _score_range(read["coleman_liau"], target_lo=6, target_hi=11, hard_lo=2, hard_hi=18)
    readability_score = statistics.mean([s_fre, s_fk, s_fog, s_smog, s_cli])

    # Sentence structure: prefer mean 14-20, discourage too many 31+
    s_mean = _score_range(sent["mean_words_per_sentence"], target_lo=14, target_hi=20, hard_lo=6, hard_hi=32)
    long_share = (sent["len_31_plus"] / stats["total_sentences"]) if stats["total_sentences"] else 0.0
    s_long = _score_low_is_good(long_share, target=0.05, hard=0.25)  # 5% ok, 25% bad
    sentence_structure_score = statistics.mean([s_mean, s_long])

    # Style: passive low, syllables per word moderate, letters per word moderate
    s_passive = _score_low_is_good(style["passive_ratio"], target=0.10, hard=0.35)
    s_syllpw = _score_range(style["avg_syllables_per_word"], target_lo=1.3, target_hi=1.6, hard_lo=1.0, hard_hi=2.2)
    s_ltrpw  = _score_range(style["avg_letters_per_word"], target_lo=4.2, target_hi=5.2, hard_lo=3.2, hard_hi=6.8)
    style_score = statistics.mean([s_passive, s_syllpw, s_ltrpw])

    # Word usage: more "you" than "we" (client focus), less hedging, less clause density
    # ratio_we_to_you: best <=0.7, ok up to 1.2, poor above 2
    ratio = usage["ratio_we_to_you"]
    if math.isinf(ratio):
        s_ratio = 10.0  # only we, no you
    else:
        # treat ratio low as good
        if ratio <= 0.7:
            s_ratio = 100.0
        elif ratio <= 1.2:
            s_ratio = 100.0 - (ratio - 0.7) / (1.2 - 0.7) * 30.0
        elif ratio <= 2.0:
            s_ratio = 70.0 - (ratio - 1.2) / (2.0 - 1.2) * 50.0
        else:
            s_ratio = 20.0
    # hedging per 1000 words
    w = stats["total_words"] or 1.0
    hedge_rate = usage["hedging_words"] / w * 1000.0
    s_hedge = _score_low_is_good(hedge_rate, target=6.0, hard=25.0)

    sub_rate = usage["subordinate_count"] / w * 1000.0
    s_sub = _score_low_is_good(sub_rate, target=10.0, hard=35.0)

    word_usage_score = statistics.mean([s_ratio, s_hedge, s_sub])

    # Overall: equal weights by default
    overall = statistics.mean([readability_score, sentence_structure_score, style_score, word_usage_score])

    improvements = build_improvements(stats, read, sent, style, usage,
                                     readability_score, sentence_structure_score, style_score, word_usage_score)

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
        improvements=improvements,
    )

def build_improvements(stats, read, sent, style, usage, rs, ss, st, wu) -> List[str]:
    tips: List[Tuple[float, str]] = []

    # Lower score -> higher priority (invert)
    tips.append((100 - rs, f"Readability is lower than ideal. Aim for a Flesch Reading Ease of ~55–75 by shortening sentences and swapping jargon for plain alternatives."))
    tips.append((100 - ss, f"Sentence structure could be tighter. Keep most sentences in the 11–20 word range and split any 31+ word sentences."))
    tips.append((100 - st, f"Style could be clearer. Reduce passive voice and favour direct, active constructions."))
    tips.append((100 - wu, f"Make it more reader-focused. Use 'you/your' to address the reader and cut hedging words (e.g. 'generally', 'perhaps')."))

    # Specific triggers
    if sent["len_31_plus"] >= 1:
        tips.append((85, f"There are {int(sent['len_31_plus'])} very long sentence(s) (31+ words). Split them and keep one idea per sentence."))

    if style["passive_ratio"] > 0.18 and stats["total_sentences"] >= 5:
        tips.append((90, f"Passive voice is relatively high (~{style['passive_ratio']*100:.0f}%). Rewrite key lines so the subject does the action (e.g. 'We improved…' -> 'You get…' or 'The team improved…')."))

    if read["flesch_reading_ease"] < 45:
        tips.append((95, f"Reading ease is quite low ({read['flesch_reading_ease']:.1f}). Reduce sentence length and break up dense paragraphs."))

    if read["gunning_fog"] > 14:
        tips.append((90, f"Gunning Fog is high ({read['gunning_fog']:.1f}). Reduce complex (3+ syllable) words and simplify phrasing."))

    if usage["hedging_words"] / (stats["total_words"] or 1.0) * 1000.0 > 12:
        tips.append((80, f"Lots of hedging words detected. Replace 'might/could/perhaps' with confident, specific claims when you have evidence."))

    if usage["ratio_we_to_you"] != 0 and not math.isinf(usage["ratio_we_to_you"]) and usage["ratio_we_to_you"] > 1.2:
        tips.append((75, f"Too much 'we' vs 'you'. Reframe benefits around the reader (outcomes, savings, time, confidence)."))

    # Sort by priority desc and dedupe
    tips_sorted = sorted(tips, key=lambda x: -x[0])
    out = []
    seen = set()
    for _, t in tips_sorted:
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= 6:
            break
    return out
