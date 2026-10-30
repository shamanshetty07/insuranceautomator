"""Lexical assertion classes for claim text: is it one plain positive statement, or something else?

Pure, deterministic, stdlib only. It describes the text and never rewrites it, never consults
evidence and never infers that damage is absent.

Safety asymmetry: only ``Assertion.POSITIVE`` may let a detection corroborate a claim. Calling a
negated, hedged or compound text POSITIVE could let a wrong ``supported`` verdict through (the
dangerous error); calling a plain positive text non-POSITIVE only sends the claim to review. So
every rule here errs toward non-POSITIVE, and ``POSITIVE`` is returned only when the text yields
no cue at all. The labels among the non-POSITIVE classes are best-effort descriptions; callers
must treat all of them alike (not corroborable) and must not read NEGATED as "no damage".
"""

import enum
import re
from dataclasses import dataclass


class Assertion(str, enum.Enum):
    POSITIVE = "positive"
    """One plain affirmative statement with no negation, hedge, contrast or second clause."""
    NEGATED = "negated"
    """The text denies something (e.g. "no dent", "appears undamaged"); not evidence of absence."""
    UNCERTAIN = "uncertain"
    """The text is hedged or speculative (e.g. "may be cracked", "appears to be dented")."""
    MIXED = "mixed"
    """Some clauses are plainly positive and others are not (e.g. "door dented, bumper not")."""
    UNCLEAR = "unclear"
    """Empty, a question, double negation, several clauses/sentences or otherwise undecidable."""


MAX_CUES = 16


@dataclass(frozen=True)
class AssertionResult:
    assertion: Assertion
    cues: tuple[str, ...]

    @property
    def is_positive(self) -> bool:
        """The only state in which a detection may corroborate the claim."""
        return self.assertion is Assertion.POSITIVE


_NEGATORS = frozenset({
    "no", "not", "never", "none", "nothing", "nowhere", "nobody", "without", "neither", "nor",
    "cannot", "zero", "0", "nil", "free", "absent", "absence", "lack", "lacks", "lacking", "devoid",
    "un", "non", "undamaged", "intact", "unharmed", "unscathed", "unblemished", "unaffected",
    "unmarked", "untouched", "unbroken", "undented", "unscratched", "uncracked", "unbent",
    "unimpaired", "flawless", "pristine",
    "isnt", "arent", "wasnt", "werent", "dont", "doesnt", "didnt", "hasnt", "havent", "hadnt",
    "cant", "wont", "couldnt", "wouldnt", "shouldnt", "aint",
})
_HEDGES = frozenset({
    "may", "might", "could", "would", "should", "must", "possibly", "possible", "perhaps", "maybe",
    "likely", "unlikely", "probably", "probable", "presumably", "presumed", "supposedly",
    "apparently", "seemingly", "suspected", "suspect", "suspicious", "potential", "potentially",
    "unclear", "uncertain", "unsure", "unknown", "undetermined", "indeterminate", "unconfirmed",
    "unverified", "ambiguous", "questionable", "if", "unless", "whether", "assume", "assumed",
    "believed", "believe", "think", "guess", "suggests", "suggest", "suggesting", "suggestive",
    "indicates", "indicate", "indicating", "indicative", "consistent", "resembles", "resemble",
    "resembling", "like", "almost", "nearly", "hardly", "barely", "scarcely", "allegedly", "alleged",
    "reportedly", "reported", "claimed", "according", "unable", "difficult",
})
_PERCEPTION = frozenset({
    "appears", "appear", "appearing", "seems", "seem", "looks", "look", "looking",
})
_EPISTEMIC = frozenset({
    "determine", "determined", "confirm", "confirmed", "verify", "verified", "assess", "assessed",
    "tell", "establish", "established", "ruled", "identify", "identified", "judge", "judged", "sure",
})
_TEMPORAL = frozenset({
    "repaired", "fixed", "restored", "refinished", "resprayed", "repainted", "previously",
    "formerly", "former", "prior", "earlier",
})
_CONDITION = frozenset({
    "fine", "good", "ok", "okay", "normal", "clean", "perfect", "acceptable", "sound",
})
_SENTENCE_SCOPED = frozenset({
    "if", "unless", "whether", "according", "reportedly", "allegedly", "alleged", "reported", "claimed",
})
_ALTERNATIVE = frozenset({"or"})
_CONTRAST = frozenset({
    "but", "however", "although", "though", "while", "whilst", "whereas", "yet", "except",
    "excepting", "despite", "nonetheless", "nevertheless", "unlike", "instead", "otherwise",
    "besides",
})
_CONTRAST_PAIRS = {"other": "than", "rather": "than", "apart": "from"}
_CLAUSE_VERBS = frozenset({
    "is", "are", "was", "were", "be", "been", "has", "have", "had", "shows", "show", "showing",
    "remains", "remain", "exhibits", "displays", "contains", "bears", "there", "there's", "it's",
    "it", "they",
}) | _PERCEPTION
_QUESTION_OPENERS = frozenset({
    "is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "can", "will", "any",
})

_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:'[^\W\d_]+)*|,")
_SENTENCE_BREAK = re.compile(r"(?:[.!]+(?=\s|$)|[;\n])+")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'"})


def _is_negator(tok: str) -> bool:
    return tok in _NEGATORS or tok.endswith("n't")


def _neg_cue(tok: str) -> str:
    return "negation:n't" if tok.endswith("n't") else f"negation:{tok}"


def _starts_new_clause(piece: list[str]) -> bool:
    if not piece:
        return False
    head = piece[0]
    if _is_negator(head) or head in _HEDGES or head in _PERCEPTION or head in _CONDITION:
        return True
    return any(t in _CLAUSE_VERBS or t.endswith("n't") for t in piece)


def _segments(tokens: list[str], cues: list[str]) -> list[list[str]]:
    """Split one sentence into clauses at contrast words, and at ","/"and" before a new clause."""
    segments: list[list[str]] = []
    current: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < n else None
        if tok in _CONTRAST or (tok in _CONTRAST_PAIRS and nxt == _CONTRAST_PAIRS[tok]):
            cues.append(f"contrast:{tok}")
            segments.append(current)
            current = []
            i += 2 if tok in _CONTRAST_PAIRS else 1
            continue
        if tok in (",", "and"):
            j = i + 1
            while j < n and tokens[j] not in (",", "and") and tokens[j] not in _CONTRAST:
                j += 1
            piece = [t for t in tokens[i + 1 : j] if t not in (",", "and")]
            if current and _starts_new_clause(piece):
                segments.append(current)
                current = []
            i += 1
            continue
        current.append(tok)
        i += 1
    segments.append(current)
    return [s for s in segments if s]


def _classify_segment(seg: list[str], cues: list[str]) -> Assertion:
    negators = [t for t in seg if _is_negator(t)]
    hedges = [t for t in seg if t in _HEDGES]
    epistemic = bool(negators) and any(t in _EPISTEMIC for t in seg)
    perception = [t for t in seg if t in _PERCEPTION]
    temporal = [t for t in seg if t in _TEMPORAL]
    alternative = [t for t in seg if t in _ALTERNATIVE]
    condition = [t for t in seg if t in _CONDITION]

    cues.extend(_neg_cue(t) for t in negators)
    cues.extend(f"hedge:{t}" for t in hedges)
    if epistemic:
        cues.append("hedge:cannot_determine")
    cues.extend(f"perception:{t}" for t in perception)
    cues.extend(f"temporal:{t}" for t in temporal)
    cues.extend(f"alternative:{t}" for t in alternative)
    cues.extend(f"condition:{t}" for t in condition)

    counted = [t for t in negators if t != "nor"]
    if len(counted) >= 2 and not epistemic:
        cues.append("double_negation")
        return Assertion.UNCLEAR
    if temporal:
        return Assertion.UNCLEAR
    if hedges or epistemic:
        return Assertion.UNCERTAIN
    if negators:
        return Assertion.NEGATED
    if alternative or condition:
        return Assertion.UNCLEAR
    if perception:
        return Assertion.UNCERTAIN
    return Assertion.POSITIVE


def _result(assertion: Assertion, cues: list[str]) -> AssertionResult:
    unique = list(dict.fromkeys(cues))
    if assertion is Assertion.POSITIVE and unique:
        assertion = Assertion.UNCLEAR
    if len(unique) > MAX_CUES:
        unique = [*unique[: MAX_CUES - 1], "cues_truncated"]
    return AssertionResult(assertion, tuple(unique))


def assertion_of(text: str) -> AssertionResult:
    """Classify the whole claim text; anything not confidently one positive assertion is non-POSITIVE."""
    norm = (text or "").casefold().translate(_APOSTROPHES)
    sentences = [_TOKEN.findall(s) for s in _SENTENCE_BREAK.split(norm)]
    sentences = [s for s in sentences if any(t != "," for t in s)]
    if not sentences:
        return _result(Assertion.UNCLEAR, ["empty"])

    cues: list[str] = []
    question = "?" in norm or sentences[0][0] in _QUESTION_OPENERS
    if question:
        cues.append("question")
    if len(sentences) > 1:
        cues.append("multiple_clauses")

    labels: list[Assertion] = []
    contrast = False
    for tokens in sentences:
        before = len(cues)
        segs = _segments(tokens, cues)
        contrast = contrast or any(c.startswith("contrast:") for c in cues[before:])
        found = [_classify_segment(seg, cues) for seg in segs]
        if _SENTENCE_SCOPED.intersection(tokens):
            found = [Assertion.UNCERTAIN if f is Assertion.POSITIVE else f for f in found]
        labels.extend(found)
    kinds = set(labels)

    if question or not labels:
        return _result(Assertion.UNCLEAR, cues)
    if Assertion.POSITIVE in kinds and len(kinds) > 1:
        return _result(Assertion.MIXED, cues)
    if kinds == {Assertion.POSITIVE}:
        if contrast or len(sentences) > 1:
            return _result(Assertion.UNCLEAR, cues)
        return _result(Assertion.POSITIVE, cues)
    if len(kinds) == 1:
        return _result(labels[0], cues)
    return _result(Assertion.UNCLEAR, cues)
