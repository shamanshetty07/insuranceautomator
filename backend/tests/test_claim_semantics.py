import itertools

import pytest

from app.services import claim_semantics as cs
from app.services.claim_semantics import MAX_CUES, Assertion, AssertionResult, assertion_of

POS, NEG, UNC, MIX, UNCL = (Assertion.POSITIVE, Assertion.NEGATED, Assertion.UNCERTAIN, Assertion.MIXED,
                            Assertion.UNCLEAR)


def a(text):
    return assertion_of(text).assertion


@pytest.mark.parametrize("text,expected,cue", [
    ("There is no dent on the door.", NEG, "negation:no"),
    ("The bumper appears undamaged.", NEG, "negation:undamaged"),
    ("No visible scratch is present.", NEG, "negation:no"),
    ("The windshield may be cracked.", UNC, "hedge:may"),
    ("The door is damaged, but the bumper is not.", MIX, "contrast:but"),
])
def test_task_examples(text, expected, cue):
    r = assertion_of(text)
    assert r.assertion is expected and cue in r.cues and not r.is_positive


@pytest.mark.parametrize("text", [
    "There is a dent on the front door.",
    "A scratch is visible on the rear bumper.",
    "The windshield is cracked.",
    "The door is dented and scratched.",
    "Dent on the left front door",
    "A 0.5 cm dent on the door.",
    "Scratch in the clear coat.",
    "Nordic notch on another note.",
    "The hood shows a dent near the badge.",
    "There's a dent on the door's lower edge.",
    "  THE DOOR IS DENTED!  ",
])
def test_plain_positive(text):
    r = assertion_of(text)
    assert r == AssertionResult(POS, ()) and r.is_positive


@pytest.mark.parametrize("text,cue", [
    ("The door is not dented.", "negation:not"),
    ("Not damaged.", "negation:not"),
    ("The door was never dented.", "negation:never"),
    ("None of the panels are dented.", "negation:none"),
    ("Nothing is broken.", "negation:nothing"),
    ("The bumper is without scratches.", "negation:without"),
    ("Neither the door nor the bumper is dented.", "negation:neither"),
    ("The panel is free of dents.", "negation:free"),
    ("Damage-free rear panel.", "negation:free"),
    ("Absence of scratches on the hood.", "negation:absence"),
    ("The windshield is intact.", "negation:intact"),
    ("The mirror is unharmed.", "negation:unharmed"),
    ("The un-damaged hood.", "negation:un"),
    ("Zero dents on the door.", "negation:zero"),
    ("0 dents visible.", "negation:0"),
    ("No dent, scratch or crack.", "negation:no"),
    ("No dent and no scratch.", "negation:no"),
    ("The door appears to be not damaged.", "negation:not"),
])
def test_negation(text, cue):
    r = assertion_of(text)
    assert r.assertion is NEG and cue in r.cues


@pytest.mark.parametrize("text", [
    "The door isn't dented.",
    "The door isn’t dented.",
    "There aren't any scratches.",
    "The bumper doesn't show damage.",
    "The hood wasn't hit.",
    "The panel hasn't cracked.",
    "The door isnt dented.",
    "The door can't be described as dented.",
    "The glass won't be cracked.",
])
def test_contractions(text):
    r = assertion_of(text)
    assert r.assertion is not POS and any(c.startswith("negation:") for c in r.cues)


@pytest.mark.parametrize("text,cue", [
    ("The windshield might be cracked.", "hedge:might"),
    ("The door could be dented.", "hedge:could"),
    ("Possibly a dent on the door.", "hedge:possibly"),
    ("Possible scratch on the bumper.", "hedge:possible"),
    ("The door is likely dented.", "hedge:likely"),
    ("The hood is probably bent.", "hedge:probably"),
    ("Suspected crack in the windshield.", "hedge:suspected"),
    ("Potential dent on the roof.", "hedge:potential"),
    ("Damage on the door is unclear.", "hedge:unclear"),
    ("Marks consistent with hail damage.", "hedge:consistent"),
    ("Looks like a dent on the door.", "hedge:like"),
    ("The door seems to be dented.", "perception:seems"),
    ("The door appears to be dented.", "perception:appears"),
    ("The door appears dented.", "perception:appears"),
    ("Whether the hood is bent cannot be determined.", "hedge:cannot_determine"),
    ("It cannot be confirmed that the door is dented.", "hedge:cannot_determine"),
    ("The door may not be dented.", "hedge:may"),
    ("If the door is dented, it is minor.", "hedge:if"),
    ("The owner reportedly dented the door.", "hedge:reportedly"),
    ("According to the owner, the door is dented.", "hedge:according"),
    ("If the door is dented and the hood is bent, both need review.", "hedge:if"),
])
def test_uncertainty(text, cue):
    r = assertion_of(text)
    assert r.assertion is UNC and cue in r.cues


@pytest.mark.parametrize("text,expected", [
    ("The door is dented and the bumper is not.", MIX),
    ("The door is dented and not scratched.", MIX),
    ("The bumper is undamaged and the door is dented.", MIX),
    ("The door is dented, the bumper is not.", MIX),
    ("The door is dented however the hood is intact.", MIX),
    ("Although the hood is intact, the door is dented.", MIX),
    ("The door is dented whereas the roof is fine.", MIX),
    ("The door is dented and the bumper may be scratched.", MIX),
    ("The door is dented; the bumper is not.", MIX),
    ("There is no dent but there is a scratch.", MIX),
    ("The door is dented but the bumper is scratched.", UNCL),
    ("The door is dented while the hood is scratched.", UNCL),
    ("Dent on the door, except near the handle.", UNCL),
])
def test_contrast_and_mixed_polarity(text, expected):
    assert a(text) is expected


@pytest.mark.parametrize("text,cue", [
    ("", "empty"),
    ("   ", "empty"),
    ("...", "empty"),
    ("Is the door dented?", "question"),
    ("The door is dented?", "question"),
    ("Is there a dent on the door", "question"),
    ("The panel is not undamaged.", "double_negation"),
    ("The door isn't intact.", "double_negation"),
    ("The bumper is not free of scratches.", "double_negation"),
    ("No longer intact.", "double_negation"),
    ("Dent on the door. Scratch on the bumper.", "multiple_clauses"),
    ("Dent on the door.\nScratch on the bumper", "multiple_clauses"),
    ("The dent has been repaired.", "temporal:repaired"),
    ("Previously dented door.", "temporal:previously"),
    ("Dent or scratch on the door.", "alternative:or"),
    ("The bumper looks fine.", "condition:fine"),
    ("The paint is in good condition.", "condition:good"),
])
def test_ambiguous_language_is_unclear(text, cue):
    r = assertion_of(text)
    assert r.assertion is UNCL and cue in r.cues


def test_multi_sentence_all_negated_stays_negated():
    r = assertion_of("No dent on the door. No scratch on the bumper.")
    assert r.assertion is NEG and "multiple_clauses" in r.cues


@pytest.mark.parametrize("text", [
    "Notch on the door edge.",
    "Nordic blue paint is scratched.",
    "Another dent on the hood.",
    "Note the dent on the door.",
    "A knot-shaped dent.",
    "The nose of the car is dented.",
    "Know that the door is dented.",
    "Mayday dent on the roof.",
    "Couldron scratch.",
    "Butterfly-shaped crack on the windshield.",
    "Ornament scratch.",
    "Freely visible dent.",
    "Intactness-gauge dent.",
])
def test_substrings_do_not_trigger(text):
    assert a(text) is POS


@pytest.mark.parametrize("text", [
    "THERE IS NO DENT ON THE DOOR",
    "There Is No Dent On The Door!!!",
    "there is no dent, on the door",
    "(no dent on the door)",
    "\"No\" dent on the door.",
])
def test_casing_and_punctuation(text):
    assert a(text) is NEG


@pytest.mark.parametrize("text", [
    "There's no doubt the door is dented.",
    "Not only dented but also scratched.",
    "No longer intact.",
])
def test_known_safe_misclassifications_are_never_positive(text):
    assert a(text) is not POS


@pytest.mark.parametrize("text", [
    "There is no dent on the door.", "The door is dented and the bumper is not.", "", "Is it dented?",
    "The door is dented.",
])
def test_deterministic(text):
    assert len({assertion_of(text) for _ in range(5)}) == 1


def test_cues_are_capped():
    r = assertion_of(" ".join(["no may might could possibly probably likely"] * 3) + " " +
                     " ".join(sorted(cs._HEDGES)))
    assert len(r.cues) == MAX_CUES and r.cues[-1] == "cues_truncated"


def test_none_text_is_unclear():
    assert assertion_of(None) == AssertionResult(UNCL, ("empty",))


_TRIGGERS = sorted(
    cs._NEGATORS | cs._HEDGES | cs._PERCEPTION | cs._TEMPORAL | cs._CONDITION | cs._ALTERNATIVE | cs._CONTRAST
    | {"isn't", "doesn't", "can't", "won't", "there's no"}
)
_TEMPLATES = (
    "{} dent on the door.", "The door is {} dented.", "The door is dented {}.", "{}",
    "There is a dent and {} the bumper.", "Scratch, {}, on the hood",
)


@pytest.mark.parametrize("token", _TRIGGERS)
def test_any_negator_or_hedge_token_is_never_positive(token):
    for template, casing in itertools.product(_TEMPLATES, (str.lower, str.upper, str.title)):
        r = assertion_of(template.format(casing(token)))
        assert r.assertion is not POS and r.cues, (template, casing)


def test_positive_iff_no_cues():
    corpus = [
        "There is a dent on the door.", "No dent.", "May be dented.", "Dent; scratch.", "Dent?", "",
        "The door is dented and the bumper is not.", "Scratch in the clear coat.", "The hood is fine.",
    ]
    for text in corpus:
        r = assertion_of(text)
        assert r.is_positive == (not r.cues)


@pytest.mark.parametrize("member", list(Assertion))
def test_assertion_values_are_stable_strings(member):
    assert member.value == member.name.lower()
