"""Instruction-following mini-suite.

A tiny fixed set of prompts with *programmatically verifiable* constraints
(word count, casing, forbidden characters, ending, ...).  This gives a
cheap, deterministic IF signal for small models long before IFBench/IFEval
are meaningful.  Every prompt, checker and the version string are frozen
here; results record the version so cross-run comparisons stay honest.
"""

from __future__ import annotations

import re
from typing import Optional

IF_MINI_VERSION = "if-mini-v1"


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


# (prompt, checker, description)
_CONSTRAINTS = [
    ("Reply with exactly three words.", lambda t: len(_words(t)) == 3, "exactly_3_words"),
    ("Reply with exactly five words.", lambda t: len(_words(t)) == 5, "exactly_5_words"),
    ("Reply with exactly one word.", lambda t: len(_words(t)) == 1, "exactly_1_word"),
    ("Answer using only lowercase letters and no punctuation.",
     lambda t: t == t.lower() and not re.search(r"[.,!?;:]", t), "lowercase_no_punct"),
    ("Answer in ALL CAPITAL LETTERS.",
     lambda t: t.upper() == t and any(c.isalpha() for c in t), "all_caps"),
    ("End your response with the word 'done'.",
     lambda t: _words(t)[-1].lower() == "done" if _words(t) else False, "ends_with_done"),
    ("End your response with a question mark.",
     lambda t: t.rstrip().endswith("?"), "ends_with_question"),
    ("Do not use the letter 'e' anywhere in your response.",
     lambda t: "e" not in t.lower(), "no_letter_e"),
    ("Reply with fewer than ten words.",
     lambda t: len(_words(t)) < 10, "under_10_words"),
    ("Reply with more than twenty words.",
     lambda t: len(_words(t)) > 20, "over_20_words"),
    ("Start your response with the word 'First'.",
     lambda t: t.lstrip().startswith("First"), "starts_with_first"),
    ("Reply with exactly two sentences.",
     lambda t: len([s for s in re.split(r"[.!?]+", t) if s.strip()]) == 2, "exactly_2_sentences"),
    ("Do not use any commas in your response.",
     lambda t: "," not in t, "no_commas"),
    ("Include the word 'spark' somewhere in your response.",
     lambda t: "spark" in t.lower(), "contains_spark"),
    ("Reply with a single word and nothing else.",
     lambda t: len(_words(t)) == 1 and len(_words(t.strip())) == 1, "single_word_only"),
    ("Answer without using the digit '5'.",
     lambda t: "5" not in t, "no_digit_5"),
    ("Write your response as one line without any newline characters.",
     lambda t: "\n" not in t, "single_line"),
    ("Reply with exactly ten words.",
     lambda t: len(_words(t)) == 10, "exactly_10_words"),
    ("Use at least two exclamation marks in your response.",
     lambda t: t.count("!") >= 2, "two_exclamations"),
    ("Respond with a rhyming pair of words separated by a comma, like 'cat, hat'.",
     lambda t: bool(re.match(r"^\s*\w+,\s*\w+\s*$", t)) and t.strip().split(",")[0].strip()[-2:] ==
                t.strip().split(",")[1].strip()[-2:], "rhyme_pair"),
]


class IFMiniEval:
    name = "if_mini"

    def run(self, model, tokenizer, device, limit: Optional[int] = None, **_) -> dict:
        prompts = _CONSTRAINTS[:limit] if limit else _CONSTRAINTS
        model.eval()
        n_pass = 0
        details = []
        for prompt, check, desc in prompts:
            # NOTE: imported here to avoid a circular import at module load
            from ..inference.generate import generate
            result = generate(model, prompt, tokenizer,
                              max_new_tokens=48, temperature=0.0)
            new = result["new_text"].strip()
            ok = bool(check(new))
            n_pass += int(ok)
            details.append({"prompt": prompt, "constraint": desc,
                            "response": new, "pass": ok})
        return {
            "task": "if_mini",
            "display_name": "IF mini-suite",
            "metric": "constraint_pass_rate",
            "acc": round(n_pass / len(prompts), 4),
            "chance": None,
            "n": len(prompts),
            "version": IF_MINI_VERSION,
            "decoding": "greedy",
            "details": details,
            "limit": limit,
        }