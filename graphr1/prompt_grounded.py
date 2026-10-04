"""Grounded, atomic extraction prompt for local Qwen graph construction.

This prompt is opt-in so the original GraphR1 baseline prompt remains intact.
"""

from __future__ import annotations

from .prompt import PROMPTS


GROUNDED_PROMPT_VERSION = "grounded-atomic-v3-review"

GROUNDED_ENTITY_EXTRACTION = """-Task-
Convert the source text into atomic hyper-relations and their entities.
Use {language} for all extracted text.

-Grounding rules-
1. Use ONLY information explicitly stated in the source text. Do not add definitions,
   background knowledge, inferred dates, inferred causes, or inferred identities.
2. Preserve uncertainty and alternatives exactly. Words such as "probably", "or",
   "until", "after", and "since" must not be strengthened or reassigned.
3. Each hyper-relation must express ONE atomic fact or one inseparable relation.
   Do not combine facts from adjacent sentences. Emit every explicit fact exactly once.
4. Never create a temporal or causal link merely because two sentences are adjacent.
5. Make every hyper-relation independently understandable. Every hyper-relation must
   state its subject by name; do not begin a record with a pronoun such as it, its,
   this, or there. Replace a pronoun only when its source reference is unambiguous.
6. Entity names must occur explicitly in the source. Entity descriptions may restate
   only the entity's explicit role in the current hyper-relation. Do not define the
   entity using outside knowledge.
7. Do not create a separate entity for an alias. Keep the alias only in the atomic
   fact about the named entity. In particular, never classify a structure alias as a
   person. Use only these broad types: person, organization, location, structure,
   event, time, object, group, or concept.
8. Do not duplicate semantically equivalent hyper-relations.
9. Output at least one hyper-relation for every non-empty source. Do not output only
   an end marker. Ignore a final source fragment if it is grammatically incomplete.

-Exact output grammar-
For each fact, output its hyper-relation first, followed immediately by the entities
that participate in that fact:
("hyper-relation"{tuple_delimiter}atomic fact{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}entity name{tuple_delimiter}entity type{tuple_delimiter}source-grounded role in this fact{tuple_delimiter}100){record_delimiter}

Use the delimiters exactly as shown. Do not quote field values. Do not add Markdown,
headings, explanations, an end marker, or any text outside the records.

-Example-
{examples}

-Source text-
{input_text}

-Output-
"""

GROUNDED_ENTITY_EXAMPLES = [
    """Text:
Harbor Hall became the first city clinic under British administration. After being
restored during the 1990s, Harbor Hall reopened as an arts centre.

Output:
("hyper-relation"{tuple_delimiter}Harbor Hall became the first city clinic under British administration.{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}Harbor Hall{tuple_delimiter}structure{tuple_delimiter}Harbor Hall became the first city clinic under British administration.{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}British administration{tuple_delimiter}organization{tuple_delimiter}British administration was the administration under which Harbor Hall became the first city clinic.{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}Harbor Hall was restored during the 1990s.{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}Harbor Hall{tuple_delimiter}structure{tuple_delimiter}Harbor Hall was restored during the 1990s.{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}1990s{tuple_delimiter}time{tuple_delimiter}The 1990s were the period explicitly stated for the restoration of Harbor Hall.{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}After its restoration during the 1990s, Harbor Hall reopened as an arts centre.{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}Harbor Hall{tuple_delimiter}structure{tuple_delimiter}Harbor Hall reopened as an arts centre after its restoration during the 1990s.{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}arts centre{tuple_delimiter}concept{tuple_delimiter}Arts centre is the stated new use of Harbor Hall after restoration.{tuple_delimiter}90){record_delimiter}"""
]

GROUNDED_REVIEW_PROMPT = """Audit the draft extraction against the source text in
the original user message, then return a COMPLETE REPLACEMENT extraction using the
same exact record grammar. Return records only.

Perform this checklist before answering:
1. Check every complete source sentence and clause; add every explicit fact that is
   missing from the draft.
2. Delete or correct every statement not entailed by the source. Pay special attention
   to who or what was seized, built, opened, restored, or renamed, and to exact dates.
3. Preserve uncertainty, alternatives, and temporal order. Do not turn "probably A or
   B" into two certain claims.
4. Every hyper-relation must state its subject by name; resolve unambiguous pronouns.
5. Split independent facts into separate hyper-relations without duplicating facts.
6. Do not create a separate entity for an alias. Never type a structure alias as a
   person. Entity descriptions must contain no facts beyond the source.
7. Ignore any incomplete final source fragment.
8. Output at least one hyper-relation and entity. Do not output an end marker.
"""


def activate_grounded_prompts() -> None:
    """Mutate the shared prompt registry for an isolated opt-in run."""
    PROMPTS.update(
        {
            "DEFAULT_TUPLE_DELIMITER": "@@",
            "DEFAULT_RECORD_DELIMITER": "##",
            "DEFAULT_COMPLETION_DELIMITER": "<COMPLETE>",
            "entity_extraction": GROUNDED_ENTITY_EXTRACTION,
            "entity_extraction_examples": GROUNDED_ENTITY_EXAMPLES,
        }
    )
