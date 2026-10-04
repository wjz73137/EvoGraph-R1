"""Prompt helpers for E-VQA multimodal tool-use QA."""

from __future__ import annotations


def build_vqa_user_prompt(
    *,
    question: str,
    image_id: str = "",
    image_path: str = "",
) -> str:
    """Build the user-side VQA prompt used by multimodal records and smoke evals."""

    lines = [
        "Answer the question using the question image and internal knowledge base.",
        "",
        (
            "Always reason first inside <think>...</think>. If additional "
            "information is required, call a tool using "
            "<tool_call>...</tool_call> per the system-provided tool "
            "specifications and formats."
        ),
        "",
        "Required sequence:",
        "- Visual grounding runs first and returns candidate entities.",
        (
            "- You MUST then call kb_search with a natural-language text query "
            "combining the best candidate entity and the question cues."
        ),
        (
            "- After text evidence is returned, give the final answer immediately "
            "as the shortest answer span supported by the evidence."
        ),
        "- Use only kb_search; never invent or call another tool.",
        "",
        (
            "When you have the final answer, output it inside "
            "<answer>...</answer>. Put only the answer span inside the tags; do not "
            "repeat the entity name, explanation, or a full sentence."
        ),
        "",
    ]
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def build_vqa_system_prompt(tools_json: str) -> str:
    """Build a compact system prompt for real multimodal VQA agent smoke tests."""

    return "\n".join(
        [
            "You are a careful VQA agent that answers using tool-retrieved evidence.",
            "",
            "# Tools",
            "You may call one function at a time.",
            "Function signatures are provided within <tools></tools>:",
            "<tools>",
            tools_json,
            "</tools>",
            "",
            (
                '1) Visual grounding: Start with kb_search({"query":"<img>"}) '
                "to retrieve candidate entities. Take the top-ranked visual entity "
                "as the anchor."
            ),
            (
                "2) Factual lookup: Form a text query using the anchoring entity "
                "name and question cues, then call kb_search. Prefer multiple "
                "rounds of refined text kb_search to gather sufficient evidence."
            ),
            (
                "3) Fallback: Use websearch only when refined kb_search attempts "
                "remain clearly insufficient, irrelevant, missing key knowledge, "
                "or conflicting."
            ),
            (
                "4) Knowledge maintenance: After websearch, resolve conflicts with "
                "the KB and integrate new information by applying graph edit operations "
                "(insert, update, delete) as needed."
            ),
            "Do not use tools that are not listed.",
        ]
    )


def build_graph_edit_user_prompt(*, question: str) -> str:
    """Build the user prompt for the original retrieve/websearch/edit baseline."""

    return "\n".join(
        [
            "Answer the question using the image and the evolving internal knowledge base.",
            "",
            "Follow this state machine:",
            "1. Visual grounding runs first. The first returned entity is the required top-1 anchor; do not substitute a lower-ranked candidate.",
            "2. Call kb_search once with that entity and the original question cues. This text lookup is mandatory even if you think you know the answer from memory.",
            "3. If the returned KB evidence directly answers the question, answer immediately.",
            "4. If the key fact is missing, irrelevant, or conflicting, call websearch once with an entity-first query containing the anchored entity, a distinguishing location or identity detail from the KB, and the exact missing fact.",
            "5. Compare web evidence with the KB. Reject evidence about a namesake or conflicting identity. Only when the evidence is specific and reliable, maintain the graph: insert a missing atomic fact; update an exact old fact with a corrected fact; or delete an exact false fact.",
            "6. After a successful graph edit, call kb_search again to verify that the edited fact is searchable, then answer.",
            "",
            "Never repeat an identical tool call unless a graph edit changed the knowledge state. Do not edit uncertain evidence. Use only listed tools.",
            "Return the shortest supported answer span inside <answer>...</answer>; do not repeat the entity name or explanation there.",
            "",
            f"Question: {question}",
        ]
    )
