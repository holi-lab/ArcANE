"""Prompt templates for the role-play call and for chapter summarization.

Prompt conventions:
  • No response-format constraints (no "respond with action/speech/thought").
  • No "do not invoke knowledge beyond" clause — the context block alone
    represents what the character is allowed to draw on.
  • A relational-axis trial mentions the interlocutor in the persona slot.
"""

from typing import Optional

from .loaders import Trial


# ── Role-play system / user templates ──────────────────────────────
def render_system(trial: Trial, context: Optional[str]) -> str:
    parts: list[str] = []
    parts.append(
        f'You are {trial.character}, from "{trial.novel}". '
        f"You are at the point in the story corresponding to chapter "
        f"{trial.query_chapter}."
    )
    if trial.axis_type == "relational" and trial.target_character:
        parts.append(f"You are interacting with {trial.target_character}.")
    if context and context.strip():
        parts.append(
            "Background you have access to:\n"
            f"<context>\n{context.strip()}\n</context>"
        )
    return "\n\n".join(parts)


def render_user(trial: Trial, hint: Optional[str] = None) -> str:
    base = f"Scenario:\n{trial.scenario}\n\nQuestion:\n{trial.question}"
    if hint and hint.strip():
        return f"{base}\n(HINT: {hint.strip()})"
    return base


# ── TimeChara expert templates ─────────────────────────────────────
# Two-stage hint pipeline mirroring ahnjaewoo/timechara's narrative-experts.
# Adapted to our single-novel, chapter-indexed setting: temporal expert maps
# the scenario to a chapter number; spatial expert decides whether the
# character is present in the scene.

def render_timechara_temporal_user(trial: Trial, total_chapters: int) -> str:
    return (
        f'You will be given a scenario and a question from the novel "{trial.novel}". '
        f"The novel has {total_chapters} chapters. Your task is to identify which "
        f"chapter contains the scene the scenario describes.\n"
        f"***\n"
        f"[Scenario]\n{trial.scenario}\n"
        f"[Question]\n{trial.question}\n"
        f"***\n"
        f"[Steps]\n"
        f"1. Recall the scene from the scenario and describe it using the six Ws "
        f"(Who, What, When, Where, Why, How).\n"
        f"2. Identify the single chapter number (1-{total_chapters}) that contains "
        f"this scene. If you cannot determine it, output \"unknown\".\n\n"
        f"First, reason step by step. Then on its own final line, output ONLY "
        f"\"Chapter: N\" (an integer) or \"Chapter: unknown\"."
    )


def render_timechara_spatial_user(trial: Trial) -> str:
    return (
        f'You will be given a scenario, a question, and a character from "{trial.novel}". '
        f"Your task is to classify whether the character is present in the scene "
        f"described.\n"
        f"***\n"
        f"[Scenario]\n{trial.scenario}\n"
        f"[Question]\n{trial.question}\n"
        f"[Character]\n{trial.character}\n"
        f"***\n"
        f"[Steps]\n"
        f"1. Recall the scene and list every character involved, including those "
        f"present but unmentioned.\n"
        f"2. Decide whether {trial.character} is among them.\n\n"
        f"First, reason step by step. Then on its own final line, output ONLY "
        f"\"Presence: present\" or \"Presence: absent\"."
    )


TIMECHARA_EXPERT_SYSTEM = "You are a helpful and accurate assistant."


def timechara_future_hint(character: str) -> str:
    return (
        f"Note that the period of the question is in the future relative to "
        f"{character}'s time point. Therefore, you should not answer the question "
        f"or mention any facts that occurred after {character}'s time point."
    )


def timechara_absent_hint(character: str) -> str:
    return (
        f"Note that {character} had not participated in the scene described in the "
        f"question. Therefore, you should not imply that {character} was present "
        f"in the scene."
    )


# ── Chapter-summary builder prompt ─────────────────────────────────
SUMMARY_SYSTEM = (
    "You are a literary summarizer. Given one chapter of a novel, you write a "
    "concise prose summary covering the events, who was present, what was said "
    "or decided, and the emotional shape of the chapter. You stay faithful to "
    "the text and do not invent details."
)

def render_summary_user(novel: str, chapter_idx: int, chapter_text: str) -> str:
    return (
        f"Novel: {novel}\nChapter: {chapter_idx}\n\n"
        f"--- chapter text ---\n{chapter_text}\n--- end ---\n\n"
        "Write the summary as one or two paragraphs of running prose."
    )
