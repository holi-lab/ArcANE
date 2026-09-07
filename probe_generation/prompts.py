"""Generator and validator prompts.

Multi-phase per probe: one (Scenario, Question) anchored at a specific
chapter, with a response from EACH phase. Each phase response is built
"as if asked at this phase's query_chapter" — i.e., the character only
knows what has happened up to that chapter.

GT structure per phase: action + (optional) speech + thought + typicality.
The thought field captures HOW this phase's character processes/construes/
ruminates on the situation — the cognitive dimension that lets adjacent
phases differentiate even when action affordance is constrained (cf.
Mischel & Shoda 1995 CAPS: situational encoding is phase-conditional).
"""

from typing import Optional


# ── Role descriptions ───────────────────────────────────────────────
SYSTEM_CONSTRUCT_ANALYST = (
    "You are a personality construct analyst. Given a Character Arc — a "
    "labeled trajectory of phases with chapter ranges and positions — you name "
    "the SINGLE behavioral decision variable whose answer flips between "
    "phases. You think in terms of observable behavior and concrete choices, "
    "not abstract personality language."
)

SYSTEM_LIFE_STAGE = (
    "You are a literary-age analyst. Given a character's arc trajectory with "
    "chapter ranges and position descriptions, you assign each phase a "
    "developmental life-stage tag from a fixed taxonomy and a brief age "
    "estimate. You read chapter ranges and developmental cues in the position "
    "descriptions (school grade, marriage, retirement, parenting, etc.) rather "
    "than guessing."
)

SYSTEM_AXIS_RE_EXP = (
    "You are an abstract psychometric analyst. Given a Character Arc framed "
    "in a specific novel's setting, you re-express the arc in era-agnostic "
    "behavioral terms — abstract psychological vocabulary that could apply "
    "to any time and place. You strip the era-specific scaffolding but "
    "preserve the trait dynamics."
)

SYSTEM_IN_TEXT_DESIGNER = (
    "You are a narrative psychologist building a behavioral probe from a "
    "verbatim source passage. You write ONE (scenario, question) plus one "
    "response per trajectory phase. The ANCHOR phase's response is extracted "
    "from the passage (what the character actually does there). Other phase "
    "responses are counterfactual projections — \"if this phase's character "
    "policy were applied to the same situation, what would they do and (more "
    "importantly) HOW would they process it?\". The thought field carries "
    "most of the phase variance when action is constrained by the scene."
)

SYSTEM_IN_WORLD_DESIGNER = (
    "You are a narrative psychologist designing a multi-phase behavioral "
    "probe in the source novel's world. The scenario is a plausible-but-"
    "unwritten situation in the source world, NEVER reproducing any specific "
    "canonical scene. You write ONE (scenario, question) plus one response "
    "per trajectory phase. Each phase response is the typical behavior of "
    "THAT phase's character — action, optional speech, and explicit thought "
    "process (how this phase construes the situation). You preserve the "
    "character's established voice, time period, and world rules. Adjacent "
    "phases may legitimately respond similarly; do not stretch."
)

SYSTEM_OOW_DESIGNER = (
    "You are a narrative psychologist transposing a character's narrative "
    "identity into a NON-source era. You receive an explicit target era and "
    "a per-phase life-stage table. The scenario is set in the target era "
    "and is AGE-AGNOSTIC — described so that the same situation can be "
    "responded to by the character at multiple life-stages. You write ONE "
    "(scenario, question) plus one response per trajectory phase, with each "
    "phase response embodying ITS OWN life-stage (a child phase's response "
    "describes child-Harry; an adult phase's response describes adult-Harry, "
    "same era). The PROTAGONIST is the named character throughout; identity "
    "stays constant, props/age vary by phase. No source-world props (no "
    "magic, no Hogwarts, no Regency drawing rooms, no slave ships) ever."
)

# ── Literary grounding (text-grounding stage; ① only) ──────────────
SYSTEM_LITERARY_LOCATOR = (
    "You are an editorial archivist for literary text. Given a brief moment "
    "description and a list of candidate events from the same chapter range, "
    "you select the one event whose content most directly enacts the moment. "
    "You reject the entire list if none enacts the moment."
)

SYSTEM_LITERARY_EXTRACTOR = (
    "You are an editorial archivist for literary text. Given a chapter and a "
    "target moment, you locate the exact passage and return it verbatim — "
    "character-for-character, no paraphrase, no compression, no invention. "
    "You always include enough surrounding text to establish the scene AND "
    "show the character's actual response."
)


# ── Validator role descriptions ─────────────────────────────────────
SYSTEM_VOICE_VALIDATOR = (
    "You are a literary editor judging whether a single phase response "
    "sounds like that character at THAT phase, in the given setting. You "
    "catch anachronistic vocabulary for the setting (e.g., 'group chat' in "
    "Regency drawing-room; 'pull camera times' in 1990s magical Britain), "
    "broken-character self-narration, generic dialogue, and tonal mismatches. "
    "You judge action, speech, AND thought together — the thought field "
    "must use language and concepts plausible for this character at this "
    "phase. gt_speech being null is NOT a failure on its own."
)

SYSTEM_PHASE_FIT_VALIDATOR = (
    "You are a psychometric reviewer. Given a Character Arc's trajectory "
    "(each phase's position_description), the decision variable, and ONE "
    "phase response (action + speech + thought), you identify the trajectory "
    "phase this response is most diagnostic of. You judge from the WHOLE "
    "response — action, speech, and especially thought (how the character "
    "construes the situation). You are NOT told which phase generated this "
    "response. Pick the phase whose POSITION DESCRIPTION best matches what "
    "the response actually expresses, not the phase whose label sounds "
    "closest."
)

SYSTEM_ANCHOR_VALIDATOR = (
    "You are an editorial fact-checker. Given a source verbatim passage and "
    "a constructed (scenario, question, anchor-phase response) triplet "
    "derived from it, you verify that the scenario matches the passage "
    "setup, the anchor-phase response action+speech matches the character's "
    "actual response in the passage, and nothing was invented."
)

SYSTEM_WORLD_VALIDATOR = (
    "You are a worldbuilding reviewer. You verify that a probe's scenario "
    "and phase responses respect the rules of the probe's setting. For "
    "In-World probes you check source-world rules (era, technology, magic) "
    "and that no canonical scene is reproduced. For Out-of-World probes you "
    "check internal consistency within the named target era, the absence of "
    "any source-world props, and that each phase response describes the "
    "character at the life-stage assigned to that phase."
)

SYSTEM_DISCRIM_VALIDATOR = (
    "You are a psychometric reviewer judging cross-phase variation within "
    "one probe. Given the decision variable, the trajectory's phase "
    "descriptions, and N phase responses to the SAME scenario+question, you "
    "judge whether each adjacent pair is differentiated on the decision-"
    "variable axis. The differentiation can live in ACTION, SPEECH, or "
    "THOUGHT — when scenes constrain action, the cognitive style often "
    "carries the phase signal. Adjacent overlap is theoretically expected "
    "when position descriptions are themselves close — this is reporting, "
    "not gatekeeping."
)


# ── Arc context block ──────────────────────────────────────────────
def _trajectory_block(arc: dict,
                      life_stages: Optional[dict[int, dict]] = None) -> str:
    lines = []
    for i, ph in enumerate(arc["trajectory"]):
        cr = ph["chapter_range"]
        kms = ph.get("key_moments", [])
        life = ""
        if life_stages and i in life_stages:
            ls = life_stages[i]
            life = f"\n    Life-stage: {ls['life_stage']} (~{ls['approx_age']})"
        lines.append(
            f"  Phase {i} (idx={i}) \"{ph['phase']}\" (chs. {cr[0]}-{cr[1]})\n"
            f"    Position: {ph['position_description']}{life}\n"
            f"    Key moments: {'; '.join(kms) if kms else '(none)'}"
        )
    return "\n".join(lines)


def _phase_contrasts_block(phase_contrasts: list[dict]) -> str:
    if not phase_contrasts:
        return ""
    lines = []
    for pc in phase_contrasts:
        f = pc.get("from_phase_idx", 0)
        t = pc.get("to_phase_idx", 0)
        lines.append(f"  P{f} → P{t}: {pc.get('contrast','')}")
    return "Phase contrasts (adjacent pairs):\n" + "\n".join(lines)


def arc_system_block(character: str, arc: dict, novel: str,
                     decision_variable: Optional[str] = None,
                     phase_contrasts: Optional[list[dict]] = None,
                     life_stages: Optional[dict[int, dict]] = None) -> str:
    is_rel = arc.get("axis_type") == "relational"
    target = arc.get("target_character", "")
    rel_str = f"  (relational target: {target})" if is_rel else ""
    parts = [
        "[CHARACTER ARC CONTEXT]",
        f"Novel: {novel.replace('_', ' ')}",
        f"Character: {character}{rel_str}",
        f"Arc: {arc['axis_name']}",
        f"Dimension: {arc.get('dimension_label', '')}",
        f"Pole start: {arc.get('pole_start', '')}",
        f"Pole end:   {arc.get('pole_end', '')}",
        f"Trajectory:\n{_trajectory_block(arc, life_stages)}",
    ]
    if decision_variable:
        parts.append(f"\nDecision variable: {decision_variable}")
    if phase_contrasts:
        parts.append(_phase_contrasts_block(phase_contrasts))
    return "\n".join(parts)


def compose_system(role: str, arc_block: str) -> str:
    return f"{role}\n\n{arc_block}"


# ── S1: decision variable extraction (user) ─────────────────────────
def step1_user(n_phases: int) -> str:
    n_pairs = max(0, n_phases - 1)
    return f"""TASK:

1. Identify the SINGLE decision variable that captures what flips between
   phases. State it as a BINARY behavioral contrast that the character
   would be answering THROUGH ACTION. It must:
     - be answerable through behavior/choice
     - be specific enough that different phases give different answers
     - be phrased neutrally (no preferred pole)
     - be ONE dimension with two contrasting poles

2. For each adjacent phase pair, describe in one sentence how the
   decision-variable answer shifts. Produce EXACTLY {n_pairs} entries —
   one per adjacent pair, in order — with 0-indexed from/to indices."""


# ── S-LifeStage (user) ──────────────────────────────────────────────
def life_stage_user(arc: dict) -> str:
    n = len(arc["trajectory"])
    lines = []
    for i, ph in enumerate(arc["trajectory"]):
        cr = ph["chapter_range"]
        lines.append(
            f"  Phase {i} (idx={i}) \"{ph['phase']}\" (chs. {cr[0]}-{cr[1]}):\n"
            f"    {ph['position_description']}"
        )
    block = "\n".join(lines)
    return f"""TASK: For EACH phase, assign:
  - life_stage: one of {{child, adolescent, young_adult, adult, older_adult}}
      child         = roughly 0–12
      adolescent    = roughly 13–17
      young_adult   = roughly 18–30
      adult         = roughly 31–50
      older_adult   = roughly 51+
  - approx_age: a brief age estimate
  - rationale: one sentence linking your tag to the phase's position_description
               and chapter range

PHASES:
{block}

Produce EXACTLY {n} entries, in phase_idx order 0..{n-1}.

- If the source story spans many years, phases will progress through stages.
- If multiple phases share the same stage, that is fine — assign honestly.
- For unusual chronologies (allegory, dream sequences), pick the stage that
  best fits the position_description rather than a literal age."""


# ── S-AxisRe (user) ─────────────────────────────────────────────────
def axis_re_exp_user(arc: dict) -> str:
    n = len(arc["trajectory"])
    return f"""Re-express the Character Arc above in ERA-AGNOSTIC abstract
psychological terms.

TASK:
1. abstract_axis: 1 sentence naming the abstract dimension (e.g. "agency vs
   communion under pressure"). No novel-specific terminology.

2. abstract_phases: EXACTLY {n} entries, one per trajectory phase in order.
   Each is a short phrase summarizing that phase's position on the abstract
   axis in era-agnostic terms."""


# ── Text grounding (reused; ① only) ─────────────────────────────────
def step2_locator_user(key_moment: str, chapter_range: list,
                       candidate_events: list[dict]) -> str:
    lines = [
        f"  Chapter {ev['chapter']:>3} · {ev.get('event_id','?')}: "
        f"{ev['description']}"
        for ev in candidate_events
    ]
    block = "\n".join(lines)
    return f"""KEY MOMENT to locate:
  "{key_moment}"

CHAPTER RANGE for this arc phase: {chapter_range[0]}–{chapter_range[1]}

CANDIDATE EVENTS:
{block}

TASK: Pick the ONE event whose CONTENT most directly enacts the key moment.
- Match by event semantics, not keyword overlap.
- For late-arc transformation moments, prefer candidates near the END of the
  chapter range.
- If no candidate genuinely depicts the moment, set event_id to null."""


def step3_extractor_user(novel: str, chapter_num: int, key_moment: str,
                         event_description: str, chapter_text: str) -> str:
    return f"""Below is Chapter {chapter_num} of "{novel.replace('_',' ')}".

You are looking for the specific passage that depicts this moment:
  Key moment   : "{key_moment}"
  Event detail : "{event_description}"

TASK: Find the passage and return:
  - verbatim_passage: EXACT source text, character-for-character. Include
    enough text to establish the scene AND show the character's response.
    Typically 2-5 paragraphs (200-1500 words).
  - first_words / last_words: opening and closing 5-8 words.

Do NOT paraphrase. Do NOT invent. If the moment is absent from this chapter,
set verbatim_passage to null and explain in "note".

CHAPTER TEXT:
{chapter_text}"""


# ── Shared style guideline ─────────────────────────────────────────
def _style_block(character: str, max_action_words: int = 30) -> str:
    return f"""STYLE GUIDELINES — apply to scenario, question, and every phase response:

- LENGTH: scenario ≤ 2 sentences (≤60 words). question ≤ 1 sentence.
  Per phase response:
    gt_action: ≤ {max_action_words} words.
    gt_speech: ≤ 25 words or null.
    gt_thought: 1–2 sentences (≤ 50 words). Captures HOW this phase
                construes/processes/ruminates on the situation.
- DICTION: plain, vivid, era- and age-appropriate language for {character}
  at THIS phase. Avoid academic vocabulary ("orchestrate", "operationalize",
  "strategically prepared", "in alignment with…"). Prefer concrete verbs
  over abstract nouns.
- SPEECH (optional): include gt_speech ONLY when {character} would naturally
  speak in this beat. Silence/action-only is fine. Awkward dialogue is
  worse than no dialogue.
- THOUGHT: first-person psychological framing OR observer voice that names
  the character's interpretation. Examples (Harry/Snape arc):
    - early phase thought: "Snape's just being unfair again, like always."
    - later phase thought: "He's pushing me on purpose — maybe this is a
      test, or he wants me angry enough to slip. Either way, react less."
  The thought is where adjacent phases differentiate most when action is
  constrained by the scene.

SCENARIO STYLE: set the tension with minimum detail. Don't write background
paragraphs. The SCENARIO is what activates the decision variable — give it
real pull. Crucially, the scenario should be SOMETHING ALL PHASES COULD
PLAUSIBLY ENCOUNTER. Don't bind it to one phase's specific life-situation
(e.g., a cupboard scene if only one phase lives in a cupboard).

QUESTION STYLE: pose an OPEN-ENDED question focused on the moment. Hard
rules:
- DO NOT enumerate options ("Should X do A or B?", "Is it X's place to ___?",
  "Whose side ___?"). Binary forks collapse phase variation into two buckets
  AND reveal the decision variable — both bad.
- DO NOT be flat ("What does X do?" with no situational focus).
- DO NOT directly name the decision variable. The axis is activated by the
  SCENARIO; the QUESTION only opens space for {character} to respond.
- Good shapes: "How does {character} respond when ___?" / "What is
  {character}'s next move?" / "Where does {character}'s attention go?" /
  "How does {character} take in what just happened?"

RESPONSE STYLE — typicality matters:
- The response is the MODAL behavior + thought for THIS phase's trait state.
  Not maximally distinctive; typical.
- ONE verb / ONE beat in action. Don't stack multiple distinct actions.
- BANNED CUES — distinctiveness markers. AVOID unless absolutely natural:
  "immediately", "at once", "decisively", "deliberately",
  "without hesitation", "instinctively", "swiftly", "instantly".
- When the response is a partial / silent / awkward reaction, write THAT.
- THE THOUGHT FIELD CARRIES PHASE VARIANCE. Adjacent phases may have nearly
  identical action+speech but differ in HOW they interpret. Use this.
- gt_typicality:
    "typical"        — modal behavior for this state distribution.
    "plausible_tail" — in distribution but not the mode. Use when the
                       situation pulls slightly off-phase, or when honoring
                       the breadth of the state distribution is more honest
                       than always writing the cleanest exemplar.
- COMPACT-TIME ARCS: if multiple phases share the same life-stage, phase
  differences may live ONLY in stance/thought, not in action-level changes."""


# ── Phase response block helper ─────────────────────────────────────
def _phase_response_instructions(arc: dict, anchor_phase_idx: int,
                                 character: str,
                                 phase_query_chapters: dict[int, int],
                                 life_stages: Optional[dict[int, dict]],
                                 probe_type: str,
                                 max_action_words: int = 30) -> str:
    n = len(arc["trajectory"])
    lines = []
    for i in range(n):
        ph = arc["trajectory"][i]
        marker = " ★ ANCHOR" if i == anchor_phase_idx else ""
        qc = phase_query_chapters.get(i, 0)
        life = ""
        if life_stages and i in life_stages and probe_type == "out_of_world":
            ls = life_stages[i]
            life = f" · life-stage: {ls['life_stage']} (~{ls['approx_age']})"
        lines.append(
            f"  phase_idx={i} \"{ph['phase']}\" — {ph['position_description'][:140]}\n"
            f"      query_chapter={qc} (knows only chs 1..{qc}){life}{marker}"
        )
    block = "\n".join(lines)
    return f"""PHASE RESPONSES — produce EXACTLY {n} entries, one per
trajectory phase, in phase_idx order 0..{n-1}.

For each entry:
  - phase_idx: 0..{n-1} (matches the trajectory order)
  - gt_action: ≤ {max_action_words} words. {character}'s action AT THIS
    PHASE, drawing ONLY on what this phase's character knows
    (events up to its query_chapter, NOT later events).
  - gt_speech: ≤ 25 words spoken naturally, or null.
  - gt_thought: 1–2 sentences. How THIS phase's character construes,
    interprets, or ruminates on the situation. THIS IS WHERE ADJACENT
    PHASES DIFFERENTIATE most when action is constrained.
  - gt_typicality: "typical" (default) or "plausible_tail".

Phase list (each phase responds AS IF asked at its own query_chapter):
{block}

CRITICAL CONSTRAINTS:
- Adjacent phases that differ on the decision variable should differ in
  KIND (different verbs OR different thought-framings) — not just intensity
  adverbs. If the scene's action affordance is narrow, put the variance in
  gt_thought.
- A phase that has not yet encountered a piece of information (because it
  comes from a later chapter) must NOT reference it.
- {character} stays in their established voice. No fourth-wall narration."""


# ── ① In-Text probe (multi-phase) ────────────────────────────────────
def in_text_user(character: str, arc: dict, anchor_phase_idx: int,
                 anchor_query_chapter: int, verbatim_passage: str,
                 phase_query_chapters: dict[int, int],
                 life_stages: Optional[dict[int, dict]]) -> str:
    anchor = arc["trajectory"][anchor_phase_idx]
    is_rel = arc.get("axis_type") == "relational"
    target = arc.get("target_character", "")
    rel_block = ""
    if is_rel and target:
        rel_block = (
            f"\nRELATIONAL TARGET: This arc measures how {character} relates "
            f"to {target}. Keep {target} present as the focal counterpart in "
            f"the scenario; every phase's response describes {character}'s "
            f"behavior directed at (or in response to) {target}.\n"
        )
    return f"""PROBE TYPE: ① In-Text (multi-phase)
ANCHOR PHASE: idx={anchor_phase_idx} "{anchor['phase']}" (chs {anchor['chapter_range'][0]}-{anchor['chapter_range'][1]})
ANCHOR QUERY CHAPTER: {anchor_query_chapter}
{rel_block}
SOURCE PASSAGE (verbatim — contains the ANCHOR phase's canonical response):
\"\"\"
{verbatim_passage}
\"\"\"

TASK: Assemble ONE probe from this passage with N phase responses.

1. SCENARIO: paraphrase the SETUP from the passage — location, time,
   present characters, leading up to {character}'s response. Stop just
   before {character} acts/speaks.

2. QUESTION: name the specific choice point (see QUESTION STYLE).

3. {_phase_response_instructions(arc, anchor_phase_idx, character,
                                  phase_query_chapters, life_stages,
                                  "in_text", max_action_words=35)}

ANCHOR phase entry (phase_idx={anchor_phase_idx}) MUST extract
{character}'s actual response from the SOURCE PASSAGE — gt_action
paraphrases the passage; gt_speech copies {character}'s spoken words
verbatim from the passage (or null if they don't speak); gt_thought
captures the construal evident in the passage. Non-anchor phases are
counterfactual projections — "if this phase's character policy were
applied to the same situation, what would they do AND THINK?".

{_style_block(character, max_action_words=35)}

NOTE for ① In-Text: anchor scenes often have narrow behavioural
affordance — late-phase trait policies may legitimately produce similar
actions to early phases in cramped scenes. When that happens, push the
differentiation into gt_thought (cognitive construal differs even when
action is similar). Do not stretch action variance to manufacture
differentiation."""


# ── ② In-World probe (multi-phase) ──────────────────────────────────
def in_world_user(character: str, arc: dict, anchor_phase_idx: int,
                  anchor_query_chapter: int,
                  phase_query_chapters: dict[int, int],
                  life_stages: Optional[dict[int, dict]]) -> str:
    anchor = arc["trajectory"][anchor_phase_idx]
    is_rel = arc.get("axis_type") == "relational"
    target = arc.get("target_character", "")
    rel_block = ""
    if is_rel and target:
        rel_block = (
            f"\nRELATIONAL TARGET: This arc measures how {character} relates "
            f"to {target}. The scenario MUST involve a direct interaction "
            f"with {target} (or a moment whose stakes come from {target}). "
            f"Every phase response describes {character}'s behavior toward "
            f"{target}.\n"
        )
    return f"""PROBE TYPE: ② In-World (multi-phase)
ANCHOR PHASE: idx={anchor_phase_idx} "{anchor['phase']}" (chs {anchor['chapter_range'][0]}-{anchor['chapter_range'][1]})
ANCHOR QUERY CHAPTER: {anchor_query_chapter}
ANCHOR PHASE POSITION: {anchor['position_description']}
{rel_block}
HARD CONSTRAINTS:
- The scenario must NOT reproduce any specific canonical scene from the
  source novel. It must be a plausible-but-unwritten moment in the source
  world.
- {character} must remain in their established voice and time period.
- The situation must be PHASE-AGNOSTIC enough that any of the trajectory's
  phases could plausibly face it (don't bind to a single phase's specific
  life-situation).
- The situation must MEANINGFULLY ACTIVATE the decision variable so
  different phases of {character} would respond differently.

TASK:

1. SCENARIO: a plausible-but-unwritten situation around chapter
   {anchor_query_chapter}. End just before {character} responds. Keep it
   phase-agnostic.

2. QUESTION: pose the choice point (see QUESTION STYLE).

3. {_phase_response_instructions(arc, anchor_phase_idx, character,
                                  phase_query_chapters, life_stages,
                                  "in_world")}

{_style_block(character)}"""


# ── ③ Out-of-World probe (multi-phase) ──────────────────────────────
def out_of_world_user(character: str, arc: dict, anchor_phase_idx: int,
                      anchor_query_chapter: int,
                      abstract_axis: str, abstract_phases: list[str],
                      era_label: str, era_description: str,
                      phase_query_chapters: dict[int, int],
                      life_stages: dict[int, dict]) -> str:
    anchor = arc["trajectory"][anchor_phase_idx]
    is_rel = arc.get("axis_type") == "relational"
    target = arc.get("target_character", "")
    rel_block = ""
    if is_rel and target:
        rel_block = (
            f"\nRELATIONAL TARGET: The arc concerns {character}'s relation to "
            f"{target}. In the transposed scenario, {target} also appears, "
            f"with the same name but a role appropriate to the target era. "
            f"Every phase response describes the interaction between "
            f"{character} and {target}.\n"
        )
    abstract_block = "\n".join(
        f"  phase_idx={i}: {p}" for i, p in enumerate(abstract_phases)
    )
    return f"""PROBE TYPE: ③ Out-of-World (multi-phase, non-source era)
ANCHOR PHASE: idx={anchor_phase_idx} "{anchor['phase']}"
ABSTRACT AXIS: {abstract_axis}

ABSTRACT PHASE POSITIONS (era-agnostic):
{abstract_block}

TARGET ERA (shared across all phase responses): {era_label}
  Description: {era_description}
{rel_block}
LIFE-STAGE LOCK — each phase response describes {character} at THAT phase's
life-stage in the target era. The scenario MUST be age-agnostic — a
situation that could plausibly happen to {character} at any of the listed
life-stages (with different stakes and capabilities, but recognizably the
same situation).

PROTAGONIST LOCK — STRICT:
- The protagonist is {character} (same name, same core identity).
- The response is about {character} in every phase entry.
- Never re-cast under another name.

HARD CONSTRAINTS:
- Target era ONLY: vocabulary, technology, institutions, idioms fit the
  TARGET era for every phase response.
- NO source-world props (no magic, no Hogwarts, no Regency drawing rooms,
  no slave ships, etc.) ever.
- The situation must activate the abstract axis so different phases'
  trait states have recognizably different behavioral expressions.

TASK:

1. SCENARIO: a brief age-agnostic situation in the target era centered on
   {character}. Set the tension in 1–2 sentences (≤60 words). End just
   before {character} responds.

2. QUESTION: pose the choice point (see QUESTION STYLE).

3. {_phase_response_instructions(arc, anchor_phase_idx, character,
                                  phase_query_chapters, life_stages,
                                  "out_of_world")}

{_style_block(character)}"""


# ── Validator user prompts ──────────────────────────────────────────
def voice_validator_user(character: str, arc: dict, phase_idx: int,
                         probe_type: str, era_label: Optional[str],
                         life_stage: Optional[str],
                         query_chapter: int,
                         scenario: str, phase_response: dict) -> str:
    phase = arc["trajectory"][phase_idx]
    if probe_type == "out_of_world":
        setting_line = (
            f"SETTING: target era '{era_label}'. The voice must fit this "
            f"era's vocabulary AND a {life_stage} version of {character}."
        )
    elif probe_type == "in_world":
        setting_line = (
            f"SETTING: the source novel's world at chs "
            f"{phase['chapter_range'][0]}-{phase['chapter_range'][1]}. "
            f"The voice must fit the source era and {character}'s phase."
        )
    else:
        setting_line = (
            f"SETTING: the source novel's world at chs "
            f"{phase['chapter_range'][0]}-{phase['chapter_range'][1]}."
        )
    return f"""CHARACTER: {character}
PHASE: idx={phase_idx} "{phase['phase']}" — {phase['position_description']}
KNOWLEDGE CUTOFF: phase responds as if asked at chapter {query_chapter}
                  (knows only events up to that chapter).
{setting_line}

PROBE SCENARIO:
  {scenario}

THIS PHASE'S RESPONSE:
  Action:  {phase_response.get('gt_action','')}
  Speech:  {phase_response.get('gt_speech') or '(none — pure action response)'}
  Thought: {phase_response.get('gt_thought','')}
  Typicality: {phase_response.get('gt_typicality','typical')}

TASK: Does this response sound like {character} at this phase, in this
setting, given only knowledge up to chapter {query_chapter}? Catch:
- anachronistic word choice for the SETTING
- references to events the phase cannot know yet
- broken-character self-narration
- generic dialogue or thought
- tonal mismatches (young character with adult articulation; vice versa)

Note: gt_speech being null is NOT a failure.

- verdict: "pass" if response is in character; "fail" otherwise.
- note: 1 sentence — what's right or wrong."""


def phase_fit_validator_user(arc: dict, target_phase_idx: int,
                             decision_variable: str,
                             scenario: str, question: str,
                             phase_response: dict) -> str:
    """Q-PhaseFit prompt — target_phase_idx is passed in but DELIBERATELY
    NOT SHOWN to the validator. Marking the target in-prompt would let the
    validator echo it without judging."""
    n = len(arc["trajectory"])
    lines = []
    for i, ph in enumerate(arc["trajectory"]):
        lines.append(
            f"  idx={i} \"{ph['phase']}\" — {ph['position_description']}"
        )
    block = "\n".join(lines)
    return f"""DECISION VARIABLE: {decision_variable}
POLE START: {arc.get('pole_start','')}
POLE END:   {arc.get('pole_end','')}

ARC PHASES (0..{n-1}):
{block}

PROBE:
  Scenario: {scenario}
  Question: {question}
  Response action:  {phase_response.get('gt_action','')}
  Response speech:  {phase_response.get('gt_speech') or '(none)'}
  Response thought: {phase_response.get('gt_thought','')}

TASK: Based on the response's ACTION + SPEECH + THOUGHT, identify which
trajectory phase this response is most diagnostic of.

- most_likely_phase_idx: integer 0..{n-1}. Pick the phase whose
  POSITION DESCRIPTION best matches what the response actually expresses.
  Use the THOUGHT field heavily — it often carries phase-distinguishing
  cognitive style when action is constrained.
- confidence: "low" | "medium" | "high".
- note: 1 sentence — the deciding cue.

You are NOT told which phase generated this probe. Judge solely on what
the response expresses. Adjacent classification is fine and expected
(phases legitimately overlap). If two adjacent phases fit equally well,
pick either and use confidence="medium" or "low"."""


def anchor_validator_user(scenario: str, question: str,
                          anchor_phase_idx: int, anchor_response: dict,
                          source_passage: str) -> str:
    return f"""SOURCE PASSAGE (verbatim):
\"\"\"
{source_passage}
\"\"\"

CONSTRUCTED PROBE:
  Scenario: {scenario}
  Question: {question}
  Anchor phase (idx={anchor_phase_idx}) response:
    Action:  {anchor_response.get('gt_action','')}
    Speech:  {anchor_response.get('gt_speech') or '(none)'}
    Thought: {anchor_response.get('gt_thought','')}

TASK: Verify scenario + anchor-phase response are faithful to the source.
- Scenario reflects the passage's setup (location, present characters, tension)?
- Anchor action+speech matches what the character actually does in the
  passage (paraphrase OK; invention NOT OK)?
- Thought is plausibly inferable from what the passage shows of the
  character's mind in this beat?

- verdict: "pass" / "fail"
- note: 1 sentence."""


def world_validator_user(character: str, novel: str, arc: dict,
                         anchor_phase_idx: int, anchor_query_chapter: int,
                         probe_type: str, era_label: Optional[str],
                         era_description: Optional[str],
                         life_stages: Optional[dict[int, dict]],
                         scenario: str, phase_responses: list[dict]) -> str:
    if probe_type == "out_of_world":
        life_block = "\n".join(
            f"  phase_idx={k}: must be {life_stages[k]['life_stage']} "
            f"(~{life_stages[k]['approx_age']})"
            for k in sorted(life_stages.keys())
        ) if life_stages else ""
        rule = f"""③ Out-of-World — the scenario is set in TARGET ERA "{era_label}".
  Description: {era_description}
  Per-phase life-stage requirements:
{life_block}

CHECK:
  (a) target-era internal consistency (no anachronisms for that era);
  (b) NO source-world props leak in (no magic, no Hogwarts, no Regency
      drawing rooms, no slave ships, etc.);
  (c) each phase response describes {character} at its required life-stage;
  (d) {character} is the named protagonist throughout."""
    else:
        rule = f"""② In-World — the scenario is in the world of
"{novel.replace('_', ' ')}" around chapter {anchor_query_chapter}.

CHECK:
  (a) scenario obeys the source world's rules (era, technology, magic
      system if any);
  (b) only characters who would exist at this point in the story are
      present;
  (c) nothing reproduces a specific canonical scene from the source."""
    resp_block = "\n".join(
        f"  phase_idx={p['phase_idx']}: action={p.get('gt_action','')[:120]}"
        f" | thought={p.get('gt_thought','')[:120]}"
        for p in phase_responses
    )
    return f"""PROBE TYPE CONTEXT:
{rule}

CHARACTER: {character}
SCENARIO:
  {scenario}

PHASE RESPONSE SUMMARIES:
{resp_block}

TASK:
- verdict: "pass" if the rules above are respected throughout, "fail" otherwise.
- note: 1 sentence — what's consistent or inconsistent."""


def discrim_validator_user(arc: dict, decision_variable: str,
                           scenario: str, question: str,
                           phase_responses: list[dict]) -> str:
    rows = []
    for p in phase_responses:
        k = p["phase_idx"]
        ph = arc["trajectory"][k]
        rows.append(
            f"Phase idx={k} \"{ph['phase']}\" "
            f"(pos: {ph['position_description'][:100]}…)\n"
            f"  Action:  {p.get('gt_action','')}\n"
            f"  Speech:  {p.get('gt_speech') or '(none)'}\n"
            f"  Thought: {p.get('gt_thought','')}"
        )
    block = "\n\n".join(rows)
    return f"""DECISION VARIABLE: {decision_variable}

SHARED SCENARIO: {scenario}
SHARED QUESTION: {question}

PHASE RESPONSES (all to the SAME scenario+question):

{block}

TASK: For each ADJACENT phase pair, judge whether the two responses are
differentiated on the decision variable. The differentiation can live in
ACTION, SPEECH, or (often, when the scene constrains action) THOUGHT —
phase-specific cognitive construal carries the trait signal when behavior
is similar.

For each adjacent pair (idx k, idx k+1) of the trajectory:
  - separation:
      "separated"  — responses commit to genuinely different decision-
                     variable answers, in the direction phases imply.
      "similar"    — responses give the same/near-same answer; this is
                     FINE if the phases' position_descriptions themselves
                     are close.
      "ambiguous"  — cannot tell because responses cite different stakes
                     or framings.
  - note: 1 sentence — what cued your judgment.

Return weak_pairs[] containing entries for pairs where separation is
"similar" or "ambiguous". If a pair is "separated", omit it from weak_pairs.

This is annotation, NOT gatekeeping — nothing drops on this verdict."""


def regen_phase_user(character: str, arc: dict, phase_idx: int,
                     query_chapter: int,
                     scenario: str, question: str,
                     prev_response: dict, all_phase_responses: list[dict],
                     failure_notes: str) -> str:
    phase = arc["trajectory"][phase_idx]
    others_block = "\n".join(
        f"  P{pr['phase_idx']}: action={pr.get('gt_action','')[:140]}"
        for pr in all_phase_responses if pr["phase_idx"] != phase_idx
    )
    return f"""A previous phase-response attempt failed validation. Generate a
NEW response for THIS PHASE ONLY, keeping the same scenario+question.

SCENARIO: {scenario}
QUESTION: {question}

TARGET PHASE: idx={phase_idx} "{phase['phase']}"
TARGET PHASE POSITION: {phase['position_description']}
QUERY CHAPTER (knowledge cutoff): {query_chapter}

OTHER PHASES' RESPONSES (your response must fit alongside these):
{others_block}

PREVIOUS RESPONSE (failed):
  Action:  {prev_response.get('gt_action','')}
  Speech:  {prev_response.get('gt_speech') or '(none)'}
  Thought: {prev_response.get('gt_thought','')}

VALIDATION FEEDBACK — fix these specific issues:
{failure_notes}

TASK: Produce one PhaseResponseOut for phase_idx={phase_idx}.

ANTI-STACKING — DO NOT compensate for the previous failure by:
  - packing multiple actions into one response,
  - stacking phase markers,
  - over-explaining,
  - making fields longer.
The new response must be JUST AS BRIEF as the original. ONE verb / ONE
beat in action. 1–2 sentences in thought. Keep it TYPICAL, not maximally
distinctive. If a clean fix isn't possible without stacking, mark
gt_typicality="plausible_tail" and write the less-tidy honest response.

- Address the feedback directly. If voice was off, use {character}'s
  phase-appropriate voice (no anachronisms, no over-articulation, no
  knowledge from after chapter {query_chapter}).
- If phase-fit was off, push the DIFFERENTIATION into the THOUGHT field
  primarily — different cognitive construal, not adverbial tweaks."""
