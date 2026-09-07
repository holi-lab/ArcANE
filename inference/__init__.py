"""Role-play probe execution.

Given probes produced by `probe_generation/`, this package elicits a free-form
response from an LLM that has been role-played as the target character at the
phase-response's `query_chapter`. The character's available context varies by
`mode`:

  vanilla  — character card only (novel, chapter, optional relation target)
  arc      — vanilla + the character's `final_auto/{slug}_auto_axes.json`,
             truncated at `query_chapter`: only phases whose `chapter_range`
             has started by that chapter are shown, and `pole_end` /
             `arc_direction` are stripped when later phases are hidden.
             literary_validation is always stripped.
  summary  — vanilla + chapter-by-chapter summaries for chapters 1..query_chapter
             (precomputed; see `inference.build_summaries`)
  rag      — vanilla + top-k chunks retrieved from chapter texts 1..query_chapter
             (precomputed embeddings; see `inference.build_index`)

The unit of work is ONE trial = (probe_id, phase_idx, mode, model). Trials are
written append-only to JSONL so a crash can be resumed: re-running skips any
trial_id already present.

The character does NOT receive any instruction about response format. Only the
scenario and question are surfaced; the natural completion is recorded as-is.
"""
