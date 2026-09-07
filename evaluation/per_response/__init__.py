"""ArcANE response evaluation.

An LLM judge scores each role-play response against the probe's ground truth
(`gt_action` / `gt_speech` / `gt_thought`) on APF, RPF, and RAE, then scores are
aggregated per (model, mode) so methods can be compared within a model.

Entry point:  uv run python -m evaluation.per_response --novel ... --character ...

Layout:
  prompt.py  — judge system/user prompt templates
  loader.py  — gather ground truth (probe files) + responses (results), joined
  runner.py  — run the judge, parse scores, aggregate, cost estimate, summary
  cli.py     — argument parsing / entry point
"""
