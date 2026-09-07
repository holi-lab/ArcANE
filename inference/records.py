"""Shared checks for inference records consumed by resume and evaluation."""

LEGACY_DRY_RUN_RESPONSE = "(dry-run, no API call)"


def is_dry_run_record(row: dict) -> bool:
    """Also recognize placeholder rows written before dry-run became stdout-only."""
    return row.get("dry_run") is True or row.get("response") == LEGACY_DRY_RUN_RESPONSE
