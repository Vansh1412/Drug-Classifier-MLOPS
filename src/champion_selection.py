"""
src/champion_selection.py — Transparent champion/challenger model selection for MedMLOps-X.

The champion is selected from evaluated candidates by:
  1. Filtering by minimum quality thresholds (accuracy, macro_f1, recall).
  2. Ranking remaining candidates by the primary_metric.
  3. Selecting the top candidate as champion.

If no candidate meets the minimum thresholds, the pipeline fails safely
with a clear error — no poor model is ever silently promoted.
"""

from __future__ import annotations

from typing import Any


class NoValidChampionError(ValueError):
    """Raised when no model candidate passes the quality gate thresholds."""
    pass


def select_champion(
    results: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select the champion from evaluated model candidates.

    The selection is deterministic: same results + same config → same champion.

    Args:
        results: List of evaluation result dicts (one per candidate).
        cfg: Full configuration dictionary. Uses cfg["champion"] for policy.

    Returns:
        Tuple of (champion_result, sorted_challengers).
        challengers is the ranked list excluding the champion.

    Raises:
        NoValidChampionError: If no candidate satisfies the minimum quality thresholds.
    """
    champ_cfg = cfg["champion"]
    min_accuracy = champ_cfg["min_accuracy"]
    min_macro_f1 = champ_cfg["min_macro_f1"]
    min_recall = champ_cfg["min_recall"]
    primary_metric = champ_cfg["primary_metric"]

    # Step 1: Apply quality gate
    passing = []
    failing = []
    for result in results:
        acc_ok = result.get("accuracy", 0) >= min_accuracy
        f1_ok = result.get("macro_f1", 0) >= min_macro_f1
        recall_ok = result.get("macro_recall", 0) >= min_recall
        if acc_ok and f1_ok and recall_ok:
            passing.append(result)
        else:
            failing.append(result)

    if not passing:
        failure_summary = "\n".join(
            f"  {r['model_name']}: accuracy={r.get('accuracy', 0):.3f} "
            f"macro_f1={r.get('macro_f1', 0):.3f} "
            f"macro_recall={r.get('macro_recall', 0):.3f}"
            for r in results
        )
        raise NoValidChampionError(
            f"NO VALID CHAMPION — No model candidate met minimum quality thresholds:\n"
            f"  Required: accuracy>={min_accuracy}, macro_f1>={min_macro_f1}, "
            f"macro_recall>={min_recall}\n"
            f"Results:\n{failure_summary}"
        )

    # Step 2: Rank by primary metric (descending)
    ranked = sorted(passing, key=lambda r: r.get(primary_metric, 0), reverse=True)
    champion = ranked[0]
    challengers = ranked[1:]

    return champion, challengers


def format_leaderboard(
    champion: dict[str, Any],
    challengers: list[dict[str, Any]],
    failing: list[dict[str, Any]] | None = None,
) -> str:
    """Produce a human-readable leaderboard string.

    Args:
        champion: Champion model result.
        challengers: Ranked challenger results.
        failing: Models that failed the quality gate.

    Returns:
        Formatted leaderboard string.
    """
    lines = ["=" * 60, "MODEL COMPETITION LEADERBOARD", "=" * 60]
    lines.append(
        f"\n[CHAMPION] {champion['model_name']}"
        f"\n   accuracy    = {champion.get('accuracy', 0):.4f}"
        f"\n   macro_f1    = {champion.get('macro_f1', 0):.4f}"
        f"\n   macro_recall= {champion.get('macro_recall', 0):.4f}"
        f"\n   cv_f1_mean  = {champion.get('cv_macro_f1_mean', 0):.4f}"
        f"\n   train_time  = {champion.get('training_time_seconds', 0):.2f}s"
    )
    for idx, chal in enumerate(challengers, start=1):
        lines.append(
            f"\n#{idx + 1} CHALLENGER: {chal['model_name']}"
            f"\n   accuracy    = {chal.get('accuracy', 0):.4f}"
            f"\n   macro_f1    = {chal.get('macro_f1', 0):.4f}"
            f"\n   macro_recall= {chal.get('macro_recall', 0):.4f}"
        )
    if failing:
        lines.append("\n[FAILED QUALITY GATE]:")
        for m in failing:
            lines.append(
                f"   {m['model_name']}: "
                f"acc={m.get('accuracy', 0):.4f} "
                f"f1={m.get('macro_f1', 0):.4f} "
                f"recall={m.get('macro_recall', 0):.4f}"
            )
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
