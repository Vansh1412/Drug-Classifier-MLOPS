"""
train.py — MedMLOps-X Model Competition Training Pipeline.

Deterministic, reproducible, multi-model training pipeline.
All randomness is seeded via configs/model_config.yaml.

Usage:
    python train.py                          # Use default config
    python train.py --config path/to/cfg.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# src/ is at the project root level
_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.champion_selection import NoValidChampionError, format_leaderboard, select_champion
from src.config import load_config
from src.data_validation import load_and_validate
from src.evaluate import train_and_evaluate
from src.model_registry import build_model_candidates
from src.preprocessing import build_preprocessor

# Use non-interactive matplotlib backend (required when running headlessly)
matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedMLOps-X Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model_config.yaml (default: configs/model_config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Step 1: Load Configuration
    # -------------------------------------------------------------------------
    cfg = load_config(args.config)
    print(f"[1/8] Configuration loaded from: {args.config or 'configs/model_config.yaml'}")

    # -------------------------------------------------------------------------
    # Step 2: Load & Validate Dataset
    # -------------------------------------------------------------------------
    data_path = cfg["data"]["data_path"]
    print(f"[2/8] Loading and validating dataset: {data_path}")
    drug_df = load_and_validate(data_path, cfg)
    print(f"      Dataset shape: {drug_df.shape} — validation PASSED")

    # -------------------------------------------------------------------------
    # Step 3: Deterministic Shuffle + Train/Test Split
    #   - sample(frac=1) uses shuffle_random_state for reproducibility
    #   - train_test_split uses split_random_state
    # -------------------------------------------------------------------------
    shuffle_seed = cfg["data"]["shuffle_random_state"]
    split_seed = cfg["data"]["split_random_state"]
    test_size = cfg["data"]["test_size"]

    print(f"[3/8] Deterministic shuffle (seed={shuffle_seed}) and split (seed={split_seed})")
    drug_df = drug_df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

    X = drug_df.drop(cfg["data"]["target_column"], axis=1).values
    y = drug_df[cfg["data"]["target_column"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_seed, stratify=y
    )
    print(f"      Train size: {len(X_train)}, Test size: {len(X_test)}")

    # -------------------------------------------------------------------------
    # Step 4: Build Model Candidates
    # -------------------------------------------------------------------------
    print("[4/8] Building model candidates from configuration...")
    candidates = build_model_candidates(cfg)
    print(f"      Candidates: {list(candidates.keys())}")

    # -------------------------------------------------------------------------
    # Step 5: Model Competition — CV + Test Evaluation
    # -------------------------------------------------------------------------
    print(f"[5/8] Running model competition ({cfg['data']['cv_folds']}-fold CV + test eval)...")
    all_results: list[dict] = []

    for name, estimator in candidates.items():
        preprocessor = build_preprocessor(cfg)
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", estimator),
        ])
        model_params = {k: v for k, v in cfg["models"][name]["params"].items()
                        if not k.startswith("#")}
        print(f"      Training: {name}...")
        result = train_and_evaluate(
            pipeline, X_train, y_train, X_test, y_test, cfg, name, model_params
        )
        all_results.append({**result, "_pipeline": pipeline})
        print(
            f"        accuracy={result['accuracy']:.4f} | "
            f"macro_f1={result['macro_f1']:.4f} | "
            f"cv_f1={result['cv_macro_f1_mean']:.4f}+/-{result['cv_macro_f1_std']:.4f} | "
            f"time={result['training_time_seconds']:.2f}s"
        )

    # -------------------------------------------------------------------------
    # Step 6: Champion Selection
    # -------------------------------------------------------------------------
    print("[6/8] Selecting champion via quality gate + primary metric ranking...")
    # Separate serializable results from pipeline objects
    serializable_results = [{k: v for k, v in r.items()
                             if not k.startswith("_")} for r in all_results]
    pipeline_map = {r["model_name"]: r["_pipeline"] for r in all_results}

    try:
        champion_result, challengers = select_champion(serializable_results, cfg)
    except NoValidChampionError as exc:
        print(f"\n[ERROR] TRAINING FAILED: {exc}")
        sys.exit(1)

    champion_pipeline = pipeline_map[champion_result["model_name"]]
    print(f"      Champion: {champion_result['model_name']}")

    # Compute failing candidates (those not in passing list)
    passing_names = {champion_result["model_name"]} | {c["model_name"] for c in challengers}
    failing_results = [r for r in serializable_results if r["model_name"] not in passing_names]
    print(format_leaderboard(champion_result, challengers, failing_results))

    # -------------------------------------------------------------------------
    # Step 7: Save Results & Artifacts
    # -------------------------------------------------------------------------
    print("[7/8] Saving results and model artifact...")
    results_dir = Path("Results")
    results_dir.mkdir(exist_ok=True)
    model_dir = Path("Model")
    model_dir.mkdir(exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    total_runtime = time.perf_counter() - run_start

    # --- metrics.json (structured, machine-readable) ---
    metrics_json_payload = {
        "run_timestamp": run_timestamp,
        "total_runtime_seconds": round(total_runtime, 4),
        "dataset": {
            "path": data_path,
            "total_records": len(drug_df),
            "train_records": len(X_train),
            "test_records": len(X_test),
            "test_size": test_size,
            "shuffle_random_state": shuffle_seed,
            "split_random_state": split_seed,
            "cv_folds": cfg["data"]["cv_folds"],
        },
        "champion": {
            **champion_result,
            "quality_gate": {
                "primary_metric": cfg["champion"]["primary_metric"],
                "min_accuracy": cfg["champion"]["min_accuracy"],
                "min_macro_f1": cfg["champion"]["min_macro_f1"],
                "min_recall": cfg["champion"]["min_recall"],
            },
        },
        "all_candidates": serializable_results,
    }

    metrics_json_path = cfg["results"]["metrics_json"]
    with open(metrics_json_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_json_payload, fh, indent=2)
    print(f"      Metrics JSON written to: {metrics_json_path}")

    # --- metrics.txt (backward compatibility with CI CML) ---
    metrics_txt_path = cfg["results"]["metrics_txt"]
    with open(metrics_txt_path, "w", encoding="utf-8") as fh:
        fh.write(f"\nChampion Model: {champion_result['model_name']}\n")
        fh.write(
            f"Accuracy = {round(champion_result['accuracy'], 4)}, "
            f"Macro F1 = {round(champion_result['macro_f1'], 4)}, "
            f"Macro Recall = {round(champion_result['macro_recall'], 4)}\n"
        )
        fh.write(f"CV Macro F1 = {round(champion_result.get('cv_macro_f1_mean', 0), 4)} "
                 f"+/- {round(champion_result.get('cv_macro_f1_std', 0), 4)}\n")
        fh.write(f"\nAll Candidates:\n")
        for r in serializable_results:
            fh.write(
                f"  {r['model_name']}: acc={r['accuracy']:.4f} "
                f"f1={r['macro_f1']:.4f} recall={r['macro_recall']:.4f}\n"
            )
    print(f"      Metrics TXT written to:  {metrics_txt_path}")

    # --- competition_report.json ---
    competition_report = {
        "run_timestamp": run_timestamp,
        "champion": champion_result["model_name"],
        "challengers": [c["model_name"] for c in challengers],
        "failed_quality_gate": [r["model_name"] for r in failing_results],
        "leaderboard": serializable_results,
    }
    competition_report_path = cfg["results"]["competition_report_json"]
    with open(competition_report_path, "w", encoding="utf-8") as fh:
        json.dump(competition_report, fh, indent=2)
    print(f"      Competition report:       {competition_report_path}")

    # --- Confusion matrix for champion ---
    champion_preds = champion_pipeline.predict(X_test)
    classes = sorted(set(y_test.tolist()))
    cm = confusion_matrix(y_test, champion_preds, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    ax.set_title(f"Champion: {champion_result['model_name']}")
    plt.tight_layout()
    cm_path = cfg["results"]["confusion_matrix_png"]
    plt.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"      Confusion matrix saved:   {cm_path}")

    # --- Model artifact (champion Skops) ---
    artifact_path = cfg["champion"]["artifact_path"]
    sio.dump(champion_pipeline, artifact_path)
    print(f"      Model artifact saved:     {artifact_path}")

    # --- Model metadata JSON ---
    metadata = {
        "model_name": champion_result["model_name"],
        "model_class": cfg["models"][champion_result["model_name"]]["class"],
        "model_params": champion_result["model_params"],
        "artifact_path": artifact_path,
        "training_timestamp": run_timestamp,
        "dataset_path": data_path,
        "shuffle_random_state": shuffle_seed,
        "split_random_state": split_seed,
        "accuracy": champion_result["accuracy"],
        "macro_f1": champion_result["macro_f1"],
        "macro_recall": champion_result["macro_recall"],
        "weighted_f1": champion_result["weighted_f1"],
        "cv_macro_f1_mean": champion_result.get("cv_macro_f1_mean"),
        "cv_macro_f1_std": champion_result.get("cv_macro_f1_std"),
    }
    metadata_path = cfg["champion"]["metadata_path"]
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"      Model metadata saved:     {metadata_path}")

    # -------------------------------------------------------------------------
    # Step 8: Summary
    # -------------------------------------------------------------------------
    print("[8/8] Training pipeline complete.")
    print(
        f"\n[OK] Champion: {champion_result['model_name']} | "
        f"Accuracy={champion_result['accuracy']:.4f} | "
        f"Macro F1={champion_result['macro_f1']:.4f} | "
        f"Total time={total_runtime:.1f}s"
    )


if __name__ == "__main__":
    main()
