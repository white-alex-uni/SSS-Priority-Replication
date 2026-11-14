#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — навчання багатошарового перцептрона з >= 3 ініціалізацій та фіксація
найкращої моделі у JSON-файлі (Роль 1, Проєкт 3).

Протокол згідно з team_config.json (команда T01):
  - train.csv (8000 записів) — навчання параметрів мережі та StandardScaler;
  - valid.csv (2000 записів) — вибір найкращої моделі за macro-F1;
  - кількість незалежних ініціалізацій >= 3 (seeds: 3001, 3002, 3003);
  - критерій проходження C1 №9: macro-F1 >= F1_min (0.56).
"""

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from .data_loader import extract_labels, load_csv, validate_dataset
from .features import FEATURE_NAMES, FeatureExtractor, StandardScaler
from .metrics import compute_macro_f1, evaluate_predictions, format_confusion_matrix
from .mlp import MLPClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODEL_JSON = Path("models") / "best_model_basic1.json"

F1_MIN_THRESHOLD = 0.56
BASELINE_MAJORITY_F1 = 0.293
BASELINE_RANDOM_F1 = 0.3119
DEFAULT_SEEDS = (3001, 3002, 3003)


def train_and_select_model(
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    output_json: Union[str, Path] = DEFAULT_MODEL_JSON,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.006,
    h1: int = 24,
    h2: int = 12,
    class_weights: Sequence[float] = (1.0, 2.0, 2.2),
    l2_reg: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Виконує повний цикл навчання з декількох ініціалізацій, обирає найкращу
    модель за macro-F1 на valid.csv та зберігає її параметри й метрики у JSON.
    """
    start_time = time.time()
    data_path = Path(data_dir)
    out_path = Path(output_json)

    train_records = load_csv(data_path / "train.csv")
    valid_records = load_csv(data_path / "valid.csv")
    session_path = data_path / "session.csv"
    session_records = load_csv(session_path) if session_path.exists() else []

    train_check = validate_dataset(train_records)
    valid_check = validate_dataset(valid_records)
    session_check = validate_dataset(session_records) if session_records else None

    if not train_check["valid"] or not valid_check["valid"]:
        raise ValueError("Виявлено помилки формату або порушення монотонності часу в даних!")

    extractor = FeatureExtractor()
    X_train_raw = extractor.extract_raw(train_records)
    X_valid_raw = extractor.extract_raw(valid_records)

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_valid = scaler.transform(X_valid_raw)

    y_train = extract_labels(train_records)
    y_valid = extract_labels(valid_records)

    in_dim = len(FEATURE_NAMES)
    runs_summary: List[Dict[str, Any]] = []

    best_overall_f1 = -1.0
    best_overall_seed = int(seeds[0])
    best_overall_epoch = 1
    best_overall_model: MLPClassifier | None = None

    if verbose:
        print("=" * 72)
        print("Навчання перцептрона (Роль 1, команда T01)")
        print(f"Ознак: {in_dim} | Архітектура: {in_dim} -> {h1} (ReLU) -> {h2} (ReLU) -> 3 (Softmax)")
        print(f"Поріг F1_min: {F1_MIN_THRESHOLD} | Ініціалізації (seeds): {list(seeds)}")
        print("=" * 72)

    for run_idx, seed in enumerate(seeds, start=1):
        model = MLPClassifier(
            in_dim=in_dim, h1=h1, h2=h2, out_dim=3, seed=seed, optimizer="adam"
        )
        shuffle_rng = random.Random(seed)

        best_seed_f1 = -1.0
        best_seed_epoch = 1
        best_seed_weights = model.copy_weights()
        epoch_history: List[Dict[str, Any]] = []

        t_seed = time.time()
        for epoch in range(1, epochs + 1):
            train_loss = model.train_epoch(
                X_train,
                y_train,
                batch_size=batch_size,
                lr=lr,
                class_weights=class_weights,
                l2_reg=l2_reg,
                rng=shuffle_rng,
            )
            val_preds = model.predict(X_valid)
            val_f1, val_per_class, _ = compute_macro_f1(y_valid, val_preds)

            epoch_history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "valid_macro_f1": round(val_f1, 6),
                    "valid_f1_per_class": [round(x, 6) for x in val_per_class],
                }
            )

            if val_f1 > best_seed_f1:
                best_seed_f1 = val_f1
                best_seed_epoch = epoch
                best_seed_weights = model.copy_weights()

        model.load_weights(best_seed_weights)
        seed_elapsed = time.time() - t_seed

        val_eval = evaluate_predictions(y_valid, model.predict(X_valid))
        train_eval = evaluate_predictions(y_train, model.predict(X_train))

        runs_summary.append(
            {
                "initialization": run_idx,
                "seed": seed,
                "best_epoch": best_seed_epoch,
                "elapsed_seconds": round(seed_elapsed, 3),
                "train_macro_f1": train_eval["macro_f1"],
                "valid_macro_f1": val_eval["macro_f1"],
                "valid_accuracy": val_eval["accuracy"],
                "valid_per_class_f1": {
                    k: v["f1"] for k, v in val_eval["per_class"].items()
                },
                "history": epoch_history,
            }
        )

        if verbose:
            print(
                f"[Ініціалізація #{run_idx} | seed={seed}] "
                f"best_epoch={best_seed_epoch:2d} | "
                f"train macro-F1={train_eval['macro_f1']:.4f} | "
                f"valid macro-F1={val_eval['macro_f1']:.4f} "
                f"(F1_0={val_eval['per_class']['0']['f1']:.4f}, "
                f"F1_1={val_eval['per_class']['1']['f1']:.4f}, "
                f"F1_2={val_eval['per_class']['2']['f1']:.4f}) "
                f"[{seed_elapsed:.2f}s]"
            )

        if best_seed_f1 > best_overall_f1:
            best_overall_f1 = best_seed_f1
            best_overall_seed = seed
            best_overall_epoch = best_seed_epoch
            best_overall_model = model

    assert best_overall_model is not None

    # Підсумкова оцінка обраної найкращої моделі
    best_train_eval = evaluate_predictions(y_train, best_overall_model.predict(X_train))
    best_valid_eval = evaluate_predictions(y_valid, best_overall_model.predict(X_valid))

    session_eval = None
    session_windowed_eval = None
    if session_records and all(r.label is not None for r in session_records):
        y_sess = extract_labels(session_records)
        X_sess = scaler.transform(extractor.extract_raw(session_records))
        session_eval = evaluate_predictions(y_sess, best_overall_model.predict(X_sess))

        # Додаткова перевірка при ізольованій подачі по 48 записів (replication_window_records=48)
        win_preds: List[int] = []
        for w_start in range(0, len(session_records), 48):
            chunk = session_records[w_start : w_start + 48]
            X_chunk = scaler.transform(extractor.extract_raw(chunk))
            win_preds.extend(best_overall_model.predict(X_chunk))
        session_windowed_eval = evaluate_predictions(y_sess, win_preds)

    passed_c1 = best_valid_eval["macro_f1"] >= F1_MIN_THRESHOLD
    total_elapsed = time.time() - start_time

    artifact_data: Dict[str, Any] = {
        "team": "T01",
        "role": "Роль 1 — Багатошаровий перцептрон та інженерія часових ознак (базовий рівень)",
        "selection_summary": {
            "selected_seed": best_overall_seed,
            "selected_epoch": best_overall_epoch,
            "num_initializations": len(seeds),
            "f1_min_threshold": F1_MIN_THRESHOLD,
            "baseline_majority_f1": BASELINE_MAJORITY_F1,
            "baseline_random_f1": BASELINE_RANDOM_F1,
            "valid_macro_f1": best_valid_eval["macro_f1"],
            "margin_above_f1_min": round(
                best_valid_eval["macro_f1"] - F1_MIN_THRESHOLD, 6
            ),
            "c1_case_9_passed": passed_c1,
            "total_training_time_seconds": round(total_elapsed, 3),
        },
        "dataset_validation": {
            "train": train_check,
            "valid": valid_check,
            "session": session_check,
        },
        "evaluation": {
            "valid": best_valid_eval,
            "train": best_train_eval,
            "session_full": session_eval,
            "session_window_48": session_windowed_eval,
        },
        "initializations_comparison": runs_summary,
        "feature_config": {
            "time_window_ms": extractor.time_window_ms,
            "short_time_window_ms": extractor.short_time_window_ms,
            "count_window": extractor.count_window,
            "short_count_window": extractor.short_count_window,
            "feature_names": list(FEATURE_NAMES),
        },
        "scaler": scaler.to_dict(),
        "model": best_overall_model.to_dict(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(artifact_data, f, ensure_ascii=False, indent=2)

    if verbose:
        print("-" * 72)
        print(f"Обрано найкращу модель: seed={best_overall_seed} (епоха {best_overall_epoch})")
        print(f"Valid macro-F1 = {best_valid_eval['macro_f1']:.4f} (поріг F1_min = {F1_MIN_THRESHOLD}, запас +{best_valid_eval['macro_f1'] - F1_MIN_THRESHOLD:.4f})")

        if session_eval is not None:
            print(f"Session (1200 записів) macro-F1 = {session_eval['macro_f1']:.4f}")
        if session_windowed_eval is not None:
            print(f"Session (25 вікон по 48 записів) macro-F1 = {session_windowed_eval['macro_f1']:.4f}")
        print(format_confusion_matrix(best_valid_eval["confusion_matrix"]))
        print(f"Дані та ваги найкращої моделі збережено у: {out_path}")
        print("=" * 72)

    return artifact_data


if __name__ == "__main__":
    train_and_select_model()