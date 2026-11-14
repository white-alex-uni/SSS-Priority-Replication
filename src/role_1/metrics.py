#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py — метрики якості класифікації та матриця помилок (Роль 1).

Реалізує обчислення macro-F1 (основний критерій вибору моделі та кейсу C1 №9),
точності (accuracy), per-class precision/recall/F1 та матриці помилок 3x3.
"""

from typing import Any, Dict, List, Sequence, Tuple

CLASS_NAMES: Dict[int, str] = {
    0: "штатна подія",
    1: "аномальна подія",
    2: "перехід до інциденту",
}


def compute_confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = 3
) -> List[List[int]]:
    """Будує матрицю помилок розміром num_classes x num_classes (рядки — істинні, стовпці — передбачені)."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Довжини y_true ({len(y_true)}) та y_pred ({len(y_pred)}) не збігаються"
        )
    cm = [[0] * num_classes for _ in range(num_classes)]
    for yt, yp in zip(y_true, y_pred):
        if 0 <= yt < num_classes and 0 <= yp < num_classes:
            cm[yt][yp] += 1
    return cm


def compute_macro_f1(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = 3
) -> Tuple[float, List[float], List[List[int]]]:
    """Повертає (macro_f1, per_class_f1, confusion_matrix)."""
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    per_class_f1: List[float] = []
    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(num_classes) if r != c)
        fn = sum(cm[c][k] for k in range(num_classes) if k != c)
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        per_class_f1.append(f1)
    macro_f1 = sum(per_class_f1) / num_classes if num_classes > 0 else 0.0
    return macro_f1, per_class_f1, cm


def evaluate_predictions(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = 3
) -> Dict[str, Any]:
    """Формує повний словник метрик якості та матриці помилок для збереження у JSON і звіту."""
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total = len(y_true)
    correct = sum(cm[c][c] for c in range(num_classes))
    accuracy = (correct / total) if total > 0 else 0.0

    per_class: Dict[str, Dict[str, Any]] = {}
    f1_list: List[float] = []

    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(num_classes) if r != c)
        fn = sum(cm[c][k] for k in range(num_classes) if k != c)
        support = sum(cm[c])

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        f1_list.append(f1)

        per_class[str(c)] = {
            "name": CLASS_NAMES.get(c, f"class_{c}"),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    macro_f1 = sum(f1_list) / num_classes if num_classes > 0 else 0.0

    return {
        "samples": total,
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def format_confusion_matrix(cm: Sequence[Sequence[int]]) -> str:
    """Форматує матрицю помилок у текстову таблицю для виводу в консоль або звіт."""
    lines = [
        "Матриця помилок (рядки — істинний клас, стовпці — передбачений клас):",
        f"{'Клас':<28} | {'Pred 0':>8} | {'Pred 1':>8} | {'Pred 2':>8} | {'Усього':>8}",
        "-" * 72,
    ]
    for idx, row in enumerate(cm):
        label_str = f"{idx} ({CLASS_NAMES.get(idx, '')})"
        row_sum = sum(row)
        lines.append(
            f"{label_str:<28} | {row[0]:>8d} | {row[1]:>8d} | {row[2]:>8d} | {row_sum:>8d}"
        )
    return "\n".join(lines)
