#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пакет Ролі 1: Багатошаровий перцептрон та інженерія часових ознак.

Публічний контракт для команди (згідно з TEAM_ROLES.md та FORMAT_SPEC_P3.md):
  - extract_features(events_window) -> list[list[float]]
  - predict_classes(events_window) -> list[int]
  - compute_event_priorities(events_window, classes=None) -> list[int]  (для Ролі 3 — ДП)
  - encode_class_table(classes) -> bytes                                (для Ролі 5 — LOG3 bit2)
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .data_loader import LogRecord, load_csv, normalize_records, validate_dataset
from .features import FEATURE_NAMES, FeatureExtractor, StandardScaler
from .metrics import (
    CLASS_NAMES,
    compute_confusion_matrix,
    compute_macro_f1,
    evaluate_predictions,
    format_confusion_matrix,
)
from .mlp import MLPClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_JSON = PROJECT_ROOT / "models" / "best_model_basic.json"

# Базові пріоритети класів і бонус critical згідно з team_config.json (команда T01)
CLASS_PRIORITY_MAP: Dict[int, int] = {
    0: 1,  # штатна подія
    1: 5,  # аномальна подія
    2: 4,  # перехід до інциденту
}
CRITICAL_BONUS: int = 5

_CACHED_BUNDLE: Optional[Tuple[FeatureExtractor, StandardScaler, MLPClassifier]] = None


def load_trained_bundle(
    model_json_path: Union[str, Path] = DEFAULT_MODEL_JSON,
) -> Tuple[FeatureExtractor, StandardScaler, MLPClassifier]:
    """Завантажує збережений екстрактор ознак, StandardScaler та MLPClassifier із JSON."""
    global _CACHED_BUNDLE
    path = Path(model_json_path)
    if _CACHED_BUNDLE is not None and path == DEFAULT_MODEL_JSON:
        return _CACHED_BUNDLE

    if not path.exists():
        from .train import train_and_select_model

        train_and_select_model(output_json=path, verbose=False)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    fcfg = data.get("feature_config", {})
    extractor = FeatureExtractor(
        time_window_ms=fcfg.get("time_window_ms", 600),
        short_time_window_ms=fcfg.get("short_time_window_ms", 300),
        count_window=fcfg.get("count_window", 8),
        short_count_window=fcfg.get("short_count_window", 4),
    )
    scaler = StandardScaler.from_dict(data["scaler"])
    model = MLPClassifier.from_dict(data["model"])

    if path == DEFAULT_MODEL_JSON:
        _CACHED_BUNDLE = (extractor, scaler, model)
    return extractor, scaler, model


def extract_features(
    events_window: Union[bytes, bytearray, Iterable[Union[LogRecord, Dict[str, Any], bytes, Any]]],
    scaler: Optional[StandardScaler] = None,
    history: Optional[Iterable[Union[LogRecord, Dict[str, Any]]]] = None,
    scale: bool = True,
    model_json_path: Union[str, Path] = DEFAULT_MODEL_JSON,
) -> List[List[float]]:
    """
    Публічний метод побудови часових ознак для вікна подій (Контракт Ролі 1).
    Приймає список LogRecord, словників або 16-байтових записів.
    """
    records = normalize_records(events_window)
    if not records:
        return []
    if scaler is not None:
        extractor = FeatureExtractor()
        raw = extractor.extract_raw(records, history=history)
        return scaler.transform(raw) if scale else raw
    if scale:
        extractor, trained_scaler, _ = load_trained_bundle(model_json_path=model_json_path)
        raw = extractor.extract_raw(records, history=history)
        return trained_scaler.transform(raw)
    extractor = FeatureExtractor()
    return extractor.extract_raw(records, history=history)


def predict_classes(
    events_window: Union[bytes, bytearray, Iterable[Union[LogRecord, Dict[str, Any], bytes, Any]]],
    history: Optional[Iterable[Union[LogRecord, Dict[str, Any]]]] = None,
    model_json_path: Union[str, Path] = DEFAULT_MODEL_JSON,
) -> List[int]:
    """
    Класифікує кожен запис вікна у класи 0 (штатна), 1 (аномальна), 2 (перехід до інциденту).
    Повертає list[int] зі значеннями 0, 1, 2 (рівно по одному класу на вхідний запис).
    """
    records = normalize_records(events_window)
    if not records:
        return []
    extractor, scaler, model = load_trained_bundle(model_json_path=model_json_path)
    raw_feats = extractor.extract_raw(records, history=history)
    scaled_feats = scaler.transform(raw_feats)
    preds = model.predict(scaled_feats)
    return [int(c) for c in preds]


def compute_event_priorities(
    events_window: Union[bytes, bytearray, Iterable[Union[LogRecord, Dict[str, Any], bytes, Any]]],
    classes: Optional[Sequence[int]] = None,
    history: Optional[Iterable[Union[LogRecord, Dict[str, Any]]]] = None,
) -> List[int]:
    """
    Інтеграційний метод для Ролі 3 (Планувальник ДП):
    Обчислює базову цінність кожного запису за формулою з FORMAT_SPEC_P3.md (розд. 3.5):
        v_i = priority(class_i) + critical_bonus * [critical_i == 1]
    """
    records = normalize_records(events_window)
    if not records:
        return []
    if classes is None:
        classes = predict_classes(records, history=history)
    if len(classes) != len(records):
        raise ValueError("Кількість класів не збігається з кількістю записів вікна")
    values: List[int] = []
    for rec, cls_id in zip(records, classes):
        prio = CLASS_PRIORITY_MAP.get(int(cls_id), 1)
        crit_add = CRITICAL_BONUS if int(rec.critical) == 1 else 0
        values.append(prio + crit_add)
    return values


def encode_class_table(classes: Sequence[int]) -> bytes:
    """
    Інтеграційний метод для Ролі 5 (Контейнер LOG3, розд. 3.3 FORMAT_SPEC_P3.md):
    Пакує послідовність класів (0, 1, 2) у побайтовий буфер class_table (1 байт uint8 на запис).
    """
    for idx, c in enumerate(classes):
        if int(c) not in (0, 1, 2):
            raise ValueError(f"Недопустимий клас {c} на позиції {idx} (дозволені 0, 1, 2)")
    return bytes(int(c) for c in classes)


__all__ = [
    "LogRecord",
    "load_csv",
    "normalize_records",
    "validate_dataset",
    "FEATURE_NAMES",
    "FeatureExtractor",
    "StandardScaler",
    "extract_features",
    "MLPClassifier",
    "CLASS_NAMES",
    "CLASS_PRIORITY_MAP",
    "CRITICAL_BONUS",
    "compute_confusion_matrix",
    "compute_macro_f1",
    "evaluate_predictions",
    "format_confusion_matrix",
    "load_trained_bundle",
    "predict_classes",
    "compute_event_priorities",
    "encode_class_table",
]
