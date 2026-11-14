#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loader.py — завантаження, валідація та представлення записів журналу (Роль 1).

Підтримує читання з CSV (train.csv, valid.csv, session.csv), словників,
а також побайтову серіалізацію/десеріалізацію (16 байтів згідно з FORMAT_SPEC_P3.md).
"""

import csv
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

# Формат 16-байтового запису згідно з FORMAT_SPEC_P3.md (Розділ 1):
# uint32 timestamp_ms, uint16 source_id, uint16 event_code, int32 metric_value,
# uint8 category, uint8 severity, uint8 critical, uint8 record_flags
RECORD_STRUCT = struct.Struct(">IHHibBBB")


@dataclass
class LogRecord:
    timestamp_ms: int
    source_id: int
    event_code: int
    metric_value: int
    category: int
    severity: int
    critical: int
    record_flags: int
    label: Optional[int] = None

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "LogRecord":
        label_val = row.get("label")
        label = int(label_val) if label_val not in (None, "") else None
        return cls(
            timestamp_ms=int(row["timestamp_ms"]),
            source_id=int(row["source_id"]),
            event_code=int(row["event_code"]),
            metric_value=int(row["metric_value"]),
            category=int(row["category"]),
            severity=int(row["severity"]),
            critical=int(row["critical"]),
            record_flags=int(row["record_flags"]),
            label=label,
        )

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "timestamp_ms": self.timestamp_ms,
            "source_id": self.source_id,
            "event_code": self.event_code,
            "metric_value": self.metric_value,
            "category": self.category,
            "severity": self.severity,
            "critical": self.critical,
            "record_flags": self.record_flags,
        }
        if self.label is not None:
            out["label"] = self.label
        return out

    def to_bytes(self) -> bytes:
        return RECORD_STRUCT.pack(
            self.timestamp_ms,
            self.source_id,
            self.event_code,
            self.metric_value,
            self.category,
            self.severity,
            self.critical,
            self.record_flags,
        )

    @classmethod
    def from_bytes(cls, data: bytes, label: Optional[int] = None) -> "LogRecord":
        if len(data) != 16:
            raise ValueError(f"Очікується 16 байтів на запис, отримано {len(data)}")
        ts, src, code, mv, cat, sev, crit, flags = RECORD_STRUCT.unpack(data)
        return cls(
            timestamp_ms=ts,
            source_id=src,
            event_code=code,
            metric_value=mv,
            category=cat,
            severity=sev,
            critical=crit,
            record_flags=flags,
            label=label,
        )


def normalize_records(
    records: Union[bytes, bytearray, Iterable[Union[LogRecord, Dict[str, Any], bytes, bytearray, Any]]]
) -> List[LogRecord]:
    """Приводить послідовність словників, 16-байтових буферів або LogRecord до списку LogRecord."""
    if isinstance(records, (bytes, bytearray)):
        if len(records) % 16 != 0:
            raise ValueError(
                f"Довжина бінарного буфера записів має бути кратною 16, отримано {len(records)}"
            )
        return [
            LogRecord.from_bytes(bytes(records[offset : offset + 16]))
            for offset in range(0, len(records), 16)
        ]

    out: List[LogRecord] = []
    for r in records:
        if isinstance(r, LogRecord):
            out.append(r)
        elif isinstance(r, dict):
            out.append(LogRecord.from_dict(r))
        elif isinstance(r, (bytes, bytearray)):
            out.append(LogRecord.from_bytes(bytes(r)))
        else:
            # Підтримка об'єктів з однойменними атрибутами від інших модулів команди
            out.append(
                LogRecord(
                    timestamp_ms=int(getattr(r, "timestamp_ms")),
                    source_id=int(getattr(r, "source_id")),
                    event_code=int(getattr(r, "event_code")),
                    metric_value=int(getattr(r, "metric_value")),
                    category=int(getattr(r, "category")),
                    severity=int(getattr(r, "severity")),
                    critical=int(getattr(r, "critical")),
                    record_flags=int(getattr(r, "record_flags")),
                    label=getattr(r, "label", None),
                )
            )
    return out


def load_csv(path: Union[str, Path]) -> List[LogRecord]:
    """Завантажує CSV-файл журналу подій у список LogRecord."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [LogRecord.from_dict(row) for row in reader]


def extract_labels(records: Iterable[LogRecord]) -> List[int]:
    """Повертає список цільових міток класів (0, 1, 2) із записів."""
    labels: List[int] = []
    for idx, r in enumerate(records):
        if r.label is None:
            raise ValueError(f"Запис #{idx} не містить цільової мітки 'label'")
        labels.append(r.label)
    return labels


def validate_dataset(records: List[LogRecord]) -> Dict[str, Any]:
    """Перевіряє коректність діапазонів полів та часової впорядкованості набору."""
    n = len(records)
    if n == 0:
        return {"count": 0, "valid": True}

    strictly_increasing = all(
        records[i].timestamp_ms < records[i + 1].timestamp_ms for i in range(n - 1)
    )
    ranges_ok = True
    for r in records:
        if not (0 <= r.timestamp_ms <= 0xFFFFFFFF):
            ranges_ok = False
        if not (1 <= r.source_id <= 4095):
            ranges_ok = False
        if not (0 <= r.event_code <= 0xFFFF):
            ranges_ok = False
        if not (-0x80000000 <= r.metric_value <= 0x7FFFFFFF):
            ranges_ok = False
        if not (0 <= r.category <= 15):
            ranges_ok = False
        if not (0 <= r.severity <= 7):
            ranges_ok = False
        if r.critical not in (0, 1):
            ranges_ok = False
        if not (0 <= r.record_flags <= 3):
            ranges_ok = False
        if r.label is not None and r.label not in (0, 1, 2):
            ranges_ok = False

    class_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
    has_labels = all(r.label is not None for r in records)
    if has_labels:
        for r in records:
            class_counts[int(r.label)] += 1  # type: ignore[arg-type]

    return {
        "count": n,
        "strictly_increasing_timestamps": strictly_increasing,
        "format_ranges_valid": ranges_ok,
        "valid": strictly_increasing and ranges_ok,
        "class_distribution": class_counts if has_labels else None,
    }
