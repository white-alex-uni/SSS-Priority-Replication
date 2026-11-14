#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features.py — побудова часових ознак на ковзному вікні історії та нормалізація (Роль 1).

Перші 7 ознак відповідають обов'язковому набору з team_config.json:
  0. dt_prev                     — інтервал до попередньої події
  1. events_in_time_window       — кількість подій у ковзному вікні часу (600 мс)
  2. cat_changes_in_window       — кількість змін category у вікні (W=8)
  3. unique_sources_in_window    — кількість різних source_id у вікні (W=8)
  4. adjacent_code_ratio         — частка пар подій із сусідніми event_code у вікні (W=8)
  5. metric_delta_same_src       — зміна metric_value відносно попередньої події того самого джерела
  6. severity                    — рівень severity

Додаткові ознаки (7..18) поглиблюють опис часової структури на короткому вікні (W=4, 300 мс)
та відхилення metric_value від базового рівня джерела без використання поля critical як мітки.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .data_loader import LogRecord, normalize_records

FEATURE_NAMES: Tuple[str, ...] = (
    "dt_prev",                      # [0] Інтервал до попередньої події (мс)
    "events_in_time_window",        # [1] Кількість подій у ковзному вікні часу (600 мс)
    "cat_changes_in_window",        # [2] Кількість змін category у вікні (W=8)
    "unique_sources_in_window",     # [3] Кількість різних source_id у вікні (W=8)
    "adjacent_code_ratio",          # [4] Частка пар подій із сусідніми event_code у вікні (W=8)
    "metric_delta_same_src",        # [5] Зміна metric_value відносно попередньої події того самого джерела
    "severity",                     # [6] Поточний рівень severity
    "log_metric_delta_same_src",    # [7] log(1 + |Δmetric_value| того самого джерела)
    "dev_from_src_baseline",        # [8] Відхилення metric_value від медіани штатного рівня джерела
    "log_dev_from_src_baseline",    # [9] log(1 + відхилення від базового рівня джерела)
    "is_high_severity",             # [10] Індикатор високої важливості (severity >= 3)
    "events_in_short_time_window",  # [11] Кількість подій у короткому вікні часу (300 мс)
    "cat_changes_short_window",     # [12] Кількість змін category у короткому вікні (W=4)
    "unique_sources_short_window",  # [13] Кількість різних source_id у короткому вікні (W=4)
    "adjacent_code_ratio_short",    # [14] Частка сусідніх event_code у короткому вікні (W=4)
    "mean_dt_short_window",         # [15] Середній інтервал часу у короткому вікні (W=4)
    "is_adjacent_code_prev",        # [16] Миттєвий перехід до сусіднього event_code (|Δcode| <= 1)
    "is_cat_changed_prev",          # [17] Миттєва зміна category відносно i-1
    "is_src_changed_prev",          # [18] Миттєва зміна source_id відносно i-1
)


class StandardScaler:
    """Стандартизація ознак: (x - mean) / std. Навчається виключно на train.csv."""

    def __init__(
        self,
        means: Optional[List[float]] = None,
        stds: Optional[List[float]] = None,
    ) -> None:
        self.means: List[float] = list(means) if means is not None else []
        self.stds: List[float] = list(stds) if stds is not None else []

    def fit(self, X: Sequence[Sequence[float]]) -> "StandardScaler":
        if not X:
            raise ValueError("Неможливо навчити StandardScaler на порожній вибірці")
        n = len(X)
        d = len(X[0])
        self.means = [0.0] * d
        self.stds = [1.0] * d
        for j in range(d):
            mean_j = sum(row[j] for row in X) / n
            var_j = sum((row[j] - mean_j) ** 2 for row in X) / n
            self.means[j] = mean_j
            self.stds[j] = math.sqrt(var_j) if var_j > 1e-12 else 1.0
        return self

    def transform(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        if not X:
            return []
        if not self.means or not self.stds:
            raise RuntimeError("StandardScaler ще не навчений (викличте fit або завантажте параметри)")
        means = self.means
        stds = self.stds
        d = len(means)
        return [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in X]

    def fit_transform(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "means": [round(m, 8) for m in self.means],
            "stds": [round(s, 8) for s in self.stds],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardScaler":
        return cls(
            means=[float(x) for x in data["means"]],
            stds=[float(x) for x in data["stds"]],
        )


class FeatureExtractor:
    """Каузальний екстрактор часових ознак на ковзному вікні історії."""

    def __init__(
        self,
        time_window_ms: int = 600,
        short_time_window_ms: int = 300,
        count_window: int = 8,
        short_count_window: int = 4,
    ) -> None:
        self.time_window_ms = time_window_ms
        self.short_time_window_ms = short_time_window_ms
        self.count_window = count_window
        self.short_count_window = short_count_window

    def extract_raw(
        self,
        records: Iterable[Union[LogRecord, Dict[str, Any]]],
        history: Optional[Iterable[Union[LogRecord, Dict[str, Any]]]] = None,
    ) -> List[List[float]]:
        """
        Обчислює ненормалізовані ознаки для списку записів.
        Якщо передано history (попередні записи сеансу), вони використовуються
        як каузальний контекст на початку вікна.
        """
        target_recs = normalize_records(records)
        if not target_recs:
            return []

        hist_recs = normalize_records(history) if history is not None else []
        all_recs = hist_recs + target_recs
        start_idx = len(hist_recs)
        total_len = len(all_recs)

        # Початкова оцінка нормального рівня для джерел, які на старті вікна ще не мали подій severity <= 1
        fallback_src_baseline: Dict[int, float] = {}
        for r in all_recs:
            if r.severity <= 1 and r.source_id not in fallback_src_baseline:
                fallback_src_baseline[r.source_id] = float(r.metric_value)

        last_src_mv: Dict[int, int] = {}
        src_normal_history: Dict[int, List[int]] = {}

        # Прогрів стану джерел на історичному префіксі (якщо передано history)
        for idx in range(start_idx):
            r = all_recs[idx]
            last_src_mv[r.source_id] = r.metric_value
            if r.severity <= 1:
                hist_list = src_normal_history.setdefault(r.source_id, [])
                if len(hist_list) < 32:
                    hist_list.append(r.metric_value)

        features: List[List[float]] = []
        t_win = self.time_window_ms
        t_short = self.short_time_window_ms
        c_win = self.count_window
        c_short = self.short_count_window

        for i in range(start_idx, total_len):
            r = all_recs[i]
            ts = r.timestamp_ms
            src = r.source_id
            code = r.event_code
            mv = r.metric_value
            cat = r.category
            sev = r.severity

            # 1. Інтервал до попередньої події
            dt_prev = float(ts - all_recs[i - 1].timestamp_ms) if i > 0 else 100.0

            # 2. Кількість подій у ковзному вікні часу [ts - t_win, ts]
            j = i
            while j > 0 and (ts - all_recs[j - 1].timestamp_ms) <= t_win:
                j -= 1
            events_in_time_win = float(i - j + 1)

            # Коротке часове вікно [ts - t_short, ts]
            j_s = i
            while j_s > 0 and (ts - all_recs[j_s - 1].timestamp_ms) <= t_short:
                j_s -= 1
            events_in_short_time_win = float(i - j_s + 1)

            # Вікно за кількістю подій W = count_window
            k0 = max(0, i - c_win + 1)
            win = all_recs[k0 : i + 1]
            m = len(win)

            # 3. Кількість змін category у вікні
            cat_changes = float(
                sum(1 for k in range(1, m) if win[k].category != win[k - 1].category)
            )

            # 4. Кількість різних source_id у вікні
            uniq_sources = float(len({w.source_id for w in win}))

            # 5. Частка пар подій із сусідніми event_code у вікні
            adj_pairs = sum(
                1 for k in range(1, m) if abs(win[k].event_code - win[k - 1].event_code) <= 1
            )
            adj_code_ratio = float(adj_pairs) / (m - 1) if m > 1 else 0.0

            # 6. Зміна metric_value відносно попередньої події того самого джерела
            if src in last_src_mv:
                metric_delta_same_src = float(abs(mv - last_src_mv[src]))
            else:
                base_init = fallback_src_baseline.get(src, float(mv))
                metric_delta_same_src = float(abs(mv - base_init))
            last_src_mv[src] = mv

            # Каузальна оцінка відхилення від нормального базового рівня джерела (severity <= 1)
            if sev <= 1:
                hist_list = src_normal_history.setdefault(src, [])
                if len(hist_list) < 32:
                    hist_list.append(mv)

            norm_hist = src_normal_history.get(src)
            if norm_hist:
                s_hist = sorted(norm_hist)
                baseline_mv = float(s_hist[len(s_hist) // 2])
            else:
                baseline_mv = fallback_src_baseline.get(src, float(mv))
            dev_from_baseline = abs(float(mv) - baseline_mv)

            # 7. Поточний рівень severity
            sev_f = float(sev)

            # Коротке вікно W_short = 4 для швидкої реакції на початок/кінець класу 2
            ks0 = max(0, i - c_short + 1)
            win_s = all_recs[ks0 : i + 1]
            ms = len(win_s)
            cat_chg_s = float(
                sum(1 for k in range(1, ms) if win_s[k].category != win_s[k - 1].category)
            )
            uniq_src_s = float(len({w.source_id for w in win_s}))
            adj_s = sum(
                1 for k in range(1, ms) if abs(win_s[k].event_code - win_s[k - 1].event_code) <= 1
            )
            adj_ratio_s = float(adj_s) / (ms - 1) if ms > 1 else 0.0
            mean_dt_s = (
                float(win_s[-1].timestamp_ms - win_s[0].timestamp_ms) / (ms - 1)
                if ms > 1
                else dt_prev
            )

            # Миттєві індикатори переходу між i-1 та i
            is_adj_prev = (
                1.0 if (i > 0 and abs(code - all_recs[i - 1].event_code) <= 1) else 0.0
            )
            is_cat_chg_prev = (
                1.0 if (i > 0 and cat != all_recs[i - 1].category) else 0.0
            )
            is_src_chg_prev = (
                1.0 if (i > 0 and src != all_recs[i - 1].source_id) else 0.0
            )

            features.append(
                [
                    dt_prev,
                    events_in_time_win,
                    cat_changes,
                    uniq_sources,
                    adj_code_ratio,
                    metric_delta_same_src,
                    sev_f,
                    math.log1p(metric_delta_same_src),
                    dev_from_baseline,
                    math.log1p(dev_from_baseline),
                    1.0 if sev >= 3 else 0.0,
                    events_in_short_time_win,
                    cat_chg_s,
                    uniq_src_s,
                    adj_ratio_s,
                    mean_dt_s,
                    is_adj_prev,
                    is_cat_chg_prev,
                    is_src_chg_prev,
                ]
            )

        return features


def extract_features(
    events_window: Iterable[Union[LogRecord, Dict[str, Any]]],
    scaler: Optional[StandardScaler] = None,
    history: Optional[Iterable[Union[LogRecord, Dict[str, Any]]]] = None,
) -> List[List[float]]:
    """
    Публічна функція побудови матриці ознак згідно з контрактом Ролі 1.
    Якщо передано навчений scaler, повертає стандартизовані ознаки.
    """
    extractor = FeatureExtractor()
    raw = extractor.extract_raw(events_window, history=history)
    if scaler is not None:
        return scaler.transform(raw)
    return raw
