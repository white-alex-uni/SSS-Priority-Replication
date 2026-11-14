#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scheduler.py — модуль планування відправки подій (Роль 3).
Реалізує точний контракт API (select_copies_dp та schedule_batches),
жадібний baseline та обчислення вартості/корисності згідно з team_config.json (T01).
"""

from typing import List, Dict, Any, Tuple, Optional

MAX_COPIES = 3
EVENT_SIZE_BYTES = 16

# Базові константи згідно з team_config.json (команда T01, узгоджено з Роллю 1)
DEFAULT_CLASS_PRIORITY: Dict[int, float] = {
    0: 1.0,  # штатна подія
    1: 5.0,  # аномальна подія
    2: 4.0,  # перехід до інциденту
}
DEFAULT_CRITICAL_BONUS: float = 5.0
DEFAULT_COPY_MULTIPLIERS: Dict[int, float] = {
    0: 0.0,
    1: 1.0,
    2: 1.5,
    3: 1.75,
}


def build_costs_and_utilities(
    events: List[Any],
    classes: Optional[List[int]] = None,
    team_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Обчислює вартість cost_i(r) = r * w_i та корисність u_i(r) для r in {0, 1, 2, 3}.
    
    :param events: Список подій (LogRecord).
    :param classes: Список класів (0, 1, 2) від Ролі 1 (predict_classes). Якщо None — береться event.label.
    :param team_config: Конфігурація з team_config.json.
    :return: (weights, utilities) розміром n x 4.
    """
    cfg = team_config or {}
    class_prio_map = cfg.get("class_priority", DEFAULT_CLASS_PRIORITY)
    critical_bonus = float(cfg.get("critical_bonus", DEFAULT_CRITICAL_BONUS))
    copy_mults = cfg.get("reliability_coeffs", DEFAULT_COPY_MULTIPLIERS)

    weights: List[List[int]] = []
    utilities: List[List[float]] = []

    for idx, event in enumerate(events):
        # Визначаємо клас події (пріоритет — передбачений клас від Ролі 1)
        if classes is not None and idx < len(classes):
            cls_id = int(classes[idx])
        else:
            raw_label = getattr(event, "label", 0)
            cls_id = int(raw_label) if raw_label is not None else 0

        # Базова цінність за формулою v_i = priority(class_i) + critical_bonus * [critical_i == 1]
        base_priority = float(class_prio_map.get(cls_id, class_prio_map.get(str(cls_id), 1.0)))
        if int(getattr(event, "critical", 0) or 0) == 1:
            base_priority += critical_bonus

        event_weights: List[int] = []
        event_utilities: List[float] = []

        for r in range(MAX_COPIES + 1):
            cost_r = r * EVENT_SIZE_BYTES
            mult_r = float(copy_mults.get(r, copy_mults.get(str(r), 0.0)))
            utility_r = base_priority * mult_r

            event_weights.append(cost_r)
            event_utilities.append(utility_r)

        weights.append(event_weights)
        utilities.append(event_utilities)

    return weights, utilities


def select_copies_dp(
    events: List[Any],
    weights: List[List[int]],
    utilities: List[List[float]],
    budget_W: int,
) -> List[int]:
    """
    Визначає оптимальну кількість копій (0..3) для кожної події за допомогою 
    динамічного програмування (багатоваріантний рюкзак, складність O(n * W * R_max)).
    """
    n = len(events)
    if n == 0:
        return []
    if budget_W <= 0:
        return [0] * n

    dp = [[0.0] * (budget_W + 1) for _ in range(n + 1)]
    choice = [[0] * (budget_W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(budget_W + 1):
            dp[i][w] = dp[i - 1][w]
            choice[i][w] = 0

            for r in range(len(weights[i - 1])):
                r_weight = weights[i - 1][r]
                r_utility = utilities[i - 1][r]

                if w >= r_weight:
                    val = dp[i - 1][w - r_weight] + r_utility
                    if val > dp[i][w]:
                        dp[i][w] = val
                        choice[i][w] = r

    copies = [0] * n
    curr_w = budget_W
    for i in range(n, 0, -1):
        r = choice[i][curr_w]
        copies[i - 1] = r
        curr_w -= weights[i - 1][r]

    return copies


def select_copies_greedy(
    events: List[Any],
    weights: List[List[int]],
    utilities: List[List[float]],
    budget_W: int,
) -> List[int]:
    """
    Жадібний baseline для вибору кількості копій (для порівняння з ДП у звіті та Кейсі №6).
    На кожному кроці додає копію події з максимальним відношенням Δu / Δcost.
    """
    n = len(events)
    if n == 0 or budget_W <= 0:
        return [0] * n

    copies = [0] * n
    remaining_w = budget_W

    while remaining_w > 0:
        best_idx = -1
        best_efficiency = 0.0

        for i in range(n):
            curr_r = copies[i]
            next_r = curr_r + 1

            if next_r < len(weights[i]):
                delta_cost = weights[i][next_r] - weights[i][curr_r]
                delta_u = utilities[i][next_r] - utilities[i][curr_r]

                if delta_cost <= remaining_w and delta_u > 0:
                    efficiency = (delta_u / delta_cost) if delta_cost > 0 else float("inf")
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_idx = i

        if best_idx == -1:
            break

        curr_r = copies[best_idx]
        remaining_w -= weights[best_idx][curr_r + 1] - weights[best_idx][curr_r]
        copies[best_idx] += 1

    return copies


def schedule_batches(
    events: List[Any],
    copies: List[int],
    node_configs: Dict[str, Any],
    max_batch_bytes: int,
) -> Dict[str, List[Any]]:
    """
    Жадібний розподіл копій по вузлах (A, B, C):
    - Сортує події за важливістю (критичні йдуть першими).
    - Кладе копії однієї події суворо на РІЗНІ вузли.
    - Враховує reliability_weight вузлів, max_batch_bytes та залишкову місткість вузла.
    """
    batches: Dict[str, List[Any]] = {node_id: [] for node_id in node_configs.keys()}
    if not node_configs or not events:
        return batches

    sorted_nodes = sorted(
        node_configs.keys(),
        key=lambda n: node_configs[n].get("reliability_weight", node_configs[n].get("reliability", 0.0)),
        reverse=True,
    )

    batch_bytes_used = {node_id: 0 for node_id in sorted_nodes}

    active_indices = [i for i, c in enumerate(copies) if c > 0]
    active_indices.sort(
        key=lambda i: (
            getattr(events[i], "critical", 0) or 0,
            copies[i],
            getattr(events[i], "severity", 0) or 0,
        ),
        reverse=True,
    )

    for idx in active_indices:
        event = events[idx]
        needed_copies = copies[idx]
        placed_copies = 0

        for node_id in sorted_nodes:
            if placed_copies >= needed_copies:
                break

            node_cfg = node_configs[node_id]
            remaining_capacity = node_cfg.get("remaining_capacity", node_cfg.get("capacity", float("inf")))

            fits_in_batch = (batch_bytes_used[node_id] + EVENT_SIZE_BYTES) <= max_batch_bytes
            fits_in_node = (batch_bytes_used[node_id] + EVENT_SIZE_BYTES) <= remaining_capacity

            if fits_in_batch and fits_in_node:
                batches[node_id].append(event)
                batch_bytes_used[node_id] += EVENT_SIZE_BYTES
                placed_copies += 1

    return batches

def _event_importance_weights(
    events: List[Any],
    classes: Optional[List[int]] = None,
    team_config: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Допоміжна функція: обчислює вагу важливості w_k кожної події для оцінки ризику втрати."""
    cfg = team_config or {}
    class_prio_map = cfg.get("class_priority", DEFAULT_CLASS_PRIORITY)
    critical_bonus = float(cfg.get("critical_bonus", DEFAULT_CRITICAL_BONUS))

    weights: List[float] = []
    for idx, event in enumerate(events):
        if classes is not None and idx < len(classes):
            cls_id = int(classes[idx])
        else:
            raw_label = getattr(event, "label", 0)
            cls_id = int(raw_label) if raw_label is not None else 0

        w = float(class_prio_map.get(cls_id, class_prio_map.get(str(cls_id), 1.0)))
        if int(getattr(event, "critical", 0) or 0) == 1:
            w += critical_bonus
        weights.append(w)
    return weights


def optimize_checkpoints_dp(
    events: List[Any],
    classes: Optional[List[int]] = None,
    snapshot_cost: float = 25.0,
    fail_prob: float = 0.15,
    team_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[int]]:
    """
    Задача Б (Просунутий рівень): оптимальні контрольні точки журналу (snapshots) через ДП.
    F[j] = min_{i < j} (F[i] + Snapshot(i, j) + RecoveryCost(i, j)), складність O(N^2).

    :return: (мінімальна сумарна вартість F[N], список індексів контрольних точок).
    """
    n = len(events)
    if n == 0:
        return 0.0, []

    w = _event_importance_weights(events, classes=classes, team_config=team_config)

    # F[j] — мінімальна вартість для префікса довжини j (зі знімком у точці j)
    F = [float("inf")] * (n + 1)
    prev_idx = [-1] * (n + 1)
    F[0] = 0.0

    for j in range(1, n + 1):
        # Накопичуємо суму w[k] * (j - k) при русі i від j-1 до 0 за O(1) на крок
        weighted_delay_sum = 0.0
        for i in range(j - 1, -1, -1):
            weighted_delay_sum += w[i] * (j - i)
            recovery_cost = fail_prob * weighted_delay_sum
            total_cost = F[i] + snapshot_cost + recovery_cost

            if total_cost < F[j]:
                F[j] = total_cost
                prev_idx[j] = i

    # Відновлення індексів контрольних точок
    checkpoints: List[int] = []
    curr = n
    while curr > 0:
        checkpoints.append(curr)
        curr = prev_idx[curr]
    checkpoints.reverse()

    return F[n], checkpoints


def evaluate_fixed_checkpoints(
    events: List[Any],
    step: int,
    classes: Optional[List[int]] = None,
    snapshot_cost: float = 25.0,
    fail_prob: float = 0.15,
    team_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[int]]:
    """
    Baseline для Задачі Б: розставляє контрольні точки з фіксованим кроком `step`
    і рахує вартість за тією ж цільовою функцією для порівняння з ДП.
    """
    n = len(events)
    if n == 0:
        return 0.0, []

    step = max(1, step)
    w = _event_importance_weights(events, classes=classes, team_config=team_config)

    checkpoints = list(range(step, n + 1, step))
    if not checkpoints or checkpoints[-1] != n:
        checkpoints.append(n)

    total_cost = 0.0
    prev = 0
    for cp in checkpoints:
        weighted_delay_sum = sum(w[k] * (cp - k) for k in range(prev, cp))
        total_cost += snapshot_cost + fail_prob * weighted_delay_sum
        prev = cp

    return total_cost, checkpoints



if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    # Додаємо корінь проєкту в sys.path для імпорту пакету role_1
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.role_1 import load_csv, predict_classes

    # 1. Завантажуємо дані Ролі 1 (session.csv або valid.csv)
    data_dir = project_root / "data"
    dataset_path = data_dir / "session.csv" if (data_dir / "session.csv").exists() else data_dir / "valid.csv"
    all_records = load_csv(dataset_path)
    print(f"Завантажено {len(all_records)} записів із {dataset_path.name}")

    # Розбиваємо на 5 вікон по 48 записів (240 подій) для експерименту
    window_size = 48
    windows = [all_records[i : i + window_size] for i in range(0, min(len(all_records), 240), window_size)]

    # Конфігурація вузлів A, B, C (штатний режим)
    nodes_normal = {
        "Node_A": {"reliability_weight": 0.99, "remaining_capacity": 2000},
        "Node_B": {"reliability_weight": 0.90, "remaining_capacity": 2000},
        "Node_C": {"reliability_weight": 0.80, "remaining_capacity": 2000},
    }
    max_batch = 1024

    # =========================================================================
    # ТАБЛИЦЯ 1: «ДП проти жадібного» (перевірка Кейсу C1 №6)
    # =========================================================================
    print("\n=== ТАБЛИЦЯ 1: ДП ПРОТИ ЖАДІБНОГО BASELINE (усереднено по вікнах) ===")
    print(f"{'Бюджет W (байт)':<18} | {'U (ДП)':<10} | {'U (Жадіб.)':<12} | {'Час ДП (мс)':<12} | {'Кейс №6 (U_dp >= U_gr)'}")
    print("-" * 78)

    for budget_W in [384, 768, 1152, 1536]:
        total_u_dp, total_u_gr, total_time_dp = 0.0, 0.0, 0.0
        case_6_ok = True

        for win in windows:
            preds = predict_classes(win)
            weights, utilities = build_costs_and_utilities(win, classes=preds)

            t0 = time.perf_counter()
            copies_dp = select_copies_dp(win, weights, utilities, budget_W)
            total_time_dp += (time.perf_counter() - t0) * 1000.0

            copies_gr = select_copies_greedy(win, weights, utilities, budget_W)

            u_dp = sum(utilities[i][copies_dp[i]] for i in range(len(win)))
            u_gr = sum(utilities[i][copies_gr[i]] for i in range(len(win)))

            if u_dp + 1e-9 < u_gr:
                case_6_ok = False

            total_u_dp += u_dp
            total_u_gr += u_gr

        n_win = len(windows)
        status = "PASS" if case_6_ok else "FAIL"
        print(f"{budget_W:<18} | {total_u_dp/n_win:<10.2f} | {total_u_gr/n_win:<12.2f} | {total_time_dp/n_win:<12.3f} | {status}")

    # =========================================================================
    # ТАБЛИЦЯ 2: «Вплив кількості копій (1/2/3) на S, R, Q» (Кейси C1 №7 та №8)
    # =========================================================================
    print("\n=== ТАБЛИЦЯ 2: ВПЛИВ КІЛЬКОСТІ КОПІЙ (r = 1, 2, 3) НА S, R, Q ===")
    print(f"{'Режим (r)':<12} | {'S (байт)':<10} | {'R_norm (Кейс №7)':<18} | {'R_degr (Кейс №8)':<18} | {'Q (Корисність)'}")
    print("-" * 82)

    for r_fixed in [1, 2, 3]:
        tot_S, tot_Q = 0, 0.0
        crit_total, crit_rec_normal, crit_rec_degraded = 0, 0, 0

        for win in windows:
            preds = predict_classes(win)
            weights, utilities = build_costs_and_utilities(win, classes=preds)
            fixed_copies = [r_fixed] * len(win)

            batches = schedule_batches(win, fixed_copies, nodes_normal, max_batch_bytes=max_batch)

            tot_S += sum(len(b) * EVENT_SIZE_BYTES for b in batches.values())
            tot_Q += sum(utilities[i][r_fixed] for i in range(len(win)))

            survived_normal = {id(ev) for node in ["Node_A", "Node_B", "Node_C"] for ev in batches[node]}
            survived_degraded = {id(ev) for node in ["Node_B", "Node_C"] for ev in batches[node]}

            for ev in win:
                if int(getattr(ev, "critical", 0) or 0) == 1:
                    crit_total += 1
                    if id(ev) in survived_normal:
                        crit_rec_normal += 1
                    if id(ev) in survived_degraded:
                        crit_rec_degraded += 1

        n_win = len(windows)
        r_norm_pct = (crit_rec_normal / crit_total * 100.0) if crit_total else 100.0
        r_degr_pct = (crit_rec_degraded / crit_total * 100.0) if crit_total else 100.0
        print(f"r = {r_fixed:<8} | {tot_S/n_win:<10.0f} | {r_norm_pct:>6.1f}% ({crit_rec_normal}/{crit_total}) | {r_degr_pct:>6.1f}% ({crit_rec_degraded}/{crit_total}) | {tot_Q/n_win:<10.2f}")
        
        # =========================================================================
    # ТАБЛИЦЯ 3: ЗАДАЧА Б — ОПТИМАЛЬНІ КОНТРОЛЬНІ ТОЧКИ (DP ПРОТИ ФІКСОВАНОГО КРОКУ)
    # =========================================================================
    print("\n=== ТАБЛИЦЯ 3: ЗАДАЧА Б — SNAPSHOTS (ДП ПРОТИ ФІКСОВАНОГО КРОКУ) ===")
    print(f"{'Фікс. крок K':<14} | {'Вартість Фікс.':<16} | {'Вартість ДП':<14} | {'Економія (%)':<14} | {'Перевірка (DP <= Fixed)'}")
    print("-" * 82)

    for fixed_step in [4, 8, 12, 16]:
        tot_dp_cost, tot_fixed_cost = 0.0, 0.0
        dp_always_better = True

        for win in windows:
            preds = predict_classes(win)
            dp_cost, _ = optimize_checkpoints_dp(win, classes=preds, snapshot_cost=25.0, fail_prob=0.15)
            fx_cost, _ = evaluate_fixed_checkpoints(win, step=fixed_step, classes=preds, snapshot_cost=25.0, fail_prob=0.15)

            if dp_cost > fx_cost + 1e-9:
                dp_always_better = False

            tot_dp_cost += dp_cost
            tot_fixed_cost += fx_cost

        n_win = len(windows)
        avg_dp = tot_dp_cost / n_win
        avg_fx = tot_fixed_cost / n_win
        saving_pct = ((avg_fx - avg_dp) / avg_fx * 100.0) if avg_fx > 0 else 0.0
        status = "PASS" if dp_always_better else "FAIL"
        print(f"K = {fixed_step:<10} | {avg_fx:<16.2f} | {avg_dp:<14.2f} | {saving_pct:>10.1f}%   | {status}")