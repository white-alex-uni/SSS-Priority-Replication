"""
Розумна захищена система пріоритетної реплікації журналів подій (LOG3).
Головна точка входу для демонстрації та роботи продукту:
Передавач -> Реплікація по вузлах A, B, C -> Відновлювач.
"""

import csv
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

from replication import Replication
from src.role_1 import load_csv, predict_classes
from src.role_3 import schedule_batches, select_copies_dp
from src.role_5.deps import serialize_records, to_record
from src.role_5.pipeline import (
    NODES,
    _BLOB_OVERHEAD,
    _best_blob,
    _values_and_weights,
    evaluate_pipeline,
)
from src.role_5.recoverer import recover_journal

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "team_config.json"
SESSION_PATH = ROOT / "data" / "session.csv"
RESULTS_DIR = ROOT / "results"
REPLICAS_DIR = RESULTS_DIR / "replicas"
RECOVERED_CSV_PATH = RESULTS_DIR / "recovered_journal.csv"

CSV_FIELDS = (
    "timestamp_ms",
    "source_id",
    "event_code",
    "metric_value",
    "category",
    "severity",
    "critical",
    "record_flags",
)


def _configure_console() -> None:
    """Налаштовує UTF-8 для коректного відображення кирилиці в консолі Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _save_node_replicas(blobs_by_node: dict[str, list[bytes]]) -> dict[str, Path]:
    """Очищує старі репліки та зберігає актуальні .bin-пакети по папках вузлів."""
    saved_dirs: dict[str, Path] = {}
    for node in NODES:
        node_dir = REPLICAS_DIR / f"node_{node}"
        node_dir.mkdir(parents=True, exist_ok=True)
        for old_file in node_dir.glob("*.bin"):
            old_file.unlink()
        for idx, blob in enumerate(blobs_by_node[node], start=1):
            (node_dir / f"replica_{idx:04d}.bin").write_bytes(blob)
        saved_dirs[node] = node_dir
    return saved_dirs


def _load_saved_replicas(node_dirs: dict[str, Path]) -> list[bytes]:
    """Зчитує збережені .bin-репліки з папок вузлів A, B, C для відновлення."""
    blobs: list[bytes] = []
    for node in NODES:
        for file_path in sorted(node_dirs[node].glob("*.bin")):
            blobs.append(file_path.read_bytes())
    return blobs


def _save_recovered_csv(records: list, output_path: Path) -> None:
    """Записує відновлений і впорядкований журнал подій у CSV-файл."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        for rec in records:
            writer.writerow(
                [
                    rec.timestamp_ms,
                    rec.source_id,
                    rec.event_code,
                    rec.metric_value,
                    rec.category,
                    rec.severity,
                    rec.critical,
                    rec.record_flags,
                ]
            )


def _compute_metrics(records, recovered, rep_stats: dict, elapsed_s: float, cfg: dict) -> dict[str, float]:
    """Обчислює метрики якості S, R, T, Q згідно з конфігурацією системи."""
    got = {serialize_records([r]) for r in recovered}
    critical = [serialize_records([r]) for r in records if r.critical]
    crit_ok = sum(1 for c in critical if c in got)
    r_val = crit_ok / len(critical) if critical else 1.0

    b_ref = 16 * len(records)
    used_bytes = int(rep_stats["global_used_bytes"])
    s_val = b_ref / (b_ref + 8 * used_bytes) if (b_ref + 8 * used_bytes) > 0 else 0.0

    t_ref = float(cfg["метрика_Q"]["t_ref_секунд"])
    t_val = t_ref / (t_ref + elapsed_s)

    m = cfg["метрика_Q"]
    q_val = m["alpha"] * s_val + m["beta"] * r_val + m["gamma"] * t_val
    if crit_ok == 0:
        q_val = 0.0

    return {
        "S": s_val,
        "R": r_val,
        "T": t_val,
        "Q": q_val,
        "time_s": elapsed_s,
        "critical_total": len(critical),
        "critical_recovered": crit_ok,
        "recovered_records": len(recovered),
        "global_used_bytes": used_bytes,
    }


def _run_primary_cycle(session_data: list, cfg: dict, scenario_name: str = "штатний") -> dict:
    """Виконує повний цикл Передавач -> Реплікація -> Відновлювач для обраного сценарію."""
    rep_cfg = cfg["реплікація"]
    scenario = rep_cfg["сценарії"][scenario_name]
    seed = int(rep_cfg["seed_моделі_реплікації"])
    window_n = int(rep_cfg["replication_window_records"])
    max_batch = int(rep_cfg["max_batch_bytes"])
    budget = int(rep_cfg["replication_budget_bits"])
    password = cfg["пароль_сеансу"]

    records = [to_record(r) for r in session_data]
    rep = Replication(scenario, seed)
    salt = hashlib.sha256(b"salt:%d" % seed).digest()[:16]

    class_counts = {0: 0, 1: 0, 2: 0}
    copies_total = [0, 0, 0, 0]
    packets_built = 0
    packed_bytes_total = 0
    counter = 0

    t0 = time.perf_counter()

    # Етап 1–2: Робота Передавача по вікнах та відправка реплік на вузли A, B, C
    for w_idx, start in enumerate(range(0, len(records), window_n)):
        win_raw = session_data[start : start + window_n]
        win = records[start : start + window_n]
        hist = session_data[max(0, start - 16) : start] or None

        classes = predict_classes(win_raw, history=hist)
        for cls_id in classes:
            class_counts[int(cls_id)] = class_counts.get(int(cls_id), 0) + 1

        weights, utilities = _values_and_weights(win, classes, cfg)

        windows_left = (len(records) - start + window_n - 1) // window_n
        left_bytes = scenario["global_replication_budget_bytes"] - rep.global_used
        per_window_bits = 8 * (left_bytes / windows_left - 3 * _BLOB_OVERHEAD)
        win_budget = int(max(0, min(budget, per_window_bits)))

        copies = select_copies_dp(win, weights, utilities, win_budget)
        for c in copies:
            copies_total[c] += 1

        nodes_state = {
            n: {
                "reliability_weight": scenario["вузли"][n]["reliability_weight"],
                "remaining_capacity": scenario["вузли"][n]["capacity_bytes"] - rep.nodes[n].used,
            }
            for n in NODES
        }
        batches = schedule_batches(win, copies, nodes_state, max_batch)

        for n in NODES:
            if not batches[n]:
                continue
            counter += 1
            nonce = struct.pack(">QI", seed, counter)
            blob, _ = _best_blob(
                batches[n],
                password,
                max_batch,
                group_id=w_idx % 65536,
                batch_seq=counter,
                salt=salt,
                nonce=nonce,
            )
            if blob is not None:
                packets_built += 1
                packed_bytes_total += len(blob)
                rep.store(n, blob)

    # Збереження вцілілих зашифрованих реплік по вузлах на диск
    node_dirs = _save_node_replicas(rep.surviving_blobs_by_node())

    # Етап 3: Робота Відновлювача (читання з диска, перевірка HMAC, дешифрування, дедуплікація)
    blobs_from_disk = _load_saved_replicas(node_dirs)
    recovered = recover_journal(blobs_from_disk, password)
    elapsed_s = time.perf_counter() - t0

    _save_recovered_csv(recovered, output_path=Path(RECOVERED_CSV_PATH.parent / f"{RECOVERED_CSV_PATH.stem}_{scenario_name}{RECOVERED_CSV_PATH.suffix}"))
    rep_stats = rep.stats()
    metrics = _compute_metrics(records, recovered, rep_stats, elapsed_s, cfg)

    return {
        "scenario_name": scenario_name,
        "seed": seed,
        "total_records": len(records),
        "windows_count": (len(records) + window_n - 1) // window_n,
        "class_counts": class_counts,
        "copies_total": copies_total,
        "packets_built": packets_built,
        "packed_bytes_total": packed_bytes_total,
        "rep_stats": rep_stats,
        "node_dirs": node_dirs,
        "metrics": metrics,
    }


def _print_report(primary: dict, all_scenarios: dict[str, dict]) -> None:
    """Друкує структурований україномовний звіт про роботу системи."""
    m = primary["metrics"]
    cls = primary["class_counts"]
    copies = primary["copies_total"]
    stats = primary["rep_stats"]
    raw_bytes = primary["total_records"] * 16

    print("=" * 74)
    print("  РОЗУМНА ЗАХИЩЕНА СИСТЕМА ПРІОРИТЕТНОЇ РЕПЛІКАЦІЇ ЖУРНАЛІВ ПОДІЙ")
    print("=" * 74)
    print(f"Вхідний журнал : {SESSION_PATH.relative_to(ROOT)} ({primary['total_records']} подій, {raw_bytes} байт)")
    print(f"Конфігурація   : {CONFIG_PATH.relative_to(ROOT)}")
    print(f"Сценарій       : {primary['scenario_name']} (seed={primary['seed']}, вікон={primary['windows_count']})")

    print("\n--- ЕТАП 1. ПЕРЕДАВАЧ (ML-класифікація, ДП-планування, LOG3) ---")
    print("Класифікація подій перцептроном:")
    print(f"  * Штатні події (клас 0)        : {cls[0]}")
    print(f"  * Аномальні події (клас 1)     : {cls[1]}")
    print(f"  * Перехід до інциденту (клас 2): {cls[2]}")
    print("Пріоритетний розподіл копій (ДП-рюкзак):")
    print(
        f"  * 0 копій: {copies[0]} | 1 копія: {copies[1]} | "
        f"2 копії: {copies[2]} | 3 копії: {copies[3]}"
    )
    print(
        f"Сформовано контейнерів LOG3 (Гаффман/RAW + AES-256-CTR + HMAC-SHA256): "
        f"{primary['packets_built']} пакетів ({primary['packed_bytes_total']} байт)"
    )

    print("\n--- ЕТАП 2. РОЗПОДІЛЕНА РЕПЛІКАЦІЯ ПО ВУЗЛАХ A, B, C ---")
    print(f"{'Вузол':<7} | {'Відправлено':<12} | {'Збережено':<10} | {'Втрачено':<9} | {'Використано / Ліміт (байт)'}")
    print("-" * 74)
    for node in NODES:
        ns = stats["вузли"][node]
        print(
            f"  {node:<5} | {ns['attempts']:<12} | {ns['stored']:<10} | "
            f"{ns['failed']:<9} | {ns['used_bytes']} / {ns['capacity_bytes']}"
        )
    print(
        f"Глобальний бюджет мережі: {stats['global_used_bytes']} / {stats['global_budget_bytes']} байт"
    )

    print("\n--- ЕТАП 3. ВІДНОВЛЮВАЧ (Верифікація HMAC, розпакування, дедуплікація) ---")
    print(f"Відновлено унікальних подій     : {m['recovered_records']} із {primary['total_records']}")
    print(
        f"Збережено критичних подій (R)   : {m['critical_recovered']} із {m['critical_total']} "
        f"({m['R'] * 100:.2f}%)"
    )
    print(
        f"Підсумкові метрики якості       : "
        f"S = {m['S']:.4f} | R = {m['R']:.4f} | T = {m['T']:.4f} | Q = {m['Q']:.4f}"
    )
    print(f"Час виконання повного циклу     : {m['time_s']:.3f} с")

    print("\n--- ЗВЕДЕНА ПЕРЕВІРКА СТІЙКОСТІ ПО ВСІХ СЦЕНАРІЯХ ---")
    print(f"{'Сценарій':<14} | {'Критичні (R)':<20} | {'Байт у мережі':<15} | {'S':<7} | {'Q':<7}")
    print("-" * 74)
    for sc_name, res in all_scenarios.items():
        crit_str = f"{res['critical_recovered']}/{res['critical_total']} ({res['R'] * 100:.1f}%)"
        print(
            f"{sc_name:<14} | {crit_str:<20} | {res['global_used_bytes']:<15} | "
            f"{res['S']:<7.4f} | {res['Q']:<7.4f}"
        )

    print("\n--- ЗБЕРЕЖЕНІ РЕЗУЛЬТАТИ ПРОДУКТУ ---")
    for node in NODES:
        rel_dir = primary["node_dirs"][node].relative_to(ROOT)
        count = len(list(primary["node_dirs"][node].glob("*.bin")))
        print(f"  * Репліки вузла {node} : {rel_dir} ({count} файлів .bin)")
    print(f"  * Відновлений журнал: {RECOVERED_CSV_PATH.relative_to(ROOT)}")
    print("=" * 74)


def main() -> None:
    _configure_console()

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Не знайдено файл конфігурації: {CONFIG_PATH}")
    if not SESSION_PATH.exists():
        raise FileNotFoundError(f"Не знайдено вхідний журнал подій: {SESSION_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    session_data = load_csv(SESSION_PATH)
    seed = int(cfg["реплікація"]["seed_моделі_реплікації"])

    primary = _run_primary_cycle(session_data, cfg, scenario_name="штатний")

    all_scenarios: dict[str, dict] = {"штатний": primary["metrics"]}
    for sc_name in cfg["реплікація"]["сценарії"]:
        if sc_name not in all_scenarios:
            all_scenarios[sc_name] = evaluate_pipeline(session_data, sc_name, seed, cfg)

    _print_report(primary, all_scenarios)


if __name__ == "__main__":
    main()
