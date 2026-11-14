"""Роль 5: наскрізний пайплайн і єдина метрика Q = alpha*S + beta*R + gamma*T."""
import hashlib
import struct
import time
from pathlib import Path

from .container import pack_replica
from .deps import huffman, serialize_records, to_record
from .recoverer import recover_journal
from ..role_3 import (build_costs_and_utilities, select_copies_dp,  # noqa: F401
                      select_copies_greedy, schedule_batches)

# replication.py лежить у корені репозиторію (видається викладачем, не змінюється)
import replication as _replication  # noqa: E402

NODES = ("A", "B", "C")
_BLOB_OVERHEAD = 75 + 14  # заголовок+HMAC+службові поля payload, байти


def _classes_for(window, history, use_perceptron):
    if not use_perceptron:
        return [0] * len(window)
    from ..role_1 import predict_classes  # Роль 1
    return predict_classes(window, history=history)


def _values_and_weights(records, classes, cfg):
    """Вага w_i (біти за довжинами Гаффмана) і корисності u_i(r) за FORMAT_SPEC_P3 п. 3.5."""
    rep = cfg["реплікація"]
    names = cfg["класифікатор"]["класи"]
    prio = cfg["пріоритети_класів"]
    bonus = float(cfg["critical_bonus"])
    mult = [0.0, 1.0, float(rep["second_copy_factor"]), float(rep["third_copy_factor"])]

    raw = serialize_records(records)
    lengths = dict(huffman.encode(raw)[0])  # {байт: довжина коду}
    weights, utilities = [], []
    for i, rec in enumerate(records):
        w = sum(lengths[b] for b in raw[16 * i:16 * i + 16])
        v = float(prio[names[str(classes[i])]]) + (bonus if rec.critical else 0.0)
        weights.append([r * w for r in range(4)])
        utilities.append([v * m for m in mult])
    return weights, utilities


def _choose_copies(records, weights, utilities, budget, use_dp):
    if use_dp:
        return select_copies_dp(records, weights, utilities, budget)
    # «без ДП»: по одній копії підряд, поки вистачає бюджету
    copies, left = [0] * len(records), budget
    for i in range(len(records)):
        if weights[i][1] <= left:
            copies[i], left = 1, left - weights[i][1]
    return copies


def _best_blob(batch, password, max_bytes, **kw):
    """Пакує batch у найкоротшому режимі; якщо більше за max_bytes - відкидає хвіст."""
    batch = list(batch)
    while batch:
        blobs = [pack_replica(batch, password, m, **kw) for m in ("huffman", "none")]
        blob = min(blobs, key=len)
        if len(blob) <= max_bytes:
            return blob, batch
        batch.pop()
    return None, []


def evaluate_pipeline(session_data, scenario, seed, config):
    """Повертає {"S","R","T","Q", ...}. scenario - назва ("штатний", ...) або словник сценарію."""
    t0 = time.perf_counter()
    if isinstance(session_data, (str, Path)):
        from ..role_1 import load_csv
        session_data = load_csv(session_data)
    rep_cfg = config["реплікація"]
    sc = rep_cfg["сценарії"][scenario] if isinstance(scenario, str) else scenario
    window_n = int(rep_cfg["replication_window_records"])
    max_batch = int(rep_cfg["max_batch_bytes"])
    budget = int(rep_cfg["replication_budget_bits"])
    password = config["пароль_сеансу"]
    use_perc, use_dp = config.get("use_perceptron", True), config.get("use_dp", True)

    records = [to_record(r) for r in session_data]
    rep = _replication.Replication(sc, seed)
    salt = hashlib.sha256(b"salt:%d" % seed).digest()[:16]  # одна сіль на сеанс (PBKDF2 один раз)
    counter, copies_total = 0, [0, 0, 0, 0]

    for w_idx, start in enumerate(range(0, len(records), window_n)):
        win_raw = session_data[start:start + window_n]
        win = records[start:start + window_n]
        hist = session_data[max(0, start - 16):start] or None
        classes = _classes_for(win_raw, hist, use_perc)
        weights, utilities = _values_and_weights(win, classes, config)
        win_budget = budget
        if config.get("adaptive_budget", True):
            # не даємо ранім вікнам вичерпати глобальний бюджет моделі відмов
            windows_left = (len(records) - start + window_n - 1) // window_n
            left_bytes = sc["global_replication_budget_bytes"] - rep.global_used
            per_window_bits = 8 * (left_bytes / windows_left - 3 * _BLOB_OVERHEAD)
            win_budget = int(max(0, min(budget, per_window_bits)))
        copies = _choose_copies(win, weights, utilities, win_budget, use_dp)
        for c in copies:
            copies_total[c] += 1
        nodes = {n: {"reliability_weight": sc["вузли"][n]["reliability_weight"],
                     "remaining_capacity": sc["вузли"][n]["capacity_bytes"] - rep.nodes[n].used}
                 for n in NODES}
        batches = schedule_batches(win, copies, nodes, max_batch)
        for n in NODES:
            if not batches[n]:
                continue
            counter += 1
            nonce = struct.pack(">QI", seed, counter)
            blob, _ = _best_blob(batches[n], password, max_batch, group_id=w_idx % 65536,
                                 batch_seq=counter, salt=salt, nonce=nonce)
            if blob is not None:
                rep.store(n, blob)

    try:
        recovered = recover_journal(rep.surviving_blobs(), password)
    except Exception:  # відновлювач не має падати; якщо впав - Q = 0
        recovered = []
    got = {serialize_records([r]) for r in recovered}
    critical = [serialize_records([r]) for r in records if r.critical]
    crit_ok = sum(1 for c in critical if c in got)
    R = crit_ok / len(critical) if critical else 1.0

    b_ref = 16 * len(records)
    stats = rep.stats()
    S = b_ref / (b_ref + 8 * stats["global_used_bytes"])
    t = time.perf_counter() - t0
    t_ref = float(config["метрика_Q"]["t_ref_секунд"])
    T = t_ref / (t_ref + t)
    m = config["метрика_Q"]
    Q = m["alpha"] * S + m["beta"] * R + m["gamma"] * T
    if crit_ok == 0:
        Q = 0.0
    return {"S": S, "R": R, "T": T, "Q": Q, "time_s": t, "critical_total": len(critical),
            "critical_recovered": crit_ok, "recovered_records": len(recovered),
            "global_used_bytes": stats["global_used_bytes"], "copies_histogram": copies_total}
