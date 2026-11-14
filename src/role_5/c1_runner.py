"""Роль 5: автоматичний прогін 12 приймальних кейсів C1. Запуск: python -m src.role_5.c1_runner"""
import glob
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.role_5.container import pack_replica, unpack_replica          # noqa: E402
from src.role_5.deps import huffman, serialize_records, deserialize_records  # noqa: E402
from src.role_5.errors import LogError                                  # noqa: E402
from src.role_5.pipeline import evaluate_pipeline, _values_and_weights  # noqa: E402
from src.role_5.recoverer import recover_journal                        # noqa: E402
from src.role_5._tmp_record import Record                               # noqa: E402

CFG = json.load(open(os.path.join(ROOT, "team_config.json"), encoding="utf-8"))
PWD = CFG["пароль_сеансу"]
REF = os.path.join(ROOT, "reference")
SEED = CFG["реплікація"]["seed_моделі_реплікації"]
SESSION = os.path.join(ROOT, "data", "session.csv")


def _rd(name):
    return open(os.path.join(REF, name), "rb").read()


def c1(): r = Record(1000, 101, 120, 2500, 2, 1, 1, 0); return unpack_replica(pack_replica([r], PWD, "huffman"), PWD) == [r]
def c2(): return unpack_replica(pack_replica([], PWD, "huffman"), PWD) == []
def c3():
    recs = unpack_replica(_rd("ref_04_session.bin"), PWD)
    return unpack_replica(pack_replica(recs, PWD, "none"), PWD) == recs
def c4():  # канонічні коди відновлюються лише з довжин
    t, _ = huffman.encode(serialize_records(unpack_replica(_rd("ref_04_session.bin"), PWD)))
    a = huffman.canonical_codes(dict(t))
    return a == huffman.canonical_codes({s: l for s, l in sorted(t, reverse=True)})
def c5():
    t, _ = huffman.encode(bytes([7]) * 16)
    return t == [(7, 1)]
def c6():
    from src.role_1 import load_csv, predict_classes
    from src.role_3 import select_copies_dp, select_copies_greedy
    recs = load_csv(SESSION)
    W = int(CFG["реплікація"]["replication_budget_bits"])
    for s in range(0, 240, 48):
        win = recs[s:s + 48]
        from src.role_5._tmp_record import to_record
        w, u = _values_and_weights([to_record(r) for r in win], predict_classes(win), CFG)
        d, g = select_copies_dp(win, w, u, W), select_copies_greedy(win, w, u, W)
        if sum(u[i][d[i]] for i in range(len(win))) + 1e-9 < sum(u[i][g[i]] for i in range(len(win))):
            return False
    return True
def _lost(sc):
    r = evaluate_pipeline(SESSION, sc, SEED, CFG)
    return 1 - r["R"], r
def c7(): return _lost("штатний")[0] == 0.0
def c8(): return _lost("деградований")[0] <= CFG["пороги_втрати_критичних_подій"]["деградований"]
def c9():
    from src.role_1 import load_csv, predict_classes
    from src.role_1.metrics import compute_macro_f1
    v = load_csv(os.path.join(ROOT, "data", "valid.csv"))
    f1 = compute_macro_f1([r.label for r in v], predict_classes(v))[0]
    return f1 >= CFG["класифікатор"]["F1_min"]  # на відкритому valid; holdout - у викладача
def c10():
    b = bytearray(_rd("ref_04_session.bin"))
    for pos in (60, len(b) - 1):  # шифротекст і HMAC
        m = bytearray(b); m[pos] ^= 1
        try:
            unpack_replica(bytes(m), PWD); return False
        except LogError as e:
            if e.code != "E_HMAC": return False
    return True
def c11():
    exp = json.load(open(os.path.join(REF, "expected_failures.json"), encoding="utf-8"))
    for name, info in exp.items():
        try:
            unpack_replica(_rd(name), PWD); return False
        except LogError as e:
            if e.code != info["очікуваний_код"]: return False
    try:
        unpack_replica(_rd("ref_01_single_record.bin"), "wrong"); return False
    except LogError as e:
        return e.code == "E_HMAC"
def c12():
    for p in sorted(glob.glob(REF + "/ref_*.bin")):
        exp = json.load(open(p[:-4] + ".json", encoding="utf-8"))["записи"]
        if [r._asdict() for r in unpack_replica(open(p, "rb").read(), PWD)] != exp: return False
    t = time.perf_counter()
    for sc in CFG["реплікація"]["сценарії"]: evaluate_pipeline(SESSION, sc, SEED, CFG)
    return time.perf_counter() - t <= CFG["ліміт_часу_повного_прогону_секунд"]


CASES = [("1 один запис", c1), ("2 порожній журнал", c2), ("3 NO_COMPRESSION", c3),
         ("4 канонічні коди з довжин", c4), ("5 вироджений алфавіт", c5), ("6 ДП >= жадібний", c6),
         ("7 штатний: всі критичні", c7), ("8 деградований: поріг", c8), ("9 macro-F1 (valid)", c9),
         ("10 підміна шифротексту/HMAC", c10), ("11 пароль/обрізка/magic/reserved", c11),
         ("12 еталони + ліміт часу", c12)]

if __name__ == "__main__":
    ok = 0
    for name, fn in CASES:
        try:
            res = bool(fn())
        except Exception as e:  # кейс не повинен ламати прогін
            res = False; name += " (виняток: %r)" % e
        ok += res
        print("%-40s %s" % (name, "PASS" if res else "FAIL"))
    print("Разом: %d/12" % ok)
