"""Роль 5: відновлення журналу з уцілілих реплік."""
from .errors import LogError
from .container import unpack_replica


def recover_journal(surviving_blobs, password):
    """Розпаковує всі репліки, пошкоджені пропускає, прибирає дублікати, сортує за часом."""
    seen, out = set(), []
    for blob in surviving_blobs:
        try:
            records = unpack_replica(blob, password)
        except LogError:
            continue  # пошкоджена репліка не повинна ламати відновлення
        for r in records:
            key = tuple(r)  # ідентичність запису = його 16 байтів
            if key not in seen:
                seen.add(key)
                out.append(r)
    out.sort(key=lambda r: r.timestamp_ms)
    return out
