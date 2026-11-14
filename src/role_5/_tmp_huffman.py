"""Канонічний Гаффман. ТИМЧАСОВА реалізація - належить Ролі 2."""
import heapq
from collections import Counter

from .errors import LogError, E_CODE, E_PAYLOAD

MAX_LEN = 15


def _lengths(freq):
    if len(freq) == 1:
        return {next(iter(freq)): 1}
    f = dict(freq)
    while True:
        heap = [(c, i, [s]) for i, (s, c) in enumerate(sorted(f.items()))]
        heapq.heapify(heap)
        depth = {s: 0 for s in f}
        n = len(heap)
        while len(heap) > 1:
            c1, _, s1 = heapq.heappop(heap)
            c2, _, s2 = heapq.heappop(heap)
            for s in s1 + s2:
                depth[s] += 1
            heapq.heappush(heap, (c1 + c2, n, s1 + s2))
            n += 1
        if max(depth.values()) <= MAX_LEN:
            return depth
        f = {s: max(1, c // 2) for s, c in f.items()}  # спрощене обмеження довжини


def canonical_codes(lengths):
    """lengths: {symbol: len} -> {symbol: (code, len)} (bl_count / next_code)."""
    bl = [0] * (MAX_LEN + 2)
    for l in lengths.values():
        bl[l] += 1
    code, nxt = 0, [0] * (MAX_LEN + 2)
    for b in range(1, MAX_LEN + 1):
        code = (code + bl[b - 1]) << 1
        nxt[b] = code
    out = {}
    for s in sorted(lengths, key=lambda x: (lengths[x], x)):
        out[s] = (nxt[lengths[s]], lengths[s])
        nxt[lengths[s]] += 1
    return out


def validate_lengths(lengths):
    if not lengths:
        return
    if any(not 1 <= l <= MAX_LEN for l in lengths.values()):
        raise LogError(E_CODE, "довжина коду поза 1..15")
    if len(lengths) == 1:
        return  # єдиний символ: довжина 1 вже перевірена
    if sum(1 << (MAX_LEN - l) for l in lengths.values()) != 1 << MAX_LEN:
        raise LogError(E_CODE, "сума Крафта != 1")


def encode(data):
    """bytes -> (table: [(sym,len)], bitstream: bytes)"""
    if not data:
        return [], b""
    lengths = _lengths(Counter(data))
    codes = canonical_codes(lengths)
    acc, nbits = 0, 0
    for b in data:
        c, l = codes[b]
        acc = (acc << l) | c
        nbits += l
    pad = (-nbits) % 8
    acc <<= pad
    return sorted(lengths.items()), acc.to_bytes((nbits + pad) // 8, "big")


def decode(table, bitstream, symbol_count):
    lengths = dict(table)
    if len(lengths) != len(table):
        raise LogError(E_CODE, "повторний символ у таблиці")
    validate_lengths(lengths)
    if symbol_count and not lengths:
        raise LogError(E_PAYLOAD, "символи без таблиці")
    rev = {(l, c): s for s, (c, l) in canonical_codes(lengths).items()}
    total_bits = len(bitstream) * 8
    value = int.from_bytes(bitstream, "big") if bitstream else 0
    out, pos, code, ln = bytearray(), 0, 0, 0
    while len(out) < symbol_count:
        if pos >= total_bits:
            raise LogError(E_PAYLOAD, "потік бітів закінчився раніше")
        code = (code << 1) | ((value >> (total_bits - 1 - pos)) & 1)
        pos += 1
        ln += 1
        if (ln, code) in rev:
            out.append(rev[(ln, code)])
            code, ln = 0, 0
        elif ln > MAX_LEN:
            raise LogError(E_PAYLOAD, "невідомий код")
    if (total_bits - pos) >= 8:
        raise LogError(E_PAYLOAD, "зайві повні байти після даних")
    if pos < total_bits and value & ((1 << (total_bits - pos)) - 1):
        raise LogError(E_PAYLOAD, "ненульовий padding")
    return bytes(out)
