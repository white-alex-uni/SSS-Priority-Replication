"""Єдине місце, де Роль 5 підключає модулі інших ролей.
Коли Роль 2 здасть свій код - змініть лише блок «Роль 2» нижче."""
from ..role_4 import crypto as crypto_p3          # Роль 4 (AES-CTR + HMAC)

# --- Роль 2 (стиснення та серіалізація записів) ---
from ..role_2 import compression as _r2
from ._tmp_record import Record, to_record
from .errors import LogError, E_CODE, E_PAYLOAD

serialize_records = _r2.serialize_records


def deserialize_records(data):
    return [to_record(r) for r in _r2.deserialize_records(data)]


class _HuffmanAdapter:
    @staticmethod
    def canonical_codes(lengths):
        try:
            return _r2._canonical_codes(lengths)
        except _r2.PayloadError as e:
            raise LogError(getattr(e, "code", E_CODE), str(e))

    @staticmethod
    def encode(data):
        if not data:
            return [], b""
        lengths = _r2._build_huffman_lengths(data)
        codes = _r2._canonical_codes(lengths)
        return sorted(lengths.items()), _r2._encode_bitstream(data, codes)

    @staticmethod
    def decode(table, bitstream, symbol_count):
        lengths = dict(table)
        if len(lengths) != len(table):
            raise LogError(E_CODE, "повторний символ у таблиці")
        try:
            codes = _r2._canonical_codes(lengths)
            if symbol_count and not codes:
                raise LogError(E_PAYLOAD, "символи без таблиці")
            return _r2._decode_bitstream(bitstream, symbol_count, codes)
        except _r2.PayloadError as e:
            raise LogError(getattr(e, "code", E_PAYLOAD), str(e))


huffman = _HuffmanAdapter()

