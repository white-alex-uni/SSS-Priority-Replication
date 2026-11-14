"""Єдине місце, де Роль 5 підключає модулі інших ролей.
Коли Роль 2 здасть свій код - змініть лише блок «Роль 2» нижче."""
from ..role_4 import crypto as crypto_p3          # Роль 4 (AES-CTR + HMAC)

# --- Роль 2 (ТИМЧАСОВО: наші заглушки) ---
from . import _tmp_huffman as huffman             # encode / decode / canonical_codes
from ._tmp_record import Record, to_record, serialize_records, deserialize_records
