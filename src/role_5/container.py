"""Роль 5: контейнер LOG3 - заголовок, pack_replica, unpack_replica."""
import os
import struct

from .deps import crypto_p3, huffman, serialize_records, deserialize_records
from .errors import (LogError, E_SHORT, E_MAGIC, E_LENGTH, E_HMAC, E_PAYLOAD)

MAGIC = b"LOG3"
VERSION = 1
HEADER_LEN = 43
MAC_LEN = 32
FLAG_CLASS_TABLE = 0x04
FLAG_NO_COMPRESSION = 0x08
_HDR = ">4sBHI16s12sI"  # magic, version, group_id, batch_seq, salt, nonce, ct_len


def build_header(group_id, batch_seq, salt, nonce, ciphertext_len):
    return struct.pack(_HDR, MAGIC, VERSION, group_id, batch_seq, salt, nonce, ciphertext_len)


def parse_header(blob):
    """Перевіряє мінімальну довжину, magic/version, точну довжину (кроки 1-3)."""
    if len(blob) < HEADER_LEN + MAC_LEN:
        raise LogError(E_SHORT, "пакет коротший за мінімум")
    magic, ver, group_id, batch_seq, salt, nonce, ct_len = struct.unpack_from(_HDR, blob)
    if magic != MAGIC or ver != VERSION:
        raise LogError(E_MAGIC, "поганий magic або version")
    if len(blob) != HEADER_LEN + ct_len + MAC_LEN:
        raise LogError(E_LENGTH, "довжина не збігається з ciphertext_len")
    return {"group_id": group_id, "batch_seq": batch_seq, "salt": salt,
            "nonce": nonce, "ct_len": ct_len}


def build_payload(records, compression_mode="huffman", class_table=None):
    raw = serialize_records(records)
    flags = 0
    if class_table is not None:
        flags |= FLAG_CLASS_TABLE
    if compression_mode == "none":
        flags |= FLAG_NO_COMPRESSION
        table, bits = [], raw
    else:
        table, bits = huffman.encode(raw)
    p = struct.pack(">HBBH", len(records), flags, 0, len(table))
    p += b"".join(struct.pack("BB", s, l) for s, l in table)
    p += struct.pack(">II", len(raw), len(bits)) + bits
    if class_table is not None:
        p += bytes(class_table)
    return p


def parse_payload(p):
    """Крок 6: сувора перевірка payload та інваріантів."""
    try:
        count, flags, reserved, nsym = struct.unpack_from(">HBBH", p, 0)
        pos = 6
        if reserved != 0 or flags & 0xF0 or flags & 0x03:
            raise LogError(E_PAYLOAD, "reserved/невідоме розширення")
        table = [struct.unpack_from("BB", p, pos + 2 * i) for i in range(nsym)]
        pos += 2 * nsym
        sym_count, bs_len = struct.unpack_from(">II", p, pos)
        pos += 8
        bits = p[pos:pos + bs_len]
        pos += bs_len
        if len(bits) != bs_len or sym_count != 16 * count:
            raise LogError(E_PAYLOAD, "розміри не збігаються")
        classes = None
        if flags & FLAG_CLASS_TABLE:
            classes = p[pos:pos + count]
            pos += count
            if len(classes) != count or any(c > 2 for c in classes):
                raise LogError(E_PAYLOAD, "поганий class_table")
        if pos != len(p):
            raise LogError(E_PAYLOAD, "зайві байти в кінці payload")
    except struct.error:
        raise LogError(E_PAYLOAD, "payload обрізано")
    if flags & FLAG_NO_COMPRESSION:
        if table or bs_len != sym_count:
            raise LogError(E_PAYLOAD, "NO_COMPRESSION: некоректні поля")
        raw = bits
    else:
        raw = huffman.decode(table, bits, sym_count)
    return deserialize_records(raw), classes


def pack_replica(records, password, compression_mode="huffman", *, class_table=None,
                 group_id=0, batch_seq=0, salt=None, nonce=None):
    salt = salt if salt is not None else os.urandom(16)
    nonce = nonce if nonce is not None else os.urandom(12)
    payload = build_payload(records, compression_mode, class_table)
    header = build_header(group_id, batch_seq, salt, nonce, len(payload))
    ct, mac = crypto_p3.encrypt_and_mac(header, payload, password, salt, nonce)
    return header + ct + mac


def unpack_replica(blob, password):
    """Повертає список записів або кидає LogError з кодом (порядок за spec п.5)."""
    h = parse_header(blob)
    header, ct, mac = blob[:HEADER_LEN], blob[HEADER_LEN:-MAC_LEN], blob[-MAC_LEN:]
    try:
        payload = crypto_p3.verify_and_decrypt(header, ct, mac, password, h["salt"], h["nonce"])
    except LogError:
        raise
    except ValueError as e:  # ReplicaAuthError (Роль 4) успадковує ValueError і має .code
        raise LogError(getattr(e, "code", E_HMAC), str(e))
    records, _ = parse_payload(payload)
    return records
