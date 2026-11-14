"""ТИМЧАСОВО (поки немає Ролі 2): 16-байтові записи. Приймає і LogRecord Ролі 1, і Record."""
import struct
from collections import namedtuple

Record = namedtuple(
    "Record",
    "timestamp_ms source_id event_code metric_value category severity critical record_flags",
)
_FMT = ">IHHiBBBB"


def to_record(r):
    return Record(int(r.timestamp_ms), int(r.source_id), int(r.event_code), int(r.metric_value),
                  int(r.category), int(r.severity), int(r.critical), int(r.record_flags))


def serialize_records(records):
    return b"".join(struct.pack(_FMT, *to_record(r)) for r in records)


def deserialize_records(data):
    if len(data) % 16:
        raise ValueError("довжина не кратна 16")
    return [Record(*struct.unpack_from(_FMT, data, i)) for i in range(0, len(data), 16)]
