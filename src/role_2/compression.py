
import heapq
import struct
from collections import Counter


RECORD_SIZE = 16
MAX_CODE_LENGTH = 15

MODE_HUFFMAN = "HUFFMAN"
MODE_NO_COMPRESSION = "NO_COMPRESSION"

FLAG_CLASS_TABLE = 0x04
FLAG_NO_COMPRESSION = 0x08
RESERVED_FLAGS_MASK = 0xF0
EXTENSION_FLAGS_MASK = 0x03


class PayloadError(ValueError):
    """Помилка структури або вмісту payload."""

    code = "E_PAYLOAD"


def _normalize_mode(mode):
    """Нормалізує назву режиму стиснення."""
    value = str(mode).strip().upper().replace("-", "_").replace(" ", "_")

    if value in ("HUFFMAN", "HUF", "CANONICAL_HUFFMAN"):
        return MODE_HUFFMAN

    if value in ("NO_COMPRESSION", "NONE", "RAW", "UNCOMPRESSED"):
        return MODE_NO_COMPRESSION

    raise ValueError("Невідомий режим стиснення: " + str(mode))


# ---------------------------------------------------------
# 1. СЕРІАЛІЗАЦІЯ ЗАПИСІВ
# ---------------------------------------------------------

def serialize_records(records):
    """
    Перетворює список LogRecord на послідовність 16-байтових записів.

    Поле label не серіалізується, оскільки воно не входить
    до формату запису LOG3.
    """
    from src.role_1 import normalize_records

    normalized = normalize_records(records)

    if len(normalized) > 65535:
        raise ValueError("Занадто багато записів для record_count")

    result = bytearray()

    for record in normalized:
        raw = record.to_bytes()

        if len(raw) != RECORD_SIZE:
            raise ValueError("Кожен запис повинен мати рівно 16 байтів")

        result.extend(raw)

    return bytes(result)


def deserialize_records(data):
    """
    Відновлює список LogRecord із послідовності 16-байтових записів.
    """
    from src.role_1 import LogRecord

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data повинно бути bytes або bytearray")

    data = bytes(data)

    if len(data) % RECORD_SIZE != 0:
        raise ValueError("Розмір даних не кратний 16 байтам")

    records = []

    for offset in range(0, len(data), RECORD_SIZE):
        raw_record = data[offset:offset + RECORD_SIZE]
        records.append(LogRecord.from_bytes(raw_record))

    return records


# ---------------------------------------------------------
# 2. ПОБУДОВА ЗВИЧАЙНОГО ДЕРЕВА ГАФФМАНА
# ---------------------------------------------------------


def _build_huffman_lengths(data):
    """
    Будує оптимальні довжини кодів Гаффмана
    з обмеженням максимальної довжини 15 біт.
    Використовує динамічне програмування.
    """
    from functools import lru_cache

    frequencies = Counter(data)

    if not frequencies:
        return {}

    if len(frequencies) == 1:
        symbol = next(iter(frequencies))
        return {symbol: 1}

    if len(frequencies) > (1 << MAX_CODE_LENGTH):
        raise ValueError("Забагато символів для заданої максимальної довжини коду")

    # Частіші символи отримують коротші коди.
    symbols = sorted(
        frequencies.keys(),
        key=lambda symbol: (-frequencies[symbol], symbol)
    )

    weights = [frequencies[symbol] for symbol in symbols]

    # Префіксні суми для швидкого обчислення сум частот.
    prefix = [0]

    for weight in weights:
        prefix.append(prefix[-1] + weight)

    n = len(symbols)
    choice = {}

    @lru_cache(maxsize=None)
    def solve(depth, assigned, slots):
        remaining = n - assigned

        if remaining == 0:
            return 0 if slots == 0 else float("inf")

        if depth > MAX_CODE_LENGTH or slots <= 0:
            return float("inf")

        if depth == MAX_CODE_LENGTH:
            if slots != remaining:
                return float("inf")

        # На останньому рівні всі доступні позиції стають листками.
            choice[(depth, assigned, slots)] = remaining

            return depth * (prefix[n] - prefix[assigned])

        # Кількість листків, які призначаються на поточній глибині.
        min_k = max(0, 2 * slots - remaining)
        max_k = min(slots, remaining)

        capacity = 1 << (MAX_CODE_LENGTH - depth)

        if capacity > 1:
            max_k = min(
                max_k,
                (capacity * slots - remaining) // (capacity - 1)
            )

        best_cost = float("inf")
        best_k = None

        for k in range(min_k, max_k + 1):
            internal_nodes = slots - k
            remaining_after = remaining - k

            if internal_nodes == 0:
                if remaining_after != 0:
                    continue
                next_slots = 0
            else:
                if remaining_after < 2 * internal_nodes:
                    continue

                if remaining_after > capacity * internal_nodes:
                    continue

                next_slots = 2 * internal_nodes

            # Вартість призначення символам поточної довжини.
            current_weight = prefix[assigned + k] - prefix[assigned]
            current_cost = depth * current_weight

            # Рекурсивно обчислюємо вартість наступного рівня.
            next_cost = solve(
                depth + 1,
                assigned + k,
                next_slots
            )

            total_cost = current_cost + next_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_k = k

        if best_k is not None:
            choice[(depth, assigned, slots)] = best_k

        return best_cost

    # Знаходимо мінімальну сумарну вартість кодів.
    minimum_cost = solve(1, 0, 2)

    if minimum_cost == float("inf"):
        raise ValueError("Не вдалося побудувати дерево кодів Гаффмана")

    # Відновлюємо отримані довжини кодів.
    lengths = {}
    depth = 1
    assigned = 0
    slots = 2

    while assigned < n:
        state = (depth, assigned, slots)

        if state not in choice:
            raise ValueError("Не вдалося відновити довжини кодів Гаффмана")

        k = choice[state]

        for index in range(assigned, assigned + k):
            lengths[symbols[index]] = depth

        assigned += k
        slots = 2 * (slots - k)
        depth += 1

    return lengths


# ---------------------------------------------------------
# 3. КАНОНІЧНІ КОДИ ГАФФМАНА
# ---------------------------------------------------------


def _canonical_codes(lengths):
    """
    Створює канонічні коди лише за довжинами кодів.

    Повертає словник:
        symbol -> (code, length)
    """
    if not lengths:
        return {}

    # Перевіряємо допустимість номерів символів і довжин кодів.
    for symbol, length in lengths.items():
        if not 0 <= symbol <= 255:
            raise PayloadError("Некоректний номер символу")

        if not 1 <= length <= MAX_CODE_LENGTH:
            raise PayloadError("Некоректна довжина коду")

    # Для одного символу використовується код 0 довжиною 1 біт.
    if len(lengths) == 1:
        symbol, length = next(iter(lengths.items()))

        if length != 1:
            raise PayloadError("Для одного символу довжина має бути 1")

        return {symbol: (0, 1)}

    # Перевірка умови Kraft = 1.
    kraft_sum = sum(
        1 << (MAX_CODE_LENGTH - length)
        for length in lengths.values()
    )

    if kraft_sum != (1 << MAX_CODE_LENGTH):
        raise PayloadError("Таблиця кодів не задає повне префіксне дерево")

    ordered = sorted(
        lengths.items(),
        key=lambda item: (item[1], item[0])
    )

    codes = {}
    code = 0
    previous_length = 0

    for symbol, length in ordered:
        code <<= length - previous_length

        if code >= (1 << length):
            raise PayloadError("Некоректна канонічна таблиця")

        codes[symbol] = (code, length)

        code += 1
        previous_length = length

    return codes



# ---------------------------------------------------------
# 4. ПАКУВАННЯ БІТІВ MSB-FIRST
# ---------------------------------------------------------

def _encode_bitstream(data, codes):
    """Кодує байти та пакує біти від старшого до молодшого."""
    output = bytearray()
    current_byte = 0
    bits_used = 0

    for symbol in data:
        code, length = codes[symbol]

        for shift in range(length - 1, -1, -1):
            bit = (code >> shift) & 1

            current_byte = (current_byte << 1) | bit
            bits_used += 1

            if bits_used == 8:
                output.append(current_byte)
                current_byte = 0
                bits_used = 0

    # Невикористані молодші біти залишаються нульовими.
    if bits_used:
        current_byte <<= 8 - bits_used
        output.append(current_byte)

    return bytes(output)


# ---------------------------------------------------------
# 5. ДЕКОДУВАННЯ БІТОВОГО ПОТОКУ
# ---------------------------------------------------------

def _decode_bitstream(bitstream, symbol_count, codes):
    """Декодує рівно symbol_count символів."""
    if symbol_count == 0:
        if bitstream:
            raise PayloadError("Для порожнього потоку bitstream має бути порожнім")
        return b""

    if not codes:
        raise PayloadError("Відсутня таблиця кодів")

    # Побудова таблиці code + length -> symbol.
    reverse_codes = {}

    for symbol, (code, length) in codes.items():
        reverse_codes[(length, code)] = symbol

    result = bytearray()
    current_code = 0
    current_length = 0
    bits_read = 0

    for byte in bitstream:
        for shift in range(7, -1, -1):
            bit = (byte >> shift) & 1

            current_code = (current_code << 1) | bit
            current_length += 1
            bits_read += 1

            found = reverse_codes.get((current_length, current_code))

            if found is not None:
                result.append(found)
                current_code = 0
                current_length = 0

                if len(result) == symbol_count:
                    break

            elif current_length >= MAX_CODE_LENGTH:
                raise PayloadError("Некоректний код у bitstream")

        if len(result) == symbol_count:
            break

    if len(result) != symbol_count:
        raise PayloadError("Недостатньо даних для декодування")

    expected_bytes = (bits_read + 7) // 8

    if expected_bytes != len(bitstream):
        raise PayloadError("Некоректна довжина bitstream")

    # Перевіряємо, що невикористані молодші біти останнього байта нульові.
    remaining_bits = bits_read % 8

    if remaining_bits:
        padding_mask = (1 << (8 - remaining_bits)) - 1

        if bitstream[-1] & padding_mask:
            raise PayloadError("Padding останнього байта має бути нульовим")

    return bytes(result)


# ---------------------------------------------------------
# 6. СТИСНЕННЯ
# ---------------------------------------------------------

def compress(data, mode):
    """
    Формує payload відповідно до FORMAT_SPEC_P3.md.

    Структура:
        record_count: 2 байти
        flags: 1 байт
        reserved: 1 байт
        table_nsym: 2 байти
        table: 2 * table_nsym байтів
        symbol_count: 4 байти
        bitstream_len: 4 байти
        bitstream: bitstream_len байтів
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data повинно бути bytes або bytearray")

    data = bytes(data)
    mode = _normalize_mode(mode)

    if len(data) % RECORD_SIZE != 0:
        raise ValueError("Дані повинні містити цілі записи по 16 байтів")

    record_count = len(data) // RECORD_SIZE

    if record_count > 65535:
        raise ValueError("Занадто багато записів")

    symbol_count = len(data)

    if mode == MODE_NO_COMPRESSION:
        flags = FLAG_NO_COMPRESSION
        table = b""
        bitstream = data

    else:
        flags = 0

        lengths = _build_huffman_lengths(data)
        codes = _canonical_codes(lengths)

        # Таблиця сортується за зростанням номера символу.
        table_data = bytearray()

        for symbol in sorted(lengths):
            table_data.extend((symbol, lengths[symbol]))

        table = bytes(table_data)
        bitstream = _encode_bitstream(data, codes)

    table_nsym = len(table) // 2

    header = struct.pack(
        ">HBBHII",
        record_count,
        flags,
        0,
        table_nsym,
        symbol_count,
        len(bitstream)
    )

    return header + table + bitstream


# ---------------------------------------------------------
# 7. РОЗПАКУВАННЯ
# ---------------------------------------------------------

def decompress(payload, mode):
    """
    Перевіряє структуру payload і відновлює початкові байти.

    Повертає саме серіалізовані записи, а не об'єкти LogRecord.
    Для отримання об'єктів використовується deserialize_records().
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload повинно бути bytes або bytearray")

    payload = bytes(payload)
    mode = _normalize_mode(mode)

    # Фіксована частина заголовка payload — 14 байтів.
    if len(payload) < 14:
        raise PayloadError("Payload занадто короткий")

    (
        record_count,
        flags,
        reserved,
        table_nsym,
        symbol_count,
        bitstream_len
    ) = struct.unpack(">HBBHII", payload[:14])

    if reserved != 0:
        raise PayloadError("Поле reserved повинно дорівнювати нулю")

    if flags & RESERVED_FLAGS_MASK:
        raise PayloadError("Зарезервовані біти flags повинні дорівнювати нулю")

    if flags & EXTENSION_FLAGS_MASK:
        raise PayloadError("Командні розширення не підтримуються базовим декодером")

    if symbol_count != RECORD_SIZE * record_count:
        raise PayloadError("symbol_count не дорівнює 16 * record_count")

    is_raw = bool(flags & FLAG_NO_COMPRESSION)
    has_class_table = bool(flags & FLAG_CLASS_TABLE)

    if mode == MODE_NO_COMPRESSION and not is_raw:
        raise PayloadError("Payload не відповідає режиму NO_COMPRESSION")

    if mode == MODE_HUFFMAN and is_raw:
        raise PayloadError("Payload не відповідає режиму HUFFMAN")

    table_start = 14
    table_end = table_start + table_nsym * 2

    if table_end > len(payload):
        raise PayloadError("Таблиця кодів виходить за межі payload")

    table_data = payload[table_start:table_end]
    bitstream_end = table_end + bitstream_len

    if bitstream_end > len(payload):
        raise PayloadError("Bitstream виходить за межі payload")

    bitstream = payload[table_end:bitstream_end]
    class_table = payload[bitstream_end:]

    if has_class_table:
        if len(class_table) != record_count:
            raise PayloadError("Некоректна довжина class_table")

        if any(value not in (0, 1, 2) for value in class_table):
            raise PayloadError("class_table містить зарезервоване значення")

    elif class_table:
        raise PayloadError("Зайві байти після bitstream")

    if is_raw:
        if table_nsym != 0 or table_data:
            raise PayloadError("NO_COMPRESSION не повинен містити таблицю кодів")

        if bitstream_len != symbol_count:
            raise PayloadError("У NO_COMPRESSION довжина bitstream має дорівнювати symbol_count")

        result = bitstream

    else:
        if symbol_count == 0:
            if table_nsym != 0 or bitstream_len != 0:
                raise PayloadError("Некоректний payload порожнього журналу")

            result = b""

        else:
            if table_nsym == 0:
                raise PayloadError("Відсутня таблиця кодів Гаффмана")

            lengths = {}
            previous_symbol = -1

            for index in range(0, len(table_data), 2):
                symbol = table_data[index]
                length = table_data[index + 1]

                if symbol <= previous_symbol:
                    raise PayloadError("Символи таблиці мають бути унікальними та відсортованими")

                if not 1 <= length <= MAX_CODE_LENGTH:
                    raise PayloadError("Довжина коду має бути від 1 до 15")

                lengths[symbol] = length
                previous_symbol = symbol

            if len(lengths) != table_nsym:
                raise PayloadError("Некоректна кількість символів у таблиці")

            codes = _canonical_codes(lengths)
            result = _decode_bitstream(bitstream, symbol_count, codes)

    if len(result) != symbol_count:
        raise PayloadError("Кількість відновлених байтів не збігається з symbol_count")

    return result
