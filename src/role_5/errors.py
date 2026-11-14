"""Коди помилок прийому репліки (FORMAT_SPEC_P3.md, розділ 5)."""

E_SHORT = "E_SHORT"
E_MAGIC = "E_MAGIC"
E_LENGTH = "E_LENGTH"
E_HMAC = "E_HMAC"
E_PAYLOAD = "E_PAYLOAD"
E_CODE = "E_CODE"


class LogError(Exception):
    """Контрольована помилка: несе один із кодів вище, замість аварійного краху."""

    def __init__(self, code, message=""):
        super().__init__("%s: %s" % (code, message))
        self.code = code
