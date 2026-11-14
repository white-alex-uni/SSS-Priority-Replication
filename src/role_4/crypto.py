import hashlib
import hmac

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

ITERATIONS = 100000
SALT_SIZE = 16
NONCE_SIZE = 12
MAC_SIZE = 32

cache = {}


class ReplicaAuthError(ValueError):
    code = "E_HMAC"


class ReplicaLengthError(ReplicaAuthError):
    code = "E_LENGTH"


def get_bytes(password):
    if isinstance(password, bytes):
        return password
    return password.encode("utf-8")


def derive_keys(password, salt):
    if len(salt) != SALT_SIZE:
        raise ValueError("Salt must be 16 bytes")

    password_bytes = get_bytes(password)
    name = password_bytes.hex() + salt.hex()
    if name in cache:
        return cache[name]

    master = hashlib.pbkdf2_hmac("sha256", password_bytes, salt, ITERATIONS, 64)
    key_enc = master[0:32]
    key_mac = master[32:64]

    if len(cache) >= 64:
        cache.clear()
    cache[name] = (key_enc, key_mac)
    return key_enc, key_mac


def aes_ctr(key_enc, nonce, data):
    counter = nonce + bytes([0, 0, 0, 1])
    cipher = Cipher(algorithms.AES(key_enc), modes.CTR(counter))
    worker = cipher.encryptor()
    result = worker.update(data) + worker.finalize()
    return result


def calc_mac(key_mac, header, cipher_text):
    message = header + cipher_text
    return hmac.new(key_mac, message, hashlib.sha256).digest()


def encrypt_and_mac(header, payload, password, salt, nonce):
    if len(nonce) != NONCE_SIZE:
        raise ValueError("Nonce must be 12 bytes")

    key_enc, key_mac = derive_keys(password, salt)
    cipher_text = aes_ctr(key_enc, nonce, payload)
    mac = calc_mac(key_mac, header, cipher_text)
    return cipher_text, mac


def verify_and_decrypt(header, ciphertext, mac, password, salt, nonce):
    if len(salt) != SALT_SIZE:
        raise ReplicaLengthError("Bad salt size")
    if len(nonce) != NONCE_SIZE:
        raise ReplicaLengthError("Bad nonce size")
    if len(mac) != MAC_SIZE:
        raise ReplicaLengthError("Bad mac size")

    key_enc, key_mac = derive_keys(password, salt)
    real_mac = calc_mac(key_mac, header, ciphertext)

    if hmac.compare_digest(real_mac, mac):
        return aes_ctr(key_enc, nonce, ciphertext)
    else:
        raise ReplicaAuthError("Mac does not match")
