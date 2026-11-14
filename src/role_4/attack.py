import random

from src.role_4.crypto import ReplicaAuthError, encrypt_and_mac, verify_and_decrypt
from src.role_4.genetic import sort_best, tournament


def weak_encrypt(data, key):
    result = bytearray()
    for i in range(len(data)):
        position = i % len(key)
        result.append(data[i] ^ key[position])
    return bytes(result)


def weak_decrypt(data, key):
    return weak_encrypt(data, key)


def text_score(data):
    score = 0
    for byte in data:
        if byte == 32:
            score = score + 3
        elif byte >= 97 and byte <= 122:
            score = score + 3
        elif byte >= 65 and byte <= 90:
            score = score + 1
        elif byte >= 48 and byte <= 57:
            score = score + 1
        elif byte >= 33 and byte <= 126:
            score = score + 1
        elif byte == 10:
            score = score + 1
        else:
            score = score - 5
    return score


def make_guess(key, ciphertext):
    text = weak_decrypt(ciphertext, bytes(key))
    guess = {"key": key, "fitness": text_score(text)}
    return guess


def attack_key(ciphertext, key_size, population=60, generations=80, mut_rate=0.3, elite=2, seed=1):
    rng = random.Random(seed)
    evals = 0

    guesses = []
    for i in range(population):
        key = []
        for j in range(key_size):
            key.append(rng.randrange(256))
        guesses.append(make_guess(key, ciphertext))
        evals = evals + 1

    for generation in range(generations):
        guesses = sort_best(guesses)
        new_guesses = []

        for i in range(elite):
            new_guesses.append(guesses[i])

        while len(new_guesses) < population:
            mom = tournament(guesses, 3, rng)
            dad = tournament(guesses, 3, rng)

            child = []
            for j in range(key_size):
                if rng.random() < 0.5:
                    child.append(mom["key"][j])
                else:
                    child.append(dad["key"][j])

            for j in range(key_size):
                if rng.random() < mut_rate:
                    if rng.random() < 0.5:
                        child[j] = rng.randrange(256)
                    else:
                        bit = rng.randrange(8)
                        child[j] = child[j] ^ (2 ** bit)

            new_guesses.append(make_guess(child, ciphertext))
            evals = evals + 1

        guesses = new_guesses

    guesses = sort_best(guesses)

    result = {}
    result["key"] = bytes(guesses[0]["key"])
    result["score"] = guesses[0]["fitness"]
    result["evals"] = evals
    return result


def flip_byte(data, index):
    changed = bytearray(data)
    changed[index] = changed[index] ^ 255
    return bytes(changed)


def tamper_test(header, payload, password, salt, nonce):
    cipher_text, mac = encrypt_and_mac(header, payload, password, salt, nonce)

    total = 0
    accepted = 0

    for i in range(len(header)):
        total = total + 1
        bad_header = flip_byte(header, i)
        try:
            verify_and_decrypt(bad_header, cipher_text, mac, password, salt, nonce)
            accepted = accepted + 1
        except ReplicaAuthError:
            pass

    for i in range(len(cipher_text)):
        total = total + 1
        bad_text = flip_byte(cipher_text, i)
        try:
            verify_and_decrypt(header, bad_text, mac, password, salt, nonce)
            accepted = accepted + 1
        except ReplicaAuthError:
            pass

    for i in range(len(mac)):
        total = total + 1
        bad_mac = flip_byte(mac, i)
        try:
            verify_and_decrypt(header, cipher_text, bad_mac, password, salt, nonce)
            accepted = accepted + 1
        except ReplicaAuthError:
            pass

    result = {}
    result["total"] = total
    result["accepted"] = accepted
    result["rejected"] = total - accepted
    return result
