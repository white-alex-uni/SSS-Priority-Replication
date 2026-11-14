import glob
import json
import os
import sys

import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.role_4.attack import attack_key, tamper_test, weak_decrypt, weak_encrypt
from src.role_4.crypto import ReplicaAuthError, derive_keys, encrypt_and_mac, verify_and_decrypt
from src.role_4.fitness import Fitness, is_bad, make_config
from src.role_4.genetic import (
    ADVANCED_CONFIG,
    DEFAULT_CONFIG,
    compare_search,
    elitism_ok,
    run_genetic_optimizer,
)
from src.role_4.pareto import compare_front, dominates, front_is_clean, get_front

PASSWORD = "algo2026"
SALT = bytes(range(16))
NONCE = bytes(range(100, 112))
HEADER = b"LOG3" + bytes(39)
PAYLOAD = b"critical event: disk failure on node B " * 5


def test_keys_sizes():
    key_enc, key_mac = derive_keys(PASSWORD, SALT)
    assert len(key_enc) == 32
    assert len(key_mac) == 32
    assert key_enc != key_mac


def test_different_salt_gives_different_keys():
    first = derive_keys(PASSWORD, SALT)
    second = derive_keys(PASSWORD, bytes(16))
    assert first != second


def test_bad_salt_size():
    with pytest.raises(ValueError):
        derive_keys(PASSWORD, b"short")


def test_counter_starts_from_one():
    key_enc, key_mac = derive_keys(PASSWORD, SALT)
    cipher_text, mac = encrypt_and_mac(HEADER, bytes(40), PASSWORD, SALT, NONCE)
    assert len(mac) == 32

    worker = Cipher(algorithms.AES(key_enc), modes.ECB()).encryptor()
    gamma = b""
    for i in range(1, 4):
        block = NONCE + i.to_bytes(4, "big")
        gamma = gamma + worker.update(block)

    assert cipher_text == gamma[0:40]


def test_roundtrip():
    cipher_text, mac = encrypt_and_mac(HEADER, PAYLOAD, PASSWORD, SALT, NONCE)
    assert cipher_text != PAYLOAD
    assert len(cipher_text) == len(PAYLOAD)
    assert len(mac) == 32
    result = verify_and_decrypt(HEADER, cipher_text, mac, PASSWORD, SALT, NONCE)
    assert result == PAYLOAD


def test_empty_payload():
    cipher_text, mac = encrypt_and_mac(HEADER, b"", PASSWORD, SALT, NONCE)
    result = verify_and_decrypt(HEADER, cipher_text, mac, PASSWORD, SALT, NONCE)
    assert result == b""


def test_wrong_password():
    cipher_text, mac = encrypt_and_mac(HEADER, PAYLOAD, PASSWORD, SALT, NONCE)
    with pytest.raises(ReplicaAuthError):
        verify_and_decrypt(HEADER, cipher_text, mac, "wrong", SALT, NONCE)


def test_wrong_salt():
    cipher_text, mac = encrypt_and_mac(HEADER, PAYLOAD, PASSWORD, SALT, NONCE)
    with pytest.raises(ReplicaAuthError):
        verify_and_decrypt(HEADER, cipher_text, mac, PASSWORD, bytes(16), NONCE)


def test_cut_ciphertext():
    cipher_text, mac = encrypt_and_mac(HEADER, PAYLOAD, PASSWORD, SALT, NONCE)
    with pytest.raises(ReplicaAuthError):
        verify_and_decrypt(HEADER, cipher_text[:-1], mac, PASSWORD, SALT, NONCE)


def test_cut_mac():
    cipher_text, mac = encrypt_and_mac(HEADER, PAYLOAD, PASSWORD, SALT, NONCE)
    with pytest.raises(ReplicaAuthError):
        verify_and_decrypt(HEADER, cipher_text, mac[:-1], PASSWORD, SALT, NONCE)


def test_error_code():
    cipher_text, mac = encrypt_and_mac(HEADER, PAYLOAD, PASSWORD, SALT, NONCE)
    try:
        verify_and_decrypt(HEADER, cipher_text, mac, "wrong", SALT, NONCE)
    except ReplicaAuthError as error:
        assert error.code == "E_HMAC"
    else:
        assert False


def test_every_changed_byte_is_rejected():
    result = tamper_test(HEADER, PAYLOAD, PASSWORD, SALT, NONCE)
    assert result["total"] == len(HEADER) + len(PAYLOAD) + 32 + 1 + 1 + 16
    assert result["accepted"] == 0
    assert result["rejected"] == result["total"]


def test_reference_packets():
    folder = os.path.join(ROOT, "reference")
    config_path = os.path.join(ROOT, "team_config.json")
    if not os.path.exists(folder) or not os.path.exists(config_path):
        pytest.skip("reference folder or team_config.json is not in the repo root")

    file = open(config_path, "r", encoding="utf-8")
    password = json.load(file)["пароль_сеансу"]
    file.close()

    names = sorted(glob.glob(folder + "/ref_*.bin"))
    assert len(names) == 6

    for name in names:
        file = open(name, "rb")
        blob = file.read()
        file.close()
        header = blob[0:43]
        salt = blob[11:27]
        nonce = blob[27:39]
        cipher_text = blob[43:-32]
        mac = blob[-32:]
        payload = verify_and_decrypt(header, cipher_text, mac, password, salt, nonce)
        assert len(payload) > 0


def test_bad_packets_are_rejected():
    folder = os.path.join(ROOT, "reference")
    config_path = os.path.join(ROOT, "team_config.json")
    if not os.path.exists(folder) or not os.path.exists(config_path):
        pytest.skip("reference folder or team_config.json is not in the repo root")

    file = open(config_path, "r", encoding="utf-8")
    password = json.load(file)["пароль_сеансу"]
    file.close()

    for name in ["bad_01_tampered_byte.bin", "bad_02_tampered_tag.bin"]:
        file = open(os.path.join(folder, name), "rb")
        blob = file.read()
        file.close()
        header = blob[0:43]
        salt = blob[11:27]
        nonce = blob[27:39]
        cipher_text = blob[43:-32]
        mac = blob[-32:]
        with pytest.raises(ReplicaAuthError):
            verify_and_decrypt(header, cipher_text, mac, password, salt, nonce)


def toy_fitness(params):
    total = 0.0
    for name in params:
        total = total - params[name] / 100.0
    return total


def toy_goals(params):
    goals = {}
    goals["S"] = 1.0 - params["budget_bits"] / 12000.0
    goals["R"] = params["budget_bits"] / 12000.0
    goals["T"] = 1.0 - params["second_copy"] / 2.0
    return goals


def test_ga_elitism_is_not_worse_than_generation_0():
    result = run_genetic_optimizer(toy_fitness, DEFAULT_CONFIG)
    assert elitism_ok(result)
    assert result["best_fitness"] >= result["best_history"][0]


def test_advanced_ga_elitism():
    result = run_genetic_optimizer(toy_fitness, ADVANCED_CONFIG)
    assert elitism_ok(result)
    assert result["local_gain"] >= 0.0


def test_ga_same_seed_same_result():
    first = run_genetic_optimizer(toy_fitness, DEFAULT_CONFIG)
    second = run_genetic_optimizer(toy_fitness, DEFAULT_CONFIG)
    assert first["best_params"] == second["best_params"]


def test_ga_genes_stay_in_limits():
    result = run_genetic_optimizer(toy_fitness, ADVANCED_CONFIG)
    limits = ADVANCED_CONFIG["limits"]
    for name in limits:
        value = result["best_params"][name]
        assert value >= limits[name]["low"]
        assert value <= limits[name]["high"]


def test_ga_and_random_search_same_evals():
    comparison = compare_search(toy_fitness, DEFAULT_CONFIG)
    assert comparison["ga"]["evals"] == comparison["random"]["evals"]


def test_elite_too_big():
    config = dict(DEFAULT_CONFIG)
    config["elite"] = config["population"]
    with pytest.raises(ValueError):
        run_genetic_optimizer(toy_fitness, config)


def test_dominates():
    good = {"S": 2, "R": 2, "T": 2}
    worse = {"S": 1, "R": 2, "T": 2}
    mixed = {"S": 3, "R": 0, "T": 2}
    assert dominates(good, worse)
    assert not dominates(worse, good)
    assert not dominates(good, good)
    assert not dominates(mixed, good)
    assert not dominates(good, mixed)


def test_pareto_front_is_clean():
    comparison = compare_front(toy_goals, DEFAULT_CONFIG)
    assert front_is_clean(comparison["ga"]["front"])
    assert front_is_clean(comparison["random"]["front"])
    assert comparison["ga"]["evals"] == comparison["random"]["evals"]
    assert len(comparison["ga"]["front"]) > 1


def test_make_config():
    base = {
        "реплікація": {
            "replication_window_records": 48,
            "replication_budget_bits": 5000,
            "second_copy_factor": 1.55,
            "third_copy_factor": 1.85,
        },
        "пріоритети_класів": {"аномальна подія": 5, "перехід до інциденту": 4},
        "critical_bonus": 5,
    }

    config = make_config(base, {"window_size": 24, "budget_bits": 3000, "critical_bonus": 9,
                                "prio_anomaly": 2, "prio_incident": 7})
    assert config["реплікація"]["replication_window_records"] == 24
    assert config["реплікація"]["replication_budget_bits"] == 3000
    assert config["critical_bonus"] == 9
    assert config["пріоритети_класів"]["аномальна подія"] == 2
    assert config["пріоритети_класів"]["перехід до інциденту"] == 7

    assert base["реплікація"]["replication_window_records"] == 48
    assert base["critical_bonus"] == 5


def test_third_copy_is_not_less_than_second():
    base = {
        "реплікація": {
            "replication_window_records": 48,
            "replication_budget_bits": 5000,
            "second_copy_factor": 1.55,
            "third_copy_factor": 1.85,
        },
        "пріоритети_класів": {"аномальна подія": 5, "перехід до інциденту": 4},
        "critical_bonus": 5,
    }
    config = make_config(base, {"second_copy": 1.95})
    assert config["реплікація"]["third_copy_factor"] == 1.95


def test_length_error_code():
    cipher_text, mac = encrypt_and_mac(HEADER, PAYLOAD, PASSWORD, SALT, NONCE)
    with pytest.raises(ReplicaAuthError) as info:
        verify_and_decrypt(HEADER, cipher_text, mac[:-1], PASSWORD, SALT, NONCE)
    assert info.value.code == "E_LENGTH"


def test_bytes_password_gives_same_keys():
    from_text = derive_keys("algo2026", SALT)
    from_bytes = derive_keys(b"algo2026", SALT)
    assert from_text == from_bytes


def test_empty_weak_key():
    with pytest.raises(ValueError):
        weak_encrypt(b"abc", b"")


def test_bad_ga_config():
    config = dict(DEFAULT_CONFIG)
    config["population"] = 1
    config["elite"] = 0
    with pytest.raises(ValueError):
        run_genetic_optimizer(toy_fitness, config)

    config = dict(DEFAULT_CONFIG)
    config["tournament"] = 0
    with pytest.raises(ValueError):
        run_genetic_optimizer(toy_fitness, config)


def test_get_front_does_not_change_ranks():
    items = []
    items.append({"goals": {"S": 1, "R": 1, "T": 1}, "rank": 7})
    items.append({"goals": {"S": 2, "R": 2, "T": 2}, "rank": 8})
    front = get_front(items)
    assert len(front) == 1
    assert front[0]["goals"]["S"] == 2
    assert items[0]["rank"] == 7
    assert items[1]["rank"] == 8


def test_unknown_parameter_name():
    base = {
        "реплікація": {
            "replication_window_records": 48,
            "replication_budget_bits": 5000,
            "second_copy_factor": 1.55,
            "third_copy_factor": 1.85,
        },
        "пріоритети_класів": {"аномальна подія": 5, "перехід до інциденту": 4},
        "critical_bonus": 5,
    }
    with pytest.raises(ValueError):
        make_config(base, {"windowsize": 24})


def test_third_extra_is_added_to_second():
    base = {
        "реплікація": {
            "replication_window_records": 48,
            "replication_budget_bits": 5000,
            "second_copy_factor": 1.55,
            "third_copy_factor": 1.85,
        },
        "пріоритети_класів": {"аномальна подія": 5, "перехід до інциденту": 4},
        "critical_bonus": 5,
    }
    config = make_config(base, {"second_copy": 1.2, "third_extra": 0.5})
    assert config["реплікація"]["second_copy_factor"] == 1.2
    assert abs(config["реплікація"]["third_copy_factor"] - 1.7) < 0.000001


def test_is_bad_uses_loss_limits():
    limits = {"штатний": 0.0, "деградований": 0.1, "аварійний": 0.45}
    assert not is_bad("штатний", {"R": 1.0}, limits)
    assert is_bad("штатний", {"R": 0.99}, limits)
    assert not is_bad("деградований", {"R": 0.9}, limits)
    assert is_bad("деградований", {"R": 0.85}, limits)
    assert not is_bad("аварійний", {"R": 0.6}, limits)
    assert is_bad("аварійний", {"R": 0.5}, limits)
    assert not is_bad("інший", {"R": 0.0}, limits)


def test_fitness_with_real_pipeline():
    config_path = os.path.join(ROOT, "team_config.json")
    if not os.path.exists(config_path):
        pytest.skip("team_config.json is not in the repo root")

    fit = Fitness("team_config.json", "data/session.csv", ["аварійний"])
    params = {"window_size": 48, "budget_bits": 5000, "second_copy": 1.55}
    first = fit.q_fixed(params)
    second = fit.q_fixed(params)
    assert first == second
    assert first > 0.0
    assert first < 1.0


def test_attack_finds_longer_keys():
    text = (b"the quick brown fox jumps over the lazy dog and the node "
            b"keeps sending log events to the central storage every second "
            b"so we can restore all critical records after a failure ") * 3
    for key in [b"Zq9r1x", b"a1B2c3D4"]:
        cipher_text = weak_encrypt(text, key)
        result = attack_key(cipher_text, len(key))
        assert result["key"] == key


def test_attack_finds_weak_key():
    text = (b"the quick brown fox jumps over the lazy dog and the node "
            b"keeps sending log events to the central storage every second "
            b"so we can restore all critical records after a failure ") * 3
    key = b"k3y"
    cipher_text = weak_encrypt(text, key)
    result = attack_key(cipher_text, 3)
    assert result["key"] == key
    assert weak_decrypt(cipher_text, result["key"]) == text
