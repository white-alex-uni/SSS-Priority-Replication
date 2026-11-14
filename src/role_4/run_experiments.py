import os
import sys
import time

from src.role_4.attack import attack_key, tamper_test, weak_decrypt, weak_encrypt
from src.role_4.fitness import Fitness, load_config
from src.role_4.genetic import ADVANCED_CONFIG, compare_search, elitism_ok, save_csv, save_plot
from src.role_4.pareto import compare_front, front_is_clean, save_front_csv, save_front_plot

CONFIG_PATH = "team_config.json"
SESSION_PATH = "data/session.csv"
OUT_DIR = os.path.join("results", "role_4")
SCENARIOS = ["штатний", "деградований", "аварійний"]
CHECK_SEEDS = [11, 22, 33, 44, 55]


def make_run_config():
    config = dict(ADVANCED_CONFIG)
    config["population"] = 16
    config["generations"] = 12
    return config


def base_params(config):
    rep = config["реплікація"]
    params = {}
    params["window_size"] = rep["replication_window_records"]
    params["budget_bits"] = rep["replication_budget_bits"]
    params["second_copy"] = rep["second_copy_factor"]
    params["third_copy"] = rep["third_copy_factor"]
    params["prio_anomaly"] = config["пріоритети_класів"]["аномальна подія"]
    params["prio_incident"] = config["пріоритети_класів"]["перехід до інциденту"]
    params["critical_bonus"] = config["critical_bonus"]
    return params


def params_text(params):
    text = ""
    for name in params:
        text = text + name + "=" + str(round(params[name], 4)) + " "
    return text


def save_block_table(base, tuned, path):
    file = open(path, "w", encoding="utf-8")
    print("scenario,variant,S,R,T,Q", file=file)

    for scenario in SCENARIOS:
        fit = Fitness(CONFIG_PATH, SESSION_PATH, [scenario], CHECK_SEEDS)

        without_ga = fit.average(base)
        with_ga = fit.average(tuned)

        line = scenario + ",without_GA,"
        line = line + str(round(without_ga["S"], 4)) + ","
        line = line + str(round(without_ga["R"], 4)) + ","
        line = line + str(round(without_ga["T"], 4)) + ","
        line = line + str(round(without_ga["Q"], 4))
        print(line, file=file)
        print(line, flush=True)

        line = scenario + ",with_GA,"
        line = line + str(round(with_ga["S"], 4)) + ","
        line = line + str(round(with_ga["R"], 4)) + ","
        line = line + str(round(with_ga["T"], 4)) + ","
        line = line + str(round(with_ga["Q"], 4))
        print(line, file=file)
        print(line, flush=True)

    file.close()


def run_ga():
    start = time.time()
    fit = Fitness(CONFIG_PATH, SESSION_PATH, SCENARIOS)
    config = make_run_config()

    comparison = compare_search(fit.q_fixed, config)
    ga = comparison["ga"]
    rs = comparison["random"]

    save_csv(comparison, os.path.join(OUT_DIR, "ga_convergence.csv"))
    save_plot(comparison, os.path.join(OUT_DIR, "ga_convergence.png"))

    base = base_params(fit.config)
    base_value = fit.q_fixed(base)

    file = open(os.path.join(OUT_DIR, "ga_summary.txt"), "w", encoding="utf-8")
    print("Fitness: 0.35*S + 0.5*R, average of 3 scenarios, open seed", file=file)
    print("Population: " + str(config["population"]), file=file)
    print("Generations: " + str(config["generations"]), file=file)
    print("Evaluations GA: " + str(ga["evals"]), file=file)
    print("Evaluations Random Search: " + str(rs["evals"]), file=file)
    print("Base config fitness: " + str(round(base_value, 5)), file=file)
    print("GA best fitness: " + str(round(ga["best_fitness"], 5)), file=file)
    print("Random Search best fitness: " + str(round(rs["best_fitness"], 5)), file=file)
    print("Winner: " + comparison["winner"], file=file)
    print("Elitism is not worse than generation 0: " + str(elitism_ok(ga)), file=file)
    print("Local search gain: " + str(round(ga["local_gain"], 5)), file=file)
    print("Base params: " + params_text(base), file=file)
    print("GA params: " + params_text(ga["best_params"]), file=file)
    file.close()

    print("GA done, winner: " + comparison["winner"], flush=True)
    save_block_table(base, ga["best_params"], os.path.join(OUT_DIR, "without_ga_table.csv"))
    print("GA part took " + str(round(time.time() - start)) + " seconds", flush=True)


def run_pareto():
    start = time.time()
    fit = Fitness(CONFIG_PATH, SESSION_PATH, SCENARIOS)
    config = make_run_config()

    comparison = compare_front(fit.goals, config)
    ga = comparison["ga"]
    rs = comparison["random"]

    save_front_csv(ga["front"], os.path.join(OUT_DIR, "pareto_front.csv"))
    save_front_plot(comparison, os.path.join(OUT_DIR, "pareto_front.png"))

    file = open(os.path.join(OUT_DIR, "pareto_summary.txt"), "w", encoding="utf-8")
    print("Goals: S, R, T, average of 3 scenarios, open seed", file=file)
    print("Evaluations GA: " + str(ga["evals"]), file=file)
    print("Evaluations Random Search: " + str(rs["evals"]), file=file)
    print("Front size GA: " + str(len(ga["front"])), file=file)
    print("Front size Random Search: " + str(len(rs["front"])), file=file)
    print("GA front covers Random Search front: " + str(round(comparison["ga_covers"], 3)), file=file)
    print("Random Search front covers GA front: " + str(round(comparison["rs_covers"], 3)), file=file)
    print("Winner: " + comparison["winner"], file=file)
    print("GA front has no dominated points: " + str(front_is_clean(ga["front"])), file=file)
    file.close()

    print("Pareto done, winner: " + comparison["winner"], flush=True)
    print("Pareto part took " + str(round(time.time() - start)) + " seconds", flush=True)


def run_attack():
    text = (b"the quick brown fox jumps over the lazy dog and the node "
            b"keeps sending log events to the central storage every second "
            b"so we can restore all critical records after a failure ") * 3
    secret = b"k3y"
    cipher_text = weak_encrypt(text, secret)
    found = attack_key(cipher_text, len(secret))

    config = load_config(CONFIG_PATH)
    password = config["пароль_сеансу"]
    header = b"LOG3" + bytes(39)
    payload = b"critical event: disk failure on node B " * 5
    salt = bytes(range(16))
    nonce = bytes(range(100, 112))
    tamper = tamper_test(header, payload, password, salt, nonce)

    file = open(os.path.join(OUT_DIR, "attack.txt"), "w", encoding="utf-8")
    print("Part 1. GA attack on a weak cipher (XOR with a short repeating key)", file=file)
    print("Real key: " + str(secret), file=file)
    print("Found key: " + str(found["key"]), file=file)
    print("Fitness evaluations: " + str(found["evals"]), file=file)
    if weak_decrypt(cipher_text, found["key"]) == text:
        print("Result: key recovered", file=file)
    else:
        print("Result: key NOT recovered", file=file)
    print("", file=file)
    print("Part 2. Replica tamper test (each byte of header, ciphertext and HMAC is changed)", file=file)
    print("Checks: " + str(tamper["total"]), file=file)
    print("Rejected: " + str(tamper["rejected"]), file=file)
    print("Accepted: " + str(tamper["accepted"]), file=file)
    if tamper["accepted"] == 0:
        print("Result: all changed replicas are rejected", file=file)
    else:
        print("Result: ERROR, some changed replicas were accepted", file=file)
    file.close()

    print("Attack done, key found: " + str(found["key"]), flush=True)
    print("Tamper test accepted: " + str(tamper["accepted"]), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    mode = "all"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == "ga":
        run_ga()
    elif mode == "pareto":
        run_pareto()
    elif mode == "attack":
        run_attack()
    else:
        run_attack()
        run_ga()
        run_pareto()
