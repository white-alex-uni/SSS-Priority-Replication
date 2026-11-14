import copy
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.role_1 import load_csv
from src.role_5.pipeline import evaluate_pipeline

KNOWN_NAMES = [
    "window_size",
    "budget_bits",
    "second_copy",
    "third_copy",
    "third_extra",
    "prio_anomaly",
    "prio_incident",
    "critical_bonus",
]


def full_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def load_config(path):
    file = open(full_path(path), "r", encoding="utf-8")
    config = json.load(file)
    file.close()
    return config


def make_config(base_config, params):
    for name in params:
        if name not in KNOWN_NAMES:
            raise ValueError("Unknown parameter: " + name)

    config = copy.deepcopy(base_config)
    rep = config["реплікація"]

    if "window_size" in params:
        rep["replication_window_records"] = params["window_size"]

    if "budget_bits" in params:
        rep["replication_budget_bits"] = params["budget_bits"]

    if "second_copy" in params:
        rep["second_copy_factor"] = params["second_copy"]

    if "third_copy" in params:
        rep["third_copy_factor"] = params["third_copy"]

    if "third_extra" in params:
        rep["third_copy_factor"] = rep["second_copy_factor"] + params["third_extra"]

    if rep["third_copy_factor"] < rep["second_copy_factor"]:
        rep["third_copy_factor"] = rep["second_copy_factor"]

    if "prio_anomaly" in params:
        config["пріоритети_класів"]["аномальна подія"] = params["prio_anomaly"]

    if "prio_incident" in params:
        config["пріоритети_класів"]["перехід до інциденту"] = params["prio_incident"]

    if "critical_bonus" in params:
        config["critical_bonus"] = params["critical_bonus"]

    return config


def is_bad(scenario, result, thresholds):
    if scenario not in thresholds:
        return False

    loss = 1.0 - result["R"]
    if loss > thresholds[scenario] + 0.000001:
        return True
    return False


class Fitness:
    def __init__(self, config_path, session_path, scenarios, seeds=None):
        self.config = load_config(config_path)
        self.records = load_csv(full_path(session_path))
        self.scenarios = scenarios
        if seeds is None:
            seeds = [self.config["реплікація"]["seed_моделі_реплікації"]]
        self.seeds = seeds
        self.calls = 0

    def average(self, params):
        config = make_config(self.config, params)
        thresholds = self.config["пороги_втрати_критичних_подій"]

        total_s = 0.0
        total_r = 0.0
        total_t = 0.0
        total_q = 0.0
        count = 0
        bad = 0

        for scenario in self.scenarios:
            for seed in self.seeds:
                result = evaluate_pipeline(self.records, scenario, seed, config)
                total_s = total_s + result["S"]
                total_r = total_r + result["R"]
                total_t = total_t + result["T"]
                total_q = total_q + result["Q"]
                count = count + 1
                if is_bad(scenario, result, thresholds):
                    bad = bad + 1

        self.calls = self.calls + 1

        average = {}
        average["S"] = total_s / count
        average["R"] = total_r / count
        average["T"] = total_t / count
        average["Q"] = total_q / count
        average["bad"] = bad
        average["count"] = count
        average["penalty"] = bad / count
        return average

    def q_fixed(self, params):
        average = self.average(params)
        alpha = self.config["метрика_Q"]["alpha"]
        beta = self.config["метрика_Q"]["beta"]
        return alpha * average["S"] + beta * average["R"] - average["penalty"]

    def goals(self, params):
        average = self.average(params)
        goals = {}
        goals["S"] = average["S"] - average["penalty"]
        goals["R"] = average["R"] - average["penalty"]
        goals["T"] = round(average["T"], 2) - average["penalty"]
        return goals
