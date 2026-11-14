import copy
import json

from src.role_1 import load_csv
from src.role_5.pipeline import evaluate_pipeline


def load_config(path):
    file = open(path, "r", encoding="utf-8")
    config = json.load(file)
    file.close()
    return config


def make_config(base_config, params):
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

    if rep["third_copy_factor"] < rep["second_copy_factor"]:
        rep["third_copy_factor"] = rep["second_copy_factor"]

    if "prio_anomaly" in params:
        config["пріоритети_класів"]["аномальна подія"] = params["prio_anomaly"]

    if "prio_incident" in params:
        config["пріоритети_класів"]["перехід до інциденту"] = params["prio_incident"]

    if "critical_bonus" in params:
        config["critical_bonus"] = params["critical_bonus"]

    return config


class Fitness:
    def __init__(self, config_path, session_path, scenarios, seeds=None):
        self.config = load_config(config_path)
        self.records = load_csv(session_path)
        self.scenarios = scenarios
        if seeds is None:
            seeds = [self.config["реплікація"]["seed_моделі_реплікації"]]
        self.seeds = seeds
        self.calls = 0

    def average(self, params):
        config = make_config(self.config, params)

        total_s = 0.0
        total_r = 0.0
        total_t = 0.0
        total_q = 0.0
        count = 0

        for scenario in self.scenarios:
            for seed in self.seeds:
                result = evaluate_pipeline(self.records, scenario, seed, config)
                total_s = total_s + result["S"]
                total_r = total_r + result["R"]
                total_t = total_t + result["T"]
                total_q = total_q + result["Q"]
                count = count + 1

        self.calls = self.calls + 1

        average = {}
        average["S"] = total_s / count
        average["R"] = total_r / count
        average["T"] = total_t / count
        average["Q"] = total_q / count
        return average

    def q_fixed(self, params):
        average = self.average(params)
        alpha = self.config["метрика_Q"]["alpha"]
        beta = self.config["метрика_Q"]["beta"]
        return alpha * average["S"] + beta * average["R"]

    def goals(self, params):
        average = self.average(params)
        goals = {}
        goals["S"] = average["S"]
        goals["R"] = average["R"]
        goals["T"] = average["T"]
        return goals
