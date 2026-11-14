import random

BASIC_LIMITS = {
    "window_size": {"low": 20, "high": 300, "whole": True},
    "threshold": {"low": 0.5, "high": 1.0, "whole": False},
    "second_copy": {"low": 1.0, "high": 2.0, "whole": False},
}

ADVANCED_LIMITS = {
    "window_size": {"low": 20, "high": 300, "whole": True},
    "threshold": {"low": 0.5, "high": 1.0, "whole": False},
    "second_copy": {"low": 1.0, "high": 2.0, "whole": False},
    "third_copy": {"low": 1.0, "high": 3.0, "whole": False},
    "hidden_1": {"low": 8, "high": 48, "whole": True},
    "hidden_2": {"low": 4, "high": 24, "whole": True},
    "weight_class": {"low": 0.5, "high": 3.0, "whole": False},
    "weight_critical": {"low": 0.5, "high": 3.0, "whole": False},
    "max_copies": {"low": 1, "high": 3, "whole": True},
}

DEFAULT_CONFIG = {
    "limits": BASIC_LIMITS,
    "population": 20,
    "generations": 15,
    "tournament": 3,
    "cross_rate": 0.8,
    "mut_rate": 0.2,
    "elite": 2,
    "seed": 42,
    "adaptive": False,
    "local_rounds": 0,
}

ADVANCED_CONFIG = {
    "limits": ADVANCED_LIMITS,
    "population": 20,
    "generations": 15,
    "tournament": 3,
    "cross_rate": 0.8,
    "mut_rate": 0.2,
    "elite": 2,
    "seed": 42,
    "adaptive": True,
    "local_rounds": 3,
}


def fix_gene(value, limit):
    if value < limit["low"]:
        value = limit["low"]
    if value > limit["high"]:
        value = limit["high"]

    if limit["whole"]:
        value = int(round(value))
    return value


def random_params(config, rng):
    limits = config["limits"]
    params = {}
    for name in limits:
        limit = limits[name]
        value = rng.uniform(limit["low"], limit["high"])
        params[name] = fix_gene(value, limit)
    return params


def make_person(params, eval_fn):
    fitness = eval_fn(params.copy())
    person = {"params": params, "fitness": fitness}
    return person


def sort_best(people):
    count = len(people)
    for i in range(count):
        for j in range(count - 1 - i):
            if people[j]["fitness"] < people[j + 1]["fitness"]:
                temp = people[j]
                people[j] = people[j + 1]
                people[j + 1] = temp
    return people


def mean_fitness(people):
    total = 0.0
    for person in people:
        total = total + person["fitness"]
    return total / len(people)


def tournament(people, size, rng):
    winner = None
    for i in range(size):
        number = rng.randrange(len(people))
        person = people[number]
        if winner is None:
            winner = person
        elif person["fitness"] > winner["fitness"]:
            winner = person
    return winner


def crossover(mom, dad, config, rng):
    if rng.random() >= config["cross_rate"]:
        return mom.copy()

    child = {}
    for name in config["limits"]:
        if rng.random() < 0.5:
            child[name] = mom[name]
        else:
            child[name] = dad[name]
    return child


def mutate(params, config, rng, rate):
    limits = config["limits"]
    result = params.copy()

    for name in limits:
        if rng.random() < rate:
            limit = limits[name]
            step = (limit["high"] - limit["low"]) * 0.1
            if limit["whole"] and step < 1:
                step = 1
            value = result[name] + rng.gauss(0, step)
            result[name] = fix_gene(value, limit)

    return result


def next_rate(rate, improved):
    if improved:
        rate = rate * 0.8
        if rate < 0.05:
            rate = 0.05
    else:
        rate = rate * 1.5
        if rate > 0.6:
            rate = 0.6
    return rate


def try_value(best, name, new_value, eval_fn):
    params = best["params"].copy()
    params[name] = new_value
    return make_person(params, eval_fn)


def local_search(person, eval_fn, config):
    limits = config["limits"]
    best = person
    evals = 0

    for round_number in range(config["local_rounds"]):
        found = False

        for name in limits:
            limit = limits[name]
            step = (limit["high"] - limit["low"]) * 0.05
            if limit["whole"] and step < 1:
                step = 1

            value = best["params"][name]
            up_value = fix_gene(value + step, limit)
            down_value = fix_gene(value - step, limit)
            up_better = False

            if up_value != value:
                candidate = try_value(best, name, up_value, eval_fn)
                evals = evals + 1
                if candidate["fitness"] > best["fitness"]:
                    best = candidate
                    found = True
                    up_better = True

            if down_value != value and not up_better:
                candidate = try_value(best, name, down_value, eval_fn)
                evals = evals + 1
                if candidate["fitness"] > best["fitness"]:
                    best = candidate
                    found = True

        if not found:
            break

    result = {"best": best, "evals": evals}
    return result


def run_genetic_optimizer(eval_fn, config):
    rng = random.Random(config["seed"])
    size = config["population"]
    elite = config["elite"]
    rate = config["mut_rate"]

    if elite >= size:
        raise ValueError("Elite must be smaller than population")

    evals = 0
    people = []
    for i in range(size):
        params = random_params(config, rng)
        person = make_person(params, eval_fn)
        evals = evals + 1
        people.append(person)
    people = sort_best(people)

    best_history = [people[0]["fitness"]]
    mean_history = [mean_fitness(people)]
    eval_history = [evals]
    rate_history = [rate]

    for generation in range(config["generations"]):
        old_best = people[0]["fitness"]
        new_people = []

        for i in range(elite):
            new_people.append(people[i])

        while len(new_people) < size:
            mom = tournament(people, config["tournament"], rng)
            dad = tournament(people, config["tournament"], rng)
            child = crossover(mom["params"], dad["params"], config, rng)
            child = mutate(child, config, rng, rate)
            person = make_person(child, eval_fn)
            evals = evals + 1
            new_people.append(person)

        people = sort_best(new_people)

        if config["adaptive"]:
            improved = people[0]["fitness"] > old_best
            rate = next_rate(rate, improved)

        best_history.append(people[0]["fitness"])
        mean_history.append(mean_fitness(people))
        eval_history.append(evals)
        rate_history.append(rate)

    best = people[0]
    local_gain = 0.0

    if config["local_rounds"] > 0:
        before = best["fitness"]
        local = local_search(best, eval_fn, config)
        best = local["best"]
        evals = evals + local["evals"]
        local_gain = best["fitness"] - before
        best_history.append(best["fitness"])
        mean_history.append(mean_fitness(people))
        eval_history.append(evals)
        rate_history.append(rate)

    result = {}
    result["best_params"] = best["params"]
    result["best_fitness"] = best["fitness"]
    result["best_history"] = best_history
    result["mean_history"] = mean_history
    result["eval_history"] = eval_history
    result["rate_history"] = rate_history
    result["local_gain"] = local_gain
    result["evals"] = evals
    return result


def run_random_search(eval_fn, config, evals):
    rng = random.Random(config["seed"] + 1)
    best_params = None
    best_fitness = None
    best_history = []

    for i in range(evals):
        params = random_params(config, rng)
        fitness = eval_fn(params.copy())

        if best_fitness is None:
            best_fitness = fitness
            best_params = params
        elif fitness > best_fitness:
            best_fitness = fitness
            best_params = params

        best_history.append(best_fitness)

    result = {}
    result["best_params"] = best_params
    result["best_fitness"] = best_fitness
    result["best_history"] = best_history
    result["evals"] = evals
    return result


def compare_search(eval_fn, config):
    ga = run_genetic_optimizer(eval_fn, config)
    rs = run_random_search(eval_fn, config, ga["evals"])

    if ga["best_fitness"] > rs["best_fitness"]:
        winner = "GA"
    elif ga["best_fitness"] < rs["best_fitness"]:
        winner = "RandomSearch"
    else:
        winner = "draw"

    comparison = {"ga": ga, "random": rs, "winner": winner}
    return comparison


def elitism_ok(ga):
    first = ga["best_history"][0]
    for value in ga["best_history"]:
        if value < first:
            return False
    return True


def save_csv(comparison, path):
    ga = comparison["ga"]
    rs = comparison["random"]

    file = open(path, "w", encoding="utf-8")
    print("step,evals,ga_best,ga_mean,random_best,mut_rate", file=file)

    for step in range(len(ga["best_history"])):
        evals = ga["eval_history"][step]
        random_best = rs["best_history"][evals - 1]

        line = str(step) + ","
        line = line + str(evals) + ","
        line = line + str(round(ga["best_history"][step], 6)) + ","
        line = line + str(round(ga["mean_history"][step], 6)) + ","
        line = line + str(round(random_best, 6)) + ","
        line = line + str(round(ga["rate_history"][step], 3))
        print(line, file=file)

    file.close()


def save_plot(comparison, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ga = comparison["ga"]
    rs = comparison["random"]

    random_x = []
    for i in range(len(rs["best_history"])):
        random_x.append(i + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(ga["eval_history"], ga["best_history"], marker="o", label="GA best")
    plt.plot(ga["eval_history"], ga["mean_history"], linestyle="--", label="GA mean")
    plt.plot(random_x, rs["best_history"], label="Random Search best")
    plt.xlabel("Fitness evaluations")
    plt.ylabel("Fitness (Q)")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
