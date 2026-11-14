import random

DEFAULT_CONFIG = {
    "population": 20,
    "generations": 15,
    "tournament": 3,
    "cross_rate": 0.8,
    "mut_rate": 0.2,
    "elite": 2,
    "seed": 42,
    "window_low": 20,
    "window_high": 300,
    "thr_low": 0.5,
    "thr_high": 1.0,
    "copy_low": 0.0,
    "copy_high": 1.0,
}


def fix_value(value, low, high):
    if value < low:
        value = low
    if value > high:
        value = high
    return value


def random_params(config, rng):
    params = {}
    window = rng.uniform(config["window_low"], config["window_high"])
    params["window_size"] = int(round(window))
    params["threshold"] = rng.uniform(config["thr_low"], config["thr_high"])
    params["second_copy"] = rng.uniform(config["copy_low"], config["copy_high"])
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

    if rng.random() < 0.5:
        child["window_size"] = mom["window_size"]
    else:
        child["window_size"] = dad["window_size"]

    if rng.random() < 0.5:
        child["threshold"] = mom["threshold"]
    else:
        child["threshold"] = dad["threshold"]

    if rng.random() < 0.5:
        child["second_copy"] = mom["second_copy"]
    else:
        child["second_copy"] = dad["second_copy"]

    return child


def mutate(params, config, rng):
    result = params.copy()

    if rng.random() < config["mut_rate"]:
        step = (config["window_high"] - config["window_low"]) * 0.1
        value = result["window_size"] + rng.gauss(0, step)
        value = fix_value(value, config["window_low"], config["window_high"])
        result["window_size"] = int(round(value))

    if rng.random() < config["mut_rate"]:
        step = (config["thr_high"] - config["thr_low"]) * 0.1
        value = result["threshold"] + rng.gauss(0, step)
        value = fix_value(value, config["thr_low"], config["thr_high"])
        result["threshold"] = value

    if rng.random() < config["mut_rate"]:
        step = (config["copy_high"] - config["copy_low"]) * 0.1
        value = result["second_copy"] + rng.gauss(0, step)
        value = fix_value(value, config["copy_low"], config["copy_high"])
        result["second_copy"] = value

    return result


def run_genetic_optimizer(eval_fn, config):
    rng = random.Random(config["seed"])
    size = config["population"]
    elite = config["elite"]

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

    for generation in range(config["generations"]):
        new_people = []

        for i in range(elite):
            new_people.append(people[i])

        while len(new_people) < size:
            mom = tournament(people, config["tournament"], rng)
            dad = tournament(people, config["tournament"], rng)
            child = crossover(mom["params"], dad["params"], config, rng)
            child = mutate(child, config, rng)
            person = make_person(child, eval_fn)
            evals = evals + 1
            new_people.append(person)

        people = sort_best(new_people)
        best_history.append(people[0]["fitness"])
        mean_history.append(mean_fitness(people))
        eval_history.append(evals)

    result = {}
    result["best_params"] = people[0]["params"]
    result["best_fitness"] = people[0]["fitness"]
    result["best_history"] = best_history
    result["mean_history"] = mean_history
    result["eval_history"] = eval_history
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
    file.write("generation,evals,ga_best,ga_mean,random_best\n")
    for gen in range(len(ga["best_history"])):
        evals = ga["eval_history"][gen]
        random_best = rs["best_history"][evals - 1]
        line = "%d,%d,%.6f,%.6f,%.6f\n" % (
            gen,
            evals,
            ga["best_history"][gen],
            ga["mean_history"][gen],
            random_best,
        )
        file.write(line)
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
    plt.title("GA vs Random Search")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
