import random

from src.role_4.genetic import random_params, crossover, mutate


def dominates(goals_a, goals_b):
    if goals_a["S"] < goals_b["S"]:
        return False
    if goals_a["R"] < goals_b["R"]:
        return False
    if goals_a["T"] < goals_b["T"]:
        return False

    better = False
    if goals_a["S"] > goals_b["S"]:
        better = True
    if goals_a["R"] > goals_b["R"]:
        better = True
    if goals_a["T"] > goals_b["T"]:
        better = True
    return better


def make_item(params, eval_multi):
    goals = eval_multi(params.copy())
    item = {"params": params, "goals": goals, "rank": 0}
    return item


def total_goals(item):
    goals = item["goals"]
    return goals["S"] + goals["R"] + goals["T"]


def set_ranks(items):
    for item in items:
        count = 0
        for other in items:
            if dominates(other["goals"], item["goals"]):
                count = count + 1
        item["rank"] = count


def is_better(item_a, item_b):
    if item_a["rank"] < item_b["rank"]:
        return True
    if item_a["rank"] > item_b["rank"]:
        return False
    if total_goals(item_a) > total_goals(item_b):
        return True
    return False


def sort_items(items):
    count = len(items)
    for i in range(count):
        for j in range(count - 1 - i):
            if is_better(items[j + 1], items[j]):
                temp = items[j]
                items[j] = items[j + 1]
                items[j + 1] = temp
    return items


def pick_winner(items, size, rng):
    winner = None
    for i in range(size):
        number = rng.randrange(len(items))
        item = items[number]
        if winner is None:
            winner = item
        elif is_better(item, winner):
            winner = item
    return winner


def get_front(items):
    set_ranks(items)
    front = []
    for item in items:
        if item["rank"] == 0:
            front.append(item)
    return front


def run_pareto(eval_multi, config):
    rng = random.Random(config["seed"])
    size = config["population"]
    elite = config["elite"]
    rate = config["mut_rate"]

    if elite >= size:
        raise ValueError("Elite must be smaller than population")

    evals = 0
    archive = []
    items = []
    for i in range(size):
        params = random_params(config, rng)
        item = make_item(params, eval_multi)
        evals = evals + 1
        items.append(item)
        archive.append(item)

    set_ranks(items)
    items = sort_items(items)
    front_history = [len(get_front(archive))]
    eval_history = [evals]

    for generation in range(config["generations"]):
        new_items = []

        for i in range(elite):
            new_items.append(items[i])

        while len(new_items) < size:
            mom = pick_winner(items, config["tournament"], rng)
            dad = pick_winner(items, config["tournament"], rng)
            child = crossover(mom["params"], dad["params"], config, rng)
            child = mutate(child, config, rng, rate)
            item = make_item(child, eval_multi)
            evals = evals + 1
            new_items.append(item)
            archive.append(item)

        set_ranks(new_items)
        items = sort_items(new_items)
        front_history.append(len(get_front(archive)))
        eval_history.append(evals)

    result = {}
    result["front"] = get_front(archive)
    result["front_history"] = front_history
    result["eval_history"] = eval_history
    result["evals"] = evals
    return result


def run_random_front(eval_multi, config, evals):
    rng = random.Random(config["seed"] + 1)
    archive = []

    for i in range(evals):
        params = random_params(config, rng)
        item = make_item(params, eval_multi)
        archive.append(item)

    result = {}
    result["front"] = get_front(archive)
    result["evals"] = evals
    return result


def cover_share(front_a, front_b):
    if len(front_b) == 0:
        return 0.0

    covered = 0
    for item_b in front_b:
        found = False
        for item_a in front_a:
            if dominates(item_a["goals"], item_b["goals"]):
                found = True
        if found:
            covered = covered + 1
    return covered / len(front_b)


def compare_front(eval_multi, config):
    ga = run_pareto(eval_multi, config)
    rs = run_random_front(eval_multi, config, ga["evals"])

    ga_covers = cover_share(ga["front"], rs["front"])
    rs_covers = cover_share(rs["front"], ga["front"])

    if ga_covers > rs_covers:
        winner = "GA"
    elif ga_covers < rs_covers:
        winner = "RandomSearch"
    else:
        winner = "draw"

    comparison = {}
    comparison["ga"] = ga
    comparison["random"] = rs
    comparison["ga_covers"] = ga_covers
    comparison["rs_covers"] = rs_covers
    comparison["winner"] = winner
    return comparison


def front_is_clean(front):
    for item_a in front:
        for item_b in front:
            if dominates(item_a["goals"], item_b["goals"]):
                return False
    return True


def save_front_csv(front, path):
    file = open(path, "w", encoding="utf-8")
    print("S,R,T,params", file=file)

    for item in front:
        text = ""
        for name in item["params"]:
            value = round(item["params"][name], 4)
            text = text + name + "=" + str(value) + " "

        s = str(round(item["goals"]["S"], 6))
        r = str(round(item["goals"]["R"], 6))
        t = str(round(item["goals"]["T"], 6))
        print(s + "," + r + "," + t + "," + text, file=file)

    file.close()


def save_front_plot(comparison, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ga_s = []
    ga_r = []
    ga_t = []
    for item in comparison["ga"]["front"]:
        ga_s.append(item["goals"]["S"])
        ga_r.append(item["goals"]["R"])
        ga_t.append(item["goals"]["T"])

    rs_s = []
    rs_r = []
    rs_t = []
    for item in comparison["random"]["front"]:
        rs_s.append(item["goals"]["S"])
        rs_r.append(item["goals"]["R"])
        rs_t.append(item["goals"]["T"])

    plt.figure(figsize=(11, 4.5))

    plt.subplot(1, 2, 1)
    plt.scatter(rs_s, rs_r, label="Random Search")
    plt.scatter(ga_s, ga_r, label="GA Pareto")
    plt.xlabel("S")
    plt.ylabel("R")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(rs_t, rs_r, label="Random Search")
    plt.scatter(ga_t, ga_r, label="GA Pareto")
    plt.xlabel("T")
    plt.ylabel("R")
    plt.legend()
    plt.grid(True)

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
