"""Small, dependency-free ride-hailing experiment. Python 3.10+.

All times are minutes. This demonstration uses synthetic data only.
Run: python simulator.py --self-test
Run: python simulator.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Request:
    id: int
    arrival: float
    origin: int
    destination: int


@dataclass
class Driver:
    # During service, zone is the scheduled destination, not current GPS.
    id: int
    zone: int
    available_at: float = 0.0


@dataclass(frozen=True)
class Config:
    duration: int = 180
    fleet: int = 8
    delta: float = 1.0
    max_wait: float = 8.0
    horizon: float = 30.0
    scenarios: int = 3
    weight: float = 3.0
    search_passes: int = 2
    candidate_budget: int = 60
    history_days: int = 10


class Travel:
    """Fixed zone table; intra-zone travel takes two minutes.

    The interface permits departure-time-dependent tables later.
    This minimal example deliberately uses time-independent travel.
    """
    coordinates = ((0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2))

    def minutes(self, origin: int, destination: int, departure: float) -> float:
        a, b = self.coordinates[origin], self.coordinates[destination]
        return 2.0 + 3.0 * (abs(a[0] - b[0]) + abs(a[1] - b[1]))


def next_epoch(t: float, delta: float) -> float:
    return math.ceil((t - 1e-10) / delta) * delta


def maximum_matching(adjacency: list[list[int]], right_size: int) -> int:
    """Maximum cardinality bipartite matching; one next trip per driver."""
    owner = [-1] * right_size

    def augment(left: int, seen: set[int]) -> bool:
        for right in adjacency[left]:
            if right in seen:
                continue
            seen.add(right)
            if owner[right] == -1 or augment(owner[right], seen):
                owner[right] = left
                return True
        return False

    return sum(augment(left, set()) for left in range(len(adjacency)))


def min_cost_maximum_matching(costs: list[list[float | None]]) -> dict[int, int]:
    """Exact maximum-cardinality, then minimum-cost bipartite matching.

    Successive shortest augmenting paths with residual reverse edges.
    Bellman-Ford handles negative reverse costs. No big-M penalty and
    no third-party solver are required. Intended for small instances.
    """
    n = len(costs)
    m = len(costs[0]) if n else 0
    source, sink = n + m, n + m + 1
    graph: list[list[list]] = [[] for _ in range(n + m + 2)]

    def edge(u: int, v: int, cost: float) -> list:
        forward = [v, len(graph[v]), 1, cost]
        reverse = [u, len(graph[u]), 0, -cost]
        graph[u].append(forward)
        graph[v].append(reverse)
        return forward

    for i in range(n):
        edge(source, i, 0.0)
    for j in range(m):
        edge(n + j, sink, 0.0)
    assignment_edges = {}
    for i, row in enumerate(costs):
        for j, cost in enumerate(row):
            if cost is not None:
                assignment_edges[i, j] = edge(i, n + j, cost)

    while True:
        distance = [math.inf] * len(graph)
        parent = [None] * len(graph)
        distance[source] = 0.0
        for _ in range(len(graph) - 1):
            changed = False
            for u, edges in enumerate(graph):
                if distance[u] == math.inf:
                    continue
                for index, (v, _, capacity, cost) in enumerate(edges):
                    if capacity and distance[u] + cost < distance[v] - 1e-10:
                        distance[v] = distance[u] + cost
                        parent[v] = (u, index)
                        changed = True
            if not changed:
                break
        if parent[sink] is None:
            break
        v = sink
        while v != source:
            u, index = parent[v]
            e = graph[u][index]
            e[2] -= 1
            graph[v][e[1]][2] += 1
            v = u
    return {i: j for (i, j), e in assignment_edges.items() if e[2] == 0}


def synthetic_day(seed: int, end: int) -> list[Request]:
    """Generate a demand stream, not vehicle trajectories or dispatch labels."""
    rng = random.Random(seed)
    requests = []
    arrival = 0.0
    while True:
        arrival += rng.expovariate(0.65)
        if arrival >= end:
            return requests
        # Demand moves between two groups of zones during the day.
        if (int(arrival) // 60) % 2 == 0:
            origin_weights = (8, 5, 2, 1, 1, 1)
            destination_weights = (1, 1, 2, 5, 5, 3)
        else:
            origin_weights = (1, 1, 1, 5, 8, 2)
            destination_weights = (5, 5, 3, 1, 1, 2)
        origin = rng.choices(range(6), weights=origin_weights)[0]
        destination = rng.choices(range(6), weights=destination_weights)[0]
        requests.append(Request(len(requests), arrival, origin, destination))


class HistoricalScenarios:
    """Only historical days enter this object. Never pass it a test stream."""
    def __init__(self, days: list[list[Request]], config: Config):
        self.days = days
        self.config = config

    def at(self, t: float) -> list[list[Request]]:
        # Same scenario sample for all candidate actions at the current epoch.
        rng = random.Random(100000 + round(t / self.config.delta))
        indexes = rng.sample(range(len(self.days)), self.config.scenarios)
        end = min(t + self.config.horizon, self.config.duration)
        return [[r for r in self.days[i] if t < r.arrival < end] for i in indexes]


def future_capacity(
    drivers: list[Driver], requests: dict[int, Request], action: dict[int, int],
    t: float, scenarios: list[list[Request]], travel: Travel, config: Config,
) -> float:
    """Scenario-wise next-service count after a candidate current action.

    Each scenario is evaluated separately with hypothetical full visibility
    inside that scenario. This is an optimistic proxy, not an executable
    future policy or an unbiased estimate of actual future completions.
    Outstanding current orders left unmatched are not included in this proxy.
    """
    availability = []
    for driver in drivers:
        if driver.id in action:
            r = requests[action[driver.id]]
            pickup = t + travel.minutes(driver.zone, r.origin, t)
            release = pickup + travel.minutes(r.origin, r.destination, pickup)
            availability.append((r.destination, release))
        else:
            availability.append((driver.zone, max(t, driver.available_at)))

    counts = []
    for scenario in scenarios:
        adjacency = []
        for zone, available_at in availability:
            feasible = []
            for j, r in enumerate(scenario):
                # No departure before request arrival; respect matching clock.
                dispatch = next_epoch(max(available_at, r.arrival), config.delta)
                pickup = dispatch + travel.minutes(zone, r.origin, dispatch)
                if dispatch < r.arrival + config.max_wait and pickup <= r.arrival + config.max_wait + 1e-9:
                    feasible.append(j)
            adjacency.append(feasible)
        counts.append(maximum_matching(adjacency, len(scenario)))
    return statistics.mean(counts) if counts else 0.0


def choose_action(
    policy: str, drivers: list[Driver], pending: list[Request], t: float,
    scenarios: list[list[Request]], travel: Travel, config: Config,
    diagnostic: dict | None = None,
) -> dict[int, int]:
    idle = [d for d in drivers if d.available_at <= t + 1e-9]
    costs = [[travel.minutes(d.zone, r.origin, t)
              if t - r.arrival + travel.minutes(d.zone, r.origin, t) <= config.max_wait + 1e-9
              else None for r in pending] for d in idle]
    indices = min_cost_maximum_matching(costs)
    baseline = {idle[i].id: pending[j].id for i, j in indices.items()}
    if diagnostic is None:
        diagnostic = {}
    diagnostic.update(time=t, idle_drivers=len(idle), pending_requests=len(pending),
                      feasible_pairs=sum(c is not None for row in costs for c in row),
                      baseline_matches=len(baseline), evaluated_candidates=0,
                      candidates_with_different_future_count=0,
                      candidates_with_higher_future_count=0,
                      baseline_future_count=None, selected_future_count=None,
                      accepted_improvements=0, changed_from_same_state_baseline=False,
                      score_gain=0.0)
    if policy == "baseline" or config.weight == 0 or not baseline:
        return baseline
    requests = {r.id: r for r in pending}
    pickups = {(idle[i].id, pending[j].id): cost
               for i, row in enumerate(costs) for j, cost in enumerate(row) if cost is not None}

    def evaluate(action):
        future = future_capacity(drivers, requests, action, t, scenarios, travel, config)
        return sum(pickups[i, j] for i, j in action.items()) - config.weight * future, future

    # Deterministic, bounded best-improvement local search. Current order
    # subset is unchanged; only drivers are substituted or swapped.
    current = baseline.copy()
    value, baseline_future = evaluate(current)
    initial_value = value
    diagnostic["baseline_future_count"] = baseline_future
    diagnostic["selected_future_count"] = baseline_future
    examined = 0
    for _ in range(config.search_passes):
        candidates = []
        assigned = sorted(current)
        for old in assigned:
            for new in idle:
                if new.id not in current and (new.id, current[old]) in pickups:
                    candidate = current.copy()
                    candidate[new.id] = candidate.pop(old)
                    candidates.append(candidate)
        for pos, a in enumerate(assigned):
            for b in assigned[pos + 1:]:
                if (a, current[b]) in pickups and (b, current[a]) in pickups:
                    candidate = current.copy()
                    candidate[a], candidate[b] = candidate[b], candidate[a]
                    candidates.append(candidate)
        best, best_value = current, value
        for candidate in candidates:
            if examined >= config.candidate_budget:
                break
            examined += 1
            score, future = evaluate(candidate)
            diagnostic["evaluated_candidates"] += 1
            diagnostic["candidates_with_different_future_count"] += int(abs(future-baseline_future) > 1e-9)
            diagnostic["candidates_with_higher_future_count"] += int(future > baseline_future+1e-9)
            if score < best_value - 1e-9:
                best, best_value = candidate, score
        if best_value >= value - 1e-9:
            break
        current, value = best, best_value
        diagnostic["accepted_improvements"] += 1
        if examined >= config.candidate_budget:
            break
    diagnostic["changed_from_same_state_baseline"] = current != baseline
    diagnostic["score_gain"] = initial_value - value
    diagnostic["selected_future_count"] = evaluate(current)[1]
    assert len(current) == len(baseline)
    assert set(current.values()) == set(baseline.values())
    return current


def overlap(start: float, end: float, horizon: float) -> float:
    return max(0.0, min(end, horizon) - max(start, 0.0))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * p
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def simulate(
    policy: str, stream: list[Request], initial_zones: list[int],
    history: HistoricalScenarios, config: Config, travel: Travel | None = None,
    diagnostics: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    travel = travel or Travel()
    decision_records = diagnostics if diagnostics is not None else []
    drivers = [Driver(i, z) for i, z in enumerate(initial_zones)]
    cohort = sorted([r for r in stream if 0 <= r.arrival < config.duration], key=lambda r: (r.arrival, r.id))
    assert len({r.id for r in cohort}) == len(cohort)
    logs = {r.id: {"request_id": r.id, "arrival": r.arrival, "origin": r.origin,
                   "destination": r.destination, "status": "pending", "driver_id": None,
                   "dispatch": None, "pickup": None, "completion": None,
                   "wait": None, "pickup_travel": None,
                   "observed_epochs": 0, "epochs_no_idle": 0,
                   "epochs_idle_but_unreachable": 0, "epochs_feasible": 0,
                   "expiry_observation": None} for r in cohort}
    pending: list[Request] = []
    cursor, epoch = 0, 0
    pickup_time = loaded_time = 0.0
    decision_ms = []
    while True:
        t = epoch * config.delta
        while cursor < len(cohort) and cohort[cursor].arrival <= t + 1e-9:
            pending.append(cohort[cursor])
            cursor += 1
        alive = []
        for r in pending:
            if t >= r.arrival + config.max_wait - 1e-9:
                logs[r.id]["status"] = "expired"
            else:
                alive.append(r)
        pending = alive
        # Observe feasibility BEFORE assignments, over each request's lifetime.
        available = [d for d in drivers if d.available_at <= t+1e-9]
        for r in pending:
            row = logs[r.id]
            row["observed_epochs"] += 1
            if not available:
                row["epochs_no_idle"] += 1
            elif any(t-r.arrival+travel.minutes(d.zone, r.origin, t) <= config.max_wait+1e-9 for d in available):
                row["epochs_feasible"] += 1
            else:
                row["epochs_idle_but_unreachable"] += 1
        if pending and any(d.available_at <= t + 1e-9 for d in drivers):
            begin = time.perf_counter()
            scenarios = history.at(t) if policy == "lookahead" and config.weight > 0 else []
            diagnostic = {}
            action = choose_action(policy, drivers, pending, t, scenarios, travel, config, diagnostic)
            decision_records.append(diagnostic)
            decision_ms.append(1000 * (time.perf_counter() - begin))
            by_id = {r.id: r for r in pending}
            assert len(set(action.values())) == len(action)
            for driver_id, request_id in action.items():
                driver, r = drivers[driver_id], by_id[request_id]
                assert driver.available_at <= t + 1e-9
                p = travel.minutes(driver.zone, r.origin, t)
                u = t + p
                f = u + travel.minutes(r.origin, r.destination, u)
                assert u - r.arrival <= config.max_wait + 1e-9
                pickup_time += overlap(t, u, config.duration)
                loaded_time += overlap(u, f, config.duration)
                driver.zone, driver.available_at = r.destination, f
                logs[r.id].update(status="served", driver_id=driver_id, dispatch=t,
                                  pickup=u, completion=f, wait=u-r.arrival, pickup_travel=p)
            selected = set(action.values())
            pending = [r for r in pending if r.id not in selected]
        # Drain all admitted requests and scheduled trips, not just [0,T).
        if cursor == len(cohort) and not pending and all(d.available_at <= t + 1e-9 for d in drivers):
            break
        epoch += 1
        if epoch > 100000:
            raise RuntimeError("Unexpected simulation length; inspect configuration.")

    rows = list(logs.values())
    for row in rows:
        if row["status"] == "expired":
            if row["observed_epochs"] == 0:
                category = "no_decision_epoch_before_expiry"
            elif row["epochs_feasible"]:
                category = "feasible_at_least_once_but_unserved"
            elif row["epochs_idle_but_unreachable"] == row["observed_epochs"]:
                category = "idle_but_unreachable_at_every_epoch"
            elif row["epochs_no_idle"] == row["observed_epochs"]:
                category = "no_idle_at_every_epoch"
            else:
                category = "mixed_no_idle_and_unreachable"
            row["expiry_observation"] = category
    served = [r for r in rows if r["status"] == "served"]
    waits = [r["wait"] for r in served]
    total_vehicle_time = len(drivers) * config.duration
    idle_time = total_vehicle_time - pickup_time - loaded_time
    assert idle_time >= -1e-7
    assert all(r["status"] in ("served", "expired") for r in rows)
    metrics = dict(policy=policy, requests=len(rows), served=len(served), expired=len(rows)-len(served),
                   service_rate=len(served)/len(rows) if rows else None,
                   mean_wait_min=statistics.mean(waits) if waits else None,
                   p90_wait_min=percentile(waits, .9),
                   mean_pickup_min=statistics.mean(r["pickup_travel"] for r in served) if served else None,
                   idle_share=idle_time/total_vehicle_time,
                   pickup_share=pickup_time/total_vehicle_time,
                   passenger_share=loaded_time/total_vehicle_time,
                   mean_decision_ms=statistics.mean(decision_ms) if decision_ms else 0,
                   max_decision_ms=max(decision_ms, default=0), drain_end_min=t)
    metrics.update(
        decision_epochs=len(decision_records),
        epochs_with_evaluated_alternatives=sum(r["evaluated_candidates"] > 0 for r in decision_records),
        epochs_with_future_count_difference=sum(r["candidates_with_different_future_count"] > 0 for r in decision_records),
        epochs_with_higher_future_count=sum(r["candidates_with_higher_future_count"] > 0 for r in decision_records),
        changed_from_same_state_baseline=sum(r["changed_from_same_state_baseline"] for r in decision_records),
        evaluated_candidates=sum(r["evaluated_candidates"] for r in decision_records))
    for category in ("no_decision_epoch_before_expiry", "feasible_at_least_once_but_unserved",
                     "idle_but_unreachable_at_every_epoch", "no_idle_at_every_epoch", "mixed_no_idle_and_unreachable"):
        metrics["expired_"+category] = sum(r["expiry_observation"] == category for r in rows)
    return metrics, rows


class ModelTests(unittest.TestCase):
    def test_solver_against_exhaustive_search(self):
        rng = random.Random(42)
        for _ in range(60):
            costs = [[None if rng.random() < .3 else rng.randint(1, 9) for _ in range(4)] for _ in range(3)]
            best = (0, 0)

            def enumerate_assignments(i, used, count, total):
                nonlocal best
                if i == len(costs):
                    best = min(best, (-count, total))
                    return
                enumerate_assignments(i+1, used, count, total)
                for j, cost in enumerate(costs[i]):
                    if cost is not None and j not in used:
                        enumerate_assignments(i+1, used | {j}, count+1, total+cost)

            enumerate_assignments(0, set(), 0, 0)
            actual = min_cost_maximum_matching(costs)
            self.assertEqual((-len(actual), sum(costs[i][j] for i, j in actual.items())), best)

    def test_scenario_counts_both_retained_and_released_capacity(self):
        class ExampleTravel:
            def minutes(self, a, b, t):
                return {(0, 2): 2, (1, 2): 4, (2, 3): 20,
                        (0, 0): 2, (1, 0): 12, (3, 0): 12,
                        (3, 3): 2, (1, 3): 12, (0, 3): 12}.get((a, b), 20)
        cfg = Config()
        drivers = [Driver(0, 0), Driver(1, 1)]
        current = {0: Request(0, 0, 2, 3)}
        scenarios = [[Request(10, 5, 0, 1), Request(11, 25, 3, 1)]]
        self.assertEqual(future_capacity(drivers, current, {0: 0}, 0, scenarios, ExampleTravel(), cfg), 1)
        self.assertEqual(future_capacity(drivers, current, {1: 0}, 0, scenarios, ExampleTravel(), cfg), 2)
        self.assertEqual(choose_action("baseline", drivers, list(current.values()), 0,
                                       scenarios, ExampleTravel(), cfg), {0: 0})
        self.assertEqual(choose_action("lookahead", drivers, list(current.values()), 0,
                                       scenarios, ExampleTravel(), cfg), {1: 0})
        # If the future residential request is earlier, both releases miss it.
        early = [[Request(11, 10, 3, 1)]]
        self.assertEqual(future_capacity(drivers, current, {1: 0}, 0, early, ExampleTravel(), cfg), 0)

    def test_no_duplicate_future_demand(self):
        cfg = Config()
        scenario = [[Request(1, 2, 0, 1)]]
        self.assertEqual(future_capacity([Driver(0, 0), Driver(1, 0)], {}, {}, 0, scenario, Travel(), cfg), 1)

    def test_no_anticipatory_departure(self):
        class LongPickup:
            def minutes(self, a, b, t):
                return 9
        result = future_capacity([Driver(0, 0)], {}, {}, 0,
                                 [[Request(1, 10, 1, 2)]], LongPickup(), Config(max_wait=8))
        self.assertEqual(result, 0)

    def test_completion_and_fixed_window_accounting(self):
        cfg = Config(duration=10, fleet=1)
        metric, rows = simulate("baseline", [Request(0, 0, 0, 2)], [0], HistoricalScenarios([[]], cfg), cfg)
        self.assertEqual(rows[0]["completion"], 10)
        self.assertAlmostEqual(metric["pickup_share"], .2)
        self.assertAlmostEqual(metric["passenger_share"], .8)
        # Completion after T is still counted as service.
        metric, rows = simulate("baseline", [Request(0, 9, 0, 2)], [0], HistoricalScenarios([[]], cfg), cfg)
        self.assertEqual(metric["served"], 1)
        self.assertGreater(rows[0]["completion"], cfg.duration)
        self.assertAlmostEqual(metric["idle_share"], .9)

    def test_expiry_before_assignment_and_pickup_at_deadline(self):
        cfg = Config(duration=10, fleet=1, delta=2, max_wait=1)
        metric, rows = simulate("baseline", [Request(0, 1, 0, 0)], [0], HistoricalScenarios([[]], cfg), cfg)
        self.assertEqual(metric["expired"], 1)
        cfg = Config(duration=10, fleet=1, max_wait=2)
        metric, rows = simulate("baseline", [Request(0, 0, 0, 0)], [0], HistoricalScenarios([[]], cfg), cfg)
        self.assertEqual(metric["served"], 1)
        self.assertEqual(rows[0]["wait"], 2)

    def test_zero_weight_is_identical_to_baseline(self):
        cfg = Config(duration=30, weight=0, history_days=3)
        history = HistoricalScenarios([synthetic_day(i, 30) for i in range(3)], cfg)
        stream = synthetic_day(999, 30)
        _, a = simulate("baseline", stream, [0, 1, 2], history, cfg)
        _, b = simulate("lookahead", stream, [0, 1, 2], history, cfg)
        self.assertEqual(a, b)

    def test_busy_vehicle_not_reassigned(self):
        cfg = Config(duration=20, fleet=1, max_wait=3)
        stream = [Request(0, 0, 0, 5), Request(1, 1, 0, 1)]
        metric, rows = simulate("baseline", stream, [0], HistoricalScenarios([[]], cfg), cfg)
        self.assertEqual([r["status"] for r in rows], ["served", "expired"])


def write_csv(path: Path, rows: list[dict]):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--minutes", type=int, default=180)
    parser.add_argument("--fleet", type=int, default=8)
    parser.add_argument("--weight", type=float, default=3.0)
    parser.add_argument("--horizon", type=float, default=30.0)
    parser.add_argument("--scenarios", type=int, default=3)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("results_v2"))
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ModelTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if min(args.minutes, args.fleet, args.days, args.horizon) <= 0 or args.weight < 0 or not 1 <= args.scenarios <= 10:
        parser.error("Use positive sizes/horizon, weight >= 0, and 1 <= scenarios <= 10.")
    cfg = Config(duration=args.minutes, fleet=args.fleet, weight=args.weight,
                 horizon=args.horizon, scenarios=args.scenarios)
    args.out.mkdir(parents=True, exist_ok=True)
    # Independent synthetic historical days; never generated from test records.
    historical = [synthetic_day(1000+i, cfg.duration) for i in range(cfg.history_days)]
    history = HistoricalScenarios(historical, cfg)
    summary = []
    print("SYNTHETIC DEMO ONLY -- not empirical evidence about Uber.")
    print("Parameters are illustrative, not validation-selected.")
    for day in range(args.days):
        seed = args.seed + day
        stream = synthetic_day(seed, cfg.duration)
        rng = random.Random(50000 + seed)
        initial = [rng.randrange(6) for _ in range(cfg.fleet)]
        write_csv(args.out / f"requests_day{day+1}.csv", [asdict(r) for r in stream])
        for policy in ("baseline", "lookahead"):
            diagnostics = []
            metric, rows = simulate(policy, stream, initial, history, cfg, diagnostics=diagnostics)
            write_csv(args.out / f"decisions_day{day+1}_{policy}.csv", diagnostics)
            metric = dict(day=day+1, seed=seed, **metric)
            summary.append(metric)
            write_csv(args.out / f"trips_day{day+1}_{policy}.csv", rows)
            wait = metric["mean_wait_min"]
            wait_text = f"{wait:.2f}" if wait is not None else "NA"
            rate = metric["service_rate"]
            rate_text = f"{rate:.1%}" if rate is not None else "NA"
            print(f"Day {day+1} | {policy:9s} | served {metric['served']}/{metric['requests']} "
                  f"({rate_text}) | mean wait {wait_text} min | idle {metric['idle_share']:.1%} "
                  f"| decision {metric['mean_decision_ms']:.2f} ms")
            if policy == "lookahead":
                print(f"  Diagnostics: alternative epochs={metric['epochs_with_evaluated_alternatives']}, "
                      f"different future-count epochs={metric['epochs_with_future_count_difference']}, "
                      f"changed decisions={metric['changed_from_same_state_baseline']}")
                print(f"  Expired with idle cars but no reachable driver at every observed epoch: "
                      f"{metric['expired_idle_but_unreachable_at_every_epoch']}/{metric['expired']}")
    write_csv(args.out / "summary.csv", summary)
    (args.out / "config.json").write_text(json.dumps({"config": asdict(cfg), "test_seed": args.seed,
        "test_days": args.days, "history_seeds": list(range(1000, 1010)),
        "data_type": "synthetic", "python_dependencies": "standard library only"}, indent=2), encoding="utf-8")
    print(f"\nSend back {args.out / 'summary.csv'} and {args.out / 'config.json'}.")


if __name__ == "__main__":
    main()
