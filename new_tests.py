"""
new_tests.py — compare ex2_331050591, ex2_ophir_v1, ex2_ophir_v2
on 20 novel problems (different layouts, probs, and start positions
from the baseline suite) × 20 seeds each.

Each (solver, seed) run is capped at TIMEOUT_PER_SEED seconds via a
subprocess pool — if it hangs the run scores 0 for that seed.
"""

import time
import concurrent.futures
import ext_elev

GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

N_SEEDS = 20
TIMEOUT_PER_SEED = 30   # seconds hard limit per (solver, seed) call

# module names (strings) so they can be passed to worker processes
SOLVERS = [
    ("mine",     "ex2_331050591"),
    ("ophir_v1", "ex2_ophir_v1"),
    ("ophir_v2", "ex2_ophir_v2"),
]

# ------------------------------------------------------------------ #
# 20 novel problems                                                    #
# ------------------------------------------------------------------ #
# Format per problem:
#   height, Elevators {id:(start, reachable_tuple, capacity)},
#   Persons {id:(start, weight, goal)},
#   elevator_chosen_action_prob {id: float},
#   person_chosen_action_prob   {id: float},
#   persons_reward {id: [milestones]},
#   goal_reward, horizon
# ------------------------------------------------------------------ #

PROBLEMS = [
    # n01 — 4-floor, 1 elevator, 2 persons, moderate probs
    ("n01", {
        "height": 4,
        "Elevators": {0: (1, (0, 1, 2, 3, 4), 12)},
        "Persons":   {10: (0, 3, 4), 11: (4, 3, 1)},
        "elevator_chosen_action_prob": {0: 0.90},
        "person_chosen_action_prob":   {10: 0.88, 11: 0.85},
        "persons_reward": {10: [4, 8], 11: [3, 7]},
        "goal_reward": 18,
        "horizon": 50,
    }),

    # n02 — 7-floor, 2 elevators partial coverage, 3 persons
    ("n02", {
        "height": 7,
        "Elevators": {0: (1, (0, 1, 2, 3, 4), 12), 1: (6, (3, 4, 5, 6, 7), 12)},
        "Persons":   {10: (0, 4, 7), 11: (7, 4, 0), 12: (2, 4, 6)},
        "elevator_chosen_action_prob": {0: 0.88, 1: 0.85},
        "person_chosen_action_prob":   {10: 0.88, 11: 0.85, 12: 0.90},
        "persons_reward": {10: [3, 6, 9], 11: [3, 6, 9], 12: [2, 5, 8]},
        "goal_reward": 25,
        "horizon": 58,
    }),

    # n03 — 9-floor, 1 full-access elevator, 3 persons spread out
    ("n03", {
        "height": 9,
        "Elevators": {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 18)},
        "Persons":   {10: (1, 5, 8), 11: (9, 5, 2), 12: (4, 5, 7)},
        "elevator_chosen_action_prob": {0: 0.95},
        "person_chosen_action_prob":   {10: 0.90, 11: 0.85, 12: 0.88},
        "persons_reward": {10: [4, 7, 10], 11: [4, 7, 10], 12: [3, 6, 9]},
        "goal_reward": 28,
        "horizon": 62,
    }),

    # n04 — 5-floor, 1 unreliable elevator (0.72), 2 persons
    ("n04", {
        "height": 5,
        "Elevators": {0: (2, (0, 1, 2, 3, 4, 5), 10)},
        "Persons":   {10: (0, 3, 5), 11: (5, 3, 1)},
        "elevator_chosen_action_prob": {0: 0.72},
        "person_chosen_action_prob":   {10: 0.90, 11: 0.88},
        "persons_reward": {10: [5, 10], 11: [4, 8]},
        "goal_reward": 20,
        "horizon": 68,
    }),

    # n05 — 6-floor, reliable + broken elevators (0.95 vs 0.32), 3 persons
    ("n05", {
        "height": 6,
        "Elevators": {0: (0, (0, 1, 2, 3, 4, 5, 6), 10),
                      1: (6, (0, 2, 4, 6), 10)},
        "Persons":   {10: (0, 4, 6), 11: (6, 4, 0), 12: (2, 4, 5)},
        "elevator_chosen_action_prob": {0: 0.95, 1: 0.32},
        "person_chosen_action_prob":   {10: 0.90, 11: 0.88, 12: 0.85},
        "persons_reward": {10: [3, 6, 9], 11: [3, 6, 9], 12: [2, 5]},
        "goal_reward": 22,
        "horizon": 60,
    }),

    # n06 — 6-floor, 3 elevators, 4 persons, varied probs
    ("n06", {
        "height": 6,
        "Elevators": {0: (0, (0, 1, 2, 3), 10),
                      1: (6, (3, 4, 5, 6), 10),
                      2: (3, (2, 3, 4), 10)},
        "Persons":   {10: (0, 3, 6), 11: (6, 3, 0),
                      12: (1, 3, 5), 13: (5, 3, 2)},
        "elevator_chosen_action_prob": {0: 0.95, 1: 0.88, 2: 0.82},
        "person_chosen_action_prob":   {10: 0.90, 11: 0.85, 12: 0.88, 13: 0.85},
        "persons_reward": {10: [3, 6, 9], 11: [3, 6, 9],
                           12: [2, 5, 8], 13: [2, 5, 8]},
        "goal_reward": 32,
        "horizon": 65,
    }),

    # n07 — reset-loop trap: 1 cheap+close person, 2 expensive far ones
    ("n07", {
        "height": 3,
        "Elevators": {0: (0, (0, 1, 2, 3), 8)},
        "Persons":   {10: (0, 3, 1),   # cheap & nearby
                      11: (0, 3, 3),   # far, low reward
                      12: (3, 3, 0)},  # far, low reward
        "elevator_chosen_action_prob": {0: 0.95},
        "person_chosen_action_prob":   {10: 0.95, 11: 0.88, 12: 0.85},
        "persons_reward": {10: [45, 45], 11: [1], 12: [1]},
        "goal_reward": 8,
        "horizon": 55,
    }),

    # n08 — 8-floor, 2 elevators meeting at floor 4 (relay), 3 persons
    ("n08", {
        "height": 8,
        "Elevators": {0: (0, (0, 1, 2, 3, 4), 12),
                      1: (8, (4, 5, 6, 7, 8), 12)},
        "Persons":   {10: (0, 4, 8), 11: (8, 4, 0), 12: (2, 4, 6)},
        "elevator_chosen_action_prob": {0: 0.90, 1: 0.85},
        "person_chosen_action_prob":   {10: 0.90, 11: 0.85, 12: 0.88},
        "persons_reward": {10: [4, 8, 12], 11: [4, 8, 12], 12: [3, 6]},
        "goal_reward": 30,
        "horizon": 65,
    }),

    # n09 — 5-floor, 2 full-access elevators, 5 persons (busy)
    ("n09", {
        "height": 5,
        "Elevators": {0: (0, (0, 1, 2, 3, 4, 5), 10),
                      1: (5, (0, 1, 2, 3, 4, 5), 10)},
        "Persons":   {10: (0, 2, 5), 11: (5, 2, 0), 12: (2, 2, 4),
                      13: (4, 2, 1), 14: (1, 2, 3)},
        "elevator_chosen_action_prob": {0: 0.90, 1: 0.88},
        "person_chosen_action_prob":   {10: 0.88, 11: 0.85, 12: 0.90,
                                        13: 0.85, 14: 0.88},
        "persons_reward": {10: [3, 7], 11: [3, 7], 12: [2, 5],
                           13: [2, 5], 14: [2, 4]},
        "goal_reward": 35,
        "horizon": 58,
    }),

    # n10 — 7-floor, asymmetric elevator floors, 3 persons
    ("n10", {
        "height": 7,
        "Elevators": {0: (0, (0, 1, 2, 3), 10),
                      1: (7, (3, 4, 5, 6, 7), 10)},
        "Persons":   {10: (0, 3, 7), 11: (7, 3, 0), 12: (2, 3, 5)},
        "elevator_chosen_action_prob": {0: 0.87, 1: 0.82},
        "person_chosen_action_prob":   {10: 0.88, 11: 0.85, 12: 0.83},
        "persons_reward": {10: [4, 8, 12], 11: [4, 8, 12], 12: [3, 6]},
        "goal_reward": 28,
        "horizon": 60,
    }),

    # n11 — 9-floor, 3 chained elevators, 4 persons (long relay)
    ("n11", {
        "height": 9,
        "Elevators": {0: (0, (0, 1, 2, 3), 12),
                      1: (5, (3, 4, 5, 6), 12),
                      2: (9, (6, 7, 8, 9), 12)},
        "Persons":   {10: (0, 4, 9), 11: (9, 4, 0),
                      12: (2, 4, 7), 13: (7, 4, 2)},
        "elevator_chosen_action_prob": {0: 0.95, 1: 0.82, 2: 0.68},
        "person_chosen_action_prob":   {10: 0.90, 11: 0.88, 12: 0.85, 13: 0.85},
        "persons_reward": {10: [4, 8, 12, 16], 11: [4, 8, 12, 16],
                           12: [3, 6, 10], 13: [3, 6, 10]},
        "goal_reward": 40,
        "horizon": 70,
    }),

    # n12 — 5-floor, 1 large-capacity elevator, 2 heavy persons
    ("n12", {
        "height": 5,
        "Elevators": {0: (0, (0, 1, 2, 3, 4, 5), 20)},
        "Persons":   {10: (0, 8, 5), 11: (5, 8, 0)},
        "elevator_chosen_action_prob": {0: 0.88},
        "person_chosen_action_prob":   {10: 0.85, 11: 0.85},
        "persons_reward": {10: [6, 12], 11: [6, 12]},
        "goal_reward": 22,
        "horizon": 52,
    }),

    # n13 — 5-floor, 3 elevators, 5 persons, varied reliability
    ("n13", {
        "height": 5,
        "Elevators": {0: (0, (0, 1, 2, 3), 8),
                      1: (5, (2, 3, 4, 5), 8),
                      2: (2, (1, 2, 3), 8)},
        "Persons":   {10: (0, 3, 5), 11: (5, 3, 0), 12: (1, 3, 4),
                      13: (4, 3, 1), 14: (2, 3, 3)},
        "elevator_chosen_action_prob": {0: 0.93, 1: 0.85, 2: 0.75},
        "person_chosen_action_prob":   {10: 0.90, 11: 0.88, 12: 0.85,
                                        13: 0.82, 14: 0.80},
        "persons_reward": {10: [3, 6, 9], 11: [3, 6, 9], 12: [2, 5],
                           13: [2, 5], 14: [2, 4]},
        "goal_reward": 30,
        "horizon": 65,
    }),

    # n14 — 3-floor dense: 2 elevators, 4 persons
    ("n14", {
        "height": 3,
        "Elevators": {0: (0, (0, 1, 2, 3), 10),
                      1: (3, (0, 1, 2, 3), 10)},
        "Persons":   {10: (0, 2, 3), 11: (3, 2, 0),
                      12: (0, 2, 2), 13: (3, 2, 1)},
        "elevator_chosen_action_prob": {0: 0.92, 1: 0.88},
        "person_chosen_action_prob":   {10: 0.90, 11: 0.88, 12: 0.85, 13: 0.82},
        "persons_reward": {10: [3, 7], 11: [3, 7], 12: [2, 5], 13: [2, 5]},
        "goal_reward": 25,
        "horizon": 48,
    }),

    # n15 — 8-floor, 1 elevator starting mid, 2 persons at extremes
    ("n15", {
        "height": 8,
        "Elevators": {0: (4, (0, 1, 2, 3, 4, 5, 6, 7, 8), 15)},
        "Persons":   {10: (0, 5, 8), 11: (8, 5, 1)},
        "elevator_chosen_action_prob": {0: 0.87},
        "person_chosen_action_prob":   {10: 0.90, 11: 0.85},
        "persons_reward": {10: [5, 10], 11: [5, 10]},
        "goal_reward": 20,
        "horizon": 56,
    }),

    # n16 — 6-floor, capacity mismatch (heavy persons need big elevator)
    ("n16", {
        "height": 6,
        "Elevators": {0: (0, (0, 1, 2, 3, 4, 5, 6), 5),   # small
                      1: (6, (0, 1, 2, 3, 4, 5, 6), 15)},  # large
        "Persons":   {10: (0, 4, 6), 11: (6, 4, 0), 12: (3, 2, 5)},
        "elevator_chosen_action_prob": {0: 0.90, 1: 0.85},
        "person_chosen_action_prob":   {10: 0.90, 11: 0.85, 12: 0.90},
        "persons_reward": {10: [4, 8, 12], 11: [4, 8, 12], 12: [2, 5]},
        "goal_reward": 28,
        "horizon": 56,
    }),

    # n17 — 5-floor, 2 full elevators, 3 persons with high per-person reward
    ("n17", {
        "height": 5,
        "Elevators": {0: (0, (0, 1, 2, 3, 4, 5), 10),
                      1: (5, (0, 1, 2, 3, 4, 5), 10)},
        "Persons":   {10: (0, 3, 5), 11: (5, 3, 0), 12: (2, 3, 4)},
        "elevator_chosen_action_prob": {0: 0.92, 1: 0.88},
        "person_chosen_action_prob":   {10: 0.88, 11: 0.85, 12: 0.90},
        "persons_reward": {10: [6, 10, 15, 20], 11: [6, 10, 15, 20],
                           12: [5, 9, 13]},
        "goal_reward": 50,
        "horizon": 55,
    }),

    # n18 — 4-floor, 4 elevators with partial overlapping coverage, 2 persons
    ("n18", {
        "height": 4,
        "Elevators": {0: (0, (0, 1, 2), 8),
                      1: (4, (2, 3, 4), 8),
                      2: (2, (0, 2, 4), 8),
                      3: (1, (1, 3, 4), 8)},
        "Persons":   {10: (0, 3, 4), 11: (4, 3, 0)},
        "elevator_chosen_action_prob": {0: 0.95, 1: 0.90, 2: 0.82, 3: 0.78},
        "person_chosen_action_prob":   {10: 0.88, 11: 0.85},
        "persons_reward": {10: [5, 10], 11: [5, 10]},
        "goal_reward": 18,
        "horizon": 52,
    }),

    # n19 — 7-floor pure relay: elevators can't reach each other's zones alone
    ("n19", {
        "height": 7,
        "Elevators": {0: (0, (0, 1, 2, 3), 10),
                      1: (7, (3, 4, 5, 6, 7), 10)},
        "Persons":   {10: (0, 3, 7), 11: (7, 3, 0)},
        "elevator_chosen_action_prob": {0: 0.90, 1: 0.85},
        "person_chosen_action_prob":   {10: 0.90, 11: 0.85},
        "persons_reward": {10: [5, 10, 15], 11: [5, 10, 15]},
        "goal_reward": 25,
        "horizon": 55,
    }),

    # n20 — 7-floor, 2 elevators (1 reliable, 1 very broken), 4 persons
    ("n20", {
        "height": 7,
        "Elevators": {0: (0, (0, 1, 2, 3, 4, 5, 6, 7), 12),
                      1: (7, (0, 1, 2, 3, 4, 5, 6, 7), 12)},
        "Persons":   {10: (0, 3, 7), 11: (7, 3, 0),
                      12: (2, 3, 5), 13: (5, 3, 2)},
        "elevator_chosen_action_prob": {0: 0.95, 1: 0.38},
        "person_chosen_action_prob":   {10: 0.90, 11: 0.90, 12: 0.85, 13: 0.85},
        "persons_reward": {10: [4, 8, 12], 11: [4, 8, 12],
                           12: [3, 6], 13: [3, 6]},
        "goal_reward": 35,
        "horizon": 62,
    }),
]


# ------------------------------------------------------------------ #
# Worker (runs in a separate process so TIMEOUT_PER_SEED can kill it) #
# ------------------------------------------------------------------ #

def _worker(module_name, problem_dict):
    import importlib
    import ext_elev as _ext
    mod = importlib.import_module(module_name)
    api = _ext.create_elevators_game(problem_dict, debug=False)
    controller = mod.Controller(api)
    for _ in range(api.get_max_steps()):
        action = controller.choose_next_action(api.get_current_state())
        api.submit_next_action(action)
        if api.get_done():
            break
    return api.get_current_reward()


def solve_with_timeout(module_name, problem_dict, timeout=TIMEOUT_PER_SEED):
    """Run solver in a child process; return reward or 0.0 on timeout/error."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_worker, module_name, problem_dict)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"    [TIMEOUT] {module_name} seed={problem_dict.get('seed')}")
            return 0.0
        except Exception as e:
            print(f"    [ERROR]   {module_name} seed={problem_dict.get('seed')}: {e}")
            return 0.0


# ------------------------------------------------------------------ #
# Main comparison runner                                               #
# ------------------------------------------------------------------ #

def run_comparison():
    solver_names = [name for name, _ in SOLVERS]
    col_w  = 9
    name_w = 5

    header = f"{'Prob':<{name_w}}"
    for sn in solver_names:
        header += f" | {sn:>{col_w}}"
    header += f" | {'Winner':>9}"
    print(f"\n{BOLD}=== NEW PROBLEM COMPARISON "
          f"({N_SEEDS} seeds × {TIMEOUT_PER_SEED}s/seed limit) ==={RESET}\n")
    print(header)
    print("-" * len(header))

    totals = {sn: 0.0 for sn in solver_names}
    wins   = {sn: 0   for sn in solver_names}

    for prob_name, base_problem in PROBLEMS:
        avgs = {sn: 0.0 for sn in solver_names}

        for i, (sn, mod_name) in enumerate(SOLVERS):
            total = 0.0
            for seed in range(N_SEEDS):
                prob = dict(base_problem)
                prob["seed"] = seed
                total += solve_with_timeout(mod_name, prob)
            avgs[sn] = total / N_SEEDS
            totals[sn] += avgs[sn]

        best_score = max(avgs.values())
        best_solvers = [sn for sn, v in avgs.items() if abs(v - best_score) < 0.01]
        for sn in best_solvers:
            wins[sn] += 1

        row = f"{prob_name:<{name_w}}"
        for sn in solver_names:
            v = avgs[sn]
            color = GREEN if sn in best_solvers else RED
            row += f" | {color}{v:>{col_w}.2f}{RESET}"
        row += f" | {GREEN}{'/'.join(best_solvers):>9}{RESET}"
        print(row)

    print("-" * len(header))

    tot_row = f"{'TOTAL':<{name_w}}"
    best_total = max(totals.values())
    for sn in solver_names:
        v = totals[sn]
        color = GREEN if abs(v - best_total) < 0.01 else RED
        tot_row += f" | {color}{v:>{col_w}.2f}{RESET}"
    tot_row += f" | {'':>9}"
    print(tot_row)

    win_row = f"{'WINS':<{name_w}}"
    best_wins = max(wins.values())
    for sn in solver_names:
        w = wins[sn]
        color = GREEN if w == best_wins else (YELLOW if w > 0 else RED)
        win_row += f" | {color}{w:>{col_w}}{RESET}"
    win_row += f" | {'':>9}"
    print(win_row)

    print()
    print(f"{BOLD}Summary:{RESET}")
    for sn in solver_names:
        print(f"  {sn:<10}: {wins[sn]:2d}/{len(PROBLEMS)} wins, "
              f"total avg = {totals[sn]:.2f}")
    print()


if __name__ == "__main__":
    run_comparison()
