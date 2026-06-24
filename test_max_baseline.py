import os

# חילצנו את ציון המקסימום מכל אלגוריתמי הבייסליין האחרים (לא כולל sol2) כדי שישמשו כרף
TARGETS = {
    "p1_easy": 111.23, "e1_easy": 114.20, "e2_easy": 97.20, "e3_easy": 64.27,
    "e4_easy": 136.93, "e5_easy": 91.67, "m1_easy": 74.87, "m2_easy": 40.47,
    "m3_easy": 56.87, "m4_easy": 109.57, "m5_easy": 92.67, "rl_easy": 471.90,
    
    "p1_med": 106.17, "e1_med": 110.20, "e2_med": 96.70, "e3_med": 76.50,
    "e4_med": 132.00, "e5_med": 89.47, "m1_med": 102.53, "m2_med": 46.67,
    "m3_med": 68.00, "m4_med": 106.10, "m5_med": 133.77, "rl_med": 458.53,
    
    "p1_hard": 104.83, "e1_hard": 109.90, "e2_hard": 71.40, "e3_hard": 48.70,
    "e4_hard": 136.53, "e5_hard": 68.97, "m1_hard": 80.43, "m2_hard": 78.43,
    "m3_hard": 57.27, "m4_hard": 163.10, "m5_hard": 96.33, "rl_hard": 458.57
}

# צבעים להדפסה יפה במסוף
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def run_test():
    solution_file = "Solution.txt"
    
    if not os.path.exists(solution_file):
        print(f"{RED}Error: File '{solution_file}' not found.{RESET}")
        print(f"{YELLOW}Please run 'python ex2_check.py' first to generate the results.{RESET}")
        return

    results = {}
    
    # קריאת התוצאות מהקובץ
    with open(solution_file, "r", encoding="utf-8") as f:
        for line in f:
            if "reward_average=" in line:
                try:
                    parts = line.split(":")
                    problem_name = parts[0].strip()
                    reward_str = parts[1].split("|")[0].split("=")[1].strip()
                    results[problem_name] = float(reward_str)
                except Exception as e:
                    continue

    if not results:
        print(f"{RED}No results found in {solution_file}. Make sure it finished running.{RESET}")
        return

    print(f"\n{BOLD}=== MAX BASELINE TEST REPORT ==={RESET}\n")
    print(f"{'Problem':<12} | {'Your Score':<12} | {'Max Target':<12} | {'Delta':<10} | {'Result'}")
    print("-" * 65)

    passed = 0
    total = len(TARGETS)

    for prob, target in TARGETS.items():
        if prob not in results:
            print(f"{prob:<12} | {YELLOW}{'NOT RUN':<12}{RESET} | {target:<12.2f} | {'-':<10} | {YELLOW}SKIPPED{RESET}")
            continue
            
        score = results[prob]
        delta = score - target
        
        if score >= target:
            status = f"{GREEN}PASSED{RESET}"
            delta_str = f"{GREEN}+{delta:.2f}{RESET}"
            passed += 1
        else:
            status = f"{RED}FAILED{RESET}"
            delta_str = f"{RED}{delta:.2f}{RESET}"

        print(f"{prob:<12} | {score:<12.2f} | {target:<12.2f} | {delta_str:<10} | {status}")

    print("-" * 65)
    print(f"\n{BOLD}Summary:{RESET} Passed {passed}/{total} instances.")
    
    if passed == total:
        print(f"{GREEN}{BOLD}🎉 AMAZING! You beat the max baseline in ALL problems! 🎉{RESET}\n")
    elif passed >= total * 0.8:
        print(f"{GREEN}Great job! You passed the baseline in most problems. Keep tweaking!{RESET}\n")
    else:
        print(f"{YELLOW}You have some room for improvement to beat the strongest baselines.{RESET}\n")

if __name__ == "__main__":
    run_test()