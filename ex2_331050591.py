"""
I used Claude Code to: Brainstorm, implement my ideas and improve code performance.

Credits: Amit Solomon and Ophir Finkelstien.
"""


import heapq
import time
import ext_elev

id = ["331050591"]

INF = float("inf")


class Plan:
    """Delivery path for a person: ordered (elevator, pickup_floor, dropoff_floor) legs."""
    __slots__ = ("legs", "intr", "suffix", "intr_min", "suffix_min")

    def __init__(self, legs, pe, q):
        self.legs = legs
        n = len(legs)
        # intr[i] = E[steps] for enter + optional carry move + exit of leg i
        self.intr = []
        for (e, a, b) in legs:
            c = 2.0 / q
            if a != b:
                c += 1.0 / pe[e]
            self.intr.append(c)
        # suffix[i] = E[total steps] from beginning of leg i to delivery
        #   = (reposition elevator to pickup: 1/pe[e]) + intr[i] + suffix[i+1]
        self.suffix = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            self.suffix[i] = self.suffix[i + 1] + 1.0 / pe[legs[i][0]] + self.intr[i]
        # Optimistic (all-succeed, unit-cost) variants for horizon feasibility checks
        self.intr_min = [2 + (1 if a != b else 0) for (e, a, b) in legs]
        self.suffix_min = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            self.suffix_min[i] = self.suffix_min[i + 1] + 1 + self.intr_min[i]


class _JointAstar:
    """Min-expected-cost A* planner for joint delivery of all target persons.
    Edge costs: 1/q per ENTER or EXIT; 1/pe per MOVE for reliable elevators
    (pe >= 0.5), 1/pe^2 for broken ones (pe < 0.5, hard-tier).
    Heuristic is admissible: per-person non-shareable enter/exit costs plus
    at most one repositioning move per elevator (the bottleneck).
    Only 'interesting' floors are MOVE targets: pickup floors, goal floors of
    current passengers, and shared transfer floors between elevators.
    """

    _DEL = -1  # sentinel: person already delivered

    def __init__(self, ctrl, plan_eids=None):
        c = ctrl
        eids = plan_eids if plan_eids is not None else tuple(c.elev_ids)
        self._eids   = eids
        self._pids   = c.person_ids
        self._eidx   = {e: i for i, e in enumerate(eids)}
        self._pidx   = {p: j for j, p in enumerate(c.person_ids)}
        self._goal   = tuple(c.goal_floor[p] for p in c.person_ids)
        self._wt     = tuple(c.weight[p]     for p in c.person_ids)
        self._cap    = tuple(c.cap[e]        for e in eids)
        self._reach  = tuple(c.reachable[e]  for e in eids)
        self._qp     = tuple(c.qp[p]         for p in c.person_ids)
        pe_raw       = tuple(c.pe[e]          for e in eids)
        self._pe_cost = tuple(1.0 / (p ** 1.3) for p in pe_raw)
        all_floors = set(c.init_efloor.values())
        for r in self._reach:
            all_floors |= r
        self._OFF = max(all_floors) + 1 if all_floors else 1
        # Shared (transfer) floors: reachable by >= 2 plan elevators
        self._shared = frozenset(
            f for f in all_floors
            if sum(1 for r in self._reach if f in r) >= 2
        )
        self._cache: dict = {}
        self._max_expand = 60000

    def get_action(self, efl, ploc, target):
        """Return the next action string, or None if planning fails/done."""
        ef  = tuple(efl[e] for e in self._eids)
        pl  = self._encode(ploc, target)
        if pl is None or all(x == self._DEL for x in pl):
            return None
        key = (ef, pl)
        act = self._cache.get(key)
        if act is not None:
            return act
        path = self._search(ef, pl)
        if not path:
            return None
        for st, a in path:
            self._cache.setdefault(st, a)
        return path[0][1]

    def _encode(self, ploc, target):
        OFF, DEL = self._OFF, self._DEL
        pl = [DEL] * len(self._pids)
        for p in target:
            loc = ploc.get(p)
            if loc is None:
                continue
            j = self._pidx[p]
            if loc[0] == 'floor':
                pl[j] = loc[1]
            else:
                e = loc[1]
                if e not in self._eidx:
                    return None  # person in non-plan elevator — can't encode
                pl[j] = OFF + self._eidx[e]
        return tuple(pl)

    def _h(self, ef, pl):
        """Admissible heuristic: non-shareable per-person costs + ≤1 move/elevator."""
        OFF, DEL = self._OFF, self._DEL
        total = 0.0
        elev_move = [False] * len(self._eids)
        for j, loc in enumerate(pl):
            if loc == DEL:
                continue
            if loc >= OFF:                        # in elevator i
                i = loc - OFF
                total += 1.0 / self._qp[j]       # exit (non-shareable)
                if ef[i] != self._goal[j]:
                    elev_move[i] = True
            else:                                 # waiting on floor f
                total += 2.0 / self._qp[j]       # enter + exit
                f = loc
                for i, r in enumerate(self._reach):
                    if f in r and self._wt[j] <= self._cap[i]:
                        if ef[i] != f:
                            elev_move[i] = True
                        break
        for i, need in enumerate(elev_move):
            if need:
                total += self._pe_cost[i]
        return total

    def _successors(self, ef, pl):
        """Yield (action_str, nef, npl, cost). Interesting-floors pruning applied."""
        OFF, DEL = self._OFF, self._DEL
        NE = len(self._eids)

        loads = [0] * NE
        for j, loc in enumerate(pl):
            if loc != DEL and loc >= OFF:
                loads[loc - OFF] += self._wt[j]

        # Mandatory: exit at goal — only action when any passenger is at their goal
        for j, loc in enumerate(pl):
            if loc == DEL or loc < OFF:
                continue
            i = loc - OFF
            if ef[i] == self._goal[j]:
                npl = pl[:j] + (self._DEL,) + pl[j + 1:]
                yield (f'EXIT{{{self._pids[j]},{self._eids[i]}}}',
                       ef, npl, 1.0 / self._qp[j])
                return

        # Exit at shared/transfer floor — only when current elevator cannot reach goal
        for j, loc in enumerate(pl):
            if loc == DEL or loc < OFF:
                continue
            i = loc - OFF
            if self._goal[j] in self._reach[i]:
                continue  # no relay needed: current elevator can deliver directly
            f = ef[i]
            if f not in self._shared:
                continue
            if any(k != i and f in self._reach[k] and self._wt[j] <= self._cap[k]
                   for k in range(NE)):
                npl = pl[:j] + (f,) + pl[j + 1:]
                yield (f'EXIT{{{self._pids[j]},{self._eids[i]}}}',
                       ef, npl, 1.0 / self._qp[j])

        # Enter: persons waiting at elevator's current floor
        for j, loc in enumerate(pl):
            if loc == DEL or loc >= OFF:
                continue
            f = loc
            for i in range(NE):
                if (ef[i] == f and f in self._reach[i]
                        and self._wt[j] + loads[i] <= self._cap[i]):
                    npl = pl[:j] + (OFF + i,) + pl[j + 1:]
                    yield (f'ENTER{{{self._pids[j]},{self._eids[i]}}}',
                           ef, npl, 1.0 / self._qp[j])

        # Move to interesting floors only (idea 4)
        move_targets: dict = {}
        for j, loc in enumerate(pl):
            if loc == DEL:
                continue
            if loc >= OFF:                         # passenger → goal or transfer floor
                i = loc - OFF
                g = self._goal[j]
                if g in self._reach[i] and g != ef[i]:
                    move_targets.setdefault(i, set()).add(g)
                elif g not in self._reach[i]:      # true relay: add shared transfer floors
                    for f in self._shared:
                        if f in self._reach[i] and f != ef[i]:
                            move_targets.setdefault(i, set()).add(f)
            else:                                  # waiting → elevator comes here
                f = loc
                for i, r in enumerate(self._reach):
                    if f in r and self._wt[j] <= self._cap[i] and ef[i] != f:
                        move_targets.setdefault(i, set()).add(f)

        for i, floors in move_targets.items():
            for f in floors:
                nef = ef[:i] + (f,) + ef[i + 1:]
                yield (f'MOVE{{{self._eids[i]},{f}}}',
                       nef, pl, self._pe_cost[i])

    def _search(self, ef0, pl0):
        """A* search. Returns [(state, action_str), ...] from start to goal."""
        DEL   = self._DEL
        start = (ef0, pl0)
        costs = {start: 0.0}
        parent: dict = {start: None}
        cnt, expanded = 0, 0
        heap = [(self._h(ef0, pl0), 0.0, cnt, start)]

        while heap:
            _, g, _, state = heapq.heappop(heap)
            if g > costs.get(state, INF) + 1e-9:
                continue
            ef, pl = state
            if all(x == DEL for x in pl):
                path = []
                cur  = state
                while parent[cur] is not None:
                    prev, act = parent[cur]
                    path.append((prev, act))
                    cur = prev
                path.reverse()
                return path
            expanded += 1
            if expanded > self._max_expand:
                return None
            for act, nef, npl, cost in self._successors(ef, pl):
                ns = (nef, npl)
                ng = g + cost
                if ng >= costs.get(ns, INF) - 1e-9:
                    continue
                costs[ns] = ng
                parent[ns] = (state, act)
                cnt += 1
                heapq.heappush(heap, (ng + self._h(nef, npl), ng, cnt, ns))
        return None


class Controller:
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game

        init_state = game.get_initial_state()
        elevators_t, persons_t, _ = init_state
        self.initial_state = init_state
        self.max_steps = game.get_max_steps()
        self.goal_reward = float(game.get_goal_reward())

        reachable = game.get_reachable()
        capacities = game.get_capacities()

        self.elev_ids = [eid for (eid, _, _) in elevators_t]
        self.reachable = {e: frozenset(reachable[e]) for e in self.elev_ids}
        self.cap = {e: capacities[e] for e in self.elev_ids}
        self.pe = {e: float(game.get_elevator_action_prob(e)) for e in self.elev_ids}
        self.init_efloor = {eid: fl for (eid, fl, _) in elevators_t}

        self.person_ids = [pid for (pid, _) in persons_t]
        self.start = {pid: loc[1] for (pid, loc) in persons_t}
        self.goal_floor = {p: game.get_person_goal(p) for p in self.person_ids}
        self.weight = {p: game.get_person_weight(p) for p in self.person_ids}
        self.qp = {p: float(game.get_person_action_prob(p)) for p in self.person_ids}
        self.Erew = {
            p: sum(game.get_person_reward(p)) / len(game.get_person_reward(p))
            for p in self.person_ids
        }
        self.all_persons = frozenset(self.person_ids)

        # Build delivery plans per person (direct + optimal relay via Dijkstra)
        self.plans = {}
        self.deliverable = set()
        for p in self.person_ids:
            plist = self._build_plans(p)
            self.plans[p] = plist
            if plist:
                self.deliverable.add(p)

        # Optimistic min-steps from initial state (for RESET viability check)
        self.min_steps_init = {
            p: (self._person_min_steps(p, ('floor', self.start[p]), self.init_efloor)
                if self.plans[p] else INF)
            for p in self.person_ids
        }

        # Choose the best cycle to farm (exhaustive subset search)
        self.target, self.allpersons_flag, self.rho = self._choose_cycle()

        # Anchor elevator: single most-reliable that can serve all target persons directly
        anchor_cands = [
            e for e in self.elev_ids
            if all(
                self.start[p] in self.reachable[e]
                and self.goal_floor[p] in self.reachable[e]
                and self.weight[p] <= self.cap[e]
                for p in self.target
            )
        ]
        if anchor_cands:
            anchor = max(anchor_cands, key=lambda e: self.pe[e])
            self._plan_eids = (anchor,)
        else:
            self._plan_eids = tuple(self.elev_ids)

        self._cache = {}
        self._t0 = time.perf_counter()
        # ~1s/step safety margin; actual grader budget ≈ 20 + 0.5*horizon
        self._time_budget = 16.0 + 0.4 * self.max_steps
        # Relay-bottleneck problems (multi-leg plans) are misled by depth > 3;
        # direct-delivery problems with many targets benefit from depth 4.
        n_t = len(self.target)
        any_relay = any(
            min((len(pl.legs) for pl in self.plans[p]), default=1) > 1
            for p in self.target
        )
        # Check if every pair of relay-target persons can ride together (any elevator)
        relay_persons = [p for p in self.target if
                         min((len(pl.legs) for pl in self.plans[p]), default=1) > 1]
        all_batchable = all(
            any(self.weight[pa] + self.weight[pb] <= self.cap[e]
                for e in self.elev_ids)
            for i, pa in enumerate(relay_persons)
            for pb in relay_persons[i+1:]
        ) if len(relay_persons) >= 2 else True
        if n_t >= 5 and any_relay:
            self._max_depth = 3   # many persons + relay: routing ambiguity makes depth 4 misleading
        elif n_t >= 4:
            self._max_depth = 4
        else:
            self._max_depth = 8
        # A* planner: engage for 2-elevator relay problems.
        # Disable only when persons can't batch AND a broken elevator forces sequential
        # solo trips (m5_hard): expectimax's adaptive lookahead handles that better.
        # 3-elevator problems (m3) stay with expectimax: A* hits 60K expansion limit
        # every step (state space too large), wasting time.
        any_broken = any(self.pe[e] < 0.5 for e in self.elev_ids)
        self._use_astar = (
            self.allpersons_flag and len(self.target) >= 4
            and any_relay
            and len(self._plan_eids) <= 2
            and (all_batchable or not any_broken)
        )
        if self._use_astar:
            self._astar_planner = _JointAstar(self, plan_eids=self._plan_eids)

    # ------------------------------------------------------------------ #
    # Plan construction                                                    #
    # ------------------------------------------------------------------ #

    def _build_plans(self, p):
        s, g, w = self.start[p], self.goal_floor[p], self.weight[p]
        seen = set()
        plans = []
        for e in self.elev_ids:
            # s accessible if in reachable set OR elevator starts there (one-time initial floor)
            s_ok = s in self.reachable[e] or s == self.init_efloor[e]
            if s_ok and g in self.reachable[e] and w <= self.cap[e]:
                legs = ((e, s, g),)
                if legs not in seen:
                    seen.add(legs)
                    plans.append(Plan(list(legs), self.pe, self.qp[p]))
        relay = self._dijkstra_plan(p)
        if relay is not None:
            key = tuple(relay)
            if key not in seen:
                seen.add(key)
                plans.append(Plan(relay, self.pe, self.qp[p]))
        return plans

    def _dijkstra_plan(self, p):
        """Min expected-cost path; handles multi-elevator relays via shared floors."""
        s, g, w, q = self.start[p], self.goal_floor[p], self.weight[p], self.qp[p]
        start_node = ('F', s)
        dist = {start_node: 0.0}
        prev = {start_node: None}
        cnt = 0
        heap = [(0.0, cnt, start_node)]
        while heap:
            d, _, node = heapq.heappop(heap)
            if d > dist.get(node, INF) + 1e-9:
                continue
            if node == 'DONE':
                break
            for nbr, cost in self._dijkstra_edges(node, g, w, q):
                nd = d + cost
                if nd < dist.get(nbr, INF) - 1e-9:
                    dist[nbr] = nd
                    prev[nbr] = node
                    cnt += 1
                    heapq.heappush(heap, (nd, cnt, nbr))
        if 'DONE' not in prev:
            return None
        path = []
        cur = 'DONE'
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        # Reconstruct (elevator, pickup, dropoff) legs from the path
        legs = []
        cur_e = cur_a = cur_f = None
        for node in path:
            if node == 'DONE':
                if cur_e is not None:
                    legs.append((cur_e, cur_a, cur_f))
                break
            if node[0] == 'F':
                if cur_e is not None:
                    legs.append((cur_e, cur_a, cur_f))
                    cur_e = None
                cur_f = node[1]
            else:
                _, e, f = node
                if cur_e is None:
                    cur_e, cur_a = e, f
                cur_f = f
        return legs if legs else None

    def _dijkstra_edges(self, node, goal, weight, q):
        """Edges with 1/prob cost = expected attempts per success."""
        out = []
        if node[0] == 'F':
            f = node[1]
            for e in self.elev_ids:
                # f accessible if in reachable set OR elevator's initial floor (cycle-start only)
                if (f in self.reachable[e] or f == self.init_efloor[e]) and weight <= self.cap[e]:
                    # Move elevator to f (1/pe) + person enters (1/q)
                    out.append((('I', e, f), 1.0 / self.pe[e] + 1.0 / q))
        else:
            _, e, f = node
            for f2 in self.reachable[e]:
                if f2 != f:
                    out.append((('I', e, f2), 1.0 / self.pe[e]))
            if f == goal:
                out.append(('DONE', 1.0 / q))
            else:
                out.append((('F', f), 1.0 / q))  # transfer: exit at non-goal floor
        return out

    # ------------------------------------------------------------------ #
    # Cycle selection: pick subset S* maximising reward/step rate          #
    # ------------------------------------------------------------------ #

    def _choose_cycle(self):
        deliverable = sorted(self.deliverable)
        n = len(deliverable)
        if n == 0:
            return frozenset(), False, 0.0
        best_rate, best_target, best_allflag = -INF, frozenset(), False
        for mask in range(1, 1 << n):
            S = frozenset(deliverable[i] for i in range(n) if (mask >> i) & 1)
            allflag = (S == self.all_persons)
            reward = sum(self.Erew[p] for p in S)
            if allflag:
                reward += self.goal_reward
            cost = self._estimate_cycle_cost(S, allflag)
            if cost <= 0 or cost == INF:
                continue
            rate = reward / cost
            if rate > best_rate:
                best_rate, best_target, best_allflag = rate, S, allflag
        return best_target, best_allflag, max(best_rate, 0.0)

    def _estimate_cycle_cost(self, S_set, allflag):
        """Greedy rollout from initial state: expected steps for one cycle."""
        efl = dict(self.init_efloor)
        ew = {e: 0 for e in self.elev_ids}
        ploc = {p: ('floor', self.start[p]) for p in self.person_ids}
        cost = 0.0
        for _ in range(600):
            present = [p for p in S_set if p in ploc]
            if not present:
                if not allflag:
                    cost += 1.0  # RESET costs 1 step
                return cost
            best_act = best_prob = None
            best_after = INF
            for act, prob in self._candidate_actions(efl, ew, ploc, present):
                if act[0] == 'RESET':
                    continue
                nefl, nw, nploc = self._apply_success(act, efl, ew, ploc)
                ng = sum(self._ctg(p, nploc[p], nefl) for p in S_set if p in nploc)
                if ng < best_after - 1e-9:
                    best_after, best_act, best_prob = ng, act, prob
            if best_act is None:
                return INF
            cost += 1.0 / best_prob
            efl, ew, ploc = self._apply_success(best_act, efl, ew, ploc)
        return INF

    # ------------------------------------------------------------------ #
    # Cost-to-go and renewal potential                                     #
    # ------------------------------------------------------------------ #

    def _ctg(self, p, loc, efl):
        """Expected steps to deliver p from location loc given elevator floors."""
        best = INF
        if loc[0] == 'in':
            e = loc[1]
            f = efl[e]
            for plan in self.plans[p]:
                for i, (le, a, b) in enumerate(plan.legs):
                    if le != e:
                        continue
                    move = 0.0 if f == b else 1.0 / self.pe[e]
                    c = move + 1.0 / self.qp[p] + plan.suffix[i + 1]
                    if c < best:
                        best = c
                    break
        else:
            f = loc[1]
            for plan in self.plans[p]:
                for i, (le, a, b) in enumerate(plan.legs):
                    if a != f:
                        continue
                    if efl[le] == f:
                        repo = 0.0
                    elif f in self.reachable[le]:
                        repo = 1.0 / self.pe[le]
                    else:
                        break  # elevator can't reach this floor; plan not executable now
                    c = repo + plan.intr[i] + plan.suffix[i + 1]
                    if c < best:
                        best = c
                    break
        return best

    def _person_min_steps(self, p, loc, efl):
        """Optimistic (all-succeed) step count to deliver p."""
        best = INF
        if loc[0] == 'in':
            e = loc[1]
            f = efl[e]
            for plan in self.plans[p]:
                for i, (le, a, b) in enumerate(plan.legs):
                    if le != e:
                        continue
                    c = (0 if f == b else 1) + 1 + plan.suffix_min[i + 1]
                    if c < best:
                        best = c
                    break
        else:
            f = loc[1]
            for plan in self.plans[p]:
                for i, (le, a, b) in enumerate(plan.legs):
                    if a != f:
                        continue
                    if efl[le] == f:
                        repo = 0
                    elif f in self.reachable[le]:
                        repo = 1
                    else:
                        break  # elevator can't reach this floor; plan not executable now
                    c = repo + plan.intr_min[i] + plan.suffix_min[i + 1]
                    if c < best:
                        best = c
                    break
        return best

    def _potential(self, efl, ploc, steps_left):
        """Renewal potential F(s) = R_remaining(s) - rho * g(s).

        Persons that can't finish within steps_left are pruned from both terms.
        Shared elevator reposition: if n persons on the same floor all need the
        same elevator to reposition there, only one reposition is needed, so we
        credit back (n-1) * 1/pe[e] that the independent ctg estimates count twice.
        """
        R = g = 0.0
        n_present = n_pursuable = 0
        reposition_group = {}   # (elevator_id, floor) -> list of person weights waiting there
        for p in self.target:
            loc = ploc.get(p)
            if loc is None:
                continue
            n_present += 1
            if self._person_min_steps(p, loc, efl) <= steps_left:
                g += self._ctg(p, loc, efl)
                R += self.Erew[p]
                n_pursuable += 1
                if loc[0] == 'floor':
                    f = loc[1]
                    for plan in self.plans[p]:
                        for (le, a, b) in plan.legs:
                            if a != f:
                                continue
                            if efl[le] != f and f in self.reachable[le]:
                                key = (le, f)
                                if key not in reposition_group:
                                    reposition_group[key] = []
                                reposition_group[key].append(self.weight[p])
                            break
        # Credit shared reposition only if at least 2 persons can ride together
        # (sum of 2 lightest weights must not exceed elevator capacity)
        redundancy_save = 0.0
        for (le, f), weights in reposition_group.items():
            if len(weights) < 2:
                continue
            weights_s = sorted(weights)
            if weights_s[0] + weights_s[1] <= self.cap[le]:
                redundancy_save += (len(weights) - 1) / self.pe[le]
        if self.allpersons_flag and n_present > 0 and n_pursuable == n_present:
            R += self.goal_reward
        return R - self.rho * (g - redundancy_save)

    # ------------------------------------------------------------------ #
    # Action generation and transition model                               #
    # ------------------------------------------------------------------ #

    def _candidate_actions(self, efl, ew, ploc, present):
        """Return (action, success_prob) for all useful actions.

        'Useful' = advances a present target person along one of its plans,
        or RESET. Multiple plans may suggest different MOVE targets for the
        same elevator -- all are included so the lookahead can compare them.
        """
        acts = []
        seen = set()

        def add(a, prob):
            if a not in seen:
                seen.add(a)
                acts.append((a, prob))

        for p in present:
            loc = ploc[p]
            if loc[0] == 'in':
                e = loc[1]
                f = efl[e]
                for plan in self.plans[p]:
                    for (le, a, b) in plan.legs:
                        if le != e:
                            continue
                        if f == b:
                            add(('EXIT', p, e), self.qp[p])
                        else:
                            add(('MOVE', e, b), self.pe[e])
                        break
            else:
                f = loc[1]
                for plan in self.plans[p]:
                    for (le, a, b) in plan.legs:
                        if a != f:
                            continue
                        if efl[le] == f:
                            if ew[le] + self.weight[p] <= self.cap[le]:
                                add(('ENTER', p, le), self.qp[p])
                        elif f in self.reachable[le]:
                            add(('MOVE', le, f), self.pe[le])
                        # else: f is not reachable and elevator isn't there — skip
                        break

        acts.append((('RESET',), 1.0))
        return acts

    def _apply_success(self, act, efl, ew, ploc):
        """Deterministic success outcome: returns (nefl, nw, nploc)."""
        kind = act[0]
        if kind == 'RESET':
            return (dict(self.init_efloor),
                    {e: 0 for e in self.elev_ids},
                    {p: ('floor', self.start[p]) for p in self.person_ids})
        nefl, nw, nploc = dict(efl), dict(ew), dict(ploc)
        if kind == 'MOVE':
            nefl[act[1]] = act[2]
        elif kind == 'ENTER':
            nploc[act[1]] = ('in', act[2])
            nw[act[2]] += self.weight[act[1]]
        else:  # EXIT
            p, e = act[1], act[2]
            f = nefl[e]
            nw[e] -= self.weight[p]
            if f == self.goal_floor[p]:
                del nploc[p]
            else:
                nploc[p] = ('floor', f)
        return nefl, nw, nploc

    def _outcomes(self, act, efl, ew, ploc):
        """Yield (prob, immediate_reward, nefl, nw, nploc) for all stochastic outcomes."""
        kind = act[0]
        if kind == 'RESET':
            yield (1.0, 0.0,
                   dict(self.init_efloor),
                   {e: 0 for e in self.elev_ids},
                   {p: ('floor', self.start[p]) for p in self.person_ids})
            return

        if kind == 'MOVE':
            _, e, target = act
            pe = self.pe[e]
            nefl = dict(efl); nefl[e] = target
            yield (pe, 0.0, nefl, ew, ploc)
            cur = efl[e]
            fail_opts = sorted({cur} | (set(self.reachable[e]) - {target}))
            pf = (1.0 - pe) / len(fail_opts)
            for fo in fail_opts:
                nf = dict(efl); nf[e] = fo
                yield (pf, 0.0, nf, ew, ploc)
            return

        if kind == 'ENTER':
            _, p, e = act
            q = self.qp[p]
            nploc = dict(ploc); nploc[p] = ('in', e)
            nw = dict(ew); nw[e] += self.weight[p]
            yield (q, 0.0, efl, nw, nploc)
            yield (1.0 - q, 0.0, efl, ew, ploc)
            return

        # EXIT
        _, p, e = act
        q = self.qp[p]
        f = efl[e]
        nw = dict(ew); nw[e] -= self.weight[p]
        if f == self.goal_floor[p]:
            nploc = dict(ploc); del nploc[p]
            r = self.Erew[p]
            if len(nploc) == 0:  # last person delivered: automatic full-clear reset
                yield (q, r + self.goal_reward,
                       dict(self.init_efloor),
                       {ee: 0 for ee in self.elev_ids},
                       {pp: ('floor', self.start[pp]) for pp in self.person_ids})
            else:
                yield (q, r, efl, nw, nploc)
        else:
            nploc = dict(ploc); nploc[p] = ('floor', f)
            yield (q, 0.0, efl, nw, nploc)
        yield (1.0 - q, 0.0, efl, ew, ploc)

    # ------------------------------------------------------------------ #
    # Depth-limited Expectimax with canonical-state cache                  #
    # ------------------------------------------------------------------ #

    def _canon(self, efl, ploc):
        return (tuple(sorted(efl.items())), tuple(sorted(ploc.items())))

    def _value(self, efl, ew, ploc, depth, steps_left):
        """max_a E[r + V(s')] with F(s) as leaf."""
        if depth <= 0 or steps_left <= 0:
            return self._potential(efl, ploc, steps_left)
        key = (self._canon(efl, ploc), depth)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        present = [p for p in self.target if p in ploc]
        acts = self._candidate_actions(efl, ew, ploc, present)
        best = -INF
        for act, _ in acts:
            ev = 0.0
            for prob, r, nefl, nw, nploc in self._outcomes(act, efl, ew, ploc):
                ev += prob * (r + self._value(nefl, nw, nploc, depth - 1, steps_left - 1))
            if ev > best:
                best = ev
        if best == -INF:
            best = self._potential(efl, ploc, steps_left)
        self._cache[key] = best
        return best

    def _endgame_eval(self, efl, ew, ploc, all_persons, steps_left):
        """Evaluate actions with target widened to all_persons and rho=0.

        Used for endgame hitchhiker planning and in-transit delivery.
        Returns the best action tuple, or None if no actions exist.
        """
        saved_target, saved_allflag, saved_rho = self.target, self.allpersons_flag, self.rho
        self.target = frozenset(all_persons)
        self.allpersons_flag = (self.target == self.all_persons)
        self.rho = 0.0
        self._cache = {}
        depth = min(steps_left, 15)
        acts = self._candidate_actions(efl, ew, ploc, all_persons)
        PRIORITY = {'EXIT': 3, 'ENTER': 2, 'MOVE': 1, 'RESET': 0}
        best_act = best_key = None
        best_val = -INF
        for act, prob in acts:
            ev = g_ev = 0.0
            for pr, r, nefl, nw, nploc in self._outcomes(act, efl, ew, ploc):
                ev += pr * (r + self._value(nefl, nw, nploc, depth - 1, steps_left - 1))
                for p in self.target:
                    loc = nploc.get(p)
                    if loc is not None and self._person_min_steps(p, loc, nefl) <= steps_left - 1:
                        g_ev += pr * self._ctg(p, loc, nefl)
            key = (-g_ev, PRIORITY[act[0]], prob)
            if best_act is None or ev > best_val + 1e-9:
                best_act, best_val, best_key = act, ev, key
            elif ev >= best_val - 1e-9 and key > best_key:
                best_act, best_key = act, key
        self.target, self.allpersons_flag, self.rho = saved_target, saved_allflag, saved_rho
        return best_act

    # ------------------------------------------------------------------ #
    # Endgame helpers                                                      #
    # ------------------------------------------------------------------ #

    def _tail_subset(self, efl, ploc, present, steps_left):
        """When individual min-steps sum exceeds steps_left, return best feasible subset."""
        plan_set = set(self._plan_eids)
        for p in present:
            if ploc[p][0] == 'in' and ploc[p][1] not in plan_set:
                return None
        min_each = {p: self._person_min_steps(p, ploc[p], efl) for p in present}
        if sum(min_each.values()) <= steps_left:
            return None  # optimistically feasible — no need to subset
        boarded = [p for p in present if ploc[p][0] == 'in']
        if boarded and sum(min_each[p] for p in boarded) > steps_left:
            return None  # can't deliver even the boarded persons
        waiting = [p for p in present if ploc[p][0] != 'in']
        n = len(waiting)
        if n == 0 or n > 10:
            return None
        candidates = []
        for mask in range(1 << n):
            subset = boarded + [waiting[i] for i in range(n) if mask >> i & 1]
            if not subset or len(subset) == len(present):
                continue
            ms = sum(min_each[p] for p in subset)
            rew = sum(self.Erew[p] for p in subset)
            candidates.append((rew, -ms, subset))
        candidates.sort(reverse=True)
        for _, _, subset in candidates:
            if sum(min_each[p] for p in subset) <= steps_left:
                return frozenset(subset)
        return None

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def choose_next_action(self, state):
        elevators_t, persons_t, _ = state
        efl = {eid: fl for (eid, fl, _) in elevators_t}
        ew = {eid: w for (eid, _, w) in elevators_t}
        ploc = {pid: loc for (pid, loc) in persons_t}

        steps_left = self.max_steps - self.game.get_current_steps()
        present = [p for p in self.target if p in ploc]

        # Fast path: A* plan-and-execute for large non-farming problems
        if self._use_astar:
            # Tail subset: when running low on steps, deliver best feasible subset
            if present:
                sub = self._tail_subset(efl, ploc, present, steps_left)
                if sub is not None:
                    act = self._astar_planner.get_action(efl, ploc, sub)
                    if act is not None:
                        return act
            act = self._astar_planner.get_action(efl, ploc, self.target)
            if act is not None:
                return act
            # Planning failed or cycle complete — fall through to expectimax

        # Target cycle complete: RESET if there's enough budget for another cycle.
        # But first check for hitchhikers still riding in an elevator — deliver
        # them before resetting rather than abandoning them mid-trip.
        if not present:
            in_transit = [p for p in self.person_ids
                          if p not in self.target and p in ploc
                          and ploc[p][0] == 'in']
            if in_transit:
                all_remaining = [p for p in self.person_ids if p in ploc]
                act = self._endgame_eval(efl, ew, ploc, all_remaining, steps_left)
                if act is not None:
                    return self._format(act)
            if self.target:
                cheapest = min((self.min_steps_init[p] for p in self.target), default=INF)
                if steps_left >= 1 + cheapest:
                    return "RESET"
            return self._fallback(elevators_t)

        # Endgame: fewer than 2 full target-cycles remain — include non-target
        # persons as hitchhikers and search exhaustively to the horizon end.
        min_cycle = min(
            (self._person_min_steps(p, ploc[p], efl) for p in present),
            default=INF
        )
        if self.target != self.all_persons:
            min_cycle += 1  # farming loop: RESET step is part of each cycle
        if min_cycle < INF and steps_left < 2 * min_cycle:
            all_remaining = [p for p in self.person_ids if p in ploc]
            if len(all_remaining) > len(present):
                act = self._endgame_eval(efl, ew, ploc, all_remaining, steps_left)
                if act is not None:
                    return self._format(act)

        elapsed = time.perf_counter() - self._t0
        remaining = max(0.0, self._time_budget - elapsed)
        step_budget = min(0.5, remaining / max(steps_left, 1))

        acts = self._candidate_actions(efl, ew, ploc, present)
        PRIORITY = {'EXIT': 3, 'ENTER': 2, 'MOVE': 1, 'RESET': 0}

        best_act = None
        t_step = time.perf_counter()

        for depth in range(1, self._max_depth + 1):
            self._cache = {}
            t_depth = time.perf_counter()

            curr_act = None
            curr_val = -INF
            curr_key = None

            for act, prob in acts:
                ev = g_ev = 0.0
                for pr, r, nefl, nw, nploc in self._outcomes(act, efl, ew, ploc):
                    child_v = self._value(nefl, nw, nploc, depth - 1, steps_left - 1)
                    ev += pr * (r + child_v)
                    for p in self.target:
                        loc = nploc.get(p)
                        if loc is not None and self._person_min_steps(p, loc, nefl) <= steps_left - 1:
                            g_ev += pr * self._ctg(p, loc, nefl)

                key = (-g_ev, PRIORITY[act[0]], prob)
                if curr_act is None or ev > curr_val + 1e-9:
                    curr_act, curr_val, curr_key = act, ev, key
                elif ev >= curr_val - 1e-9 and key > curr_key:
                    curr_act, curr_key = act, key

            best_act = curr_act

            depth_time = time.perf_counter() - t_depth
            step_time = time.perf_counter() - t_step
            # Stop if predicted next-depth time would exceed the per-step budget
            # (factor of 4 estimates branching growth per level)
            if step_time + depth_time * 4 > step_budget:
                break

        return self._format(best_act) if best_act else self._fallback(elevators_t)

    def _fallback(self, elevators_t):
        for (eid, cur_f, _) in elevators_t:
            for f in self.reachable[eid]:
                if f != cur_f:
                    return f"MOVE{{{eid},{f}}}"
        return "RESET"

    def _format(self, act):
        kind = act[0]
        if kind == 'RESET':
            return "RESET"
        if kind == 'MOVE':
            return f"MOVE{{{act[1]},{act[2]}}}"
        if kind == 'ENTER':
            return f"ENTER{{{act[1]},{act[2]}}}"
        return f"EXIT{{{act[1]},{act[2]}}}"
