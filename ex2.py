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

        # Depth scales down with more targets (more branching per step)
        n_t = len(self.target)
        self.depth = 5 if n_t <= 1 else (4 if n_t <= 2 else 3)
        self._cache = {}
        self._t0 = time.perf_counter()

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
        """
        R = g = 0.0
        n_present = n_pursuable = 0
        for p in self.target:
            loc = ploc.get(p)
            if loc is None:
                continue
            n_present += 1
            if self._person_min_steps(p, loc, efl) <= steps_left:
                g += self._ctg(p, loc, efl)
                R += self.Erew[p]
                n_pursuable += 1
        if self.allpersons_flag and n_present > 0 and n_pursuable == n_present:
            R += self.goal_reward
        return R - self.rho * g

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

        # Target cycle complete: RESET if there's enough budget for another cycle
        if not present:
            if self.target:
                cheapest = min((self.min_steps_init[p] for p in self.target), default=INF)
                if steps_left >= 1 + cheapest:
                    return "RESET"
            return self._fallback(elevators_t)

        self._cache = {}
        # Throttle depth if cumulative wall-clock time is getting large
        depth = self.depth
        elapsed = time.perf_counter() - self._t0
        if elapsed > 0.8:
            depth = min(depth, 2)
        if elapsed > 1.3:
            depth = 1

        acts = self._candidate_actions(efl, ew, ploc, present)
        PRIORITY = {'EXIT': 3, 'ENTER': 2, 'MOVE': 1, 'RESET': 0}
        best_act = None
        best_val = -INF
        best_key = None

        for act, prob in acts:
            ev = g_ev = 0.0
            for pr, r, nefl, nw, nploc in self._outcomes(act, efl, ew, ploc):
                child_v = self._value(nefl, nw, nploc, depth - 1, steps_left - 1)
                ev += pr * (r + child_v)
                for p in self.target:
                    loc = nploc.get(p)
                    if loc is not None and self._person_min_steps(p, loc, nefl) <= steps_left - 1:
                        g_ev += pr * self._ctg(p, loc, nefl)

            # Primary: max expected value. Tie-break: min remaining g, then action type, then reliability.
            key = (-g_ev, PRIORITY[act[0]], prob)
            if best_act is None or ev > best_val + 1e-9:
                best_act, best_val, best_key = act, ev, key
            elif ev >= best_val - 1e-9 and key > best_key:
                best_act, best_key = act, key

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
