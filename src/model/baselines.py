"""Baseline planners for the planning model (formulation in Section S5 of the Supplementary Materials of the paper).

Three practitioner-style planners, all built through ``PlanBuilder`` so that every constraint of the
model (one option per unit-bench, lanes, ownership, pads, rigs, crews, capital, lease deadlines,
acquisition budget) holds by construction; ``Model.feasible`` is the independent check in the tests.

* ``heuristic_plan``  rank-and-fill with recourse: drill-to-hold units first, then by profitability
  index or NPV; spacing and lateral length chosen per unit-bench by value (interference is already in
  the revenue); benches co-developed from one pad; packages bought by value-to-price.
  ``tune_heuristic`` grid-searches its planning deck, rig shadow price and ranking metric against the
  model's own objective (the strongest fair version of the heuristic).
* ``pso_plan``        binary particle swarm optimization (Kennedy and Eberhart's sigmoid-velocity
  BPSO, the form used in well-placement papers) over the first-stage option and package bits, with
  the heuristic's recourse rule.
* ``ga_plan``         a genetic algorithm with the same encoding, decoding and fitness.

``recourse_fill`` is the shared per-node fill routine. Plans are dicts as in ``Model.empty_plan``:
z0 (n0,), zk (K, n1), y (G,), w0 (U,), wk (K, U, T-1).

Economics used for ranking (per option o, node k): value = deck * revenue - cost - pad (if the unit has
no pad yet) - mu_rig * rig-days, with revenue = M.revk[o, k] in node k and the probability-weighted
M.revk[o, :] @ prob for the shared first stage. ``deck`` is an extra multiplier on top of whatever deck
``Model(deck=...)`` already folded into revk. With ``terminal=True`` the terminal option value the
model credits to an undeveloped unit-bench is subtracted as an opportunity cost (off by default: a
practitioner's rank-and-fill does not price the option to wait).
"""
import time

import numpy as np

from instance import NB, T
from model import PlanBuilder


# ================================================================================================
# tables and the rank-and-fill core
# ================================================================================================
def _tables(M):
    """Lookup tables for the fill routine, computed once per Model and cached on the instance."""
    tb = getattr(M, "_baseline_tables", None)
    if tb is not None:
        return tb
    p = M.I["prob"]
    by_t = [np.flatnonzero(M.o_t == t) for t in range(T)]
    by_ub_t = {}
    for t in range(T):
        ub_t = M.o_ub[by_t[t]]
        for ub in range(M.nub):
            by_ub_t[ub, t] = by_t[t][ub_t == ub]
    inf = np.inf
    tb = dict(
        rev_exp=M.revk @ p,                                  # expected option revenue (n,)
        tau_exp=M.tau @ p,                                   # expected terminal value per unit-bench (nub,)
        by_t=by_t, by_ub_t=by_ub_t,
        min_rig=np.array([M.rigdays[i].min() if len(i) else inf for i in by_t]),
        min_wells=np.array([M.wells[i].min() if len(i) else inf for i in by_t]),
        min_cost=np.array([M.C[i].min() if len(i) else inf for i in by_t]),
    )
    M._baseline_tables = tb
    return tb


def _values(M, B, k, t, cand, deck=1.0, mu_rig=0.0, terminal=False, mu_crew=0.0):
    """Value of adding each option of ``cand`` (all of year t) in node k (None: first stage).

    value = deck * revenue - cost - pad (if the unit has no pad yet) - mu_rig * rig-days - mu_crew * lateral-miles
            [- the terminal value forfeited by developing the unit-bench, if ``terminal``].
    ``mu_crew`` (2026-11-02, NDIC rule check) is a shadow price per lateral mile of completion capacity; its default
    0 leaves every earlier call unchanged.

    Returns (ok, value, basis): ``ok`` marks options admissible now (unit owned or in a bought package,
    not past the lease deadline, unit-bench not yet developed in the node, no pad planned after year
    t) with positive value; ``basis`` = cost + pad charged, the denominator of the profitability index.
    The first stage reads node 0's state, which equals every node's state while no recourse exists.
    """
    tb = _tables(M)
    kk = 0 if k is None else k
    rev = tb["rev_exp"][cand] if k is None else M.revk[cand, k]
    u, ub = M.o_u[cand], M.o_ub[cand]
    owned = M.ub_owner_ok[cand].copy()
    if M.G:
        pk = M.pkg[cand]
        owned |= (pk >= 0) & (B.plan["y"][np.maximum(pk, 0)] > 0.5)
    py = B.pad_year[kk, u]
    ok = owned & ~M.dead[cand] & ~B.used_ub[kk, ub] & (py <= t)
    pad = np.where(py < 0, M.pad_cost[u, t], 0.0)
    value = deck * rev - M.C[cand] - pad - mu_rig * M.rigdays[cand] - mu_crew * M.lat_miles[cand]
    if terminal:
        value = value - (tb["tau_exp"][ub] if k is None else M.tau[ub, k])
    ok &= value > 0
    return ok, value, M.C[cand] + pad


def _exhausted(M, B, kk, t):
    """True when year t of node kk cannot take even the smallest option (rig, crew or capital)."""
    tb, cap = _tables(M), M.I["cap"]
    return (cap["rig"][t] - B.rig[kk, t] < tb["min_rig"][t] - 1e-9
            or cap["crew"][t] - B.crew[kk, t] < tb["min_wells"][t] - 1e-9
            or cap["capital"][t] - B.cap[kk, t] < tb["min_cost"][t] - 1e-9)


def _try_add(B, opts, k):
    """Add the first option of ``opts`` (in preference order) that PlanBuilder accepts; return it or None."""
    for o in opts:
        if B.can_add(o, k):
            B.add(o, k)
            return int(o)
    return None


def _codevelop(M, B, k, t, o, deck, mu_rig, terminal, mu_crew=0.0):
    """Try the best admissible option of the other bench of option o's unit in the same year (same pad)."""
    kk = 0 if k is None else k
    ub2 = M.o_u[o] * NB + (1 - M.o_b[o])
    if B.used_ub[kk, ub2]:
        return None
    c2 = _tables(M)["by_ub_t"][ub2, t]
    if len(c2) == 0:
        return None
    ok, v2, _ = _values(M, B, k, t, c2, deck, mu_rig, terminal, mu_crew)
    if not ok.any():
        return None
    c2, v2 = c2[ok], v2[ok]
    return _try_add(B, c2[np.argsort(-v2, kind="stable")], k)


def _fill_year(M, B, k, t, deck=1.0, mu_rig=0.0, metric="pi", concept=True, terminal=False, mu_crew=0.0):
    """Rank-and-fill year t in node k (None: the shared first stage). Returns the options added.

    Unit-benches are ranked by the score of their best option (by value): units whose lease deadline
    falls in year t (drill-to-hold, now or never) first, then by the profitability index value/(cost +
    pad) for metric 'pi' or by value for 'npv'. Each unit-bench gets its best admissible option (spacing
    and lateral length by value; the first option PlanBuilder accepts, in value order, is the repair).
    With ``concept`` the other bench of the same unit is tried right after, from the same pad.
    """
    tb, I = _tables(M), M.I
    kk = 0 if k is None else k
    cand = tb["by_t"][t]
    ok, value, basis = _values(M, B, k, t, cand, deck, mu_rig, terminal, mu_crew)
    cand, value, basis = cand[ok], value[ok], basis[ok]
    added = []
    if len(cand) == 0:
        return added
    score = value / basis if metric == "pi" else value
    ub = M.o_ub[cand]
    order = np.lexsort((-value, ub))                              # by unit-bench, best value first
    cand, value, score, ub = cand[order], value[order], score[order], ub[order]
    starts = np.flatnonzero(np.r_[True, ub[1:] != ub[:-1]])
    ends = np.r_[starts[1:], len(ub)]
    hold = I["deadline"][M.o_u[cand[starts]]] == t                # drill-to-hold: deadline this year
    rank = np.lexsort((-score[starts], (~hold).astype(int)))      # hold first, then by score
    for j in rank:
        if _exhausted(M, B, kk, t):
            break
        if B.used_ub[kk, ub[starts[j]]]:
            continue
        o = _try_add(B, cand[starts[j]:ends[j]], k)
        if o is None:
            continue
        added.append(o)
        if concept:
            o2 = _codevelop(M, B, k, t, o, deck, mu_rig, terminal, mu_crew)
            if o2 is not None:
                added.append(o2)
    return added


def recourse_fill(M, B, k, deck=1.0, mu_rig=0.0, metric="pi", concept=True, terminal=False, mu_crew=0.0):
    """Fill years 1..T-1 of node k by rank-and-fill with node-k economics (M.revk[:, k]).

    ``B`` is a PlanBuilder holding the year-0 decisions (and any packages). Years are filled in order,
    so pads are never planned after a well. Returns the list of options added.
    """
    added = []
    for t in range(1, T):
        added += _fill_year(M, B, k, t, deck, mu_rig, metric, concept, terminal, mu_crew)
    return added


# ================================================================================================
# packages
# ================================================================================================
def package_values(M, deck=1.0, mu_rig=0.0, terminal=False, mu_crew=0.0):
    """Year-0 development value of each package ($M, same economics as the ranking).

    Over the package's units: the best year-0 option value of each unit-bench floored at zero, summed
    over the two benches, minus one pad, floored at zero [plus, if ``terminal``, the expected terminal
    value of the unit-benches, whose option values are then net of it].
    """
    tb, I = _tables(M), M.I
    o0 = tb["by_t"][0]
    v = deck * tb["rev_exp"][o0] - M.C[o0] - mu_rig * M.rigdays[o0] - mu_crew * M.lat_miles[o0]
    if terminal:
        v = v - tb["tau_exp"][M.o_ub[o0]]
    best = np.zeros(M.nub)
    np.maximum.at(best, M.o_ub[o0], v)
    unit = np.maximum(best.reshape(M.U, NB).sum(1) - M.pad_cost[:, 0], 0.0)
    if terminal:
        unit = unit + tb["tau_exp"].reshape(M.U, NB).sum(1)
    val = np.zeros(M.G)
    sel = I["pkg_of"] >= 0
    np.add.at(val, I["pkg_of"][sel], unit[sel])
    return val


def buy_packages(M, B, deck=1.0, mu_rig=0.0, terminal=False, mu_crew=0.0):
    """Buy packages in descending value-to-price ratio while the ratio exceeds 1 and the budget allows.

    A package that does not fit the remaining acquisition budget is skipped (greedy knapsack).
    Returns the packages bought.
    """
    if M.G == 0:
        return []
    ratio = package_values(M, deck, mu_rig, terminal, mu_crew) / np.maximum(M.I["pkg_cost"], 1e-12)
    bought = []
    for g in np.argsort(-ratio, kind="stable"):
        if ratio[g] <= 1.0:
            break
        if B.can_buy(g):
            B.buy(g)
            bought.append(int(g))
    return bought


# ================================================================================================
# 1. rank-and-fill heuristic with recourse, and 2. its tuned version
# ================================================================================================
def heuristic_plan(M, deck=1.0, mu_rig=0.0, metric="pi", concept=True, terminal=False, packages=True,
                   verbose=False, mu_crew=0.0):
    """Practitioner rank-and-fill with recourse; returns a feasible plan dict.

    Year 0 (shared by all nodes): buy packages by value-to-price, drill-to-hold units (deadline 0)
    first, then fill by rank with expected revenue. Then every node k and year t = 1..T-1 is filled
    with node-k economics (``recourse_fill``); a unit with deadline 1 that was not developed in year 0
    is drill-to-hold in year 1 of every node.

    deck     extra revenue multiplier (a planning deck; 1.0 = the model's own prices)
    mu_rig   shadow price per rig-day ($M), charged against value
    metric   'pi' ranks unit-benches by value/(cost + pad), 'npv' by value
    concept  co-develop both benches of a unit from one pad
    terminal subtract the forfeited terminal option value from an option's value
    packages buy acreage packages before filling
    mu_crew  shadow price per lateral mile of completion capacity ($M), charged against value (default 0:
             the registered rule; added 2026-11-02 for the NDIC rule check)
    """
    B = PlanBuilder(M)
    bought = buy_packages(M, B, deck, mu_rig, terminal, mu_crew) if packages else []
    a0 = _fill_year(M, B, None, 0, deck, mu_rig, metric, concept, terminal, mu_crew)
    ak = [len(recourse_fill(M, B, k, deck, mu_rig, metric, concept, terminal, mu_crew)) for k in range(M.K)]
    if verbose:
        print(f"heuristic deck={deck} mu_rig={mu_rig} mu_crew={mu_crew} {metric}: packages {bought}, year-0 options {len(a0)} "
              f"(wells {int(B.plan['z0'] @ M.wells[M.idx0])}), recourse options per node {ak}")
    return B.plan


def tune_heuristic(M, by="ce", decks=(0.7, 0.8, 0.9, 1.0), mus=(0.0, 0.05, 0.1, 0.2, 0.4), metrics=("pi", "npv"),
                   concepts=(True,), terminals=(False, True), verbose=False, mus_crew=(0.0,)):
    """Grid search over the heuristic's settings, scored by M.evaluate(plan)[by] ('ce' or 'enpv').

    Returns (best plan, its config dict, its score). The grid is the product of the arguments, so the
    default call includes the untuned heuristic (deck 1.0, mu_rig 0.0, 'pi', terminal False); the
    ``terminal=True`` half of the grid lets the heuristic price the option value of leaving a unit-bench
    undeveloped, which the model credits at the horizon (80 configs, a few milliseconds each).
    ``mus_crew`` (default (0.0,): the registered 80-setting grid) adds lateral-mile charges to the grid;
    the config dict carries ``mu_crew`` only when the grid has more than the default value.
    """
    best_plan, best_cfg, best = None, None, -np.inf
    for deck in decks:
        for mu in mus:
            for metric in metrics:
                for concept in concepts:
                    for terminal in terminals:
                        for mu_crew in mus_crew:
                            plan = heuristic_plan(M, deck, mu, metric, concept, terminal, mu_crew=mu_crew)
                            score = float(M.evaluate(plan)[by])
                            if verbose:
                                print(f"  deck={deck} mu_rig={mu} mu_crew={mu_crew} {metric} concept={concept} terminal={terminal}: {by}={score:.3f}")
                            if score > best:
                                best_plan, best = plan, score
                                best_cfg = dict(deck=deck, mu_rig=mu, metric=metric, concept=concept, terminal=terminal)
                                if tuple(mus_crew) != (0.0,):
                                    best_cfg["mu_crew"] = mu_crew
    return best_plan, best_cfg, best


# ================================================================================================
# shared encoding, decoding and fitness for the metaheuristics
# ================================================================================================
FILL_DEFAULT = dict(deck=1.0, mu_rig=0.0, metric="pi", concept=True)


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


class _Decoder:
    """Decodes a bit vector over (year-0 options, packages), length n0 + G, into a feasible plan.

    Marked packages are bought in descending priority (budget permitting); marked year-0 options are
    added in descending priority through PlanBuilder.can_add/add (infeasible ones are skipped: the
    repair); then every node's recourse is filled by the heuristic rule (``recourse='heuristic'``) or
    left empty (``'none'``). Fitness = M.evaluate(plan)[objective]. Scores are cached by the decoded
    insertion order (which determines the plan), so repeated positions cost nothing.
    """
    def __init__(self, M, objective="ce", fill=None, recourse="heuristic"):
        self.M, self.objective, self.recourse = M, objective, recourse
        self.fill = dict(FILL_DEFAULT if fill is None else fill)
        self.cache, self.evals = {}, 0

    def order(self, bits, priority):
        n0 = self.M.n0
        sel = np.flatnonzero(bits[:n0])
        opts = sel[np.argsort(-priority[sel], kind="stable")]
        pk = np.flatnonzero(bits[n0:])
        pkgs = pk[np.argsort(-priority[n0 + pk], kind="stable")]
        return opts, pkgs

    def decode(self, bits, priority):
        M = self.M
        opts, pkgs = self.order(bits, priority)
        B = PlanBuilder(M)
        for g in pkgs:
            if B.can_buy(g):
                B.buy(g)
        for j in opts:
            if _exhausted(M, B, 0, 0):
                break
            o = M.idx0[j]
            if B.used_ub[0, M.o_ub[o]]:
                continue
            if B.can_add(o):
                B.add(o)
        if self.recourse == "heuristic":
            for k in range(M.K):
                recourse_fill(M, B, k, **self.fill)
        return B.plan

    def __call__(self, bits, priority):
        """(score, plan); plan is None on a cache hit (a cached score can never be a new best)."""
        opts, pkgs = self.order(bits, priority)
        key = np.r_[opts, self.M.n0 + pkgs].astype(np.int32).tobytes()
        if key in self.cache:
            return self.cache[key], None
        plan = self.decode(bits, priority)
        score = float(self.M.evaluate(plan)[self.objective])
        self.cache[key] = score
        self.evals += 1
        return score, plan


def _seed_bits(M, fill):
    """The heuristic's year-0 decisions and packages as a bit vector (the swarm/population seed)."""
    plan = heuristic_plan(M, **fill)
    return np.r_[plan["z0"], plan["y"]] > 0.5


def _init_population(rng, x_seed, P, p_init, with_seed=True):
    """Row 0 the seed; rows 1..P/2-1 the seed with each bit flipped with probability up to p_init; the
    rest uniform random bits (the standard BPSO/GA initialization), for diversity. With ``with_seed=False``
    every row is uniform random bits (runs started without the heuristic seed)."""
    n = len(x_seed)
    if not with_seed:
        return rng.random((P, n)) < 0.5
    X = np.zeros((P, n), bool)
    X[0] = x_seed
    for i in range(1, P):
        if i < max(1, P // 2):
            X[i] = x_seed ^ (rng.random(n) < p_init * rng.uniform(0.2, 1.0))
        else:
            X[i] = rng.random(n) < 0.5
    return X


MAX_STALL = 200     # with an evaluation budget: stop after this many iterations without a new (uncached) evaluation


class _Budget:
    """Stopping rule of the metaheuristics: a time budget (seconds, None = no cap) and/or an evaluation budget
    (``max_evals``, None = no cap; a decoder cache hit does not count). The first evaluation of a run (the seed
    or the first random individual) always happens, so a run with max_evals=0 still returns a plan."""
    def __init__(self, dec, time_budget, max_evals):
        if time_budget is None and max_evals is None:
            raise ValueError("give time_budget or max_evals")
        self.t0, self.dec = time.time(), dec
        self.time_budget = np.inf if time_budget is None else float(time_budget)
        self.max_evals = max_evals

    def elapsed(self):
        return time.time() - self.t0

    def out(self):
        return (self.elapsed() > self.time_budget
                or (self.max_evals is not None and self.dec.evals >= self.max_evals))

    def progress(self):
        """Fraction of the budget used, in [0, 1] (the larger of the time and evaluation fractions)."""
        f = self.elapsed() / self.time_budget if np.isfinite(self.time_budget) else 0.0
        if self.max_evals is not None:
            f = max(f, self.dec.evals / max(1, self.max_evals))
        return float(min(1.0, f))


# ================================================================================================
# 3. binary PSO
# ================================================================================================
def pso_plan(M, time_budget=60.0, seed=0, objective="ce", n_particles=40, w=0.7, c1=1.5, c2=1.5,
             recourse="heuristic", fill=None, vmax=4.0, p_init=0.05, verbose=False, max_evals=None, with_seed=True):
    """Binary PSO (Kennedy and Eberhart 1997) over the first-stage option and package bits.

    v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x), clipped to [-vmax, vmax]; x_i = 1 with probability
    sigmoid(v_i). A particle decodes to a feasible plan through PlanBuilder (packages, then year-0
    options in descending sigmoid(v), then the heuristic's recourse rule in every node); fitness is
    M.evaluate(plan)[objective]. One particle is the heuristic's plan; its velocity is +-vmax so its
    position starts exactly there. Runs until ``time_budget`` seconds elapse.

    ``fill`` is the recourse rule's config (keys of ``recourse_fill``: deck, mu_rig, metric, concept,
    terminal); None is the untuned rule (deck 1.0, mu_rig 0.0, 'pi', concept). The seed is
    ``heuristic_plan(M, **fill)``, so a tuned config from ``tune_heuristic`` can be passed as is.

    Optional (defaults keep the original behavior): ``max_evals`` also stops the run after that many uncached
    evaluations (``time_budget=None`` then means no time cap; with an evaluation budget the run also stops
    after MAX_STALL iterations without a new evaluation); ``with_seed=False`` starts from uniform random
    bits only, and seed_score is then the heuristic plan's score computed outside the budget.

    Returns (best plan, info) with info = dict(iters, evals, best_history, time, seed_score, best_score,
    eval_history), eval_history a list of (evals, seconds, best) after the initial swarm and each iteration.
    """
    rng = np.random.default_rng(seed)
    fill = dict(FILL_DEFAULT if fill is None else fill)
    dec = _Decoder(M, objective, fill, recourse)
    bud = _Budget(dec, time_budget, max_evals)
    x_seed = _seed_bits(M, fill)
    n, P = len(x_seed), n_particles
    X = _init_population(rng, x_seed, P, p_init, with_seed)
    V = np.where(X, 1.0, -1.0) * vmax * rng.uniform(0.5, 1.0, (P, n))
    if with_seed:
        V[0] = np.where(X[0], vmax, -vmax)
    prob = _sigmoid(V)
    F = np.full(P, -np.inf)
    best_f, best_plan = -np.inf, None
    for i in range(P):
        if i > 0 and bud.out():
            break
        F[i], plan = dec(X[i], prob[i])
        if F[i] > best_f and plan is not None:
            best_f, best_plan = F[i], plan
    seed_score = float(F[0]) if with_seed else float(M.evaluate(heuristic_plan(M, **fill))[objective])
    pbest, pbest_f = X.copy(), F.copy()
    gbest = pbest[np.argmax(pbest_f)].astype(float)
    hist, iters, stall = [best_f], 0, 0
    ehist = [(dec.evals, bud.elapsed(), float(best_f))]
    while not bud.out():
        before = dec.evals
        r1, r2 = rng.random((P, n)), rng.random((P, n))
        Xf = X.astype(float)
        V = w * V + c1 * r1 * (pbest.astype(float) - Xf) + c2 * r2 * (gbest - Xf)
        np.clip(V, -vmax, vmax, out=V)
        prob = _sigmoid(V)
        X = rng.random((P, n)) < prob
        for i in range(P):
            if bud.out():
                break
            f, plan = dec(X[i], prob[i])
            if f > pbest_f[i]:
                pbest[i], pbest_f[i] = X[i], f
            if f > best_f and plan is not None:
                best_f, best_plan = f, plan
        gbest = pbest[np.argmax(pbest_f)].astype(float)
        iters += 1
        hist.append(best_f)
        ehist.append((dec.evals, bud.elapsed(), float(best_f)))
        if verbose:
            print(f"  pso it{iters}: best {objective}={best_f:.3f} evals={dec.evals} t={bud.elapsed():.1f}s")
        stall = stall + 1 if dec.evals == before else 0
        if max_evals is not None and stall >= MAX_STALL:
            break
    info = dict(iters=iters, evals=dec.evals, best_history=hist, time=bud.elapsed(),
                seed_score=seed_score, best_score=float(best_f), eval_history=ehist)
    return best_plan, info


# ================================================================================================
# 4. genetic algorithm
# ================================================================================================
def _static_priority(M):
    """Decode order for the GA: year-0 profitability index of each option (expected revenue less cost
    and pad over cost plus pad) and value-to-price of each package."""
    tb = _tables(M)
    o0 = M.idx0
    pad = M.pad_cost[M.o_u[o0], 0]
    pi = (tb["rev_exp"][o0] - M.C[o0] - pad) / (M.C[o0] + pad)
    ratio = package_values(M) / np.maximum(M.I["pkg_cost"], 1e-12) if M.G else np.zeros(0)
    return np.r_[pi, ratio]


def ga_plan(M, time_budget=60.0, seed=0, objective="ce", pop=40, p_mut=0.02, p_cx=0.8, recourse="heuristic",
            fill=None, tournament=2, n_elite=2, p_init=0.05, verbose=False, max_evals=None, with_seed=True):
    """Genetic algorithm with the encoding, decoding and fitness of ``pso_plan``.

    Tournament selection (size ``tournament``), uniform crossover with probability ``p_cx``, bit-flip
    mutation with probability ``p_mut`` per bit, elitism of ``n_elite``, until the time budget. Marked
    bits decode in a static priority order (year-0 profitability index; packages by value-to-price).
    One individual is the heuristic's plan (``heuristic_plan(M, **fill)``, see ``pso_plan``).
    ``max_evals`` and ``with_seed`` as in ``pso_plan`` (the defaults keep the original behavior).
    Returns (best plan, info) as in ``pso_plan`` (iters = generations).
    """
    rng = np.random.default_rng(seed)
    fill = dict(FILL_DEFAULT if fill is None else fill)
    dec = _Decoder(M, objective, fill, recourse)
    bud = _Budget(dec, time_budget, max_evals)
    prio = _static_priority(M)
    x_seed = _seed_bits(M, fill)
    n, P = len(x_seed), pop
    X = _init_population(rng, x_seed, P, p_init, with_seed)
    F = np.full(P, -np.inf)
    best_f, best_plan = -np.inf, None

    def evaluate(i):
        nonlocal best_f, best_plan
        F[i], plan = dec(X[i], prio)
        if F[i] > best_f and plan is not None:
            best_f, best_plan = F[i], plan

    for i in range(P):
        if i > 0 and bud.out():
            break
        evaluate(i)
    seed_score = float(F[0]) if with_seed else float(M.evaluate(heuristic_plan(M, **fill))[objective])
    hist, gens, stall = [best_f], 0, 0
    ehist = [(dec.evals, bud.elapsed(), float(best_f))]
    n_elite = min(n_elite, P)

    def pick():
        cand = rng.integers(P, size=tournament)
        return cand[np.argmax(F[cand])]

    while not bud.out():
        before = dec.evals
        elite = np.argsort(-F, kind="stable")[:n_elite]
        children = np.zeros((P - n_elite, n), bool)
        for c in range(P - n_elite):
            child = X[pick()].copy()
            if rng.random() < p_cx:
                mask = rng.random(n) < 0.5
                child[mask] = X[pick()][mask]
            child ^= rng.random(n) < p_mut
            children[c] = child
        newX = np.vstack([X[elite], children])
        newF = np.r_[F[elite], np.full(P - n_elite, -np.inf)]
        X, F = newX, newF
        for i in range(n_elite, P):
            if bud.out():
                break
            evaluate(i)
        gens += 1
        hist.append(best_f)
        ehist.append((dec.evals, bud.elapsed(), float(best_f)))
        if verbose:
            print(f"  ga gen{gens}: best {objective}={best_f:.3f} evals={dec.evals} t={bud.elapsed():.1f}s")
        stall = stall + 1 if dec.evals == before else 0
        if max_evals is not None and stall >= MAX_STALL:
            break
    info = dict(iters=gens, evals=dec.evals, best_history=hist, time=bud.elapsed(),
                seed_score=seed_score, best_score=float(best_f), eval_history=ehist)
    return best_plan, info
