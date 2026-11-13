"""Particle-swarm and genetic-algorithm baselines. Numpy only: imports instance, model and baselines, never solve or bench_v3 (gurobipy), so
it runs on a machine without a Gurobi license.

Why a rebuild. The sigmoid binary PSO of ``baselines.pso_plan`` (Kennedy and Eberhart 1997) samples every bit
afresh each iteration with probability sigmoid(v). For a bit on which a particle agrees with its personal and
the global best, both pull terms vanish, so v <- w v decays toward zero and the bit is set with probability
drifting to 0.5: the swarm keeps dropping and re-drawing the seed's options at random and rarely leaves the
seed's basin. Retuning vmax, c1 or c2 cannot fix this; a V-shaped transfer or a continuous encoding can.

Three methods, all over the same first-stage encoding as ``baselines`` (one bit per year-0 option, then one per
acreage package; length n0 + G), all decoded by ``baselines._Decoder`` (packages bought in priority order
within the budget, then marked year-0 options added in priority order through PlanBuilder, infeasible ones
skipped, then every node's recourse filled by the heuristic's rule ``fill``) and all scored by
``M.evaluate(plan)[objective]``:

* ``bpso_v_plan``  binary PSO with a V-shaped transfer function (Mirjalili and Lewis 2013): a bit flips with
  probability |tanh(v)|, so an agreeing bit with v near 0 stays put instead of being re-drawn.
* ``rk_pso_plan``  continuous PSO with constriction coefficients (Clerc and Kennedy 2002) over random keys
  (Bean 1994): key > 0.5 marks the bit, the key itself is the decode priority.
* ``ga2_plan``     genetic algorithm with per-bit mutation 1/n (Baeck 1993), a swap move, uniform crossover,
  binary tournament and two elites.

Common arguments: ``max_evals`` (the evaluation budget; a decoder cache hit does not count; the first
evaluation of a run always happens, so max_evals=0 returns the seed's plan), ``time_budget`` (optional cap in
seconds), ``seed`` (numpy Generator seed; a run with only an evaluation budget is deterministic), ``objective``
('ce' or 'enpv'), ``fill`` (the recourse rule's config, as for ``baselines.pso_plan``; also the seed's config),
``with_seed`` (True: individual 0 is the heuristic plan ``heuristic_plan(M, **fill)``, individuals 1..P/2-1 are
it with each bit flipped with probability up to 0.05, the rest uniform random; False: all uniform random),
the population size (40), ``hist_every`` (None: one history entry per iteration; k: an entry when at
least k evaluations have passed since the last one, and at the end) and ``verbose``.

Each returns (best plan, info) with info = dict(iters, evals, time, seed_score, best_score, best_history, stop,
calls): best_history is a list of (evals, seconds, best score so far), starting after the initial population;
stop is 'evals', 'time' or 'stall' (MAX_STALL iterations without a new evaluation); calls counts decoder calls
including cache hits. With with_seed=False, seed_score is the heuristic plan's score, computed outside the
budget, for reference.

References.
  Baeck, T. (1993). Optimal mutation rates in genetic search. ICGA 5, 2-8.
  Bean, J. C. (1994). Genetic algorithms and random keys for sequencing and optimization. ORSA Journal on
    Computing 6(2), 154-160.
  Clerc, M. and Kennedy, J. (2002). The particle swarm: explosion, stability, and convergence in a
    multidimensional complex space. IEEE Transactions on Evolutionary Computation 6(1), 58-73.
  Helwig, S., Branke, J. and Mostaghim, S. (2013). Experimental analysis of bound handling techniques in
    particle swarm optimization. IEEE Transactions on Evolutionary Computation 17(2), 259-271.
  Kennedy, J. and Eberhart, R. C. (1997). A discrete binary version of the particle swarm algorithm.
    IEEE SMC, 4104-4108.
  Mirjalili, S. and Lewis, A. (2013). S-shaped versus V-shaped transfer functions for binary particle swarm
    optimization. Swarm and Evolutionary Computation 9, 1-14.
  Shi, Y. and Eberhart, R. C. (1998). A modified particle swarm optimizer. IEEE ICEC, 69-73 (inertia 0.9 to 0.4).
"""
import hashlib
import time

import numpy as np

from instance import T
from baselines import FILL_DEFAULT, MAX_STALL, _Decoder, _init_population, _static_priority, heuristic_plan

P_INIT = 0.05


# ================================================================================================
# plan serialization (copied from bench_v3 so that this module does not import gurobipy)
# ================================================================================================
def sparse_plan(plan):
    """A plan as sparse index lists: z0, the positions in M.idx0 of the year-0 options taken (option M.idx0[i]); zk,
    per node, the positions in M.idx1; y, the packages bought; w0, the units padded in year 0; wk, per node, the flat
    indices u * (T - 1) + (t - 1) of the pads built in year t >= 1. dense_plan(M, .) inverts it."""
    nz = lambda a: np.flatnonzero(np.asarray(a).ravel() > 0.5).tolist()  # noqa: E731
    return dict(z0=nz(plan["z0"]), zk=[nz(r) for r in plan["zk"]], y=nz(plan["y"]), w0=nz(plan["w0"]),
                wk=[nz(w) for w in plan["wk"]])


def dense_plan(M, sp_plan):
    """The plan dict of a sparse_plan on Model M (same instance and tree)."""
    ix = lambda v: np.asarray(v, dtype=np.intp)  # noqa: E731
    q = M.empty_plan()
    q["z0"][ix(sp_plan["z0"])] = 1.0
    q["y"][ix(sp_plan["y"])] = 1.0
    q["w0"][ix(sp_plan["w0"])] = 1.0
    for k in range(M.K):
        q["zk"][k, ix(sp_plan["zk"][k])] = 1.0
        w = np.zeros(M.U * (T - 1))
        w[ix(sp_plan["wk"][k])] = 1.0
        q["wk"][k] = w.reshape(M.U, T - 1)
    return q


def options_sha1(M):
    """SHA-1 of the option table (o_u, o_b, o_len, o_k, o_t as int64, in Model's enumeration order), as
    bench_v3.plan_format computes it."""
    tab = np.ascontiguousarray(np.c_[M.o_u, M.o_b, M.o_len, M.o_k, M.o_t].astype(np.int64))
    return hashlib.sha1(tab.tobytes()).hexdigest()


# ================================================================================================
# shared run state: budget, best-so-far, history
# ================================================================================================
class _Run:
    def __init__(self, M, max_evals, time_budget, seed, objective, fill, with_seed, hist_every, verbose, name):
        if max_evals is None and time_budget is None:
            raise ValueError("give max_evals or time_budget")
        if objective not in ("ce", "enpv"):
            raise ValueError("objective must be 'ce' or 'enpv'")
        self.t0 = time.time()
        self.M, self.name, self.objective, self.verbose = M, name, objective, verbose
        self.max_evals, self.with_seed, self.hist_every = max_evals, with_seed, hist_every
        self.time_budget = np.inf if time_budget is None else float(time_budget)
        self.rng = np.random.default_rng(seed)
        self.fill = dict(FILL_DEFAULT if fill is None else fill)
        self.dec = _Decoder(M, objective, self.fill, "heuristic")
        self.prio = _static_priority(M)
        self.seed_plan = heuristic_plan(M, **self.fill)
        self.x_seed = np.r_[self.seed_plan["z0"], self.seed_plan["y"]] > 0.5
        self.n = len(self.x_seed)
        self.best_f, self.best_plan = -np.inf, None
        self.calls, self.iters, self.stall, self.stop = 0, 0, 0, None
        self.hist, self._last_rec = [], -np.inf
        self.seed_score = None if with_seed else float(M.evaluate(self.seed_plan)[objective])

    def elapsed(self):
        return time.time() - self.t0

    def out(self):
        if self.max_evals is not None and self.dec.evals >= self.max_evals:
            self.stop = "evals"
        elif self.elapsed() > self.time_budget:
            self.stop = "time"
        return self.stop is not None

    def progress(self):
        """Fraction of the budget used, in [0, 1]: the larger of the evaluation and time fractions."""
        f = self.elapsed() / self.time_budget if np.isfinite(self.time_budget) else 0.0
        if self.max_evals is not None:
            f = max(f, self.dec.evals / max(1, self.max_evals))
        return float(min(1.0, f))

    def evaluate(self, bits, priority):
        """Score of a position (cached by the decoder); keeps the best plan."""
        self.calls += 1
        f, plan = self.dec(bits, priority)
        if f > self.best_f and plan is not None:
            self.best_f, self.best_plan = f, plan
        return f

    def initial(self, X, priorities):
        """Evaluate the initial population (the first individual always, the rest within the budget)."""
        F = np.full(len(X), -np.inf)
        for i in range(len(X)):
            if i > 0 and self.out():
                break
            F[i] = self.evaluate(X[i], priorities[i])
        if self.with_seed:
            self.seed_score = float(F[0])
        self.record(force=True)
        return F

    def record(self, force=False):
        e = self.dec.evals
        if force or self.hist_every is None or e - self._last_rec >= self.hist_every:
            self.hist.append((int(e), round(self.elapsed(), 3), float(self.best_f)))
            self._last_rec = e

    def end_iteration(self, evals_before):
        """Bookkeeping after one iteration; returns True when the run should stop."""
        self.iters += 1
        self.record()
        if self.verbose:
            print(f"  {self.name} it{self.iters}: best {self.objective}={self.best_f:.3f} evals={self.dec.evals} "
                  f"t={self.elapsed():.1f}s", flush=True)
        self.stall = self.stall + 1 if self.dec.evals == evals_before else 0
        if self.stall >= MAX_STALL:
            self.stop = "stall"
            return True
        return self.out()

    def result(self):
        if self.stop is None:
            self.out()
        if not self.hist or self.hist[-1][0] != self.dec.evals or self.hist[-1][2] != self.best_f:
            self.record(force=True)
        info = dict(iters=self.iters, evals=self.dec.evals, time=self.elapsed(), seed_score=self.seed_score,
                    best_score=float(self.best_f), best_history=self.hist, stop=self.stop, calls=self.calls)
        return self.best_plan, info


def _initial_bits(R, P):
    return _init_population(R.rng, R.x_seed, P, P_INIT, R.with_seed)


# ================================================================================================
# 1. binary PSO with a V-shaped transfer function
# ================================================================================================
def bpso_v_plan(M, max_evals, time_budget=None, seed=0, objective="ce", fill=None, with_seed=True, n_particles=40,
                c1=2.0, c2=2.0, w_start=0.9, w_end=0.4, vmax=6.0, hist_every=None, verbose=False):
    """Binary PSO with the V-shaped transfer V2(v) = |tanh(v)| (Mirjalili and Lewis 2013).

    Velocity: v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x), clipped to [-vmax, vmax], with r1, r2 uniform per
    bit and the inertia w decreasing linearly from w_start = 0.9 to w_end = 0.4 over the budget (Shi and Eberhart
    1998; the budget fraction is the larger of the evaluation and time fractions). Position: bit i flips with
    probability |tanh(v_i)| and otherwise keeps its value, so a bit on which the particle agrees with both bests
    and whose velocity has decayed stays put (the defect of the sigmoid rule does not arise). Velocities
    start at zero, so the seed particle starts exactly at the heuristic plan.

    Decoding uses the static priority of ``baselines._static_priority`` (year-0 profitability index of each option,
    value-to-price of each package), the same order as the GA's, so the decode order never depends on the
    velocity's sign. c1 = c2 = 2 and vmax = 6 are the published settings. See the module docstring for the
    common arguments and the returned info.
    """
    R = _Run(M, max_evals, time_budget, seed, objective, fill, with_seed, hist_every, verbose, "bpso_v")
    n, P, rng = R.n, n_particles, R.rng
    X = _initial_bits(R, P)
    V = np.zeros((P, n))
    prio = [R.prio] * P
    F = R.initial(X, prio)
    pbest, pbest_f = X.copy(), F.copy()
    if not R.out():
        while True:
            before = R.dec.evals
            w = w_start - (w_start - w_end) * R.progress()
            gbest = pbest[int(np.argmax(pbest_f))].astype(float)
            Xf = X.astype(float)
            r1, r2 = rng.random((P, n)), rng.random((P, n))
            V = w * V + c1 * r1 * (pbest - Xf) + c2 * r2 * (gbest - Xf)
            np.clip(V, -vmax, vmax, out=V)
            X = X ^ (rng.random((P, n)) < np.abs(np.tanh(V)))
            for i in range(P):
                if R.out():
                    break
                f = R.evaluate(X[i], R.prio)
                if f > pbest_f[i]:
                    pbest[i], pbest_f[i] = X[i], f
            if R.end_iteration(before):
                break
    return R.result()


# ================================================================================================
# 2. random-key PSO with constriction
# ================================================================================================
def rk_pso_plan(M, max_evals, time_budget=None, seed=0, objective="ce", fill=None, with_seed=True, n_particles=40,
                chi=0.7298, c1=2.05, c2=2.05, vclamp=0.5, hist_every=None, verbose=False):
    """Continuous PSO with Clerc and Kennedy's (2002) constriction over random keys (Bean 1994) in [0, 1]^(n0 + G).

    v <- chi (v + c1 r1 (pbest - x) + c2 r2 (gbest - x)), clipped to [-vclamp, vclamp]; x <- x + v, and a key that
    leaves [0, 1] is reflected back into it with its velocity component reversed (Helwig et al. 2013). Decoding:
    bit i is set when key_i > 0.5 and the decode priority is the key itself, so the swarm searches both the set
    of marked options and the order in which the repair adds them. Velocities start at zero.

    The heuristic seed is encoded as keys 0.75 (set bits) and 0.25 (unset bits): the seed's options then tie in
    priority and are added in index order, which reproduces the heuristic's year-0 plan since they fit together.
    The seed-perturbed particles get the same encoding of their bits plus a uniform jitter in [-0.2, 0.2] (which
    keeps their bits and spreads their priorities); the random particles get uniform keys. See the module
    docstring for the common arguments and the returned info.
    """
    R = _Run(M, max_evals, time_budget, seed, objective, fill, with_seed, hist_every, verbose, "rk_pso")
    n, P, rng = R.n, n_particles, R.rng
    bits = _initial_bits(R, P)
    X = np.where(bits, 0.75, 0.25)
    if with_seed:
        half = max(1, P // 2)
        X[1:half] += rng.uniform(-0.2, 0.2, (max(0, half - 1), n))
        X[half:] = rng.random((P - half, n))
    else:
        X = rng.random((P, n))
    V = np.zeros((P, n))
    F = R.initial(X > 0.5, X)
    pbest, pbest_f = X.copy(), F.copy()
    if not R.out():
        while True:
            before = R.dec.evals
            gbest = pbest[int(np.argmax(pbest_f))]
            r1, r2 = rng.random((P, n)), rng.random((P, n))
            V = chi * (V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X))
            np.clip(V, -vclamp, vclamp, out=V)
            X = X + V
            lo, hi = X < 0.0, X > 1.0
            X[lo], X[hi] = -X[lo], 2.0 - X[hi]
            V[lo | hi] *= -1.0
            for i in range(P):
                if R.out():
                    break
                f = R.evaluate(X[i] > 0.5, X[i])
                if f > pbest_f[i]:
                    pbest[i], pbest_f[i] = X[i].copy(), f
            if R.end_iteration(before):
                break
    return R.result()


# ================================================================================================
# 3. genetic algorithm with 1/n mutation and a swap move
# ================================================================================================
def _swap(rng, child, n0):
    """Move one set bit to a random unset bit of the same block (year-0 options, or packages)."""
    on = np.flatnonzero(child)
    if len(on) == 0:
        return
    j = on[rng.integers(len(on))]
    lo, hi = (0, n0) if j < n0 else (n0, len(child))
    off = lo + np.flatnonzero(~child[lo:hi])
    if len(off) == 0:
        return
    child[j] = False
    child[off[rng.integers(len(off))]] = True


def ga2_plan(M, max_evals, time_budget=None, seed=0, objective="ce", fill=None, with_seed=True, pop=40, p_cx=0.8,
             p_mut=None, p_swap=0.5, tournament=2, n_elite=2, hist_every=None, verbose=False):
    """Generational GA: binary tournament selection, uniform crossover with probability p_cx = 0.8, bit-flip
    mutation with probability 1/n per bit (Baeck 1993; p_mut overrides it), then with probability p_swap = 0.5
    per child a swap move that moves one set bit to a random unset bit of the same block (year-0 options or
    packages), which changes an option's spacing, length or unit without changing the count; two elites are kept.

    Decoding uses the static priority of ``baselines._static_priority`` (as ``ga_plan`` and ``bpso_v_plan``). See
    the module docstring for the common arguments and the returned info.
    """
    R = _Run(M, max_evals, time_budget, seed, objective, fill, with_seed, hist_every, verbose, "ga2")
    n, P, rng, n0 = R.n, pop, R.rng, M.n0
    pm = 1.0 / n if p_mut is None else p_mut
    n_elite = min(n_elite, P)
    X = _initial_bits(R, P)
    F = R.initial(X, [R.prio] * P)

    def pick():
        cand = rng.integers(P, size=tournament)
        return cand[np.argmax(F[cand])]

    if not R.out():
        while True:
            before = R.dec.evals
            elite = np.argsort(-F, kind="stable")[:n_elite]
            children = np.zeros((P - n_elite, n), bool)
            for c in range(P - n_elite):
                child = X[pick()].copy()
                if rng.random() < p_cx:
                    mask = rng.random(n) < 0.5
                    child[mask] = X[pick()][mask]
                child ^= rng.random(n) < pm
                if rng.random() < p_swap:
                    _swap(rng, child, n0)
                children[c] = child
            X = np.vstack([X[elite], children])
            F = np.r_[F[elite], np.full(P - n_elite, -np.inf)]
            for i in range(n_elite, P):
                if R.out():
                    break
                F[i] = R.evaluate(X[i], R.prio)
            if R.end_iteration(before):
                break
    return R.result()


METHODS = dict(bpso_v=bpso_v_plan, rk_pso=rk_pso_plan, ga2=ga2_plan)
