"""Synthetic Bakken-like instance generator (v3.1):

- rigs, crews and capital calibrated so that each binds in some year; field range 3 miles;
- ownership and parents correlated with rock quality (core acreage sits in the sweet spots, knob core_mix);
- the field (mu, Sigma) is the posterior given production already observed at parented unit-benches
  (the operator has learned from its parents), with the prior kept for reference;
- costs discounted at spud, revenue at first production (nine-month lag);
- every economic constant lives in the instance dict so it can be perturbed for paired robustness scoring.

Geometry: an nx x ny grid of one-mile sections; unit u = (column i, pair j) covers sections
(i, 2j) and (i, 2j+1); its in-line neighbor n(u) = (i, j+1). Each section-bench has L lanes.
"""
import copy

import numpy as np
from scipy.spatial.distance import cdist

L_LANES = 6                        # lanes per section-bench (about 880-ft spacing across a mile)
NB = 2                             # benches: 0 = Middle Bakken, 1 = Three Forks 1
T = 4                              # planning years
LENGTHS = (2, 3, 4)                # lateral lengths, miles
PAD_COST = 2.5                     # $M per unit pad
ROYALTY = 0.20
RF = 0.05                          # risk-free discount rate (strip valuation)
DISC = (1 + RF) ** -(np.arange(T) + 0.75)      # revenue: spud-to-sales lag of about nine months
DISC_COST = (1 + RF) ** -np.arange(T)          # costs at spud
F0_HEDGE = 0.60                    # hedgeable share of a committed (year-0) well's revenue within the liquid tenor
A_BENCH = np.array([1.0, 1.25])    # type-curve factor loading by bench
A_LEN = {2: 1.0, 3: 1.10, 4: 1.25} # type-curve factor loading by lateral length (long-lateral uplift is uncertain)
NU_TERM = 0.5                      # terminal option value share for undeveloped unit-benches
TERM_YEARS = 1.5                   # terminal inventory credited up to this many years of rig capacity

DEFAULTS = dict(
    eur_mult={2: 1.00, 3: 1.40, 4: 1.55},     # lateral-length EUR multiplier
    cost={2: 7.0, 3: 8.5, 4: 10.0},           # $M per well
    rig_days={2: 25, 3: 32, 4: 40},           # rig-days per well
    k_by_len={2: (2, 3, 4, 5, 6), 3: (2, 3, 4, 5), 4: (2, 3, 4)},
    h_k={2: 1.00, 3: 0.96, 4: 0.90, 5: 0.83, 6: 0.75},   # interference: EUR multiplier for k new wells per unit-bench
    parent_pen=0.05,                          # EUR loss per existing parent well in the same bench (floor 0.70)
    cross_bench_pen=0.5,                      # weight of parents in the other bench of the same unit
    sig_theta=0.15, sig_p=0.25, sig_eps=0.40, # log-sd: type-curve level, long-dated price, per-well noise
    range_mi=3.0, bench_corr=0.7, sd_bench=(3.0, 2.4),
    f_hedge=(F0_HEDGE, 0.0, 0.0, 0.0),        # only committed wells are pre-hedged; recourse wells see their node's price
    pdp_hedge=0.5,                            # share of the producing base's remaining revenue inside the liquid tenor
    core_mix=0.5,                             # weight of rock quality (vs a spatial draw) in ownership and parent placement
    condition_on_parents=True,                # field covariance = posterior given parent production
    condition_mean=False,                     # keep the prior mean (see condition_field)
    mu_scale=1.0,                             # multiplies the revenue map (0.75: a map net of operating costs and production taxes)
    codevelop=False,                          # a unit's two benches are developed in the same year or one bench only (co-development variant)
)


def gh_nodes(n, sig):
    z, w = np.polynomial.hermite_e.hermegauss(n)
    return np.exp(sig * z - sig**2 / 2), w / w.sum()


def field_cov(X, range_mi, bench_corr, sd_bench):
    d = cdist(X, X) / range_mi
    Cs = (1 + np.sqrt(3) * d) * np.exp(-np.sqrt(3) * d)                       # Matern 3/2
    sb = np.asarray(sd_bench)
    return Cs, np.kron(Cs, np.outer(sb, sb) * np.array([[1, bench_corr], [bench_corr, 1]]))


def build_nodes(I, n_nodes):
    th, wth = gh_nodes(n_nodes, I["sig_theta"])
    pr, wpr = gh_nodes(n_nodes, I["sig_p"])
    I["theta"], I["price"], I["prob"] = np.repeat(th, n_nodes), np.tile(pr, n_nodes), np.outer(wth, wpr).ravel()
    I["n_nodes"] = n_nodes
    return I


def condition_field(mu_prior, Sigma_prior, parents, sig_eps, rng, condition_mean=False):
    """Posterior (mu, Sigma) of the field given noisy observations at parented unit-benches."""
    nub = Sigma_prior.shape[0]
    g_true = np.linalg.cholesky(Sigma_prior + 1e-9 * np.eye(nub)) @ rng.standard_normal(nub)
    P = np.flatnonzero(parents.ravel() > 0)
    if len(P) == 0:
        return mu_prior.copy(), Sigma_prior.copy(), g_true
    npar = parents.ravel()[P].astype(float)
    D = (sig_eps * mu_prior.ravel()[P]) ** 2 / npar                           # noise of the per-ub average
    obs = g_true[P] + rng.standard_normal(len(P)) * np.sqrt(D)
    S_PP = Sigma_prior[np.ix_(P, P)] + np.diag(D)
    Kg = Sigma_prior[:, P] @ np.linalg.inv(S_PP)
    # The posterior covariance is what the risk term needs (less uncertainty near parents). The posterior mean is
    # kept at the prior unless condition_mean is set: a random draw of the parents' outcomes would shift every
    # instance's economics by a seed-dependent amount and make instances incomparable.
    mu_post = (mu_prior.ravel() + Kg @ obs) if condition_mean else mu_prior.ravel().copy()
    Sigma_post = Sigma_prior - Kg @ Sigma_prior[P, :]
    Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)
    return mu_post.reshape(mu_prior.shape), Sigma_post, g_true


def make_instance(nx, ny, seed=0, own_frac=0.65, deadline_frac=0.25, parent_frac=0.35, rig_mult=2.5, n_nodes=3, **over):
    assert ny % 2 == 0
    rng = np.random.default_rng(seed)
    I = copy.deepcopy(DEFAULTS); I.update(over)
    npair = ny // 2
    U = nx * npair
    uu = np.arange(U)
    col, pair = uu // npair, uu % npair
    secs = np.stack([col * ny + 2 * pair, col * ny + 2 * pair + 1], 1)           # sections of each unit (south, north)
    nbr = np.where(pair + 1 < npair, uu + 1, -1)                                  # in-line neighbor to the north
    X = np.c_[col + 0.5, 2 * pair + 1.0]                                          # unit centers, miles
    # mean net revenue per well at reference price by bench ($M): two sweet spots, TF = 0.8 MB
    c1, c2 = np.array([0.35 * nx, 0.6 * ny]), np.array([0.75 * nx, 0.25 * ny])
    mu_mb = 8 + 7 * np.exp(-((X - c1) ** 2).sum(1) / (2 * (0.25 * nx) ** 2)) + 4 * np.exp(-((X - c2) ** 2).sum(1) / (2 * (0.18 * nx) ** 2))
    mu = np.stack([mu_mb, 0.8 * mu_mb], 1) * I.get("mu_scale", 1.0)              # U x NB (prior); mu_scale: net-of-opex variant
    Cs, Sigma = field_cov(X, I["range_mi"], I["bench_corr"], I["sd_bench"])
    # ownership and parents: a mix of rock quality and a clustered spatial draw (core acreage in the sweet spots)
    zs = np.linalg.cholesky(Cs + 1e-8 * np.eye(U)) @ rng.standard_normal(U)
    zs = (zs - zs.mean()) / zs.std()
    mq = (mu_mb - mu_mb.mean()) / mu_mb.std()
    zf = I["core_mix"] * mq + (1 - I["core_mix"]) * zs
    owned = zf > np.quantile(zf, 1 - own_frac)
    WI = np.where(owned, rng.uniform(0.5, 1.0, U), 0.75)                          # acquired units come at 75% WI
    NRI = WI * (1 - ROYALTY)
    mu_rank = (mu_mb - mu_mb.min()) / (mu_mb.max() - mu_mb.min() + 1e-9)
    p_par = np.clip(parent_frac * (0.5 + mu_rank), 0, 1)
    parents = np.zeros((U, NB), int)
    has_parent = owned & (rng.random(U) < p_par)
    parents[has_parent, 0] = rng.integers(2, 5, has_parent.sum())                 # 2-4 Middle Bakken parents
    tf = has_parent & (rng.random(U) < 0.5)
    parents[tf, 1] = rng.integers(1, 3, tf.sum())                                 # 1-2 Three Forks parents in half of them
    deadline = np.full(U, T - 1)
    fringe = owned & ~has_parent & (rng.random(U) < deadline_frac)
    deadline[fringe] = rng.integers(0, 2, fringe.sum())
    # acreage packages: 2x2 blocks of unowned units, priced by quality ($/acre 1,000-5,400)
    blk = (col // 2) * 1000 + (pair // 2)
    pkg_ids = {p: i for i, p in enumerate(np.unique(blk[~owned]))}
    pkg_of = np.array([pkg_ids[p] if not o else -1 for p, o in zip(blk, owned)])
    acre_price = 1000 + 400 * (mu_mb - 8)
    pkg_cost = np.zeros(len(pkg_ids))
    for u in np.flatnonzero(~owned):
        pkg_cost[pkg_of[u]] += 1280 * acre_price[u] / 1e6
    # capacities per year: wpy two-mile wells of rig time; crews in lateral-miles; capital set to bind
    wpy = rig_mult * 0.30 * U
    cap = dict(rig=np.full(T, wpy * I["rig_days"][2]), crew=np.full(T, wpy * 2.0 * 1.15),
               capital=np.full(T, 0.9 * wpy * I["cost"][2] * 0.8 + PAD_COST * 0.15 * U), acq=0.6 * U)
    # the field the operator plans with: posterior given its parents' production
    mu_prior, Sigma_prior = mu.copy(), Sigma.copy()
    g_true = None
    if I["condition_on_parents"]:
        mu, Sigma, g_true = condition_field(mu_prior, Sigma_prior, parents, I["sig_eps"], rng, I.get("condition_mean", False))
    I.update(dict(nx=nx, ny=ny, U=U, secs=secs, nbr=nbr, X=X, mu=mu, Sigma=Sigma, mu_prior=mu_prior, Sigma_prior=Sigma_prior,
                  g_true=g_true, owned=owned, WI=WI, NRI=NRI, parents=parents, deadline=deadline, pkg_of=pkg_of, pkg_cost=pkg_cost,
                  cap=cap, n_sec=nx * ny, seed=seed, own_frac=own_frac, deadline_frac=deadline_frac, parent_frac=parent_frac,
                  rig_mult=rig_mult))
    return build_nodes(I, n_nodes)


def perturb(I, mu_bias=1.0, h_slope=1.0, range_mult=1.0, eur4=None, sig_p=None, sig_theta=None, n_nodes=None, cv_field=False):
    """A copy of the instance with a perturbed 'truth' for paired robustness scoring of fixed plans."""
    J = copy.deepcopy(I)
    J["mu"] = J["mu"] * mu_bias
    J["h_k"] = {k: max(0.3, 1 - h_slope * (1 - v)) for k, v in I["h_k"].items()}
    if range_mult != 1.0:
        J["range_mi"] = I["range_mi"] * range_mult
        _, Sp = field_cov(I["X"], J["range_mi"], I["bench_corr"], I["sd_bench"])
        sc = np.sqrt(np.maximum(np.diag(I["Sigma"]), 1e-12) / np.diag(Sp))
        J["Sigma"] = Sp * np.outer(sc, sc)                                     # same marginal variances as the planning field
    if cv_field:                                                               # constant coefficient of variation: sd proportional to mu
        sd_old = np.sqrt(np.maximum(np.diag(J["Sigma"]), 1e-12)); sd_new = sd_old.mean() / J["mu"].mean() * J["mu"].ravel()
        s = sd_new / sd_old
        J["Sigma"] = J["Sigma"] * np.outer(s, s)
    if eur4 is not None:
        J["eur_mult"] = dict(I["eur_mult"]); J["eur_mult"][4] = eur4
    if sig_p is not None: J["sig_p"] = sig_p
    if sig_theta is not None: J["sig_theta"] = sig_theta
    if sig_p is not None or sig_theta is not None or n_nodes is not None:
        build_nodes(J, n_nodes or I["n_nodes"])
    return J


def single_node(I, k):
    """The instance restricted to one node of the tree (for perfect-information bounds)."""
    J = copy.deepcopy(I)
    J["theta"], J["price"], J["prob"] = I["theta"][[k]], I["price"][[k]], np.array([1.0])
    J["n_nodes"] = 1
    return J
