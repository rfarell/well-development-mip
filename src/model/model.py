"""Option tables, node coefficients, exact evaluation, feasibility and an incremental plan builder for
the planning model (Section S5 of the Supplementary Materials of the paper), v3.1:

- long-lateral economics are length-weighted over the units a lateral traverses, as sums of rock x interest
  (revenue sum_i alpha_i mu_i NRI_i), with the field exposure split over the traversed unit-benches;
- parents in both benches interfere (same bench fully, the other bench with weight cross_bench_pen);
- costs are discounted at spud, revenue at first production; crews in lateral-miles;
- only committed (year-0) wells are pre-hedged (f = (f0, 0, 0, 0));
- terminal inventory value: capacity-feasible options only, credited to at most TERM_YEARS of rig capacity
  (a plan-independent credited set), gated for fringe units by a held indicator, and netted for lanes that
  a neighbor's long lateral consumes (credit fraction iota = min(own - dev, 1 - consumed));
- the producing base (parents) enters every node as a background cash flow and field exposure;
- gaussian_wells=True replaces the exact per-well CGF by its second-cumulant truncation (the seed's treatment);
- v3.2 (conventions study): an optional I["nu_term"] overrides the terminal credit share instance.NU_TERM
  (instances without the key behave exactly as before); the hedge share is already I["f_hedge"].
- v3.3 (2026-10-04): a long lateral into an owned fringe neighbor after that neighbor's
  deadline requires the neighbor to be held (Hold rows cover host and neighbor); the terminal credit is the best
  two-mile option in the unit's own lanes, net of the pad, with field exposure only where the credit is positive;
  the producing base can be restricted to the operator's own parents (base_parents, pdp_remain).

A plan is a dict: z0 (n0,) first-stage options (year 0); zk (K, n1) recourse options (years >= 1)
per node; y (G,) packages; w0 (U,) pads built in year 0; wk (K, U, T-1) pads built in years >= 1.
"""
import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp

from instance import A_BENCH, A_LEN, DISC, DISC_COST, L_LANES, LENGTHS, NB, NU_TERM, PAD_COST, T, TERM_YEARS

PDP_REMAIN = 0.6          # remaining PV of a parent well's revenue, as a share of a new well's mean revenue
MODEL_VERSION = "v3.3"    # stamped into every run file; the reporting scripts reject runs of another version


class Model:
    def __init__(self, I, lam, deck=1.0, gaussian_wells=False):
        self.I, self.lam, self.deck, self.gaussian_wells = I, lam, deck, gaussian_wells
        U, K = I["U"], len(I["prob"])
        self.U, self.K = U, K
        self.nub, self.nsb = U * NB, I["n_sec"] * NB
        # ---- lane capacity per section-bench (parents occupy lanes) ---------------------------------
        self.lane_cap = np.full(self.nsb, float(L_LANES))
        for u in range(U):
            for b in range(NB):
                for sec in I["secs"][u]:
                    self.lane_cap[sec * NB + b] -= I["parents"][u, b]
        # ---- enumerate options (u, b, len, k, t) with their traversed sections and units ------------
        rows, trav = [], []
        for u in range(U):
            for b in range(NB):
                for ln in LENGTHS:
                    n_u = I["nbr"][u]
                    if ln > 2 and n_u < 0:
                        continue
                    secs = list(I["secs"][u]) + ([I["secs"][n_u][0]] if ln >= 3 else []) + ([I["secs"][n_u][1]] if ln == 4 else [])
                    units = [(u, 1.0)] if ln == 2 else ([(u, 2 / 3), (n_u, 1 / 3)] if ln == 3 else [(u, 0.5), (n_u, 0.5)])
                    cap = min(self.lane_cap[s * NB + b] for s in secs)
                    for k in I["k_by_len"][ln]:
                        if k > cap:
                            continue
                        for t in range(T):
                            rows.append((u, b, ln, k, t)); trav.append((secs, units))
        R = np.array(rows)
        self.o_u, self.o_b, self.o_len, self.o_k, self.o_t = R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4]
        self.o_ub = self.o_u * NB + self.o_b
        self.n = len(R)
        self.idx0 = np.flatnonzero(self.o_t == 0)
        self.idx1 = np.flatnonzero(self.o_t >= 1)
        self.n0, self.n1 = len(self.idx0), len(self.idx1)
        self.pos1 = np.full(self.n, -1); self.pos1[self.idx1] = np.arange(self.n1)
        self.pos0 = np.full(self.n, -1); self.pos0[self.idx0] = np.arange(self.n0)
        # ---- lane usage matrix: option o uses k lanes in each traversed section-bench ----------------
        r_idx, c_idx, vals = [], [], []
        for o, (secs, units) in enumerate(trav):
            for s in secs:
                r_idx.append(s * NB + self.o_b[o]); c_idx.append(o); vals.append(float(self.o_k[o]))
        self.Lanes = sp.csr_matrix((vals, (r_idx, c_idx)), shape=(self.nsb, self.n))
        self._Lcsc = self.Lanes.tocsc()
        # ---- length-weighted economics ($M per well): revenue sum_i alpha_i mu_i NRI_i ------------------
        cb = I.get("cross_bench_pen", 0.5)
        mun_o = np.zeros(self.n); wi_o = np.zeros(self.n); pen_o = np.ones(self.n)
        ex_r, ex_c, ex_v = [], [], []
        for o, (secs, units) in enumerate(trav):
            b = self.o_b[o]
            for (uu, wt) in units:
                mun_o[o] += wt * I["mu"][uu, b] * I["NRI"][uu]; wi_o[o] += wt * I["WI"][uu]
                npar = I["parents"][uu, b] + cb * I["parents"][uu, 1 - b]
                pen_o[o] = min(pen_o[o], max(0.70, 1 - I["parent_pen"] * npar))
        hk = np.array([I["h_k"][k] for k in self.o_k])
        mult = np.array([I["eur_mult"][l] for l in self.o_len])
        base = DISC[self.o_t] * mult * hk * pen_o                            # per well, before rock, interest and factors
        self.R = base * mun_o                                               # mean net revenue per well at reference price
        for o, (secs, units) in enumerate(trav):
            for (uu, wt) in units:
                ex_r.append(uu * NB + self.o_b[o]); ex_c.append(o); ex_v.append(self.o_k[o] * base[o] * wt * I["NRI"][uu])
        self.Wexp = sp.csr_matrix((ex_v, (ex_r, ex_c)), shape=(self.nub, self.n))   # option exposure over traversed unit-benches
        self.C = DISC_COST[self.o_t] * wi_o * self.o_k * np.array([I["cost"][l] for l in self.o_len])
        self.pad_cost = np.outer(I["WI"], DISC_COST) * PAD_COST             # U x T, discounted at spud
        self.rigdays = self.o_k * np.array([I["rig_days"][l] for l in self.o_len], float)
        self.lat_miles = self.o_k * self.o_len.astype(float)
        self.wells = self.o_k.astype(float)
        # ---- node scaling: theta^(a_b * a_len) * (f_t + (1 - f_t) P_k) -------------------------------------
        th, pr = I["theta"], I["price"]
        f = np.asarray(I["f_hedge"])[self.o_t]
        aexp = (A_BENCH[self.o_b] * np.array([A_LEN[l] for l in self.o_len]))[:, None]
        self.scale = (th[None, :] ** aexp) * (f[:, None] + (1 - f[:, None]) * pr[None, :])
        self.Rk = deck * self.R[:, None] * self.scale                       # n x K, per well
        self.revk = self.o_k[:, None] * self.Rk                             # option revenue per node
        # ---- per-well CGF table K(a) = log E exp(-a (eps - 1)), eps lognormal(sig_eps) ------------------
        ze, we = np.polynomial.hermite_e.hermegauss(80)
        se = I["sig_eps"]
        eps = np.exp(se * ze - se**2 / 2) - 1
        self.a_grid = np.linspace(0, 30, 30001)
        self.K_grid = logsumexp(-self.a_grid[:, None] * eps[None, :] + np.log(we / we.sum())[None, :], axis=1)
        assert lam * self.Rk.max() < self.a_grid[-1], "CGF table range exceeded: extend a_grid"
        s2 = np.exp(se**2) - 1
        if gaussian_wells:
            self.cgf_k = self.o_k[:, None] * (lam * self.Rk) ** 2 * s2 / 2                        # second-cumulant truncation
        else:
            self.cgf_k = self.o_k[:, None] * np.interp(lam * self.Rk, self.a_grid, self.K_grid)   # n x K, exact
        self.var_idio_k = self.o_k[:, None] * self.Rk**2 * s2
        # ---- ownership status of unit-benches: 0 owned (held beyond horizon), 1 owned fringe, 2 unowned ----
        self.fringe_u = I["deadline"] < T - 1
        self.kind = np.repeat(np.where(I["owned"], np.where(self.fringe_u, 1, 0), 2), NB)
        self.G = len(I["pkg_cost"])
        self.pkg = I["pkg_of"][self.o_u]
        self.ub_owner_ok = I["owned"][self.o_u]
        self.post = self.o_t > I["deadline"][self.o_u]                                   # options after the host's lease deadline
        self.hold_ok = ~self.post
        self.dead = np.zeros(self.n, bool)
        # v3.3 (2026-10-04): a 3- or 4-mile lateral reaches into the in-line neighbor; if that neighbor is an owned fringe
        # unit and the option comes after the neighbor's deadline, the neighbor must have been held by a well of its own
        # by its deadline (the same rule as for the host unit). Before v3.3 only the host's deadline was checked.
        nbr_u = I["nbr"][self.o_u]
        has_nbr = (self.o_len >= 3) & (nbr_u >= 0)
        nb = np.maximum(nbr_u, 0)
        self.nbr_post = has_nbr & I["owned"][nb] & (self.o_t > I["deadline"][nb])
        self.nbr_hold_unit = np.where(self.nbr_post, nb, -1)
        post_idx = np.flatnonzero(self.post); nbr_post_idx = np.flatnonzero(self.nbr_post)
        pairs = [(o, self.o_u[o]) for o in post_idx] + [(o, self.nbr_hold_unit[o]) for o in nbr_post_idx]
        self.post_rows = np.arange(len(pairs))
        r_idx, c_idx, vals = [], [], []
        for r, (o, unit) in enumerate(pairs):
            r_idx.append(r); c_idx.append(o); vals.append(1.0)
            for o2 in np.flatnonzero((self.o_u == unit) & self.hold_ok):
                r_idx.append(r); c_idx.append(o2); vals.append(-1.0)
        self.Hold = sp.csr_matrix((vals, (r_idx, c_idx)), shape=(len(pairs), self.n))
        hu = self.hold_ok & self.fringe_u[self.o_u]
        self.HoldU = sp.csr_matrix((np.ones(hu.sum()), (self.o_u[hu], np.flatnonzero(hu))), shape=(U, self.n))
        nbr_u = I["nbr"][self.o_u]
        self.nbr_pkg = np.where((self.o_len >= 3) & (nbr_u >= 0) & ~I["owned"][np.maximum(nbr_u, 0)], I["pkg_of"][np.maximum(nbr_u, 0)], -1)
        # ---- terminal inventory value: capacity-feasible year-(T-1) options, capped at TERM_YEARS of rig time ----
        # v3.2: the credit share nu is I["nu_term"] when the instance sets it, else instance.NU_TERM (unchanged
        # default). The credited set is the same for every nu > 0 (its order and budget do not scale with nu);
        # nu = 0 credits nothing.
        nu = float(I.get("nu_term", NU_TERM))
        self.nu_term = nu
        # v3.3 (2026-10-04): the credited configuration is the unit-bench's best two-mile option in its own lanes (a long
        # option would need the neighbor's lanes and lease, which the horizon does not check), net of the pad the undeveloped
        # unit still has to build; and no field exposure is attached in nodes where the credit is zero.
        last = (self.o_t == T - 1) & (self.o_len == 2)
        cap_ok = (self.rigdays <= I["cap"]["rig"][T - 1]) & (self.lat_miles <= I["cap"]["crew"][T - 1]) & (self.C + self.pad_cost[self.o_u, T - 1] <= I["cap"]["capital"][T - 1])
        p = I["prob"]
        tau = np.zeros((self.nub, K)); best_o = np.full(self.nub, -1)
        for ub in range(self.nub):
            cand = np.flatnonzero(last & (self.o_ub == ub) & cap_ok)
            if len(cand) == 0:
                continue
            npv = self.revk[cand] - self.C[cand][:, None] - self.pad_cost[self.o_u[cand], T - 1][:, None]
            exp_npv = npv @ p
            j = int(np.argmax(exp_npv))
            if exp_npv[j] > 0:
                tau[ub] = nu * np.maximum(0.0, npv[j]); best_o[ub] = cand[j]
        order = np.argsort(-(tau @ p) / np.where(best_o >= 0, self.rigdays[np.maximum(best_o, 0)], 1.0))
        budget, credited = TERM_YEARS * I["cap"]["rig"][T - 1], np.zeros(self.nub, bool)
        for ub in order:
            if best_o[ub] < 0 or tau[ub] @ p <= 0:
                continue
            if self.rigdays[best_o[ub]] <= budget:
                credited[ub] = True; budget -= self.rigdays[best_o[ub]]
        self.tau = tau * credited[:, None]
        self.tau_best = np.where(credited, best_o, -1)
        self.Texp = np.zeros((K, self.nub, self.nub))                       # column ub = exposure vector of ub's credited option
        for ub in np.flatnonzero(credited):
            o = best_o[ub]
            col = self.Wexp[:, o].toarray().ravel()
            for k in range(K):
                if self.tau[ub, k] > 0:                                          # exposure only where the credit is positive
                    self.Texp[k][:, ub] = nu * col * self.scale[o, k]
        # lanes of a credited unit-bench consumed by other units' long laterals: credit fraction lost per option
        r_idx, c_idx, vals = [], [], []
        for o, (secs, units) in enumerate(trav):
            if self.o_len[o] == 2:
                continue
            for (uu, wt) in units[1:]:                                       # the neighbor(s), not the home unit
                ub = uu * NB + self.o_b[o]
                if credited[ub]:
                    r_idx.append(ub); c_idx.append(o); vals.append(min(1.0, self.o_k[o] / self.o_k[best_o[ub]]))
        self.Cons = sp.csr_matrix((vals, (r_idx, c_idx)), shape=(self.nub, self.n))
        # ---- producing base (parents): background cash flow B_k and exposure b_k per node -----------------
        # The operator's base counts I["base_parents"] (its own producing wells per unit-bench; default: every parent,
        # as in the synthetic generator) at I["pdp_remain"] of a new well's mean revenue (default PDP_REMAIN). Lane
        # occupancy and interference above use every parent, whoever operates it.
        par = np.asarray(I.get("base_parents", I["parents"])).ravel().astype(float)
        self.pdp_remain = float(I.get("pdp_remain", PDP_REMAIN))
        nri_ub = np.repeat(I["NRI"], NB)
        mu_ub = I["mu"].ravel()
        ph = I["pdp_hedge"]
        pdp_scale = (th[None, :] ** np.tile(A_BENCH, U)[:, None]) * (ph + (1 - ph) * pr[None, :])   # nub x K
        self.B_k = ((par * self.pdp_remain * nri_ub * mu_ub)[:, None] * pdp_scale).sum(0)
        self.b_k = (par * self.pdp_remain * nri_ub)[:, None] * pdp_scale
        # ---- bookkeeping matrices --------------------------------------------------------------------
        o = np.arange(self.n)
        self.Mub = sp.csr_matrix((np.ones(self.n), (self.o_ub, o)), shape=(self.nub, self.n))
        self.Mubt = sp.csr_matrix((np.ones(self.n), (self.o_ub * T + self.o_t, o)), shape=(self.nub * T, self.n))
        self.Sigma = I["Sigma"]

    # ----------------------------------------------------------------------------------------------
    def empty_plan(self):
        return dict(z0=np.zeros(self.n0), zk=np.zeros((self.K, self.n1)), y=np.zeros(self.G),
                    w0=np.zeros(self.U), wk=np.zeros((self.K, self.U, T - 1)))

    def full_z(self, plan, k):
        z = np.zeros(self.n); z[self.idx0] = plan["z0"]; z[self.idx1] = plan["zk"][k]
        return z

    def own_vec(self, plan, z):
        """own_ub in {0,1}: owned non-fringe -> 1; owned fringe -> held by a well by its deadline; unowned -> package bought."""
        I = self.I
        held = (self.HoldU @ z) > 0.5
        bought = (plan["y"][np.maximum(I["pkg_of"], 0)] > 0.5) if self.G else np.zeros(self.U, bool)
        own_u = np.where(I["owned"], np.where(self.fringe_u, held, True), bought)
        return np.repeat(own_u, NB).astype(float)

    def inventory(self, plan, z):
        """Credit fraction of each unit-bench's terminal inventory: min(own - dev, 1 - lanes consumed by neighbors), in [0, 1]."""
        own = self.own_vec(plan, z)
        dev = self.Mub @ z
        return np.clip(np.minimum(own - dev, 1 - self.Cons @ z), 0.0, 1.0)

    def node_terms(self, plan, k):
        """Deterministic NPV D_k (incl. the producing base), field exposure w_k (nub,), CGF sum, idiosyncratic variance."""
        I = self.I
        z = self.full_z(plan, k)
        inv = self.inventory(plan, z)
        pads = self.pad_cost[:, 0] @ plan["w0"] + (self.pad_cost[:, 1:] * plan["wk"][k]).sum()
        D = (self.revk[:, k] - self.C) @ z - pads - I["pkg_cost"] @ plan["y"] + self.tau[:, k] @ inv + self.B_k[k]
        w = self.Wexp @ (z * self.scale[:, k]) + self.Texp[k] @ inv + self.b_k[:, k]
        return D, w, float(self.cgf_k[:, k] @ z), float(self.var_idio_k[:, k] @ z)

    def evaluate(self, plan, Sigma=None):
        """Exact CE under the model (Gaussian field, K-node tree, exact per-well CGF), with E[NPV] and lambda*sigma_W."""
        S = self.Sigma if Sigma is None else Sigma
        lam, K, p = self.lam, self.K, self.I["prob"]
        q, D, V = np.zeros(K), np.zeros(K), np.zeros(K)
        for k in range(K):
            Dk, w, cg, vi = self.node_terms(plan, k)
            fv = float(w @ S @ w)
            q[k] = -lam * Dk + 0.5 * lam**2 * fv + cg
            D[k], V[k] = Dk, fv + vi
        phi = logsumexp(q, b=p)
        enpv = float(p @ D)
        var = float(p @ V + p @ (D - enpv) ** 2)
        return dict(ce=float(-phi / lam), enpv=enpv, sigma=np.sqrt(var), lam_sigma=lam * np.sqrt(var), q=q, D=D)

    def sigma_decomposition(self, plan):
        """Variance of NPV by source: type-curve factor, long-dated price, their interaction, field, per-well."""
        I, p = self.I, self.I["prob"]
        n = I["n_nodes"]; K = self.K
        D, Fv, Iv = np.zeros(K), np.zeros(K), np.zeros(K)
        for k in range(K):
            Dk, w, cg, vi = self.node_terms(plan, k)
            D[k], Fv[k], Iv[k] = Dk, float(w @ self.Sigma @ w), vi
        P = p.reshape(n, n); Dm = D.reshape(n, n)
        pth, ppr = P.sum(1), P.sum(0)
        m = float(p @ D)
        Eth = (P * Dm).sum(1) / pth; Epr = (P * Dm).sum(0) / ppr
        v_theta = float(pth @ (Eth - m) ** 2); v_price = float(ppr @ (Epr - m) ** 2)
        v_fac = float(p @ (D - m) ** 2)
        return dict(theta=v_theta, price=v_price, interaction=v_fac - v_theta - v_price, field=float(p @ Fv), idio=float(p @ Iv),
                    total=v_fac + float(p @ Fv) + float(p @ Iv))

    def implied_tilt(self, plan):
        """Softmax node weights of the plan and the implied deck (price and type-curve tilt)."""
        ev = self.evaluate(plan)
        pi = self.I["prob"] * np.exp(ev["q"] - logsumexp(ev["q"], b=self.I["prob"]))
        return dict(pi=pi, implied_price=float(pi @ self.I["price"]), implied_theta=float(pi @ self.I["theta"]))

    def ce_mc(self, plan, n=2000, seed=123, lognormal=True, return_se=False):
        """Monte Carlo CE with a lognormal field (cv-matched) and exact lognormal per-well noise; Gaussian field if lognormal=False."""
        rng = np.random.default_rng(seed)
        Lc = np.linalg.cholesky(self.Sigma + 1e-9 * np.eye(self.nub))
        mu_ub = np.maximum(self.I["mu"].ravel(), 1e-6); sd = np.sqrt(np.maximum(np.diag(self.Sigma), 1e-12)); cv = sd / mu_ub
        se = self.I["sig_eps"]
        xs, pw = [], []
        for k in range(self.K):
            Dk, w, cg, vi = self.node_terms(plan, k)
            g = (Lc @ rng.standard_normal((self.nub, n))).T
            if lognormal:
                g = mu_ub * (np.exp(cv * g / sd - cv**2 / 2) - 1)
            z = self.full_z(plan, k)
            act = np.flatnonzero(z > 0.5)
            idio = np.zeros(n)
            if len(act):
                counts = self.o_k[act]; Rw = np.repeat(self.Rk[act, k], counts)
                eps = np.exp(se * rng.standard_normal((len(Rw), n)) - se**2 / 2) - 1
                idio = Rw @ eps
            xs.append(Dk + g @ w + idio); pw.append(np.full(n, self.I["prob"][k] / n))
        x = np.concatenate(xs); pwv = np.concatenate(pw)
        ce = float(-logsumexp(-self.lam * x, b=pwv) / self.lam)
        if not return_se:
            return ce
        bs = []
        for j in range(10):
            sel = np.concatenate([np.arange(j * n // 10, (j + 1) * n // 10) + k * n for k in range(self.K)])
            bs.append(float(-logsumexp(-self.lam * x[sel], b=pwv[sel] * 10) / self.lam))
        return ce, float(np.std(bs, ddof=1) / np.sqrt(10))

    # ----------------------------------------------------------------------------------------------
    def feasible(self, plan, verbose=False):
        I, U, K = self.I, self.U, self.K
        bad = []
        ybuy = (plan["y"][np.maximum(self.pkg, 0)] > 0.5) if self.G else np.zeros(self.n, bool)
        ynbr = (plan["y"][np.maximum(self.nbr_pkg, 0)] > 0.5) if self.G else np.zeros(self.n, bool)
        for k in range(K):
            z = self.full_z(plan, k)
            if len(self.post_rows) and np.any(self.Hold @ z > 1e-9): bad.append(f"drill-to-hold k{k}")
            if np.any((self.nbr_pkg >= 0) & (z > 0.5) & ~ynbr): bad.append(f"neighbor-ownership k{k}")
            if np.any(self.Mub @ z > 1 + 1e-9): bad.append(f"ub k{k}")
            if np.any(self.Lanes @ z > self.lane_cap + 1e-9): bad.append(f"lanes k{k}")
            nonown = (~self.ub_owner_ok) & (z > 0.5)
            if np.any(nonown & ~ybuy): bad.append(f"ownership k{k}")
            pads = np.c_[plan["w0"], plan["wk"][k]]
            if np.any(pads.sum(1) > 1 + 1e-9): bad.append(f"pads>1 k{k}")
            cum = np.cumsum(pads, 1)
            need = (self.Mubt @ z).reshape(U, NB, T).max(1)
            if np.any((need > 0.5) & (cum < 0.5)): bad.append(f"pad-before-well k{k}")
            for t in range(T):
                sel = z * (self.o_t == t)
                if self.rigdays @ sel > I["cap"]["rig"][t] + 1e-6: bad.append(f"rig k{k} t{t}")
                if self.lat_miles @ sel > I["cap"]["crew"][t] + 1e-6: bad.append(f"crew k{k} t{t}")
                if self.C @ sel + self.pad_cost[:, t] @ pads[:, t] > I["cap"]["capital"][t] + 1e-6: bad.append(f"capital k{k} t{t}")
        if I["pkg_cost"] @ plan["y"] > I["cap"]["acq"] + 1e-6: bad.append("acq budget")
        if I.get("codevelop", False):                                            # v3.3 variant: both benches in one year, or one bench
            for k in range(K):
                z = self.full_z(plan, k)
                yr = np.full((U, NB), -1)
                for o in np.flatnonzero(z > 0.5):
                    yr[self.o_u[o], self.o_b[o]] = self.o_t[o]
                both = (yr >= 0).all(1)
                if np.any(both & (yr[:, 0] != yr[:, 1])): bad.append(f"codevelop k{k}")
        if verbose and bad: print("infeasible:", bad[:8])
        return len(bad) == 0


class PlanBuilder:
    """Incremental construction with feasibility checks (for heuristics, rounding and repair).

    First-stage options (t = 0) apply in every node; recourse options (t >= 1) apply in one node.
    Pads are added automatically with the first well in a unit and charged to that year's capital.
    """
    def __init__(self, M):
        self.M, I = M, M.I
        K, U = M.K, M.U
        self.plan = M.empty_plan()
        self.used_ub = np.zeros((K, M.nub), bool)
        self.dev_year = np.full((K, M.nub), -1)
        self.lanes = np.zeros((K, M.nsb))
        self.rig = np.zeros((K, T)); self.crew = np.zeros((K, T)); self.cap = np.zeros((K, T))
        self.pad_year = np.full((K, U), -1)
        self.acq = 0.0

    def _rows(self, o):
        Lc = self.M._Lcsc
        return Lc.indices[Lc.indptr[o]:Lc.indptr[o + 1]]

    def nodes_of(self, o, k=None):
        return range(self.M.K) if self.M.o_t[o] == 0 else [k]

    def owned_ok(self, o):
        M = self.M
        return M.ub_owner_ok[o] or (M.pkg[o] >= 0 and self.plan["y"][M.pkg[o]] > 0.5)

    def can_add(self, o, k=None):
        M, I = self.M, self.M.I
        if not self.owned_ok(o):
            return False
        if M.nbr_pkg[o] >= 0 and self.plan["y"][M.nbr_pkg[o]] < 0.5:
            return False
        t, u, ub = M.o_t[o], M.o_u[o], M.o_ub[o]
        rows = self._rows(o)
        for kk in self.nodes_of(o, k):
            if self.used_ub[kk, ub]:
                return False
            if M.post[o] and not (0 <= self.pad_year[kk, u] <= I["deadline"][u]):
                return False
            if M.nbr_post[o]:                                                    # v3.3: the fringe neighbor must be held too
                nu = M.nbr_hold_unit[o]
                if not (0 <= self.pad_year[kk, nu] <= I["deadline"][nu]):
                    return False
            if I.get("codevelop", False):                                        # v3.3 variant: the other bench, if developed, in the same year
                ub2 = u * NB + (1 - M.o_b[o])
                if self.used_ub[kk, ub2] and self.dev_year[kk, ub2] != t:
                    return False
            if np.any(self.lanes[kk, rows] + M.o_k[o] > M.lane_cap[rows] + 1e-9):
                return False
            padc = 0.0 if 0 <= self.pad_year[kk, u] <= t else M.pad_cost[u, t]
            if self.pad_year[kk, u] > t:
                return False
            if self.rig[kk, t] + M.rigdays[o] > I["cap"]["rig"][t] + 1e-9: return False
            if self.crew[kk, t] + M.lat_miles[o] > I["cap"]["crew"][t] + 1e-9: return False
            if self.cap[kk, t] + M.C[o] + padc > I["cap"]["capital"][t] + 1e-9: return False
        return True

    def add(self, o, k=None):
        M = self.M
        t, u, ub = M.o_t[o], M.o_u[o], M.o_ub[o]
        rows = self._rows(o)
        for kk in self.nodes_of(o, k):
            self.used_ub[kk, ub] = True; self.dev_year[kk, ub] = t
            self.lanes[kk, rows] += M.o_k[o]
            self.rig[kk, t] += M.rigdays[o]; self.crew[kk, t] += M.lat_miles[o]; self.cap[kk, t] += M.C[o]
            if self.pad_year[kk, u] < 0:
                self.pad_year[kk, u] = t
                self.cap[kk, t] += M.pad_cost[u, t]
                if t == 0: self.plan["w0"][u] = 1
                else: self.plan["wk"][kk, u, t - 1] = 1
        if t == 0: self.plan["z0"][M.pos0[o]] = 1
        else: self.plan["zk"][k, M.pos1[o]] = 1

    def can_buy(self, g):
        return self.plan["y"][g] < 0.5 and self.acq + self.M.I["pkg_cost"][g] <= self.M.I["cap"]["acq"] + 1e-9

    def buy(self, g):
        self.plan["y"][g] = 1; self.acq += self.M.I["pkg_cost"][g]
