"""The lower-margin (net-economics) control scales the prior mean by 0.75. The generator conditions the field on the parents'
production with an observation noise proportional to the mean, so the posterior covariance the plans face falls too (only the
prior covariance is unchanged). This script measures how much of the control's result that explains, without any solver:

  - the posterior covariance of each control instance against the same instance at mu_scale = 1;
  - the control's two stored plans (the strip plan rn and the risk-averse plan ce) re-scored with the mean scaled but the
    posterior covariance frozen at the mu_scale = 1 value. The strip plan does not depend on the covariance, so the re-scored
    gain is a lower bound on the gain an optimizer would find under the frozen covariance; the run's upper bound less the
    re-scored strip plan is an upper bound on it, because the frozen covariance dominates the control's in the Loewner order.

Usage: rescore_net_frozen_cov.py [--paper ../../paper]     (numpy only; reads results/replicates/net_s*_r100.json)
Writes results/net_frozen_cov_rescore.json and, with --paper, sections/results_numbers_netcov.tex (macros \\netcov...).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from instance import make_instance, T
from model import Model

assert "gurobipy" not in sys.modules
HERE = os.path.dirname(os.path.abspath(__file__))


def dense(M, s):
    ix = lambda v: np.asarray(v, dtype=np.intp)  # noqa: E731
    q = M.empty_plan()
    q["z0"][ix(s["z0"])] = 1.0; q["y"][ix(s["y"])] = 1.0; q["w0"][ix(s["w0"])] = 1.0
    for k in range(M.K):
        q["zk"][k, ix(s["zk"][k])] = 1.0
        w = np.zeros(M.U * (T - 1)); w[ix(s["wk"][k])] = 1.0
        q["wk"][k] = w.reshape(M.U, T - 1)
    return q


def boot(v, n=4000, seed=0):
    v = np.asarray(v, float); idx = np.random.default_rng(seed).integers(len(v), size=(n, len(v)))
    b = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", default=None)
    args = ap.parse_args()
    rows = []
    for fn in sorted(glob.glob(os.path.join(HERE, "results", "replicates", "net_s*_r100.json"))):
        d = json.load(open(fn)); r = d["runs"][0]; kn = dict(r["knobs"]); lam = r["lam"]
        assert kn.get("mu_scale") == 0.75, kn
        Ic = make_instance(d["nx"], d["ny"], seed=r["seed"], **kn)                      # the control instance as run
        Ib = make_instance(d["nx"], d["ny"], seed=r["seed"], **dict(kn, mu_scale=1.0))   # the same draws at the base mean
        Sc, Sb = Ic["Sigma"], Ib["Sigma"]
        assert np.array_equal(Ic["Sigma_prior"], Ib["Sigma_prior"]) and np.array_equal(Ic["parents"], Ib["parents"])
        assert np.linalg.eigvalsh(Sb - Sc).min() > -1e-8                              # frozen covariance dominates the control's
        par = Ic["parents"].ravel() > 0
        M = Model(Ic, lam)
        zN, zA = dense(M, r["plans"]["rn"]), dense(M, r["plans"]["ce"])
        aN, aA = M.evaluate(zN), M.evaluate(zA)                                         # as run
        fN, fA = M.evaluate(zN, Sigma=Sb), M.evaluate(zA, Sigma=Sb)                     # mean scaled, posterior covariance frozen
        assert abs(aN["ce"] - r["methods"]["rn"]["ce"]) < 1e-6 and abs(aA["ce"] - r["methods"]["ce"]["ce"]) < 1e-6, "decoding does not reproduce the run"
        E = aN["enpv"]
        rows.append(dict(seed=r["seed"], E=E, noise_ratio=0.75 ** 2,
                         var_ratio_parented=float(np.diag(Sc)[par].sum() / np.diag(Sb)[par].sum()),
                         var_ratio_other=float(np.diag(Sc)[~par].sum() / np.diag(Sb)[~par].sum()),
                         trace_ratio=float(np.trace(Sc) / np.trace(Sb)),
                         gain_as_run=100 * (aA["ce"] - aN["ce"]) / E, gain_same_plans_frozen=100 * (fA["ce"] - fN["ce"]) / E,
                         gain_upper_frozen=100 * (r["methods"]["ce"]["ce_ub"] - fN["ce"]) / E,
                         regime_as_run=aN["lam_sigma"], regime_frozen=fN["lam_sigma"]))
        print(f"seed {r['seed']}: gain as run {rows[-1]['gain_as_run']:.3f}%, same plans under the frozen covariance {rows[-1]['gain_same_plans_frozen']:.3f}%, "
              f"upper bound {rows[-1]['gain_upper_frozen']:.3f}%; posterior variance at parented unit-benches x{rows[-1]['var_ratio_parented']:.3f}", flush=True)
    col = lambda k: np.array([x[k] for x in rows])  # noqa: E731
    ga, gf = col("gain_as_run"), col("gain_same_plans_frozen")
    S = dict(n=len(rows), gain_as_run=boot(ga), gain_same_plans_frozen=boot(gf), gain_diff=boot(gf - ga), gain_upper_frozen=float(col("gain_upper_frozen").mean()),
             seeds_up=int((gf > ga).sum()), var_ratio_parented=float(col("var_ratio_parented").mean()), var_ratio_other=float(col("var_ratio_other").mean()),
             trace_ratio=float(col("trace_ratio").mean()), regime_as_run=float(col("regime_as_run").mean()), regime_frozen=float(col("regime_frozen").mean()))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, "results", "net_frozen_cov_rescore.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))
    if args.paper:
        f2 = lambda x: f"{x:.2f}"  # noqa: E731
        mac = [("N", str(S["n"])), ("NoiseRatio", f2(0.75 ** 2)), ("VarDropParented", f"{100 * (1 - S['var_ratio_parented']):.0f}"), ("VarDropOther", f"{100 * (1 - S['var_ratio_other']):.0f}"),
               ("GainAsRun", f2(S["gain_as_run"][0])), ("GainSamePlans", f2(S["gain_same_plans_frozen"][0])), ("GainSamePlansLo", f2(S["gain_same_plans_frozen"][1])),
               ("GainSamePlansHi", f2(S["gain_same_plans_frozen"][2])), ("GainDiff", f2(S["gain_diff"][0])), ("GainDiffLo", f2(S["gain_diff"][1])), ("GainDiffHi", f2(S["gain_diff"][2])),
               ("GainUpper", f2(S["gain_upper_frozen"])), ("SeedsUp", str(S["seeds_up"])), ("RegimeAsRun", f2(S["regime_as_run"])), ("RegimeFrozen", f2(S["regime_frozen"]))]
        with open(os.path.join(args.paper, "sections", "results_numbers_netcov.tex"), "w") as fh:
            fh.write("% generated by src/model/rescore_net_frozen_cov.py; do not edit\n")
            for k, v in mac:
                fh.write(f"\\newcommand{{\\netcov{k}}}{{{v}}}\n")
        print(f"wrote {len(mac)} macros")
