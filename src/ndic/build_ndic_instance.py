"""Build a 40-unit development-planning instance calibrated on public North Dakota (NDIC) data.

The instance follows the contract of src/model/instance.py (keys nx, ny, U, secs, nbr, X, mu, Sigma, mu_prior,
Sigma_prior, g_true, owned, WI, NRI, parents, deadline, pkg_of, pkg_cost, cap, n_sec, seed, the DEFAULTS constants
and the node tree of build_nodes), plus a 'provenance' key holding every data choice.

What comes from public records: the PLSS lattice and the spacing units, the existing laterals by unit and bench,
the planning operator and its footprint, the type-curve level by unit, the field's range and standard deviation,
the state-lease deadlines, the rig count behind the capacities, and the auction bonuses behind the package prices.
What stays at the generator's defaults is listed in the provenance and in Section S1.4 of the Supplementary Materials.

Run from anywhere:  python src/ndic/build_ndic_instance.py
Memory stays below about 1 GB: every parquet read projects columns and filters rows.
No solver is called; the validation uses the numpy-only parts of src/model.
"""
import os
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")   # return freed Arrow memory to the OS
import copy
import json
import pickle
import re
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ndic"
DER = ROOT / "data" / "derived"
FIG = ROOT / "paper" / "figures"
REP = ROOT / "docs" / "reports"
sys.path.insert(0, str(ROOT / "src" / "model"))
from instance import DEFAULTS, NB, PAD_COST, ROYALTY, T, build_nodes, condition_field, field_cov  # noqa: E402

# ---------------------------------------------------------------------------------------------------------
# Choices (every one is copied into the provenance)
# ---------------------------------------------------------------------------------------------------------
PLANNING_DATE = pd.Timestamp("2024-12-31")
PLAN_YEAR0 = 2025                       # year t = 0 of the plan
NX, NY = 8, 10                          # columns x section rows; units are N-S pairs of sections
COUNTY = "McKenzie"
EAST_CENTRAL = dict(rng=(94, 98), twp=(148, 152))   # 'east-central McKenzie': ranges 94-98 W, townships 148-152 N
RMAX, TMIN = 105, 145                   # lattice origin for column / row indices
CAL_YEARS = (2018, 2024)                # first-production years of the calibration wells
TC_YEARS = (2018, 2019)                 # vintages of the type curve: the oldest calibration wells, with 6.5+ years of history
CAL_RADIUS_MI = 6.0                     # wells within this distance of a unit center inform its mean
KERNEL_SD_MI = 2.0                      # Gaussian kernel of the local mean
CLASSIFY_BUFFER_MI = 10.0               # wells used for bench classification: block + this buffer
TWO_MILE_CLASS = (1.5, 2.5)             # lateral miles of the calibration sample (the two-mile class)
TVD_BAKKEN = (9400.0, 11700.0)          # lateral TVD window of the Bakken petroleum system in this area
LANE_CAP = 6                            # parents per unit-bench are capped at the lane count (instance L_LANES)
P_REF = 60.0                            # $/bbl at the wellhead: end-2024 WTI strip (~$65-70) less the Bakken differential
SEV_TAX = 0.10                          # ND oil gross production tax 5% + oil extraction tax 5%
OPEX_BBL = 10.0                         # $/bbl lease operating, gathering and transport (assumption)
RF = 0.05                               # risk-free rate, as in instance.py
TERMINAL_DECLINE = 0.06                 # per year, the tail of the type curve beyond the observed months
EUR_YEARS = 30
WI_OWNED = 0.75                         # documented default: disclosed average operated WI (no public WI field)
WI_ACQ = 0.75                           # acquired units, as instance.py
RIG_UTIL = 0.90
RIG_YEAR = 2022                         # x_nd_daily_rigs covers 2021-08 to 2022-12 only; 2022 is the last full year
AUCTION_YEARS = (2019, 2024)
PREV_AUCTION_YEARS = (2013, 2018)      # fallback window when no bids fall near the block in AUCTION_YEARS
PRIMARY_TERM_YRS = 5                    # state lease primary term, for auctioned tracts without an expiry date
SEED = 0
N_NODES = 3

LAT0 = LON0 = None                      # set to the block's SW corner


def to_xy(lon, lat):
    return ((np.asarray(lon) - LON0) * np.cos(np.radians(LAT0)) * 69.17, (np.asarray(lat) - LAT0) * 69.05)


def geom_xy(g):
    return shapely.transform(g, lambda c: np.c_[to_xy(c[:, 0], c[:, 1])])


def norm_op(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = re.sub(r"[.,&]", " ", str(s).upper())
    s = re.sub(r"\b(LLC|L L C|LP|L P|INC|CORP|CORPORATION|CO|COMPANY|LTD)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def mrss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # MB on macOS


def read(name, columns, filters=None):
    return pq.read_table(RAW / f"{name}.parquet", columns=columns, filters=filters).to_pandas()


def read_batched(name, columns, expr, batch=100_000):
    """Read a large table batch by batch, keeping only the rows that pass the pyarrow expression (bounded memory)."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    parts = []
    for b in ds.dataset(RAW / f"{name}.parquet").to_batches(columns=columns, filter=expr, batch_size=batch):
        if b.num_rows:
            parts.append(b)
    if not parts:
        return pd.DataFrame(columns=columns)
    tbl = pa.Table.from_batches(parts)
    if "api10" in tbl.column_names:                           # integer API numbers: a fraction of the memory of strings
        tbl = tbl.set_column(tbl.column_names.index("api10"), "api10", pc.cast(tbl["api10"], pa.int64()))
    out = tbl.to_pandas()
    del tbl
    del parts
    pa.default_memory_pool().release_unused()
    return out


P = {}          # provenance
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


# ---------------------------------------------------------------------------------------------------------
# 1. Section lattice and the choice of block
# ---------------------------------------------------------------------------------------------------------
def load_sections():
    s = read("x_nd_plss_sections", ["section", "township", "range", "area_acres", "geometry"],
             [("township", ">=", TMIN), ("township", "<=", 153), ("range", ">=", 94), ("range", "<=", RMAX)])
    s = s[(s.section >= 1) & (s.section <= 36)].reset_index(drop=True)
    r, c = (s.section - 1) // 6, (s.section - 1) % 6                       # row from the north, boustrophedon order
    s["gc"] = (RMAX - s["range"]) * 6 + np.where(r % 2 == 0, 5 - c, c)     # column from the west
    s["gr"] = (s.township - TMIN) * 6 + (5 - r)                            # row from the south
    assert not s.duplicated(["gc", "gr"]).any()
    s["g"] = shapely.from_wkb(s.geometry.values)
    cty = read("x_nd_cat_county_boundaries", ["name", "geometry"])
    mck = shapely.from_wkb(cty[cty.name == COUNTY].geometry.values[0])
    s["in_county"] = shapely.within(shapely.centroid(s.g.values), mck)
    return s.drop(columns="geometry")


def dsu_match(s):
    d = read("x_nd_drilling_spacing_units", ["dsu_id", "dsu_acres", "geometry"])
    dg = shapely.from_wkb(d.geometry.values)
    tree = shapely.STRtree(dg)
    cen = shapely.centroid(s.g.values)
    key = {(a, b): i for i, (a, b) in enumerate(zip(s.gc, s.gr))}
    acres = d.dsu_acres.values
    out = {}
    for (gc, gr), i in key.items():
        if gr % 2:
            continue
        j = key.get((gc, gr + 1))
        if j is None:
            continue
        a = set(tree.query(cen[i], predicate="within")); b = set(tree.query(cen[j], predicate="within"))
        both = a & b
        out[(gc, gr)] = dict(m1280=any(acres[k] == 1280 for k in both), any=len(both) > 0,
                             split=any(acres[k] == 640 for k in a | b),
                             over2560=any(acres[k] >= 2560 for k in both))
    P["dsu_rows"] = int(len(d))
    return out


# ---------------------------------------------------------------------------------------------------------
# 2. Laterals, surveys, wells, production
# ---------------------------------------------------------------------------------------------------------
def load_laterals(bbox):
    l = read("x_nd_laterals", ["api10", "well_sub", "geometry"], pc.field("well_sub") == "LAT1")
    g = shapely.from_wkb(l.geometry.values)
    b = shapely.bounds(g)
    m = (b[:, 0] > bbox[0]) & (b[:, 2] < bbox[2]) & (b[:, 1] > bbox[1]) & (b[:, 3] < bbox[3])
    l = l[m].drop(columns="geometry").reset_index(drop=True)
    l["g"] = g[m]
    l = l.drop_duplicates("api10").reset_index(drop=True)
    w = read("wells", ["api10", "source_id", "well_name", "operator", "status", "well_type", "spud_date"])
    l = l.merge(w, on="api10", how="left")
    l["api10"] = pd.to_numeric(l.api10).astype("int64")
    l["file_no"] = pd.to_numeric(l.source_id, errors="coerce")
    l["mid"] = shapely.line_interpolate_point(l.g.values, 0.5, normalized=True)
    return l


def production_first12(apis):
    p = read_batched("production_monthly", ["api10", "year", "month", "oil_bbl"], pc.field("api10").isin([str(a) for a in apis]))
    P["production_rows_read"] = int(len(p))
    p["m"] = p.year * 12 + p.month - 1
    p = p.groupby(["api10", "m"], as_index=False).oil_bbl.sum(min_count=1)
    m0 = p.groupby("api10").m.min().rename("m0")
    p = p.join(m0, on="api10")
    p["k"] = p.m - p.m0
    last_m = int(p.m.max())
    P["production_last_month"] = f"{last_m // 12}-{last_m % 12 + 1:02d}"
    oil2024 = p[(p.m >= 2024 * 12) & (p.m < 2025 * 12)].groupby("api10").oil_bbl.sum()
    return p, m0, oil2024, last_m


def survey_aggregates(apis):
    s = read_batched("directional_surveys", ["api10", "well_sub", "md_ft", "inclination", "tvd_ft", "lat", "lon"],
                     (pc.field("api10").isin([str(a) for a in apis])) & (pc.field("inclination") > 85.0) & (pc.field("well_sub") == "LAT1"))
    s = s[s.well_sub == "LAT1"]
    P["survey_rows_read"] = int(len(s))
    g = s.groupby("api10")
    a = g.agg(tvd_med=("tvd_ft", "median"), md_min=("md_ft", "min"), md_max=("md_ft", "max"), n_st=("tvd_ft", "size"),
              s_lat=("lat", "median"), s_lon=("lon", "median"))
    a["lat_ft_svy"] = a.md_max - a.md_min
    return a[a.n_st >= 5]


# ---------------------------------------------------------------------------------------------------------
# 3. Bench classification: operator name tokens as labels, local TVD surfaces for the rest
# ---------------------------------------------------------------------------------------------------------
TF_RE = r"(?:\dTFH|TFH\b|\bTF\d?H?\b|-TF\b|THREE FORKS|\d+T\d?$)"
MB_RE = r"(?:MBH\b|\bMB\d?H?\b|\d+B$)"


def classify_bench(x):
    n = x.well_name.fillna("").str.upper().str.strip()
    lab = np.where(n.str.contains(TF_RE), 1, np.where(n.str.contains(MB_RE), 0, -1))
    xy = np.c_[x.mx, x.my]; t = x.tvd.values

    def fit(i, mask, R, kmin=5):
        d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
        idx = np.flatnonzero(mask & (d < R)); idx = idx[idx != i]
        if len(idx) < kmin:
            return np.nan
        A = np.c_[np.ones(len(idx)), xy[idx] - xy[i]]; wg = 1 / (0.25 + d[idx])
        return np.linalg.lstsq(A * wg[:, None], t[idx] * wg, rcond=None)[0][0]

    def two_surface(i, labels, R):
        a, b = fit(i, labels == 0, R), fit(i, labels == 1, R)
        if np.isfinite(a) and np.isfinite(b):
            return int(abs(t[i] - b) < abs(t[i] - a)), b - a
        return -1, np.nan

    # leave-one-out accuracy on the labeled wells (radius 2.5 then 4 miles)
    loo, offs = [], []
    for i in np.flatnonzero(lab >= 0):
        p, o = two_surface(i, lab, 2.5)
        if p < 0:
            p, o = two_surface(i, lab, 4.0)
        if p >= 0:
            loo.append(p == lab[i]); offs.append(o)
    off_med = float(np.nanmedian(offs))
    bench = lab.copy(); how = np.array(["name" if v >= 0 else "" for v in lab], dtype=object)
    for R, tag in ((2.5, "tvd2.5"), (4.0, "tvd4")):
        for i in np.flatnonzero(bench < 0):
            p, _ = two_surface(i, lab, R)
            if p >= 0:
                bench[i] = p; how[i] = tag
    for i in np.flatnonzero(bench < 0):                       # one labeled surface nearby: use the median offset
        a, b = fit(i, lab == 0, 4.0), fit(i, lab == 1, 4.0)
        if np.isfinite(a):
            bench[i] = int(t[i] - a > off_med / 2); how[i] = "tvd_mb_offset"
        elif np.isfinite(b):
            bench[i] = int(t[i] - b > -off_med / 2); how[i] = "tvd_tf_offset"
    for i in np.flatnonzero(bench < 0):                       # last resort: pseudo-labels of classified neighbors
        p, _ = two_surface(i, np.where(bench >= 0, bench, -1), 4.0)
        bench[i] = max(p, 0); how[i] = "tvd_pseudo" if p >= 0 else "default_mb"
    P["bench_classifier"] = dict(
        labels_from_names=dict(MB=int((lab == 0).sum()), TF=int((lab == 1).sum()), unlabeled=int((lab < 0).sum())),
        name_regex=dict(TF=TF_RE, MB=MB_RE), loo_accuracy=round(float(np.mean(loo)), 3), loo_n=int(len(loo)),
        tf_minus_mb_tvd_ft_median=round(off_med, 1), method_counts=pd.Series(how).value_counts().to_dict(),
        tvd_source="median TVD of LAT1 survey stations with inclination > 85 deg (bottom-hole TVD if no survey)")
    return bench, how


# ---------------------------------------------------------------------------------------------------------
# 4. Type curve: first-year share of EUR and PV factor, from the area's long histories
# ---------------------------------------------------------------------------------------------------------
def type_curve(p, wells_ok, last_m):
    """Mean monthly profile (share of first-12-month oil) of the calibration vintages, an Arps tail beyond the months
    every well has observed, a terminal exponential decline, and the first-year share and PV factor of the EUR."""
    q = p[p.api10.isin(wells_ok)]
    kmax = q.groupby("api10").k.max()
    n12 = q[q.k < 12].groupby("api10").oil_bbl.count()
    ids = n12[n12 == 12].index
    K = int(min(120, kmax.reindex(ids).quantile(0.05) + 1))   # months observed by 95% of the wells
    q = q[q.api10.isin(ids) & (q.k < K)]
    piv = q.pivot(index="api10", columns="k", values="oil_bbl")
    q12w = piv.loc[:, :11].sum(1)
    piv = piv[q12w > 0].div(q12w[q12w > 0], axis=0)
    prof = piv.mean(0).fillna(0).values                       # mean share of first-12-month oil, by month
    K = len(prof)
    obs_ratio = {m: round(float(prof[:m].sum()), 3) for m in (24, 36, 60) if m <= K}
    kk = np.arange(12, K); y = np.log(np.maximum(prof[12:], 1e-6))
    best = None
    for b in np.linspace(0.3, 1.8, 31):                       # Arps hyperbolic fit on months 12+ (log least squares)
        for Di in np.exp(np.linspace(np.log(0.01), np.log(0.5), 60)):
            f = -np.log1p(b * Di * kk) / b
            c = np.mean(y - f); r = np.sum((y - c - f) ** 2)
            if best is None or r < best[0]:
                best = (r, b, Di, c)
    _, b, Di, c = best
    months = np.arange(EUR_YEARS * 12)
    tail = np.exp(c - np.log1p(b * Di * months) / b)
    Dterm = 1 - (1 - TERMINAL_DECLINE) ** (1 / 12)
    Dhyp = Di / (1 + b * Di * months)
    sw = np.argmax(Dhyp < Dterm) if np.any(Dhyp < Dterm) else len(months)
    if sw < len(months):
        tail[sw:] = tail[sw] * (1 - Dterm) ** (months[sw:] - sw)
    full_prof = np.r_[prof, tail[K:]]
    eur = full_prof.sum()                                     # in units of the first-12-month oil
    pv = (full_prof * (1 + RF) ** (-(months + 0.5) / 12)).sum()
    P["type_curve"] = dict(wells=int(len(ids)), months_observed=int(K), arps_b=round(b, 2), arps_Di_per_month=round(Di, 4),
                           terminal_decline=TERMINAL_DECLINE, eur_years=EUR_YEARS,
                           first_year_share_of_eur=round(1 / eur, 4), pv5_over_eur=round(pv / eur, 4),
                           cum_over_q12_observed=obs_ratio,
                           sample=f"wells near the block, first production {TC_YEARS[0]}-{TC_YEARS[1]}, complete first 12 months")
    return 1 / eur, pv / eur, full_prof


# ---------------------------------------------------------------------------------------------------------
# 5. Variogram (Matern 3/2 with nugget, the field_cov parametrization)
# ---------------------------------------------------------------------------------------------------------
def fit_variogram(xy, e, hmax=8.0, bw=0.25):
    from scipy.spatial.distance import pdist
    d = pdist(xy); g = 0.5 * pdist(e[:, None], "sqeuclidean")
    m = d < hmax
    d, g = d[m], g[m]
    bins = np.arange(0, hmax + bw, bw)
    idx = np.digitize(d, bins) - 1
    hb = np.array([d[idx == i].mean() for i in range(len(bins) - 1) if (idx == i).sum() >= 30])
    gb = np.array([g[idx == i].mean() for i in range(len(bins) - 1) if (idx == i).sum() >= 30])
    nb = np.array([(idx == i).sum() for i in range(len(bins) - 1) if (idx == i).sum() >= 30])
    best = None
    from scipy.optimize import nnls
    for r in np.exp(np.linspace(np.log(0.05), np.log(20), 150)):
        z = np.sqrt(3) * hb / r
        A = np.c_[np.ones_like(hb), 1 - (1 + z) * np.exp(-z)]
        W = np.sqrt(nb)
        coef, res = nnls(A * W[:, None], gb * W)
        if best is None or res < best[0]:
            best = (res, r, coef)
    _, r, (nug, psill) = best
    return dict(range_mi=float(r), nugget=float(nug), psill=float(psill), lags=hb.tolist(), gamma=gb.tolist(), npairs=nb.tolist(),
                sample_var=float(np.var(e)))


def _anova_sigma_b(e, cells):
    d = pd.DataFrame(dict(e=e, c=cells))
    g = d.groupby("c").e
    n = g.size(); k = len(n); N = len(d); m = g.mean()
    msb = float((n * (m - d.e.mean()) ** 2).sum() / (k - 1))
    msw = float(((d.e - d.c.map(m)) ** 2).sum() / (N - k))
    n0 = (N - float((n ** 2).sum()) / N) / (k - 1)
    return (msb - msw) / n0, msw, k


def _matern32(d, r):
    z = np.sqrt(3) * np.asarray(d) / r
    return (1 + z) * np.exp(-z)


def unit_field(mx, my, e, n_boot=300, seed=1):
    """Unit-scale field from variance components of the log residuals.

    sigma_u^2: between-cell variance of 1 x 2 mile cells on the block's lattice (one-way random effects); sigma_22^2 the
    same on 2 x 2-unit cells. The mean correlation of the six unit pairs inside a 2 x 2 block (distances 1, 2 and
    sqrt(5) miles, two pairs each) is (4 sigma_22^2 / sigma_u^2 - 1) / 3, and the Matern 3/2 range of field_cov is the
    one that reproduces it. Cell-bootstrap (resampling 2 x 2 blocks) gives the intervals.
    """
    cu = [f"{a}_{b}" for a, b in zip(np.floor(mx).astype(int), np.floor(my / 2).astype(int))]
    c4 = np.array([f"{a}_{b}" for a, b in zip(np.floor(mx / 2).astype(int), np.floor(my / 4).astype(int))])
    dpairs = np.array([1.0, 1.0, 2.0, 2.0, np.sqrt(5), np.sqrt(5)])

    def est(idx):
        su, sw, ku = _anova_sigma_b(e[idx], [cu[i] for i in idx])
        s4, _, k4 = _anova_sigma_b(e[idx], list(c4[idx]))
        rho = (4 * s4 / su - 1) / 3 if su > 0 else np.nan
        if not np.isfinite(rho) or rho <= _matern32(dpairs, 0.05).mean():
            r = 0.05
        elif rho >= _matern32(dpairs, 50).mean():
            r = 50.0
        else:
            lo, hi = 0.05, 50.0
            for _ in range(60):
                mid = np.sqrt(lo * hi)
                lo, hi = (mid, hi) if _matern32(dpairs, mid).mean() < rho else (lo, mid)
            r = np.sqrt(lo * hi)
        return su, s4, sw, rho, r, ku, k4

    su, s4, sw, rho, r, ku, k4 = est(np.arange(len(e)))
    rng = np.random.default_rng(seed)
    blocks = pd.Series(np.arange(len(e))).groupby(c4).apply(list).tolist()
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        bsu, _, _, _, br, _, _ = est(idx)
        bs.append((np.sqrt(max(bsu, 0)), br))
    bs = np.array(bs)
    return dict(sd_unit_log=float(np.sqrt(max(su, 0))), var_unit=float(su), var_2x2=float(s4), sd_within_log=float(np.sqrt(sw)),
                mean_corr_2x2=float(rho), range_mi=float(r), cells_unit=int(ku), cells_2x2=int(k4),
                sd_unit_ci90=np.round(np.percentile(bs[:, 0], [5, 95]), 3).tolist(),
                range_ci90=np.round(np.percentile(bs[:, 1], [5, 95]), 2).tolist())


# ---------------------------------------------------------------------------------------------------------
def main():
    global LAT0, LON0
    say(f"planning date {PLANNING_DATE.date()}, block {NX} x {NY} sections, county {COUNTY}")
    # ---- 1. lattice and block -------------------------------------------------------------------------
    s = load_sections()
    M = dsu_match(s)
    key = {(a, b): i for i, (a, b) in enumerate(zip(s.gc, s.gr))}
    lon_c = shapely.get_x(shapely.centroid(s.g.values)); lat_c = shapely.get_y(shapely.centroid(s.g.values))
    LAT0, LON0 = float(lat_c.mean()), float(lon_c.mean())          # provisional origin for the window scoring
    ec = s["range"].between(*EAST_CENTRAL["rng"]) & s.township.between(*EAST_CENTRAL["twp"])
    bb = shapely.bounds(s.g.values[ec.values])
    l = load_laterals((bb[:, 0].min() - 0.25, bb[:, 1].min() - 0.18, bb[:, 2].max() + 0.25, bb[:, 3].max() + 0.18))
    P["laterals_rows_in_bbox"] = int(len(l))
    p, m0, oil2024, last_m = production_first12(l.api10.unique())
    l["m0"] = l.api10.map(m0)
    l["fp_year"] = l.m0 // 12
    tree = shapely.STRtree(s.g.values)
    hit = tree.query(l.mid.values, predicate="within")
    sec_of = np.full(len(l), -1); sec_of[hit[0]] = hit[1]
    l["gc"] = np.where(sec_of >= 0, s.gc.values[np.maximum(sec_of, 0)], -999)
    l["gr"] = np.where(sec_of >= 0, s.gr.values[np.maximum(sec_of, 0)], -999)
    cands = []
    for c0 in range(s.gc.min(), s.gc.max() - NX + 2):
        for r0 in range(0, s.gr.max() - NY + 2, 2):          # pairs start on township rows (6,5), (4,3), (2,1)
            ids = [key.get((c0 + i, r0 + j)) for i in range(NX) for j in range(NY)]
            if any(v is None for v in ids):
                continue
            sub = s.iloc[ids]
            if not sub.in_county.all():
                continue
            if not (sub["range"].between(*EAST_CENTRAL["rng"]).all() and sub.township.between(*EAST_CENTRAL["twp"]).all()):
                continue
            mm = [M.get((c0 + i, r0 + 2 * j), {}).get("m1280", False) for i in range(NX) for j in range(NY // 2)]
            inw = (l.gc >= c0) & (l.gc < c0 + NX) & (l.gr >= r0) & (l.gr < r0 + NY)
            cands.append(dict(c0=c0, r0=r0, match=float(np.mean(mm)), n_lat=int(inw.sum()),
                              n_cal=int((inw & l.fp_year.between(*CAL_YEARS)).sum())))
    C = pd.DataFrame(cands)
    C = C[C.match == C.match.max()].sort_values(["n_cal", "n_lat"], ascending=False)
    P["block_selection"] = dict(rule=(f"8x10 windows with pairs on township rows, every section in {COUNTY} County and in "
                                      f"R{EAST_CENTRAL['rng'][0]}-{EAST_CENTRAL['rng'][1]}W, T{EAST_CENTRAL['twp'][0]}-"
                                      f"{EAST_CENTRAL['twp'][1]}N; keep the windows with the highest 1280-acre DSU match; "
                                      f"choose the one with most laterals first producing {CAL_YEARS[0]}-{CAL_YEARS[1]}"),
                                n_windows_considered=int(len(cands)), n_windows_best_match=int(len(C)),
                                runner_up=C.iloc[1].to_dict() if len(C) > 1 else None)
    c0, r0 = int(C.iloc[0].c0), int(C.iloc[0].r0)
    blk_ids = np.array([key[(c0 + i, r0 + j)] for i in range(NX) for j in range(NY)])   # section index = col*NY + row
    B = s.iloc[blk_ids].reset_index(drop=True)
    bxy = shapely.bounds(B.g.values)
    LON0, LAT0 = float(bxy[:, 0].min()), float(bxy[:, 1].min())      # block SW corner
    B["gx"] = geom_xy(B.g.values)
    twps = sorted(B.township.unique()); rngs = sorted(B["range"].unique())
    area = f"T{twps[0]}-{twps[-1]}N, R{rngs[0]}-{rngs[-1]}W, {COUNTY} County, ND"
    say("block:", area, "| lattice origin col", c0, "row", r0, "| SW corner", round(LAT0, 5), round(LON0, 5))
    # units
    npair = NY // 2; U = NX * npair
    uu = np.arange(U); col, pair = uu // npair, uu % npair
    secs = np.stack([col * NY + 2 * pair, col * NY + 2 * pair + 1], 1)
    nbr = np.where(pair + 1 < npair, uu + 1, -1)
    ug = [shapely.union(B.gx.values[a], B.gx.values[b]) for a, b in secs]
    X = np.array([[shapely.get_x(shapely.centroid(g)), shapely.get_y(shapely.centroid(g))] for g in ug])
    Xideal = np.c_[col + 0.5, 2 * pair + 1.0]
    unit_acres = np.array([B.area_acres.values[a] + B.area_acres.values[b] for a, b in secs])
    um = [M[(c0 + int(col[u]), r0 + 2 * int(pair[u]))] for u in range(U)]
    match = dict(m1280=float(np.mean([m["m1280"] for m in um])), any_dsu=float(np.mean([m["any"] for m in um])),
                 also_in_2560_or_larger=int(sum(m["over2560"] for m in um)), touching_640=int(sum(m["split"] for m in um)))
    say("DSU match (two sections in one 1280-acre DSU):", match)
    P.update(area=area, townships=[int(t) for t in twps], ranges=[int(r) for r in rngs], sw_corner_latlon=[LAT0, LON0],
             sections=B.apply(lambda r: f"{int(r.township)}N-{int(r['range'])}W-{int(r.section)}", axis=1).tolist(),
             dsu_match=match, unit_acres=dict(min=float(unit_acres.min()), max=float(unit_acres.max())),
             center_offset_from_ideal_grid_mi=float(np.abs(X - Xideal).max()))

    # ---- 2. wells near the block: geometry, surveys, bench, parents ------------------------------------
    l["mx"], l["my"] = to_xy(shapely.get_x(l.mid.values), shapely.get_y(l.mid.values))
    near = (l.mx > -CLASSIFY_BUFFER_MI) & (l.mx < NX + CLASSIFY_BUFFER_MI) & (l.my > -CLASSIFY_BUFFER_MI) & (l.my < NY + CLASSIFY_BUFFER_MI)
    x = l[near].reset_index(drop=True)
    x["gx"] = geom_xy(x.g.values)
    x["len_geom_mi"] = shapely.length(x.gx.values)
    sv = survey_aggregates(x.api10.unique())
    bh = read("x_bottomhole_locations", ["api10", "well_sub", "tvd_ft"], pc.field("api10").isin([str(a) for a in x.api10]))
    bh["api10"] = pd.to_numeric(bh.api10).astype("int64")
    bh = bh[bh.well_sub == "LAT1"].drop_duplicates("api10").set_index("api10").tvd_ft
    x = x.join(sv, on="api10")
    x["tvd"] = x.tvd_med.fillna(x.api10.map(bh))
    x["lat_mi"] = (x.lat_ft_svy / 5280).where(x.lat_ft_svy.notna(), x.len_geom_mi)
    keep = x.tvd.between(*TVD_BAKKEN) & (x.lat_mi >= 0.5)
    P["wells_near_block"] = dict(lat1_laterals=int(len(x)), with_survey=int(x.tvd_med.notna().sum()),
                                 kept_bakken_tvd_and_len_ge_0p5mi=int(keep.sum()))
    x = x[keep].reset_index(drop=True)
    x["bench"], x["bench_how"] = classify_bench(x)
    say("bench classifier:", P["bench_classifier"])
    # operator at the planning date: roll back changes of operator reported after it
    tr = read("x_nd_daily_transfers", ["report_date", "from_operator", "file_no", "kind"],
              pc.field("file_no").isin([int(v) for v in x.file_no.dropna()]))
    tr = tr[(pd.to_datetime(tr.report_date) > PLANNING_DATE) & (tr.kind == "CHANGE_OF_OPERATOR")].sort_values("report_date")
    first_after = tr.groupby("file_no").from_operator.first()
    x["op_cut"] = [norm_op(first_after.get(f, o)) for f, o in zip(x.file_no, x.operator)]
    P["operator_rollback"] = dict(wells_with_change_after_planning_date=int(len(first_after)))
    # producing at the planning date
    cut_m = PLANNING_DATE.year * 12 + PLANNING_DATE.month - 1
    x["producing"] = (x.m0 <= cut_m) & ((x.api10.map(oil2024).fillna(0) > 0) | (x.status == "A"))
    # unit assignment by length inside each unit
    utree = shapely.STRtree(ug)
    assign = []
    for i, g in enumerate(x.gx.values):
        cand = utree.query(g)
        if len(cand) == 0:
            continue
        lens = np.array([shapely.length(shapely.intersection(g, ug[u])) for u in cand])
        j = int(np.argmax(lens))
        if lens[j] >= 0.5:
            assign.append((i, int(cand[j]), "max"))
        for u, ln in zip(cand, lens):
            if u != cand[j] and ln >= 1.0:
                assign.append((i, int(u), "extra"))
    A = pd.DataFrame(assign, columns=["i", "u", "how"])
    A = A.join(x[["api10", "bench", "producing", "op_cut", "m0", "status"]], on="i")
    par_rows = A[A.producing]
    parents_raw = np.zeros((U, NB), int)
    for (u, b), n in par_rows.groupby(["u", "bench"]).size().items():
        parents_raw[u, b] = n
    parents = np.minimum(parents_raw, LANE_CAP)
    excluded = A[~A.producing].drop_duplicates("api10")
    P["parents"] = dict(rule=("LAT1 laterals in the Bakken TVD window, first production on or before the planning date and "
                              "oil reported in 2024 or status A; a lateral is a parent of the unit holding most of its length "
                              "(>= 0.5 mi) and of any other unit holding >= 1.0 mi"),
                        laterals_in_block=int(A.api10.nunique()), parent_laterals=int(par_rows.api10.nunique()),
                        extra_unit_assignments=int((par_rows.how == "extra").sum()),
                        not_parents=excluded.assign(after=excluded.m0 > cut_m).groupby("after").size().rename(
                            {True: "first_production_after_planning_date", False: "not_producing_in_2024"}).to_dict(),
                        status_of_parents=par_rows.drop_duplicates("api10").status.value_counts().to_dict(),
                        unit_benches_capped_at_lane_count=int((parents_raw > LANE_CAP).sum()),
                        parent_count_before_cap=int(parents_raw.sum()), parent_count_after_cap=int(parents.sum()))
    say("parents:", P["parents"])

    # ---- planning operator and ownership --------------------------------------------------------------
    opc = par_rows.drop_duplicates("api10").op_cut.value_counts()
    op = opc.index[0]
    unit_ops = par_rows.groupby("u").op_cut.agg(lambda v: v.value_counts().to_dict())
    operates = np.array([op in unit_ops.get(u, {}) for u in range(U)])
    pm = read("permits", ["api10", "operator", "permit_date", "lat", "lon", "cancellation_announced_date", "cancellation_stands"])
    pm = pm[(pd.to_datetime(pm.permit_date) <= PLANNING_DATE) & (pm.operator.map(norm_op) == op)].copy()
    pm["api10"] = pd.to_numeric(pm.api10, errors="coerce").fillna(-1).astype("int64")
    pm = pm[~((pd.to_datetime(pm.cancellation_announced_date) <= PLANNING_DATE) & (pm.cancellation_stands == True))]  # noqa: E712
    permit_units = set()
    lat_unit = A[A.how == "max"].drop_duplicates("api10").set_index("api10").u
    for a, la, lo in zip(pm.api10, pm.lat, pm.lon):
        if a in lat_unit.index and not x.loc[x.api10 == a, "producing"].any():
            permit_units.add(int(lat_unit[a]))
        elif a not in lat_unit.index and np.isfinite(la):
            pt = shapely.points(*to_xy(lo, la))
            hit = [u for u in utree.query(pt, predicate="within")]
            permit_units.update(int(u) for u in hit)
    owned = operates.copy()
    owned[list(permit_units)] = True
    mixed = [u for u in range(U) if len(unit_ops.get(u, {})) > 1]
    WI = np.where(owned, WI_OWNED, WI_ACQ)
    NRI = WI * (1 - ROYALTY)
    P["ownership"] = dict(planning_operator=op, parent_laterals_by_operator=opc.head(8).to_dict(),
                          units_operated=int(operates.sum()), units_added_by_permit=int((owned & ~operates).sum()),
                          owned_units=int(owned.sum()), units_with_two_or_more_operators=len(mixed),
                          wi_rule=f"no public working-interest field: owned {WI_OWNED} (disclosed operated average), acquired {WI_ACQ}",
                          royalty=ROYALTY)
    say("ownership:", P["ownership"])

    # ---- 3. type-curve level by unit-bench ---------------------------------------------------------------
    cal = x[x.fp_year.between(*CAL_YEARS) & x.lat_mi.between(*TWO_MILE_CLASS)].copy()
    pp = p[p.api10.isin(cal.api10)]
    cal = cal[cal.m0 + 11 <= last_m]
    w12 = pp[pp.k < 12].pivot(index="api10", columns="k", values="oil_bbl")
    w12 = w12.reindex(columns=range(12)).reindex(cal.api10)
    complete = w12.notna().all(1)
    share = (w12[complete].div(w12[complete].sum(1), axis=0)).mean(0).values
    obs_share = (w12.notna().values * share).sum(1)
    q12 = np.where(complete, w12.sum(1, min_count=1), w12.sum(1, min_count=1) / np.where(obs_share > 0, obs_share, np.nan))
    cal["q12"] = q12; cal["complete"] = complete.values; cal["obs_share"] = obs_share
    n_before = len(cal)
    cal = cal[(cal.obs_share >= 0.5) & (cal.q12 > 0)]
    cal["qpm"] = cal.q12 / cal.lat_mi
    P["calibration_sample"] = dict(rule=(f"wells within {CLASSIFY_BUFFER_MI} mi of the block, first production "
                                         f"{CAL_YEARS[0]}-{CAL_YEARS[1]}, lateral {TWO_MILE_CLASS[0]}-{TWO_MILE_CLASS[1]} mi, "
                                         "12 months of history"),
                                   wells=int(n_before), complete_first_12=int(cal.complete.sum()),
                                   imputed_withheld_months=int((~cal.complete).sum()), dropped_under_half_observed=int(n_before - len(cal)),
                                   imputation=("months withheld while confidential are filled by ratio: observed oil / mean share "
                                               "of those months in complete wells' first-12-month oil"),
                                   month_share_complete_wells=np.round(share, 4).tolist(),
                                   by_vintage_median_q12_per_mile=cal.groupby("fp_year").qpm.median().round(0).to_dict(),
                                   by_bench=cal.groupby("bench").size().to_dict())
    say("calibration sample:", {k: v for k, v in P["calibration_sample"].items() if k != "month_share_complete_wells"})
    # trend + bench + vintage regression on log oil per mile
    yv = np.log(cal.qpm.values)
    cx, cy = cal.mx.values - NX / 2, cal.my.values - NY / 2
    vint = pd.get_dummies(cal.fp_year.astype(int), drop_first=True).values.astype(float)
    Xr = np.c_[np.ones(len(cal)), cx, cy, cx ** 2, cy ** 2, cx * cy, cal.bench.values, vint]
    beta, *_ = np.linalg.lstsq(Xr, yv, rcond=None)
    resid = yv - Xr @ beta
    se_tf = float(np.sqrt(np.var(resid) * np.linalg.inv(Xr.T @ Xr)[6, 6]))
    ratio = float(np.exp(beta[6]))
    P["bench_ratio"] = dict(tf_over_mb=round(ratio, 3), ci95=[round(float(np.exp(beta[6] - 1.96 * se_tf)), 3),
                                                               round(float(np.exp(beta[6] + 1.96 * se_tf)), 3)],
                            model="log(q12 per mile) ~ quadratic trend in (x, y) + bench + first-production year", generator_default=0.8)
    say("bench ratio TF/MB:", P["bench_ratio"])
    vg = fit_variogram(np.c_[cal.mx, cal.my], resid)
    say("well-level variogram:", {k: round(v, 3) for k, v in vg.items() if not isinstance(v, list)})
    # type curve and conversion to $M
    s1, pvf, full_prof = type_curve(p, x.api10[x.fp_year.between(*TC_YEARS)], last_m)
    usd_per_pvbbl = P_REF * (1 - SEV_TAX) - OPEX_BBL
    conv = (1 / s1) * pvf * usd_per_pvbbl / 1e6                   # $M per bbl of first-12-month oil
    say(f"type curve: first-year share {s1:.3f}, PV5/EUR {pvf:.3f}, net $/bbl {usd_per_pvbbl:.1f} -> $M per first-year bbl {conv:.3e}")
    # local mean of MB-equivalent first-12-month oil per 2-mile well
    qeq = cal.qpm.values * 2.0 / np.where(cal.bench.values == 1, ratio, 1.0)
    cxy = np.c_[cal.mx, cal.my]
    mu_mb = np.zeros(U); nwell = np.zeros(U, int); q_mb = np.zeros(U)
    for u in range(U):
        d = np.hypot(*(cxy - X[u]).T)
        m = d <= CAL_RADIUS_MI
        wgt = np.exp(-0.5 * (d[m] / KERNEL_SD_MI) ** 2)
        q_mb[u] = float(wgt @ qeq[m] / wgt.sum()); nwell[u] = int(m.sum())
    mu_mb = q_mb * conv
    mu = np.stack([mu_mb, ratio * mu_mb], 1)
    uf = unit_field(cal.mx.values, cal.my.values, resid)
    say("unit-scale field:", uf)
    sd_log = uf["sd_unit_log"]
    sd_bench = (sd_log * float(mu[:, 0].mean()), sd_log * float(mu[:, 1].mean()))
    P["field"] = dict(range_mi=round(uf["range_mi"], 3), range_ci90=uf["range_ci90"], sd_log=round(sd_log, 4), sd_log_ci90=uf["sd_unit_ci90"],
                      within_unit_sd_log=round(uf["sd_within_log"], 4), sig_eps_default=DEFAULTS["sig_eps"],
                      mean_corr_2x2=round(uf["mean_corr_2x2"], 3), cells=dict(unit=uf["cells_unit"], block_2x2=uf["cells_2x2"]),
                      sd_bench_musd=[round(v, 3) for v in sd_bench], bench_corr=DEFAULTS["bench_corr"],
                      bench_corr_source="generator default (not estimated)",
                      well_variogram=dict(fit_range_mi=round(vg["range_mi"], 3), nugget=round(vg["nugget"], 4), psill=round(vg["psill"], 4),
                                          sample_var=round(vg["sample_var"], 4), lags=np.round(vg["lags"], 3).tolist(),
                                          gamma=np.round(vg["gamma"], 4).tolist(), npairs=vg["npairs"],
                                          note="flat beyond 0.5 mi: the well-level variogram does not identify a range at unit scale"),
                      rule=("residuals of log(q12 per mile) after a quadratic trend, bench and vintage effects; sd = between-unit "
                            "variance component (1 x 2 mi cells); range = Matern 3/2 range reproducing the variance of 2 x 2-unit "
                            "means; sd_bench = sd_log x mean mu of the bench (delta method)"))
    P["mu"] = dict(kernel=f"Gaussian, sd {KERNEL_SD_MI} mi, wells within {CAL_RADIUS_MI} mi of the unit center",
                   wells_per_unit=dict(min=int(nwell.min()), max=int(nwell.max())),
                   q12_mb_equiv_per_2mile_well_bbl=dict(min=round(q_mb.min()), max=round(q_mb.max()), mean=round(q_mb.mean())),
                   conversion=dict(p_ref_usd_bbl=P_REF, severance_tax=SEV_TAX, opex_usd_bbl=OPEX_BBL, net_usd_per_bbl=usd_per_pvbbl,
                                   first_year_share_of_eur=round(s1, 4), pv5_over_eur=round(pvf, 4), musd_per_first_year_bbl=conv,
                                   gas_and_ngl_revenue="not included"),
                   mu_mb_musd=dict(min=round(mu[:, 0].min(), 2), max=round(mu[:, 0].max(), 2), mean=round(mu[:, 0].mean(), 2)),
                   mu_tf_musd=dict(min=round(mu[:, 1].min(), 2), max=round(mu[:, 1].max(), 2), mean=round(mu[:, 1].mean(), 2)))
    say("mu:", P["mu"])

    # remaining value of the producing base implied by the type curve (model.PDP_REMAIN is fixed at 0.6)
    kk_all = np.arange(len(full_prof))
    pv_new = float((full_prof * (1 + RF) ** (-(kk_all + 0.5) / 12)).sum())
    ages = (cut_m - par_rows.drop_duplicates("api10").m0.values).astype(int)
    rem = [float((full_prof[a + 1:] * (1 + RF) ** (-(kk_all[a + 1:] - a - 0.5) / 12)).sum()) / pv_new for a in ages]
    P["producing_base"] = dict(model_pdp_remain=0.6, implied_remaining_share_mean=round(float(np.mean(rem)), 3),
                               implied_remaining_share_median=round(float(np.median(rem)), 3),
                               parent_age_years_median=round(float(np.median(ages)) / 12, 1),
                               note=("share of a new well's PV5 still to come from a parent of the type-curve shape at its age on the "
                                     "planning date; the model counts every parent in the block, owned or not, at PDP_REMAIN = 0.6"))
    say("producing base:", P["producing_base"])

    # ---- 4. deadlines ------------------------------------------------------------------------------------
    ls = read("leases_state", ["lease_no", "expiration_date", "status", "twp", "rng", "sec", "acres"])
    au = read("x_nd_lease_auction_results", ["auction_year", "auction_month", "lease_no", "county", "township", "range", "section",
                                             "net_acres", "bonus_per_acre"])
    sec_unit = {}
    for u in range(U):
        for sidx in secs[u]:
            r = B.iloc[sidx]
            sec_unit[(int(r.township), int(r["range"]), int(r.section))] = u
    horizon = (pd.Timestamp(f"{PLAN_YEAR0}-01-01"), pd.Timestamp(f"{PLAN_YEAR0 + T - 1}-12-31"))
    exp_by_unit = {}
    lsb = ls[[(t, r, s_) in sec_unit for t, r, s_ in zip(ls.twp, ls.rng, ls.sec)]]
    for t, r, s_, e, st in zip(lsb.twp, lsb.rng, lsb.sec, pd.to_datetime(lsb.expiration_date), lsb.status):
        if st == "LEASED" and pd.notna(e) and horizon[0] <= e <= horizon[1]:
            exp_by_unit.setdefault(sec_unit[(t, r, s_)], []).append(e)
    aub = au[[(t, r, s_) in sec_unit for t, r, s_ in zip(au.township, au["range"], au.section)]]
    for t, r, s_, yr, ln in zip(aub.township, aub["range"], aub.section, aub.auction_year, aub.lease_no):
        if ln in set(lsb.lease_no):
            continue                                            # the lease record carries the expiry
        e = pd.Timestamp(f"{int(yr) + PRIMARY_TERM_YRS}-12-31")
        if horizon[0] <= e <= horizon[1]:
            exp_by_unit.setdefault(sec_unit[(t, r, s_)], []).append(e)
    deadline = np.full(U, T - 1)
    hbp = parents.sum(1) > 0
    for u, es in exp_by_unit.items():
        if owned[u] and not hbp[u]:
            deadline[u] = min(T - 1, min(es).year - PLAN_YEAR0)
    P["deadlines"] = dict(state_lease_rows_in_block=int(len(lsb)), state_lease_status=lsb.status.value_counts().to_dict(),
                          state_lease_expiry_years=pd.to_datetime(lsb.expiration_date).dt.year.value_counts().sort_index().to_dict(),
                          auction_rows_in_block=int(len(aub)), auction_years_in_block=aub.auction_year.value_counts().sort_index().to_dict(),
                          units_with_expiry_in_horizon=int(len(exp_by_unit)),
                          units_with_expiry_in_horizon_but_held_by_production=int(sum(hbp[u] for u in exp_by_unit)),
                          deadline_counts={int(k): int(v) for k, v in zip(*np.unique(deadline, return_counts=True))},
                          rule=("owned unit without producing parents and a state lease (or auctioned tract, primary term "
                                f"{PRIMARY_TERM_YRS} years) expiring in {horizon[0].year}-{horizon[1].year}: deadline = expiry year - "
                                f"{PLAN_YEAR0}; else T-1. Units with producing parents are held by production."))
    say("deadlines:", P["deadlines"])

    # ---- 5. capacities ------------------------------------------------------------------------------------
    rg = read("x_nd_daily_rigs", ["report_date", "operator", "county", "township", "range", "section"])
    rg["y"] = pd.to_datetime(rg.report_date).dt.year
    ry = rg[rg.y == RIG_YEAR]
    nrep = ry.report_date.nunique()
    mine = ry[(ry.operator.map(norm_op) == op) & (ry.county.str.upper() == COUNTY.upper())]
    rigs = len(mine) / nrep
    in_block = sum((int(t), int(r), int(s_)) in sec_unit for t, r, s_ in zip(mine.township.fillna(0), mine["range"].fillna(0),
                                                                            mine.section.fillna(0)))
    rig_days = rigs * 365 * RIG_UTIL
    wpy = rig_days / DEFAULTS["rig_days"][2]
    cap = dict(rig=np.full(T, rig_days), crew=np.full(T, wpy * 2.0 * 1.15),
               capital=np.full(T, 0.9 * wpy * DEFAULTS["cost"][2] * 0.8 + PAD_COST * 0.15 * U), acq=0.6 * U)
    P["capacities"] = dict(rig_source=f"x_nd_daily_rigs {RIG_YEAR}: rows for {op} in {COUNTY} County / {nrep} daily reports",
                           rigs=round(rigs, 3), rig_rows_in_block_share=round(in_block / max(len(mine), 1), 3), utilization=RIG_UTIL,
                           rig_days=round(rig_days, 1), wells_per_year=round(wpy, 2), crew_lateral_miles=round(float(cap["crew"][0]), 2),
                           capital_musd=round(float(cap["capital"][0]), 2), acq_budget_musd=cap["acq"],
                           note="formulas of instance.make_instance with wpy = rig-days / 25; daily rig lists exist only for 2021-08 to 2022-12")
    say("capacities:", P["capacities"])

    # ---- 6. packages -------------------------------------------------------------------------------------
    blk = (col // 2) * 1000 + (pair // 2)
    pkg_ids = {q: i for i, q in enumerate(np.unique(blk[~owned]))}
    pkg_of = np.array([pkg_ids[q] if not o else -1 for q, o in zip(blk, owned)])
    a2 = au[au.auction_year.between(*AUCTION_YEARS) & au.bonus_per_acre.notna()]
    price_src = {}

    a_all = au[au.bonus_per_acre.notna() & (au.auction_year <= PLANNING_DATE.year)]

    def bonus(twp, rng):
        near = lambda d, rad: d[(d.township - twp).abs().le(rad) & (d["range"] - rng).abs().le(rad)]  # noqa: E731
        for rad, tag in ((0, "same township"), (1, "3x3 townships"), (2, "5x5 townships")):
            v = near(a2, rad).bonus_per_acre
            if len(v) >= 3:
                return float(v.median()), f"{tag}, {AUCTION_YEARS[0]}-{AUCTION_YEARS[1]}", int(len(v))
        prev = a_all[a_all.auction_year.between(*PREV_AUCTION_YEARS)]
        for rad, tag in ((2, "5x5 townships"), (3, "7x7 townships")):  # the previous six years, the same window for every township
            v = near(prev, rad).bonus_per_acre
            if len(v) >= 3:
                return float(v.median()), f"{tag}, {PREV_AUCTION_YEARS[0]}-{PREV_AUCTION_YEARS[1]} (no nearby bids in {AUCTION_YEARS[0]}-{AUCTION_YEARS[1]})", int(len(v))
        v = a2[a2.county.str.upper() == COUNTY.upper()].bonus_per_acre
        return float(v.median()), "county", int(len(v))
    cty_med = float(a2[a2.county.str.upper() == COUNTY.upper()].bonus_per_acre.median())
    pkg_cost = np.zeros(len(pkg_ids))
    unit_price = np.zeros(U)
    for u in np.flatnonzero(~owned):
        r = B.iloc[secs[u][0]]
        bpa, tag, n = bonus(int(r.township), int(r["range"]))
        price_src[f"{int(r.township)}N-{int(r['range'])}W"] = dict(median_bonus_per_acre=round(bpa, 1), source=tag, bids=n)
        unit_price[u] = 1280 * bpa / 1e6
        pkg_cost[pkg_of[u]] += unit_price[u]
    P["packages"] = dict(n=int(len(pkg_ids)), units_in_packages=int((~owned).sum()), price_by_township=price_src,
                         county_median_bonus_2019_2024=cty_med,
                         bids_in_5x5_after_planning_date=au[au.auction_year > PLANNING_DATE.year].pipe(
                             lambda d: d[(d.township - int(B.township.min())).abs().le(3) & (d["range"] - int(B["range"].min())).abs().le(3)]
                         ).groupby("auction_year").bonus_per_acre.median().to_dict(),
                         cost_musd=np.round(pkg_cost, 3).tolist(),
                         rule=(f"unowned units in 2x2 blocks (two columns x two pairs); price = median bonus per acre of "
                               f"x_nd_lease_auction_results {AUCTION_YEARS[0]}-{AUCTION_YEARS[1]} in the unit's township "
                               "(else the 3x3 or 5x5 township neighborhood; else the 5x5 or 7x7 neighborhood in "
                               f"{PREV_AUCTION_YEARS[0]}-{PREV_AUCTION_YEARS[1]}; else the county) x 1280 acres; nominal dollars"))
    say("packages:", {k: v for k, v in P["packages"].items() if k != "cost_musd"}, "cost range", pkg_cost.min() if len(pkg_cost) else None,
        pkg_cost.max() if len(pkg_cost) else None)

    # ---- assemble ------------------------------------------------------------------------------------------
    I = copy.deepcopy(DEFAULTS)
    I.update(range_mi=float(uf["range_mi"]), sd_bench=tuple(float(v) for v in sd_bench))
    _, Sigma_prior = field_cov(X, I["range_mi"], I["bench_corr"], I["sd_bench"])
    mu_prior = mu.copy()
    rng = np.random.default_rng(SEED)
    mu_post, Sigma, g_true = condition_field(mu_prior, Sigma_prior, parents, I["sig_eps"], rng, False)
    sd_prior = np.sqrt(np.diag(Sigma_prior)); sd_post = np.sqrt(np.maximum(np.diag(Sigma), 0))
    I.update(dict(nx=NX, ny=NY, U=U, secs=secs, nbr=nbr, X=X, mu=mu_post, Sigma=Sigma, mu_prior=mu_prior, Sigma_prior=Sigma_prior,
                  g_true=g_true, owned=owned, WI=WI, NRI=NRI, parents=parents, deadline=deadline, pkg_of=pkg_of, pkg_cost=pkg_cost,
                  cap=cap, n_sec=NX * NY, seed=SEED, own_frac=float(owned.mean()), deadline_frac=float((deadline < T - 1).mean()),
                  parent_frac=float((parents.sum(1) > 0).mean()), rig_mult=float(wpy / (0.30 * U))))
    build_nodes(I, N_NODES)
    P["posterior"] = dict(rule="condition_field(mu_prior, Sigma_prior, parents, sig_eps, rng(seed 0)); posterior mean kept at the prior",
                          sd_prior_mean=round(float(sd_prior.mean()), 3), sd_post_mean=round(float(sd_post.mean()), 3))
    P["defaults_kept"] = {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in DEFAULTS.items()
                          if k not in ("range_mi", "sd_bench")}
    P["module_constants_kept"] = dict(L_LANES=LANE_CAP, T=T, PAD_COST=PAD_COST, ROYALTY=ROYALTY, A_BENCH="(1.0, 1.25)", RF=RF)
    P["inputs"] = {f: dict(rows=int(pq.ParquetFile(RAW / f"{f}.parquet").metadata.num_rows))
                   for f in ["x_nd_plss_sections", "x_nd_drilling_spacing_units", "x_nd_laterals", "wells", "directional_surveys",
                             "x_bottomhole_locations", "production_monthly", "permits", "x_nd_daily_transfers", "leases_state",
                             "x_nd_lease_auction_results", "x_nd_daily_rigs", "x_nd_cat_county_boundaries"]}
    P["data_copied"] = "2026-10-02 / 2026-10-04 from a local copy of the public NDIC layers"
    P["choices"] = dict(PLANNING_DATE=str(PLANNING_DATE.date()), PLAN_YEAR0=PLAN_YEAR0, CAL_YEARS=CAL_YEARS, CAL_RADIUS_MI=CAL_RADIUS_MI,
                        KERNEL_SD_MI=KERNEL_SD_MI, CLASSIFY_BUFFER_MI=CLASSIFY_BUFFER_MI, TWO_MILE_CLASS=TWO_MILE_CLASS,
                        TVD_BAKKEN=TVD_BAKKEN, P_REF=P_REF, SEV_TAX=SEV_TAX, OPEX_BBL=OPEX_BBL, RF=RF,
                        TERMINAL_DECLINE=TERMINAL_DECLINE, EUR_YEARS=EUR_YEARS, WI_OWNED=WI_OWNED, WI_ACQ=WI_ACQ,
                        RIG_UTIL=RIG_UTIL, RIG_YEAR=RIG_YEAR, AUCTION_YEARS=AUCTION_YEARS, PREV_AUCTION_YEARS=PREV_AUCTION_YEARS, PRIMARY_TERM_YRS=PRIMARY_TERM_YRS,
                        SEED=SEED, N_NODES=N_NODES)
    I["provenance"] = P

    # ---- 8. validation (numpy only) -----------------------------------------------------------------------
    from model import Model
    import baselines
    Mo = Model(I, 1 / 100)
    plan = baselines.heuristic_plan(Mo)
    ok = Mo.feasible(plan, verbose=True)
    ev = Mo.evaluate(plan)
    empty = Mo.evaluate(Mo.empty_plan())
    wells0 = int(plan["z0"] @ Mo.wells[Mo.idx0])
    V = dict(options=int(Mo.n), first_year_options=int(Mo.n0), recourse_options=int(Mo.n1),
             options_by_bench={b: int((Mo.o_b == b).sum()) for b in range(NB)},
             unit_benches_with_options=int(len(np.unique(Mo.o_ub))), lane_cap_total=float(Mo.lane_cap.sum()),
             heuristic_feasible=bool(ok), heuristic_first_year_wells=wells0, heuristic_packages=int(plan["y"].sum()),
             heuristic_ce=round(ev["ce"], 3), heuristic_enpv=round(ev["enpv"], 3), heuristic_lam_sigma=round(ev["lam_sigma"], 4),
             empty_plan_enpv=round(empty["enpv"], 3), empty_plan_ce=round(empty["ce"], 3),
             empty_plan_lam_sigma=round(empty["lam_sigma"], 4), lam=0.01)
    assert ok and np.isfinite(ev["ce"]) and np.isfinite(ev["enpv"])
    P["validation"] = V
    say("validation:", V)

    # ---- 7. outputs ----------------------------------------------------------------------------------------
    DER.mkdir(parents=True, exist_ok=True)
    with open(DER / "ndic_mckenzie_40u.pkl", "wb") as f:
        pickle.dump(I, f)
    summ = dict(area=area, units=U, planning_operator=op, owned_units=int(owned.sum()),
                parents_by_bench=dict(MB=int(parents[:, 0].sum()), TF=int(parents[:, 1].sum())),
                unit_benches_with_parents=int((parents > 0).sum()), mu=P["mu"], bench_ratio=P["bench_ratio"],
                field={k: v for k, v in P["field"].items() if k != "variogram"}, deadlines=P["deadlines"]["deadline_counts"],
                capacities={k: v for k, v in P["capacities"].items()}, packages=dict(n=P["packages"]["n"],
                cost_musd=P["packages"]["cost_musd"]), dsu_match=match, validation=V,
                per_unit=[dict(u=int(u), col=int(col[u]), pair=int(pair[u]), sections=[P["sections"][secs[u][0]], P["sections"][secs[u][1]]],
                               x_mi=round(float(X[u, 0]), 3), y_mi=round(float(X[u, 1]), 3), mu_mb=round(float(mu[u, 0]), 3),
                               mu_tf=round(float(mu[u, 1]), 3), parents_mb=int(parents[u, 0]), parents_tf=int(parents[u, 1]),
                               parents_raw=[int(v) for v in parents_raw[u]], owned=bool(owned[u]), deadline=int(deadline[u]),
                               package=int(pkg_of[u]), operators=unit_ops.get(u, {})) for u in range(U)],
                peak_rss_mb=round(mrss(), 1))
    with open(DER / "ndic_mckenzie_40u_summary.json", "w") as f:
        json.dump(summ, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    make_figure(B, ug, X, parents, owned, pkg_of, mu, x, A, U, area, op)
    say("peak RSS MB", round(mrss(), 1))
    return I


def make_figure(B, ug, X, parents, owned, pkg_of, mu, x, A, U, area, op):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    fig, ax = plt.subplots(figsize=(6.2, 6.6))
    norm = Normalize(mu[:, 0].min(), mu[:, 0].max())
    cmap = matplotlib.colormaps["Blues"]
    polys = [shapely.get_coordinates(shapely.convex_hull(g)) for g in ug]
    ax.add_collection(PolyCollection(polys, facecolors=[cmap(0.12 + 0.6 * norm(v)) for v in mu[:, 0]], edgecolors="#fcfcfb", linewidths=1.5))
    for g in B.gx.values:
        c = shapely.get_coordinates(g)
        ax.plot(c[:, 0], c[:, 1], color="#ffffff", lw=0.4, alpha=0.6)
    for u in np.flatnonzero(owned):
        c = polys[u]
        ax.plot(c[:, 0], c[:, 1], color="#0b0b0b", lw=1.6)
    for gidx in np.unique(pkg_of[pkg_of >= 0]):
        hull = shapely.unary_union([ug[u] for u in np.flatnonzero(pkg_of == gidx)]).buffer(-0.04)
        for part in getattr(hull, "geoms", [hull]):
            c = shapely.get_coordinates(part.exterior)
            ax.plot(c[:, 0], c[:, 1], color="#eb6834", lw=1.2, ls=(0, (3, 2)))
    par = x[x.api10.isin(A[A.producing].api10)]
    for g, b in zip(par.gx.values, par.bench.values):
        c = shapely.get_coordinates(g)
        ax.plot(c[:, 0], c[:, 1], color="#52514e" if b == 0 else "#8a6d1f", lw=0.5, alpha=0.7, ls="-" if b == 0 else (0, (2, 1)))
    for u in range(U):
        ax.text(X[u, 0], X[u, 1], f"{parents[u, 0]}|{parents[u, 1]}\n${mu[u, 0]:.1f}M", ha="center", va="center", fontsize=6.5,
                color="#0b0b0b", bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.85))
    ax.set_xlim(-0.3, 8.3); ax.set_ylim(-0.3, 10.3); ax.set_aspect("equal")
    ax.set_xlabel("miles east of the block's SW corner"); ax.set_ylabel("miles north")
    ax.set_title(f"NDIC-calibrated block: {area}\nlabels: parents MB | TF, and mean MB revenue per two-mile well", fontsize=8.5)
    sm = matplotlib.cm.ScalarMappable(norm=Normalize(norm.vmin, norm.vmax), cmap=matplotlib.colors.ListedColormap(
        [cmap(0.12 + 0.6 * v) for v in np.linspace(0, 1, 256)]))
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Middle Bakken mean, $M per two-mile well", fontsize=7); cb.ax.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#0b0b0b", lw=1.6, label=f"owned unit ({op.title()})"),
         Line2D([], [], color="#eb6834", lw=1.2, ls=(0, (3, 2)), label="acreage package"),
         Line2D([], [], color="#52514e", lw=0.8, label="parent lateral, Middle Bakken"),
         Line2D([], [], color="#8a6d1f", lw=0.8, ls=(0, (2, 1)), label="parent lateral, Three Forks")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=6.5, frameon=False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "ndic_area.pdf")
    fig.savefig(DER / "ndic_area.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
