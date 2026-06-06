# -*- coding: utf-8 -*-
"""Temporal Alignment of Vessels and Truck Bookings.ipynb
"""

pip install pandas numpy scipy statsmodels matplotlib openpyxl

import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.ardl import ARDL, UECM
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson


# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE  = "PortBotany_ARDL_Paper.xlsx"
OUTPUT_DIR  = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vessel–truck pairing: which vessel event to test each truck type against
PAIRINGS = {
    "IMPORT":  "Arrival",
    "EXPORT":  "Departure",
    "STKFLIN": "Departure",   # Stack Run-In FULL  = export → vessel departure
    "STKMTIN": "Departure",   # Stack Run-In EMPTY = export → vessel departure
    "STKOUT":  "Arrival",     # Stack Run Out      = import → vessel arrival
}

# Free-storage policy windows (days) + 2-day buffer
MAX_LAG = {
    "IMPORT":  5,   # 3 days free storage + 2
    "EXPORT":  9,   # 7 days free storage + 2
    "STKFLIN": 9,
    "STKMTIN": 9,
    "STKOUT":  5,
}

TRUCK_TYPES = list(PAIRINGS.keys())

# Commodity codes to analyse
COMMODITIES = ['GEN', 'REF', 'HAZ', 'MT']   # OOG and NEST excluded (sparse)

# Movement groupings → vessel column
IMPORT_TYPES = ['IMPORT', 'STKOUT']
EXPORT_TYPES = ['EXPORT', 'STKFLIN', 'STKMTIN']

# Free-storage lag windows
MAX_LAG_IMPORT = 5   # 3-day free storage + 2-day buffer
MAX_LAG_EXPORT = 9   # 7-day free storage + 2-day buffer

def build_daily_series(trucks_df, vessel_df):
    dates = pd.date_range("2022-09-01", "2022-12-31", freq="D")
    trucks_df = trucks_df.copy()
    trucks_df["date"] = trucks_df["TIMESLOTDATE"].dt.normalize()
    out = pd.DataFrame(index=dates, dtype=float)
    out.index.name = "date"

    for typ in TRUCK_TYPES:
        cnt = (trucks_df[trucks_df["Timeslot type"] == typ]
               .groupby("date").size()
               .reindex(dates, fill_value=0))
        out[typ] = cnt.values

    arr = (pd.to_datetime(vessel_df["Arrival"]).dt.normalize()
           .value_counts().reindex(dates, fill_value=0))
    dep = (pd.to_datetime(vessel_df["Departure"]).dt.normalize()
           .value_counts().reindex(dates, fill_value=0))
    out["VESSEL_ARR"] = arr.values
    out["VESSEL_DEP"] = dep.values

    return out

def build_commodity_daily(trucks_df, vessel_df, commodity, movement):
    df = trucks_df.copy()
    df['date'] = df['TIMESLOTDATE'].dt.normalize()

    types = IMPORT_TYPES if movement == 'import' else EXPORT_TYPES
    vessel_col = 'Arrival'   if movement == 'import' else 'Departure'

    sub = df[
        (df['COMMODITYCODE'] == commodity) &
        (df['Timeslot type'].isin(types))
    ]

    truck_series = (sub.groupby('date').size()
                       .reindex(DATES, fill_value=0)
                       .astype(float))

    vessel_dt = pd.to_datetime(vessel_df[vessel_col]).dt.normalize()
    vessel_series = (vessel_dt.value_counts()
                               .reindex(DATES, fill_value=0)
                               .astype(float))

    return truck_series, vessel_series, vessel_col

# ADF test
def unit_root_tests(daily, term_name):
    print(f"\n{'─'*50}")
    print(f"Unit Root Tests  —  {term_name}")
    print(f"{'─'*50}")
    print(f"  {'Series':<18} {'ADF p':>8} {'KPSS p':>8}  {'Order'}")

    results = {}
    cols = TRUCK_TYPES + ["VESSEL_ARR","VESSEL_DEP"]
    for col in cols:
        series = daily[col].values.astype(float)
        adf_p  = adfuller(series, autolag="AIC")[1]
        try:
            kpss_p = kpss(series, regression="c", nlags="auto")[1]
        except:
            kpss_p = 0.10

        order = "I(1)" if (adf_p > 0.05 or kpss_p < 0.05) else "I(0)"
        results[col] = {"adf_p": adf_p, "kpss_p": kpss_p, "order": order}
        print(f"  {col:<18} {adf_p:>8.4f} {kpss_p:>8.4f}  {order}")

    return results

# Same Day Analysis
def same_day_diffs(trucks_df, vessel_df, truck_type, vessel_col):
    sub = trucks_df[trucks_df["Timeslot type"] == truck_type].copy()
    sub["date"] = sub["TIMESLOTDATE"].dt.date

    vdf = vessel_df.copy()
    vdf["date"]        = pd.to_datetime(vdf[vessel_col]).dt.date
    vdf["vessel_hour"] = (pd.to_datetime(vdf[vessel_col]).dt.hour +
                          pd.to_datetime(vdf[vessel_col]).dt.minute / 60)

    diffs = []
    for _, vrow in vdf.iterrows():
        day_trucks = sub[sub["date"] == vrow["date"]]
        if len(day_trucks) > 0:
            diffs.extend((day_trucks["TIMEZONE"].values - vrow["vessel_hour"]).tolist())

    return np.array(diffs)

# same-day t-test and Wilcoxon for all truck types
def run_samedaytest(trucks_df, vessel_df, term_name):
    print(f"\n{'─'*60}")
    print(f"Same-Day Matched Time Differences  —  {term_name}")
    print(f"{'─'*60}")
    print(f"  {'Series':<35} {'n':>8} {'mean δ':>8} {'med δ':>7} "
          f"{'lead%':>6} {'lag%':>6} {'t-p':>10} {'W-p':>10}")

    results = {}
    for truck_type in TRUCK_TYPES:
        vcol  = PAIRINGS[truck_type]
        diffs = same_day_diffs(trucks_df, vessel_df, truck_type, vcol)
        if len(diffs) == 0:
            continue

        t_stat, t_p = stats.ttest_1samp(diffs, 0)
        try:
            _, w_p = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        except:
            w_p = np.nan

        results[truck_type] = dict(
            n       = len(diffs),
            mean    = float(np.mean(diffs)),
            median  = float(np.median(diffs)),
            std     = float(np.std(diffs)),
            pct_lead= float((diffs < 0).mean() * 100),
            pct_lag = float((diffs > 0).mean() * 100),
            t_stat  = float(t_stat),
            t_p     = float(t_p),
            w_p     = float(w_p),
            diffs   = diffs,
            vcol    = vcol,
        )

        def sp(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        label = f"{truck_type} ~ {vcol}"
        print(f"  {label:<35} {len(diffs):>8,} {np.mean(diffs):>+8.3f} "
              f"{np.median(diffs):>+7.3f} {(diffs<0).mean()*100:>6.1f} "
              f"{(diffs>0).mean()*100:>6.1f} {t_p:>10.3e} "
              f"{'' if np.isnan(w_p) else f'{w_p:.3e}'}{sp(t_p)}")

    return results

# Hourly cross-correlation Function (CCF)

def hourly_ccf(trucks_df, vessel_df, truck_type, vessel_col, max_lag_h=48):

    idx = pd.date_range("2022-09-01", "2022-12-31 23:00", freq="h")
    n   = len(idx)

    # Truck hourly counts
    sub = trucks_df[trucks_df["Timeslot type"] == truck_type].copy()
    sub["slot_h"] = sub["slot_dt"].dt.floor("h")
    truck_ts = sub.groupby("slot_h").size().reindex(idx, fill_value=0).values.astype(float)

    # Vessel hourly events
    vessel_h = pd.to_datetime(vessel_df[vessel_col]).dt.floor("h")
    vessel_ts = vessel_h.value_counts().reindex(idx, fill_value=0).values.astype(float)

    # Normalised CCF
    yn = (truck_ts  - truck_ts.mean())  / (truck_ts.std()  + 1e-12)
    xn = (vessel_ts - vessel_ts.mean()) / (vessel_ts.std() + 1e-12)
    full = np.correlate(yn, xn, mode="full") / n
    mid  = len(full) // 2
    lags = np.arange(-max_lag_h, max_lag_h + 1)
    ccf_vals = full[mid - max_lag_h : mid + max_lag_h + 1]
    ci = 1.96 / np.sqrt(n)

    peak_idx = int(np.argmax(np.abs(ccf_vals)))
    return lags, ccf_vals, ci, int(lags[peak_idx]), float(ccf_vals[peak_idx])

# RAYLEIGH TEST FOR CIRCULAR NON-UNIFORMITY

def rayleigh_test(angles_rad):
    """
    Rayleigh test for uniformity of a circular distribution.
    H0: angles are uniformly distributed on the circle.

    Returns:
      R     mean resultant length (0=uniform, 1=concentrated)
      z     test statistic = n * R²
      p     p-value (Mardia 1972 approximation)
      mu_h  mean direction (hours, 0–24)
    """
    n = len(angles_rad)
    if n == 0:
        return 0.0, 0.0, 1.0, 0.0

    C   = np.cos(angles_rad).mean()
    S   = np.sin(angles_rad).mean()
    R   = np.sqrt(C**2 + S**2)
    z   = n * R**2

    # p-value approximation
    p = np.exp(-z) * (1
        + (2*z - z**2) / (4*n)
        - (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288*n**2))
    p = float(np.clip(p, 0, 1))

    mu_rad = np.arctan2(S, C)
    mu_h   = mu_rad * 12 / np.pi
    if mu_h < 0:
        mu_h += 24

    return float(R), float(z), p, float(mu_h)


def run_rayleigh(trucks_df, vessel_df, term_name):
    print(f"\n{'─'*60}")
    print(f"Rayleigh Test  —  {term_name}")
    print(f"{'─'*60}")
    print(f"  {'Series':<35} {'R':>7} {'p(tz)':>10} {'modal':>6} "
          f"{'R_diff':>7} {'p(diff)':>10}")

    results = {}
    for truck_type in TRUCK_TYPES:
        vcol = PAIRINGS[truck_type]
        sub  = trucks_df[trucks_df["Timeslot type"] == truck_type].copy()
        sub["date"] = sub["TIMESLOTDATE"].dt.date
        v_days = set(pd.to_datetime(vessel_df[vcol]).dt.date)
        on_vdays = sub[sub["date"].isin(v_days)]

        # Test 1: TIMEZONE distribution on vessel-event days
        tz_rad = on_vdays["TIMEZONE"].values * (2 * np.pi / 24)
        R, z, p, mu_h = rayleigh_test(tz_rad)

        # Test 2: Time differences (mod 24h) distribution
        # Reuse diffs computed in same-day test
        # Compute here again for self-containedness
        vdf = vessel_df.copy()
        vdf["date"]        = pd.to_datetime(vdf[vcol]).dt.date
        vdf["vessel_hour"] = (pd.to_datetime(vdf[vcol]).dt.hour +
                              pd.to_datetime(vdf[vcol]).dt.minute / 60)
        diffs = []
        for _, vrow in vdf.iterrows():
            dt = sub[sub["date"] == vrow["date"]]
            if len(dt) > 0:
                diffs.extend((dt["TIMEZONE"].values - vrow["vessel_hour"]).tolist())
        diffs = np.array(diffs)
        if len(diffs) > 0:
            diffs_circ = (diffs % 24) * (2 * np.pi / 24)
            R_d, _, p_d, mu_d = rayleigh_test(diffs_circ)
        else:
            R_d, p_d, mu_d = 0.0, 1.0, 0.0

        results[truck_type] = dict(
            R=R, z=z, p_ray=p, mu_h=mu_h,
            R_diff=R_d, p_diff=p_d, mu_diff_h=mu_d,
            n=len(on_vdays)
        )

        def sp(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        label = f"{truck_type} ~ {vcol}"
        print(f"  {label:<35} {R:>7.4f} {p:>10.4g}{sp(p)} {mu_h:>6.1f} "
              f"{R_d:>7.4f} {p_d:>10.4g}{sp(p_d)}")

    return results

# ARDL + DISTRIBUTED LAG + GRANGER CAUSALITY
def run_ardl(daily, term_name, truck_col, vessel_col, max_lag):
    y     = daily[truck_col].astype(float)
    x     = daily[[vessel_col]].astype(float)
    n     = len(y)
    xname = vessel_col

    print(f"\n  {term_name} | {truck_col} ~ {vessel_col} | max_lag={max_lag}")

    # Step 1: AIC model selection
    best_aic, best_p, best_q = np.inf, 1, 0
    for p in range(1, 5):
        for q in range(0, max_lag + 1):
            try:
                m = ARDL(y, lags=p, order={xname: q}, exog=x)
                r = m.fit()
                if r.aic < best_aic:
                    best_aic, best_p, best_q = r.aic, p, q
            except:
                pass

    # Step 2: Fit ARDL
    model = ARDL(y, lags=best_p, order={xname: best_q}, exog=x)
    res   = model.fit(cov_type="HC3")

    lr_x  = sum(v for k, v in res.params.items() if vessel_col in str(k))
    lr_ar = sum(v for k, v in res.params.items()
                if truck_col in str(k) and "L" in str(k))
    lr = lr_x / (1 - lr_ar) if abs(1 - lr_ar) > 1e-9 else np.nan

    ss_res = (res.resid**2).sum()
    ss_tot = ((y.values[best_p:] - y.values[best_p:].mean())**2).sum()
    r2 = max(0.0, 1 - ss_res / ss_tot)
    dw = durbin_watson(res.resid)

    # Step 3: Bounds test
    bounds_stat = bounds_p = crit_I0 = crit_I1 = np.nan
    if best_q >= 1:         # bounds test requires q ≥ 1
        try:
            uecm  = UECM(y, lags=best_p, order={xname: best_q}, exog=x)
            ures  = uecm.fit()
            bt    = ures.bounds_test(case=3)
            bounds_stat = float(bt.stat)
            bounds_p    = float(bt.p_value)
            crit_I0     = float(bt.critical_values.iloc[2, 0])  # 5% I(0)
            crit_I1     = float(bt.critical_values.iloc[2, 1])  # 5% I(1)
        except:
            pass

    # Step 4: AR(1)-DL model
    trim = max_lag + 1
    Y    = y.values[trim:]
    Yar  = y.values[trim - 1 : n - 1]
    Xdl  = np.column_stack([x.values[trim - k : n - k, 0]
                             for k in range(max_lag + 1)])
    Xall = np.column_stack([np.ones(len(Y)), Yar, Xdl])
    ols  = OLS(Y, Xall).fit(cov_type="HC3")
    coef = ols.params[2:]
    pval = ols.pvalues[2:]
    se   = ols.bse[2:]
    dl_r2 = max(0.0, 1 - (ols.resid**2).sum() / ((Y - Y.mean())**2).sum())

    sig_lags = [k for k in range(max_lag + 1) if pval[k] < 0.05]

    # Step 5: Granger causality
    gc_data = np.column_stack([y.values, x.values[:, 0]])
    gc_pv   = {}
    first_sig_gc = None
    for lag in range(1, max_lag + 1):
        try:
            gc    = grangercausalitytests(gc_data, maxlag=lag, verbose=False)
            p_gc  = gc[lag][0]["ssr_ftest"][1]
            F_gc  = gc[lag][0]["ssr_ftest"][0]
            gc_pv[lag] = (F_gc, p_gc)
            if p_gc < 0.05 and first_sig_gc is None:
                first_sig_gc = lag
        except:
            pass

    # Step 6: Daily CCF
    yv = y.values.astype(float)
    xv = x.values[:, 0].astype(float)
    yn = (yv - yv.mean()) / (yv.std() + 1e-12)
    xn = (xv - xv.mean()) / (xv.std() + 1e-12)
    full = np.correlate(yn, xn, mode="full") / n
    mid  = len(full) // 2
    lags_ccf = np.arange(-max_lag, max_lag + 1)
    ccf_vals = full[mid - max_lag : mid + max_lag + 1]
    ccf_ci   = 1.96 / np.sqrt(n)
    peak_idx = int(np.argmax(np.abs(ccf_vals)))
    peak_lag = int(lags_ccf[peak_idx])
    peak_r   = float(ccf_vals[peak_idx])

    def sp(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    print(f"    ARDL({best_p},{best_q})  AIC={best_aic:.1f}  "
          f"R²={r2:.3f}  DW={dw:.2f}  LR={lr:.1f}")
    print(f"    Sig DL lags: {sig_lags}  |  "
          f"1st Granger lag: {first_sig_gc}  |  "
          f"CCF peak: lag={peak_lag}d r={peak_r:.3f}")
    if not np.isnan(bounds_stat):
        print(f"    Bounds test F={bounds_stat:.3f} p={bounds_p:.4f}{sp(bounds_p)}"
              f"  I(0) 5%={crit_I0:.3f}  I(1) 5%={crit_I1:.3f}")

    return dict(
        term=term_name, truck_col=truck_col, vessel_col=vessel_col,
        max_lag=max_lag, best_p=best_p, best_q=best_q, best_aic=best_aic,
        lr=lr, r2=r2, dl_r2=dl_r2, dw=dw,
        bounds_stat=bounds_stat, bounds_p=bounds_p,
        crit_I0=crit_I0, crit_I1=crit_I1,
        dl_coef=coef, dl_pval=pval, dl_se=se,
        sig_lags=sig_lags, gc_pv=gc_pv, first_sig_gc=first_sig_gc,
        ccf_lags=lags_ccf, ccf_vals=ccf_vals, ccf_ci=ccf_ci,
        peak_lag=peak_lag, peak_r=peak_r,
    )


def run_all_ardl(daily, term_name):
   # Run ARDL analysis for all five truck booking types at one terminal.
    print(f"\n{'─'*60}")
    print(f"ARDL Analysis  —  {term_name}")
    print(f"{'─'*60}")
    results = {}
    for truck_type in TRUCK_TYPES:
        vcol    = "VESSEL_ARR" if PAIRINGS[truck_type] == "Arrival" else "VESSEL_DEP"
        max_lag = MAX_LAG[truck_type]
        results[truck_type] = run_ardl(daily, term_name, truck_type, vcol, max_lag)
    return results

def compute_score(result):
    if result is None:
        return 0
    score = 0
    if result['sig_lags']:
        score += 1
    if result['first_sig_gc'] is not None:
        score += 1
    if abs(result['peak_r']) > result['ccf_ci']:
        score += 1
    if not np.isnan(result['lr']) and abs(result['lr']) > 10:
        score += 1
    return score
