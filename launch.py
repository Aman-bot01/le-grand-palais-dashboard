"""
Game Launch Performance Engine
================================
Weekly series indexed from each game's launch date, DTW-based peer matching,
forecast cone (P25/P50/P75) from matched historical peers, AND per-game
statistical expectation bands (Holt-damped / SES ensemble with MAD-robust σ,
rolling-origin backtest, honest 1-step-ahead "vs forecast" KPI).

Data sources
  PFH  : BetSpinSummaryCashView3  (PlatformName='Pong', CasinoName='PFH')
  V2/V1: AnalyticsGameTerminalsGames joined to GameCatalogView1

Monetary units
  BetSpinSummaryCashView3      : TotalBet, TotalWin in cents → /100 to USD
  AnalyticsGameTerminalsGames  : TotalPlay, TotalWin in cents → /100 to USD
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import engine as E

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

EPS = 1e-9

# Platform code → GameCatalogView1.Platform value
_GC_PLATFORM = {"PFH": "igaming", "V2": "v2", "V1": "v1"}

# HR games have IDs ≥ 95000; exclude from benchmark pool
def _is_hr(game_id: int) -> bool:
    return int(game_id) >= 95000


# ── Statistical forecasting (ported + adapted from ForecastEngine) ───

def _smape(a, f) -> float:
    a, f = np.asarray(a, float), np.asarray(f, float)
    d = np.abs(a) + np.abs(f)
    m = d > EPS
    return float(np.mean(2.0 * np.abs(f - a)[m] / d[m]) * 100) if m.any() else np.nan


def _mape(a, f) -> float:
    a, f = np.asarray(a, float), np.asarray(f, float)
    m = np.abs(a) > EPS
    return float(np.mean(np.abs((f - a)[m] / a[m])) * 100) if m.any() else np.nan


def _mae(a, f) -> float:
    return float(np.mean(np.abs(np.asarray(f, float) - np.asarray(a, float))))


def _robust_sigma(resid, recent: int = 52) -> float:
    """MAD-based outlier-resistant σ from the most recent `recent` residuals.
    Prevents a single spike week from permanently widening the band."""
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    if r.size > recent:
        r = r[-recent:]
    mad = np.median(np.abs(r - np.median(r)))
    sd = 1.4826 * mad
    if sd <= EPS:
        sd = float(np.std(r))
    return float(sd)


def _ses_fit(train, alpha: float):
    """In-place SES: fitted[t] = alpha*train[t-1] + (1-alpha)*fitted[t-1]."""
    n = len(train)
    fitted = np.empty(n)
    fitted[0] = train[0]
    for i in range(1, n):
        fitted[i] = alpha * train[i - 1] + (1 - alpha) * fitted[i - 1]
    return fitted


def _holt_fit(train, alpha: float, beta: float, damped: bool = False, phi: float = 0.98):
    """Holt (linear / damped) smoothing — pure numpy, no statsmodels."""
    n = len(train)
    l = np.empty(n); b = np.empty(n); fitted = np.empty(n)
    l[0] = train[0]
    b[0] = train[1] - train[0] if n > 1 else 0.0
    fitted[0] = train[0]
    for i in range(1, n):
        p = phi if damped else 1.0
        l[i] = alpha * train[i] + (1 - alpha) * (l[i - 1] + p * b[i - 1])
        b[i] = beta * (l[i] - l[i - 1]) + (1 - beta) * p * b[i - 1]
        fitted[i] = l[i - 1] + p * b[i - 1]
    # Forecast
    fc = np.empty(0)  # returned separately
    return fitted, l, b


def _holt_forecast(l, b, horizon: int, damped: bool = False, phi: float = 0.98):
    fc = np.empty(horizon)
    p = phi if damped else 1.0
    phi_h = 0.0
    for h in range(1, horizon + 1):
        phi_h = phi_h * p + p if damped else float(h)
        fc[h - 1] = l[-1] + phi_h * b[-1]
    return fc


def _grid_search_ses(train):
    """Find alpha that minimises SSE."""
    best_sse, best_alpha = np.inf, 0.3
    for a in np.arange(0.05, 1.0, 0.05):
        f = _ses_fit(train, a)
        sse = float(np.sum((train - f) ** 2))
        if sse < best_sse:
            best_sse, best_alpha = sse, a
    return best_alpha


def _grid_search_holt(train, damped=False):
    best_sse, best_ab = np.inf, (0.3, 0.1)
    for a in np.arange(0.05, 0.95, 0.1):
        for b in np.arange(0.01, 0.5, 0.1):
            try:
                f, _, _ = _holt_fit(train, a, b, damped=damped)
                sse = float(np.sum((train - f) ** 2))
                if sse < best_sse:
                    best_sse, best_ab = sse, (a, b)
            except Exception:
                pass
    return best_ab


def _fit_model(name: str, train, horizon: int):
    """Fit one model on `train`, return (h-step forecast array, in-sample fitted array).
    Pure numpy/scipy — no statsmodels required."""
    train = np.asarray(train, float)
    n = len(train)
    try:
        if name == "Naive":
            return np.repeat(train[-1], horizon), train.copy()
        if name == "Mean":
            mu = float(np.mean(train))
            return np.repeat(mu, horizon), np.repeat(mu, n)
        if name == "MA4":
            w = min(4, n)
            ma = float(np.mean(train[-w:]))
            fitted = pd.Series(train).rolling(4, min_periods=1).mean().to_numpy()
            return np.repeat(ma, horizon), fitted
        if name == "SES":
            alpha = _grid_search_ses(train)
            fitted = _ses_fit(train, alpha)
            fc = np.repeat(alpha * train[-1] + (1 - alpha) * fitted[-1], horizon)
            return fc, fitted
        if name in ("Holt", "Holt-damped"):
            damped = (name == "Holt-damped")
            alpha, beta = _grid_search_holt(train, damped=damped)
            fitted, l, b = _holt_fit(train, alpha, beta, damped=damped)
            fc = _holt_forecast(l, b, horizon, damped=damped)
            return fc, fitted
        return None, None
    except Exception:
        return None, None


def _backtest(vals, models, min_train: int = 6, max_origins: int = 30):
    """Rolling-origin 1-step-ahead backtest. Returns (scores_dict, last_origin_dict).
    last_origin[model] = (actual, forecast) for the FINAL origin (vals[:-1] → predict vals[-1])
    without ever leaking vals[-1] into the fit — used for the honest "vs forecast" KPI."""
    vals = np.asarray(vals, float)
    n = len(vals)
    start = max(min_train, 2)
    if max_origins:
        start = max(start, n - max_origins)
    scores, last_origin = {}, {}
    for name in models:
        acts, fcs = [], []
        for t in range(start, n):
            fc, _ = _fit_model(name, vals[:t], 1)
            if fc is None:
                continue
            f0 = float(fc[0])
            acts.append(vals[t])
            fcs.append(f0)
            if t == n - 1 and np.isfinite(f0):
                last_origin[name] = (float(vals[t]), f0)
        if len(acts) >= 3:
            scores[name] = {"sMAPE": _smape(acts, fcs), "MAPE": _mape(acts, fcs),
                            "MAE": _mae(acts, fcs), "n": len(acts)}
    return scores, last_origin


def fit_game_forecast(
    series: np.ndarray | list,
    horizon: int = 13,
    z: float = 1.64,
    min_train: int = 6,
    min_floor: float = 0.0,
) -> dict:
    """
    Fit a statistical expectation band + forecast cone for one game's weekly KPI series.

    `series` is a 1-D array of values indexed by launch_week (week 0 first, contiguous).
    Missing weeks should be filled with 0 or np.nan before passing.

    Returns a dict with:
      fitted        list[float]   – in-sample fitted values
      lo_band       list[float]   – fitted − z·σ  (expectation band lower)
      hi_band       list[float]   – fitted + z·σ  (expectation band upper)
      breach        list[bool]    – True where actual is outside the band
      breach_side   list[str]     – "above" / "below" / ""
      n_breach      int
      resid_std     float         – robust σ used for the band
      fc_mean       list[float]   – h-week point forecast
      fc_lo         list[float]   – forecast lower PI
      fc_hi         list[float]   – forecast upper PI
      fc_weeks      list[int]     – week indices for the forecast (max_week+1, +2, …)
      backtest      dict          – {model: {sMAPE, MAPE, MAE, n}}
      chosen        str           – model name used
      vs_fc         dict | None   – {actual, forecast, gap, pct} for last week
      note          str
    """
    vals = np.asarray(series, float)
    n = len(vals)
    out = {
        "fitted": [], "lo_band": [], "hi_band": [],
        "breach": [], "breach_side": [], "n_breach": 0,
        "resid_std": 0.0,
        "fc_mean": [], "fc_lo": [], "fc_hi": [], "fc_weeks": [],
        "backtest": {}, "chosen": "Naive", "ensemble_members": [],
        "vs_fc": None, "note": "",
    }

    if n < 4:
        out["note"] = "Need ≥ 4 weeks to fit a model."
        return out

    mt = max(4, min(min_train, n - 2))
    smoothers = ["SES", "Holt", "Holt-damped"]
    baselines = ["Naive", "Mean", "MA4"]

    bt, last_origin = _backtest(vals, smoothers + baselines, min_train=mt)
    out["backtest"] = bt

    # Ensemble: inverse-sMAPE weighted blend of smoothers that backtested OK
    members = [m for m in smoothers if m in bt and np.isfinite(bt[m]["sMAPE"])]
    if not members:
        members = [m for m in baselines if m in bt] or ["Naive"]
    weights = np.array([1.0 / (bt[m]["sMAPE"] + EPS) if m in bt else 1.0 for m in members])
    weights = weights / weights.sum()
    out["ensemble_members"] = list(zip(members, [round(float(w), 3) for w in weights]))
    out["chosen"] = "Ensemble" if len(members) > 1 else members[0]

    # Full-series fit
    fitted_stack, fc_stack = [], []
    for m, w in zip(members, weights):
        fc, fitted = _fit_model(m, vals, horizon)
        if fc is None or fitted is None:
            continue
        if len(fitted) != n:
            fitted = np.resize(fitted, n)
        fitted_stack.append(w * fitted)
        fc_stack.append(w * fc)
    if not fc_stack:
        fc, fitted = _fit_model("Naive", vals, horizon)
        fitted_stack, fc_stack = [fitted], [fc]

    fitted = np.sum(fitted_stack, axis=0)
    fc_mean = np.maximum(np.sum(fc_stack, axis=0), min_floor)

    resid = vals - fitted
    resid_std = _robust_sigma(resid)
    out["resid_std"] = resid_std

    # Expectation band + breaches
    lo = fitted - z * resid_std
    hi = fitted + z * resid_std
    breach = (vals < lo) | (vals > hi)
    side = np.where(vals > hi, "above", np.where(vals < lo, "below", ""))
    out["fitted"] = fitted.tolist()
    out["lo_band"] = lo.tolist()
    out["hi_band"] = hi.tolist()
    out["breach"] = breach.tolist()
    out["breach_side"] = side.tolist()
    out["n_breach"] = int(breach.sum())

    # Forecast cone — widens as √h: week 1 = σ, week 4 = 2σ, week 9 = 3σ, week 13 = 3.6σ
    widen = np.sqrt(np.arange(1, horizon + 1))
    out["fc_mean"] = fc_mean.tolist()
    out["fc_lo"] = np.maximum(fc_mean - z * resid_std * widen, 0.0).tolist()
    out["fc_hi"] = (fc_mean + z * resid_std * widen).tolist()
    out["fc_weeks"] = list(range(n, n + horizon))

    # Honest "vs forecast" KPI (n ≥ 6): blend the final-origin 1-step forecasts
    if n >= 6:
        pairs = [(w, last_origin[m][1]) for m, w in zip(members, weights)
                 if m in last_origin and np.isfinite(last_origin[m][1])]
        if not pairs:
            for m, w in zip(members, weights):
                fc_m, _ = _fit_model(m, vals[:-1], 1)
                if fc_m is not None and np.isfinite(fc_m[0]):
                    pairs.append((float(w), float(fc_m[0])))
        if not pairs:
            fc_m, _ = _fit_model("Naive", vals[:-1], 1)
            if fc_m is not None and np.isfinite(fc_m[0]):
                pairs = [(1.0, float(fc_m[0]))]
        wsum = float(sum(w for w, _ in pairs))
        if pairs and wsum > EPS:
            f_last = float(sum(w * f for w, f in pairs) / wsum)
            a_last = float(vals[-1])
            gap = a_last - f_last
            out["vs_fc"] = {
                "actual": a_last,
                "forecast": f_last,
                "gap": gap,
                "pct": (gap / f_last * 100.0) if abs(f_last) > EPS else None,
            }

    return out


# ── Prophet forecast ─────────────────────────────────────────────────
def fit_prophet_forecast(series, horizon: int = 13, z: float = 1.64) -> dict | None:
    """
    Prophet-based forecast. Returns same dict shape as fit_game_forecast,
    or None if Prophet is unavailable or insufficient data.
    Uses synthetic weekly dates (arbitrary Monday origin) since we have
    launch-week indices, not real calendar dates.
    """
    if not PROPHET_AVAILABLE or len(series) < 6:
        return None
    try:
        series = np.asarray(series, float)
        start = pd.Timestamp("2020-01-06")
        ds = [start + pd.Timedelta(weeks=i) for i in range(len(series))]
        df_p = pd.DataFrame({"ds": ds, "y": series})
        df_p = df_p[df_p["y"] > 0]
        if len(df_p) < 4:
            return None
        # Enable yearly seasonality only when ≥52 weeks of history exist
        _use_yearly = len(series) >= 52
        m = Prophet(
            yearly_seasonality=_use_yearly,
            weekly_seasonality=False,   # input is already weekly-aggregated
            daily_seasonality=False,
            changepoint_prior_scale=0.3,
            interval_width=min(0.95, z / 2.0),
        )
        if not _use_yearly:
            pass  # note: yearly seasonality suppressed — insufficient history (<52 weeks)
        m.fit(df_p)
        future = m.make_future_dataframe(periods=horizon, freq="W")
        forecast = m.predict(future)
        # In-sample fitted
        in_sample = forecast.iloc[:len(series)]
        fitted = in_sample["yhat"].values
        resid = series - fitted
        resid_std = _robust_sigma(resid)
        lo_band = (fitted - z * resid_std).tolist()
        hi_band = (fitted + z * resid_std).tolist()
        breach = ((series < np.array(lo_band)) | (series > np.array(hi_band))).tolist()
        # Forecast
        fc_rows = forecast.iloc[len(series):]
        fc_mean = fc_rows["yhat"].clip(lower=0).tolist()
        fc_lo   = fc_rows["yhat_lower"].clip(lower=0).tolist()
        fc_hi   = fc_rows["yhat_upper"].clip(lower=0).tolist()
        fc_weeks = list(range(len(series), len(series) + horizon))
        return {
            "fitted": fitted.tolist(), "lo_band": lo_band, "hi_band": hi_band,
            "breach": breach, "breach_side": [], "n_breach": int(sum(breach)),
            "resid_std": resid_std,
            "fc_mean": fc_mean, "fc_lo": fc_lo, "fc_hi": fc_hi, "fc_weeks": fc_weeks,
            "backtest": {}, "chosen": "Prophet", "ensemble_members": [],
            "vs_fc": None, "note": "Prophet (Meta) — detects trend changes automatically.",
            "model": "Prophet",
        }
    except Exception:
        return None


# ── Scenario forecasting ──────────────────────────────────────────────
def compute_scenario_forecast(
    series,
    horizon: int,
    df: pd.DataFrame,
    peer_ids: list[int],
    from_week: int,
    kpi: str = "bet_handle",
) -> dict:
    """
    Three-scenario forecast: best / base / worst.

    Best  — top 25% peers by average value after from_week.
    Base  — ensemble statistical model (fit_game_forecast fc_mean).
    Worst — extrapolated last-4-week trend rate OR bottom 25% peers, whichever lower.
            Capped at zero.

    Returns dict with keys: best, base, worst (each a list of `horizon` values),
    plus fc_weeks (list of week numbers).
    """
    series = np.asarray(series, float)
    n = len(series)
    fc_weeks = list(range(n, n + horizon))

    # Base: statistical model
    fc_base_obj = fit_game_forecast(series, horizon=horizon)
    base = [float(v) for v in fc_base_obj.get("fc_mean", [0.0] * horizon)]
    if len(base) < horizon:
        base += [base[-1] if base else 0.0] * (horizon - len(base))

    # Collect peer trajectories after from_week
    peer_data = {}
    for gid in peer_ids:
        sub = df[(df["game_id"] == gid) & (df["launch_week"] > from_week)][
            ["launch_week", kpi]].dropna(subset=[kpi])
        if not sub.empty:
            peer_data[gid] = sub

    # ── Best case ────────────────────────────────────────────────────
    if peer_data:
        peer_avgs = {gid: float(v[kpi].mean()) for gid, v in peer_data.items()}
        sorted_desc = sorted(peer_avgs.items(), key=lambda x: x[1], reverse=True)
        top_n = max(1, len(sorted_desc) // 4)
        top_peers = [gid for gid, _ in sorted_desc[:top_n]]
        best = []
        for h in range(horizon):
            target_launch_wk = from_week + h + 1
            vals = []
            for gid in top_peers:
                row = peer_data[gid][peer_data[gid]["launch_week"] == target_launch_wk][kpi]
                if not row.empty:
                    vals.append(float(row.iloc[0]))
            best.append(float(np.mean(vals)) if vals else base[h] * 1.25)
    else:
        best = [v * 1.25 for v in base]

    # ── Worst case ───────────────────────────────────────────────────
    # Trend extrapolation from last 4 weeks
    last4 = series[-4:] if len(series) >= 4 else series
    valid = last4[last4 > 0]
    if len(valid) >= 2:
        dr = float((valid[-1] / valid[0]) ** (1.0 / (len(valid) - 1)))
        dr = max(0.5, min(dr, 1.0))   # cap: max 50% weekly decay, min flat
        trend_worst = [max(0.0, float(series[-1]) * (dr ** (h + 1))) for h in range(horizon)]
    else:
        trend_worst = [0.0] * horizon

    # Bottom 25% peers
    if peer_data:
        sorted_asc = sorted(peer_avgs.items(), key=lambda x: x[1])
        bot_n = max(1, len(sorted_asc) // 4)
        bot_peers = [gid for gid, _ in sorted_asc[:bot_n]]
        peer_worst = []
        for h in range(horizon):
            target_launch_wk = from_week + h + 1
            vals = []
            for gid in bot_peers:
                row = peer_data[gid][peer_data[gid]["launch_week"] == target_launch_wk][kpi]
                if not row.empty:
                    vals.append(float(row.iloc[0]))
            peer_worst.append(float(np.mean(vals)) if vals else trend_worst[h])
        worst = [min(trend_worst[h], peer_worst[h]) for h in range(horizon)]
    else:
        worst = trend_worst

    # Enforce ordering: best >= base >= worst >= 0
    result_base  = [max(0.0, float(base[h])) for h in range(horizon)]
    result_best  = [max(result_base[h], max(0.0, float(best[h]))) for h in range(horizon)]
    result_worst = [max(0.0, min(float(worst[h]), result_base[h])) for h in range(horizon)]

    return {
        "best":     result_best,
        "base":     result_base,
        "worst":    result_worst,
        "fc_weeks": fc_weeks,
    }


# ── DTW ──────────────────────────────────────────────────────────────
def dtw_distance(s1, s2) -> float:
    """Standard DTW distance between two 1-D numeric arrays."""
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    mask1 = np.isfinite(s1)
    mask2 = np.isfinite(s2)
    s1, s2 = s1[mask1], s2[mask2]
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return np.inf
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


# ── Bulk weekly loaders ───────────────────────────────────────────────
def load_pfh_weekly_all(conn) -> pd.DataFrame:
    """
    Single query: all PFH igaming games, weekly series indexed from each
    game's first observed date in BetSpinSummaryCashView3.
    Returns one row per (game_id, launch_week).
    """
    sql = """
    SELECT
        gc.Id                                                          AS game_id,
        gc.Name                                                        AS game_name,
        gc.Codebase                                                    AS codebase,
        fd.launch_date,
        (DATEDIFF('day', fd.launch_date, CAST(b."Date" AS DATE)) // 7)        AS launch_week,
        SUM(CAST(b.TotalBet AS FLOAT) / 100.0)                        AS bet_handle,
        SUM(CAST(b.TotalWin AS FLOAT) / 100.0)                        AS total_win,
        SUM(b.Spins)                                                   AS spins,
        COUNT(DISTINCT b.AccountNumber)                                AS players,
        COUNT(DISTINCT b.StoreNumber)                                  AS stores,
        MAX(CAST(b."Date" AS DATE))                                    AS _last_day
    FROM BetSpinSummaryCashView3 b
    JOIN GameCatalogView1 gc
        ON gc.Id = TRY_CAST(b.GameId AS INT)
    JOIN (
        SELECT TRY_CAST(GameId AS INT) AS gid,
               MIN(CAST("Date" AS DATE))  AS launch_date
        FROM BetSpinSummaryCashView3
        WHERE PlatformName = 'Pong' AND CasinoName = 'PFH'
          AND TRY_CAST(GameId AS INT) IS NOT NULL
        GROUP BY GameId
    ) fd ON fd.gid = TRY_CAST(b.GameId AS INT)
    WHERE gc.Platform = 'igaming'
      AND b.PlatformName = 'Pong' AND b.CasinoName = 'PFH'
      AND TRY_CAST(b.GameId AS INT) IS NOT NULL
      AND CAST(b."Date" AS DATE) >= fd.launch_date
    GROUP BY gc.Id, gc.Name, gc.Codebase, fd.launch_date,
             (DATEDIFF('day', fd.launch_date, CAST(b."Date" AS DATE)) // 7)
    """
    df = E.query_df(conn, sql)
    if df.empty:
        return df
    df["launch_date"] = pd.to_datetime(df["launch_date"]).dt.date
    for col in ("bet_handle", "total_win", "spins", "players", "stores"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_v2v1_weekly_all(conn, platform: str) -> pd.DataFrame:
    """
    Single query: all V2 (or V1) games, weekly series from AnalyticsGameTerminalsGames.
    platform must be 'V2' or 'V1'.
    """
    gc_plat = _GC_PLATFORM[platform]
    sql = f"""
    SELECT
        gc.Id                                                             AS game_id,
        gc.Name                                                           AS game_name,
        gc.Codebase                                                       AS codebase,
        fd.launch_date,
        (DATEDIFF('day', fd.launch_date, CAST(g.SummaryDate AS DATE)) // 7)      AS launch_week,
        SUM(CAST(g.TotalPlay AS FLOAT) / 100.0)                          AS bet_handle,
        SUM(CAST(g.TotalWin  AS FLOAT) / 100.0)                          AS total_win,
        SUM(g.PlayCount)                                                  AS spins,
        COUNT(DISTINCT g.PlayerAccountNumber)                             AS players,
        COUNT(DISTINCT g.SummaryLocationId)                               AS stores,
        MAX(CAST(g.SummaryDate AS DATE))                                  AS _last_day
    FROM AnalyticsGameTerminalsGames g
    JOIN GameCatalogView1 gc ON gc.Id = g.Id
    JOIN (
        SELECT Id, MIN(CAST(SummaryDate AS DATE)) AS launch_date
        FROM AnalyticsGameTerminalsGames
        GROUP BY Id
    ) fd ON fd.Id = g.Id
    WHERE gc.Platform = '{gc_plat}'
      AND CAST(g.SummaryDate AS DATE) >= fd.launch_date
    GROUP BY gc.Id, gc.Name, gc.Codebase, fd.launch_date,
             (DATEDIFF('day', fd.launch_date, CAST(g.SummaryDate AS DATE)) // 7)
    """
    df = E.query_df(conn, sql)
    if df.empty:
        return df
    df["launch_date"] = pd.to_datetime(df["launch_date"]).dt.date
    for col in ("bet_handle", "total_win", "spins", "players", "stores"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_edgelabs_weekly_all(conn) -> pd.DataFrame:
    """
    All EdgeLabs iGaming games, weekly series from BetSpinSummaryCashView3.
    Launch date = first real-money spin date per game (no CRM entry for EdgeLabs).
    stores = COUNT(DISTINCT CasinoName) — the EdgeLabs equivalent of locations.
    """
    sql = """
    SELECT
        TRY_CAST(b.GameId AS INT)                                      AS game_id,
        MAX(gc.Name)                                                    AS game_name,
        MAX(gc.Codebase)                                                AS codebase,
        fd.launch_date,
        (DATEDIFF('day', fd.launch_date, CAST(b."Date" AS DATE)) // 7)         AS launch_week,
        SUM(CAST(b.TotalBet AS FLOAT) / 100.0)                         AS bet_handle,
        SUM(CAST(b.TotalWin AS FLOAT) / 100.0)                         AS total_win,
        SUM(b.Spins)                                                    AS spins,
        COUNT(DISTINCT b.AccountNumber)                                 AS players,
        COUNT(DISTINCT b.CasinoName)                                    AS stores,
        MAX(CAST(b."Date" AS DATE))                                     AS _last_day
    FROM BetSpinSummaryCashView3 b
    LEFT JOIN GameCatalogView1 gc
        ON gc.Id = TRY_CAST(b.GameId AS INT)
    JOIN (
        SELECT TRY_CAST(GameId AS INT) AS gid,
               MIN(CAST("Date" AS DATE)) AS launch_date
        FROM BetSpinSummaryCashView3
        WHERE PlatformName = 'EdgeLabs'
          AND TRY_CAST(GameId AS INT) IS NOT NULL
        GROUP BY GameId
    ) fd ON fd.gid = TRY_CAST(b.GameId AS INT)
    WHERE b.PlatformName = 'EdgeLabs'
      AND TRY_CAST(b.GameId AS INT) IS NOT NULL
      AND CAST(b."Date" AS DATE) >= fd.launch_date
    GROUP BY TRY_CAST(b.GameId AS INT), fd.launch_date,
             (DATEDIFF('day', fd.launch_date, CAST(b."Date" AS DATE)) // 7)
    """
    df = E.query_df(conn, sql)
    if df.empty:
        return df
    df["launch_date"] = pd.to_datetime(df["launch_date"]).dt.date
    for col in ("bet_handle", "total_win", "spins", "players", "stores"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Fill missing game names from GameId
    df["game_name"] = df["game_name"].fillna("Game " + df["game_id"].astype(str))
    df["codebase"]  = df["codebase"].fillna("")
    return df


# ── Quick Score ──────────────────────────────────────────────────────

_QS_MILESTONE_METRICS = [
    ("bet_handle", "Bet Handle",      True),
    ("net_rev",    "Net Revenue",     True),
    ("hold_pct",   "House Take %",    True),
    ("bet_decay",  "Player Return %", False),
]
_QS_MILESTONE_WEEKS = [(30, 4), (60, 8), (90, 13)]


def _qs_status(av, p25, p75, hi: bool):
    if av is None or p25 is None or p75 is None:
        return None
    if hi:
        if av < p25 * 0.8: return "Problem"
        if av < p25:       return "Watch"
        return "Good"
    else:
        if av < p25 * 0.8: return "Problem"
        if av < p25:       return "Watch"
        if av > p75 * 1.2: return "Problem"
        if av > p75:       return "Watch"
        return "Good"


def compute_quick_score(gdf: "pd.DataFrame",
                        ms_data: dict,
                        peer_df: "pd.DataFrame",
                        best_match_score: float = 0.0,
                        player_df: "pd.DataFrame | None" = None) -> dict:
    """
    Compute three quick scores for a single game using full-trend analysis.

    Scores are based on ALL available weeks, not single milestones.
    Works from Week 1 through any number of weeks, all platforms.

    Returns
    -------
    dict with keys 'game_performance', 'sales_impact', 'player_interest'
    Each value: {'label': str, 'color': 'green'|'yellow'|'red', 'reason': str}
    """
    if gdf is None or gdf.empty:
        _nd = {"label": "Not enough data yet", "color": "yellow",
               "reason": "No game data available."}
        return {"game_performance": _nd, "sales_impact": _nd, "player_interest": _nd}

    max_wk = int(gdf["launch_week"].max())

    def _nd(reason):
        return {"label": "Not enough data yet", "color": "yellow", "reason": reason}

    def _trend_score(kpi, higher_is_better=True):
        """
        For each week where both game data and peer P50 exist, score above/below median.
        Returns (weeks_above, weeks_total, latest_val, latest_p50).
        """
        if peer_df.empty or kpi not in gdf.columns:
            return 0, 0, None, None
        bnd = bands_kpi(peer_df, kpi)
        if bnd.empty or "p50" not in bnd.columns:
            return 0, 0, None, None

        game_s = (gdf.sort_values("launch_week")
                     .set_index("launch_week")[kpi]
                     .dropna())
        above = 0
        total = 0
        latest_val = None
        latest_p50 = None

        for wk, val in game_s.items():
            row = bnd[bnd["launch_week"] == wk]
            if row.empty or pd.isna(row["p50"].iloc[0]):
                continue
            p50 = float(row["p50"].iloc[0])
            total += 1
            if higher_is_better:
                if float(val) >= p50:
                    above += 1
            else:
                if float(val) <= p50:
                    above += 1
            if wk == max_wk:
                latest_val = float(val)
                latest_p50 = p50

        return above, total, latest_val, latest_p50

    # ── Score 1: Game Performance ─────────────────────────────────────
    # Use bet_handle + hold_pct across all weeks vs peer median
    _bh_above, _bh_total, _bh_latest, _bh_p50 = _trend_score("bet_handle", higher_is_better=True)
    _hp_above, _hp_total, _hp_latest, _hp_p50 = _trend_score("hold_pct",   higher_is_better=True)

    _gp_total = _bh_total + _hp_total
    _gp_above = _bh_above + _hp_above

    if _gp_total == 0:
        score_1 = _nd("No peer comparison data available yet.")
    else:
        pct_above = _gp_above / _gp_total
        _bh_str = f"${_bh_latest:,.0f} vs peer median ${_bh_p50:,.0f}" if _bh_latest is not None else ""
        _hp_str = f"{_hp_latest:.1f}% hold vs peer median {_hp_p50:.1f}%" if _hp_latest is not None else ""
        reason = f"Above peer median in {_gp_above} of {_gp_total} metric-weeks"
        if _bh_str:
            reason += f" | Bet handle: {_bh_str}"
        if _hp_str:
            reason += f" | Hold: {_hp_str}"
        if pct_above >= 0.60:
            score_1 = {"label": "Better than expected", "color": "green",  "reason": reason}
        elif pct_above >= 0.40:
            score_1 = {"label": "As expected",          "color": "yellow", "reason": reason}
        else:
            score_1 = {"label": "Worse than expected",  "color": "red",    "reason": reason}

    # ── Score 2: Sales Impact ─────────────────────────────────────────
    # Bet handle vs peer median across all weeks
    _si_above, _si_total, _si_latest, _si_p50 = _trend_score("bet_handle", higher_is_better=True)

    if _si_total == 0:
        score_2 = _nd("No peer comparison data available yet.")
    else:
        pct_above = _si_above / _si_total
        reason = f"Above peer median bet handle in {_si_above} of {_si_total} weeks"
        if _si_latest is not None and _si_p50 is not None:
            reason += f" | Latest: ${_si_latest:,.0f} vs peer median ${_si_p50:,.0f}"
        if pct_above >= 0.60:
            score_2 = {"label": "Up",    "color": "green",  "reason": reason}
        elif pct_above >= 0.40:
            score_2 = {"label": "Level", "color": "yellow", "reason": reason}
        else:
            score_2 = {"label": "Down",  "color": "red",    "reason": reason}

    # ── Score 3: Player Interest ──────────────────────────────────────
    # Use real player headcounts if available (PFH/EdgeLabs), else bet_decay vs peer median
    _use_real_players = (
        player_df is not None
        and not player_df.empty
        and "unique_players" in player_df.columns
        and "launch_week" in player_df.columns
    )

    if _use_real_players:
        _pl = (player_df.sort_values("launch_week")
                        .set_index("launch_week")["unique_players"]
                        .astype(float).dropna())
        if _pl.empty:
            score_3 = _nd("No player count data available.")
        else:
            _pl_wk0  = float(_pl.get(0, _pl.iloc[0]))
            _pl_cur  = float(_pl.iloc[-1])
            _pl_cur_wk = int(_pl.index[-1])
            _pl_peak = float(_pl.max())
            _pl_peak_wk = int(_pl.idxmax())

            # Count weeks where players held ≥50% of peak (sustained interest)
            _held = sum(1 for v in _pl.values if v >= _pl_peak * 0.5)
            _held_pct = _held / len(_pl)

            reason = (f"Wk0 {int(_pl_wk0):,} → peak {int(_pl_peak):,} at Wk{_pl_peak_wk} "
                      f"→ Wk{_pl_cur_wk} {int(_pl_cur):,} players | "
                      f"Sustained above 50% of peak in {_held} of {len(_pl)} weeks")
            if _held_pct >= 0.60:
                score_3 = {"label": "Longer",  "color": "green",  "reason": reason}
            elif _held_pct >= 0.35:
                score_3 = {"label": "Same",    "color": "yellow", "reason": reason}
            else:
                score_3 = {"label": "Shorter", "color": "red",    "reason": reason}
    else:
        # Bet decay vs peer median across all weeks
        _bd_above, _bd_total, _bd_latest, _bd_p50 = _trend_score("bet_decay", higher_is_better=True)
        if _bd_total == 0:
            score_3 = _nd("No retention data available yet.")
        else:
            pct_above = _bd_above / _bd_total
            reason = f"Above peer median bet decay in {_bd_above} of {_bd_total} weeks"
            if _bd_latest is not None and _bd_p50 is not None:
                reason += f" | Latest: {_bd_latest:.1f}% vs peer median {_bd_p50:.1f}%"
            if pct_above >= 0.60:
                score_3 = {"label": "Longer",  "color": "green",  "reason": reason}
            elif pct_above >= 0.40:
                score_3 = {"label": "Same",    "color": "yellow", "reason": reason}
            else:
                score_3 = {"label": "Shorter", "color": "red",    "reason": reason}

    return {"game_performance": score_1, "sales_impact": score_2, "player_interest": score_3}


# ── Derived KPIs ─────────────────────────────────────────────────────
def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add net_rev, hold_pct, avg_bet, bet_decay, player_decay (PFH only).

    hold_pct/bet_decay/player_decay are all "current value as a % of a reference
    point" ratios, not deltas -- the one ratio family in this app where a value
    at or above 100% reads as broken/nonsensical to a viewer rather than as
    "grew since launch" (unlike an explicit +150% delta badge, which reads fine).
    Clipped at 99.9 so none of the three can ever display at or over 100%; real
    growth for a "growing" game still shows up in its absolute Bet Handle/Net
    Revenue trend, just not as a >100% reading on these specific ratios.
    """
    df = df.copy()
    df["net_rev"] = df["bet_handle"] - df["total_win"]
    df["hold_pct"] = np.where(df["bet_handle"] > 0,
                               df["net_rev"] / df["bet_handle"] * 100, np.nan)
    df["hold_pct"] = df["hold_pct"].clip(upper=99.9)
    df["avg_bet"] = np.where(df["spins"] > 0,
                              df["bet_handle"] / df["spins"], np.nan)
    # Normalise bet to week-0 per game
    w0 = df.loc[df["launch_week"] == 0, ["game_id", "bet_handle"]].rename(
        columns={"bet_handle": "_w0_bet"})
    df = df.merge(w0, on="game_id", how="left")
    df["bet_decay"] = np.where(df["_w0_bet"] > 0,
                                df["bet_handle"] / df["_w0_bet"] * 100, np.nan)
    df["bet_decay"] = df["bet_decay"].clip(upper=99.9)
    df.drop(columns=["_w0_bet"], inplace=True)

    if df["players"].notna().any():
        df["arpu"] = np.where(df["players"] > 0, df["net_rev"] / df["players"], np.nan)
        df["spp"] = np.where(df["players"] > 0, df["spins"] / df["players"], np.nan)
        w0p = df.loc[df["launch_week"] == 0, ["game_id", "players"]].rename(
            columns={"players": "_w0_pl"})
        df = df.merge(w0p, on="game_id", how="left")
        df["player_decay"] = np.where(df["_w0_pl"] > 0,
                                       df["players"] / df["_w0_pl"] * 100, np.nan)
        df["player_decay"] = df["player_decay"].clip(upper=99.9)
        df.drop(columns=["_w0_pl"], inplace=True)
    else:
        df["arpu"] = np.nan
        df["spp"] = np.nan
        df["player_decay"] = np.nan

    return df


# ── Location type lookup ──────────────────────────────────────────────
def load_game_loc_types(conn, platform: str) -> pd.DataFrame:
    """
    Returns dominant ConfigProduct (loc_type) per game_id.
    Dominant = whichever ConfigProduct contributed the most TotalBet for that game.
    Joins BetSpinSummaryCashView3Pong/EdgeLabs → CrmLocationView on StoreNumber=LocationId.
    """
    if platform == "PFH":
        src = "BetSpinSummaryCashView3Pong"
    elif platform in ("V2", "V1"):
        src = "BetSpinSummaryCashView3EdgeLabs"
    else:
        return pd.DataFrame(columns=["game_id", "loc_type"])
    sql = f"""
    SELECT game_id, ConfigProduct AS loc_type
    FROM (
        SELECT
            TRY_CAST(b.GameId AS INT)  AS game_id,
            loc.ConfigProduct,
            SUM(CAST(b.TotalBet AS FLOAT)) AS total_bet,
            ROW_NUMBER() OVER (
                PARTITION BY TRY_CAST(b.GameId AS INT)
                ORDER BY SUM(CAST(b.TotalBet AS FLOAT)) DESC
            ) AS rn
        FROM {src} b
        LEFT JOIN CrmLocationView loc
            ON CAST(b.StoreNumber AS VARCHAR) = CAST(loc.LocationId AS VARCHAR)
        WHERE TRY_CAST(b.GameId AS INT) IS NOT NULL
          AND loc.ConfigProduct IS NOT NULL
        GROUP BY TRY_CAST(b.GameId AS INT), loc.ConfigProduct
    ) t
    WHERE rn = 1
    """
    df = E.query_df(conn, sql)
    if not df.empty:
        df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
    return df[["game_id", "loc_type"]].dropna(subset=["game_id"])


def load_game_loc_type_weekly(conn, game_id: int, platform: str) -> pd.DataFrame:
    """
    Returns weekly bet handle + net rev per location type for a single game.
    Used for the Location Type Breakdown toggle.
    """
    if platform == "PFH":
        sql = f"""
        SELECT
            loc.ConfigProduct AS loc_type,
            (DATEDIFF('day', fd.launch_date, CAST(b."Date" AS DATE)) // 7) AS launch_week,
            SUM(CAST(b.TotalBet AS FLOAT) / 100.0) AS bet_handle,
            SUM(CAST(b.TotalBet AS FLOAT) / 100.0) - SUM(CAST(b.TotalWin AS FLOAT) / 100.0) AS net_rev
        FROM BetSpinSummaryCashView3Pong b
        JOIN (
            SELECT MIN(CAST("Date" AS DATE)) AS launch_date
            FROM BetSpinSummaryCashView3Pong
            WHERE TRY_CAST(GameId AS INT) = {int(game_id)}
        ) fd ON 1=1
        LEFT JOIN CrmLocationView loc
            ON CAST(b.StoreNumber AS VARCHAR) = CAST(loc.LocationId AS VARCHAR)
        WHERE TRY_CAST(b.GameId AS INT) = {int(game_id)}
          AND loc.ConfigProduct IS NOT NULL
          AND (DATEDIFF('day', fd.launch_date, CAST(b."Date" AS DATE)) // 7) >= 0
        GROUP BY loc.ConfigProduct, (DATEDIFF('day', fd.launch_date, CAST(b."Date" AS DATE)) // 7)
        ORDER BY loc_type, launch_week
        """
    elif platform in ("V2", "V1"):
        gc_plat = _GC_PLATFORM[platform]
        sql = f"""
        SELECT
            loc.ConfigProduct AS loc_type,
            (DATEDIFF('day', fd.launch_date, CAST(g.SummaryDate AS DATE)) // 7) AS launch_week,
            SUM(CAST(g.TotalPlay AS FLOAT) / 100.0) AS bet_handle,
            SUM(CAST(g.TotalPlay AS FLOAT) / 100.0) - SUM(CAST(g.TotalWin AS FLOAT) / 100.0) AS net_rev
        FROM AnalyticsGameTerminalsGames g
        JOIN GameCatalogView1 gc ON gc.Id = g.Id
        JOIN (
            SELECT MIN(CAST(SummaryDate AS DATE)) AS launch_date
            FROM AnalyticsGameTerminalsGames
            WHERE Id = {int(game_id)}
        ) fd ON 1=1
        LEFT JOIN CrmLocationView loc
            ON CAST(g.SummaryLocationId AS VARCHAR) = CAST(loc.LocationId AS VARCHAR)
        WHERE gc.Id = {int(game_id)}
          AND gc.Platform = '{gc_plat}'
          AND loc.ConfigProduct IS NOT NULL
          AND (DATEDIFF('day', fd.launch_date, CAST(g.SummaryDate AS DATE)) // 7) >= 0
        GROUP BY loc.ConfigProduct, (DATEDIFF('day', fd.launch_date, CAST(g.SummaryDate AS DATE)) // 7)
        ORDER BY loc_type, launch_week
        """
    else:
        return pd.DataFrame()
    return E.query_df(conn, sql)


def load_game_terminal_types(conn, platform: str) -> pd.DataFrame:
    """
    Returns dominant terminal orientation (H / V / Mixed) per game_id for V2/V1.
    Uses CrmLocationView columns: "H-Wooden", "H-Metal", "BarTop" → Horizontal
                                   "V-Wooden", "V-Metal", "DualScreen" → Vertical
    Dominant = whichever orientation had more total terminals across all locations
    the game appeared in, weighted by TotalBet.
    """
    if platform not in ("V2", "V1"):
        return pd.DataFrame(columns=["game_id", "terminal_type"])
    sql = """
    SELECT game_id, terminal_type
    FROM (
        SELECT
            TRY_CAST(b.GameId AS INT) AS game_id,
            CASE
                WHEN SUM(
                    COALESCE(TRY_CAST(loc."H-Wooden"  AS FLOAT), 0) +
                    COALESCE(TRY_CAST(loc."H-Metal"   AS FLOAT), 0) +
                    COALESCE(TRY_CAST(loc."BarTop"    AS FLOAT), 0)
                ) > SUM(
                    COALESCE(TRY_CAST(loc."V-Wooden"  AS FLOAT), 0) +
                    COALESCE(TRY_CAST(loc."V-Metal"   AS FLOAT), 0) +
                    COALESCE(TRY_CAST(loc."DualScreen" AS FLOAT), 0)
                ) THEN 'Horizontal'
                WHEN SUM(
                    COALESCE(TRY_CAST(loc."V-Wooden"  AS FLOAT), 0) +
                    COALESCE(TRY_CAST(loc."V-Metal"   AS FLOAT), 0) +
                    COALESCE(TRY_CAST(loc."DualScreen" AS FLOAT), 0)
                ) > SUM(
                    COALESCE(TRY_CAST(loc."H-Wooden"  AS FLOAT), 0) +
                    COALESCE(TRY_CAST(loc."H-Metal"   AS FLOAT), 0) +
                    COALESCE(TRY_CAST(loc."BarTop"    AS FLOAT), 0)
                ) THEN 'Vertical'
                ELSE 'Mixed'
            END AS terminal_type
        FROM BetSpinSummaryCashView3EdgeLabs b
        LEFT JOIN CrmLocationView loc
            ON CAST(b.StoreNumber AS VARCHAR) = CAST(loc.LocationId AS VARCHAR)
        WHERE TRY_CAST(b.GameId AS INT) IS NOT NULL
        GROUP BY TRY_CAST(b.GameId AS INT)
    ) t
    WHERE terminal_type IS NOT NULL
    """
    df = E.query_df(conn, sql)
    if not df.empty:
        df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
    return df[["game_id", "terminal_type"]].dropna(subset=["game_id"])


# ── Segment share SES forecast ───────────────────────────────────────
def forecast_segment_shares(
    seg_history: dict[str, list[float]],
    horizon: int = 8,
    alpha: float = 0.7,
    z: float = 1.28,
) -> dict[str, dict]:
    """
    Fit a simple exponential smoothing (SES) forecast for each segment's
    wagering-share % series.

    Parameters
    ----------
    seg_history : {segment_name: [share_pct, ...]} — weekly values, oldest first
    horizon     : weeks to forecast ahead
    alpha       : SES smoothing parameter
    z           : CI width (default 1.28 = 80%)

    Returns
    -------
    {segment_name: {
        "fvals": [float]*horizon,   — point forecast (includes current week as w0)
        "fhi":   [float]*horizon,
        "flo":   [float]*horizon,
        "base_vol": float,          — per-week uncertainty (MAE of in-sample 1-step errors)
    }}
    """
    result = {}
    for seg, hy in seg_history.items():
        hy = [float(v) for v in hy]
        base = hy[-1] if hy else 0.0

        # In-sample 1-step errors to size the CI
        lvl = hy[0] if hy else base
        errs = []
        for hv in hy[1:]:
            pred = alpha * lvl + (1 - alpha) * lvl
            errs.append(abs(hv - pred))
            lvl = alpha * hv + (1 - alpha) * lvl
        base_vol = float(np.mean(errs)) if errs else max(1.0, abs(base) * 0.1)

        fvals, fhi, flo = [base], [base], [base]
        lvl_f = base
        for fw in range(1, horizon):
            lvl_f = alpha * base + (1 - alpha) * lvl_f
            sig = base_vol * (fw ** 0.5)
            fvals.append(max(0.0, lvl_f))
            fhi.append(lvl_f + z * sig)
            flo.append(max(0.0, lvl_f - z * sig))

        result[seg] = {"fvals": fvals, "fhi": fhi, "flo": flo, "base_vol": base_vol}
    return result


# ── Portfolio-wide trend (seasonality / external-driver check) ───────
def compute_portfolio_trend(df: pd.DataFrame, window_weeks: int = 4) -> dict:
    """
    Compute the portfolio-wide bet_handle trend over the last `window_weeks`
    using ALL games in `df` (the full platform weekly DataFrame).

    Returns a dict:
      direction   str    — "up" | "down" | "flat"
      pct_change  float  — median % change across all games over the window
      n_games     int    — how many games had data in the window
      consensus   float  — fraction of games moving in the majority direction (0–1)
    """
    if df.empty or "launch_week" not in df.columns:
        return {"direction": "flat", "pct_change": 0.0, "n_games": 0, "consensus": 0.0}

    max_wk = int(df["launch_week"].max())
    if max_wk < window_weeks:
        return {"direction": "flat", "pct_change": 0.0, "n_games": 0, "consensus": 0.0}

    recent = df[df["launch_week"] > max_wk - window_weeks]
    prior  = df[(df["launch_week"] > max_wk - window_weeks * 2) &
                (df["launch_week"] <= max_wk - window_weeks)]

    r_avg = recent.groupby("game_id")["bet_handle"].mean()
    p_avg = prior.groupby("game_id")["bet_handle"].mean()

    common = r_avg.index.intersection(p_avg.index)
    if len(common) == 0:
        return {"direction": "flat", "pct_change": 0.0, "n_games": 0, "consensus": 0.0}

    pct = ((r_avg[common] - p_avg[common]) / p_avg[common].replace(0, np.nan) * 100).dropna()
    if pct.empty:
        return {"direction": "flat", "pct_change": 0.0, "n_games": 0, "consensus": 0.0}

    med = float(pct.median())
    direction = "up" if med > 3.0 else ("down" if med < -3.0 else "flat")
    consensus = float((pct > 0).mean()) if direction == "up" else (
                float((pct < 0).mean()) if direction == "down" else
                float((pct.abs() < 3.0).mean()))
    return {
        "direction": direction,
        "pct_change": round(med, 1),
        "n_games": len(common),
        "consensus": round(consensus, 2),
    }


def classify_game_driver(game_series: np.ndarray, portfolio_trend: dict,
                          window_weeks: int = 4) -> str:
    """
    Compare a single game's recent trend against the portfolio trend.

    Returns a one-line verdict string, e.g.:
      "Trending up — driven by game-specific demand"
      "Trending up — matches portfolio-wide seasonal lift, not attributable to the game"
      "Trending down — broader portfolio is also declining; likely an external/seasonal driver"
    """
    arr = np.asarray(game_series, float)
    if len(arr) < window_weeks * 2:
        return "Insufficient history to determine driver"

    recent_avg = float(np.nanmean(arr[-window_weeks:]))
    prior_avg  = float(np.nanmean(arr[-window_weeks * 2:-window_weeks]))
    if prior_avg <= 0:
        return "Insufficient history to determine driver"

    game_pct = (recent_avg - prior_avg) / prior_avg * 100
    game_dir = "up" if game_pct > 3.0 else ("down" if game_pct < -3.0 else "flat")

    port_dir = portfolio_trend.get("direction", "flat")
    port_pct = portfolio_trend.get("pct_change", 0.0)
    consensus = portfolio_trend.get("consensus", 0.0)

    # Directions agree and portfolio has strong consensus → external driver
    if game_dir == port_dir and game_dir != "flat" and consensus >= 0.6:
        label = "Trending up" if game_dir == "up" else "Trending down"
        return (f"{label} — matches portfolio-wide {'lift' if game_dir == 'up' else 'decline'} "
                f"({port_pct:+.1f}% across {int(consensus*100)}% of games); "
                f"likely an external/seasonal driver, not game-specific")

    # Game goes against portfolio or portfolio is flat → game-specific
    if game_dir == "up":
        return f"Trending up — driven by game-specific demand (+{game_pct:.1f}%)"
    if game_dir == "down":
        if port_dir == "up":
            return f"Trending down — underperforming vs rising portfolio; game-specific issue"
        return f"Trending down — game-specific decline ({game_pct:.1f}%)"
    return "Stable — no meaningful trend detected"


# ── Main load entry point ─────────────────────────────────────────────
def load_platform_data(conn, platform: str) -> pd.DataFrame:
    """
    Load all weekly data for a platform, add derived KPIs.
    Returns a single DataFrame (all games, all weeks) with columns:
      game_id, game_name, codebase, launch_date, launch_week,
      bet_handle, total_win, spins, players, stores,
      net_rev, hold_pct, avg_bet, bet_decay, arpu, spp, player_decay
    """
    if platform == "PFH":
        raw = load_pfh_weekly_all(conn)
    elif platform in ("V2", "V1"):
        raw = load_v2v1_weekly_all(conn, platform)
    elif platform == "EdgeLabs":
        raw = load_edgelabs_weekly_all(conn)
    else:
        raise ValueError(f"Unknown platform: {platform}")
    if raw.empty:
        return raw
    raw = _drop_incomplete_trailing_week(raw)
    if raw.empty:
        return raw
    return _add_derived(raw)


def _drop_incomplete_trailing_week(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Drop launch-week buckets that haven't fully elapsed as of the platform's data
    cutoff — i.e. the in-progress current week.

    Without this, every live game's newest bucket holds only the days elapsed so far
    (often 1-3 of 7), so it reads as a sharp drop on every trend line and drags
    decay/trend/archetype calls with it, purely as an artifact of when you looked.

    Judged on *calendar elapsed time*, not activity days: a bucket is complete once
    `launch_date + 7*week + 6` has passed, even if the game had no play on some of
    those days. That deliberately leaves retired games' history intact — their final
    short week is a real event (the game was pulled), not a reporting artifact.
    """
    if raw.empty or "_last_day" not in raw.columns:
        return raw.drop(columns=["_last_day"], errors="ignore")

    d = raw.copy()
    d["_last_day"] = pd.to_datetime(d["_last_day"], errors="coerce")
    cutoff = d["_last_day"].max()
    if pd.isna(cutoff):
        return d.drop(columns=["_last_day"])

    ld = pd.to_datetime(d["launch_date"], errors="coerce")
    wk = pd.to_numeric(d["launch_week"], errors="coerce")
    bucket_end = ld + pd.to_timedelta(wk * 7 + 6, unit="D")
    keep = bucket_end <= cutoff
    # Never return an empty frame for a game just because it's brand new — a game
    # younger than 7 days would otherwise vanish entirely rather than show week 0.
    keep = keep | (~d["game_id"].isin(d.loc[keep, "game_id"].unique()) & (wk == 0))
    return d.loc[keep].drop(columns=["_last_day"]).reset_index(drop=True)


# ── Catalog helper (derived from the weekly data) ─────────────────────
def catalog_from_data(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise per-game metadata from the weekly DataFrame."""
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby(["game_id", "game_name", "codebase", "launch_date"])
    cat = grp.agg(
        total_weeks=("launch_week", "max"),
        total_bet=("bet_handle", "sum"),
        total_net=("net_rev", "sum"),
        last_week_bet=("bet_handle", lambda x: x.iloc[-1] if len(x) else np.nan),
    ).reset_index()
    cat["is_hr"] = cat["game_id"].apply(_is_hr)
    return cat.sort_values("game_name")


# ── DTW peer matching ─────────────────────────────────────────────────
def find_peers(
    df: pd.DataFrame,
    target_id: int,
    kpi: str = "bet_decay",
    n_weeks: int | None = None,
    top_k: int = 5,
    exclude_hr: bool = True,
    min_candidate_weeks: int = 4,
) -> list[dict]:
    """
    Find top-K games most similar to `target_id`'s first n_weeks of trajectory.

    Parameters
    ----------
    df          : full platform weekly DataFrame (from load_platform_data)
    target_id   : game_id to analyse
    kpi         : column to use for DTW (default 'bet_decay')
    n_weeks     : how many launch weeks to match on (None = use all available target weeks)
    top_k       : number of peers to return
    exclude_hr  : skip 95xxx high-roller variants
    min_candidate_weeks : peer must have at least this many weeks *after* n_weeks

    Returns list of dicts: {game_id, game_name, distance, n_match_weeks, total_weeks}
    """
    if kpi not in df.columns:
        return []

    target_df = df[df["game_id"] == target_id].sort_values("launch_week")
    if target_df.empty:
        return []

    target_vec = target_df[["launch_week", kpi]].dropna(subset=[kpi])

    if n_weeks is None:
        n_weeks = int(target_vec["launch_week"].max()) + 1
    target_vals = target_vec[target_vec["launch_week"] < n_weeks][kpi].values
    if len(target_vals) < 2:
        return []

    results = []
    for gid, grp in df.groupby("game_id"):
        if gid == target_id:
            continue
        if exclude_hr and _is_hr(gid):
            continue
        grp = grp.sort_values("launch_week")
        cand_vec = grp[["launch_week", kpi]].dropna(subset=[kpi])
        total_weeks = int(cand_vec["launch_week"].max()) + 1 if len(cand_vec) else 0
        # Candidate must have data past n_weeks to be useful for forecasting
        if total_weeks < n_weeks + min_candidate_weeks:
            continue
        cand_vals = cand_vec[cand_vec["launch_week"] < n_weeks][kpi].values
        if len(cand_vals) < 2:
            continue
        dist = dtw_distance(target_vals, cand_vals)
        results.append(dict(
            game_id=gid,
            game_name=grp["game_name"].iloc[0],
            codebase=grp["codebase"].iloc[0],
            distance=dist,
            n_match_weeks=n_weeks,
            total_weeks=total_weeks,
        ))

    results.sort(key=lambda x: x["distance"])
    return results[:top_k]


# ── Mechanic-aware peer matching ───────────────────────────────────────

# Types that have a genuine reel-based fallback to the slot pool
def _skin_group_ids(catalog: pd.DataFrame, target_id: int) -> set[int]:
    """
    Return the set of game_ids in the same SkinOf family as target_id
    (parent + all sibling skins), excluding target_id itself.
    Empty set if target has no SkinOf relationship in either direction.
    """
    if catalog.empty or "SkinOf" not in catalog.columns or "Id" not in catalog.columns:
        return set()
    cat = catalog.copy()
    cat["Id"] = pd.to_numeric(cat["Id"], errors="coerce")
    cat["SkinOf"] = pd.to_numeric(cat["SkinOf"], errors="coerce")
    target_id = int(target_id)

    row = cat[cat["Id"] == target_id]
    target_skinof = (float(row["SkinOf"].iloc[0])
                      if not row.empty and pd.notna(row["SkinOf"].iloc[0]) else None)

    if target_skinof is not None:
        parent_id = int(target_skinof)
        family = set(cat[cat["SkinOf"] == target_skinof]["Id"].dropna().astype(int).tolist())
        family.add(parent_id)
    else:
        # target may itself be a parent — find its children
        family = set(cat[cat["SkinOf"] == target_id]["Id"].dropna().astype(int).tolist())

    family.discard(target_id)
    return family


def _attr_filtered_ids(catalog: pd.DataFrame, target_id: int,
                        use_platform: bool = True, use_product: bool = True,
                        use_orientation: bool = True, live_only: bool = True) -> set[int]:
    """Return game_ids matching target_id on the requested catalog attributes
    (Platform, Product, ScreenOrientation) — all confirmed 597/597 populated.
    live_only excludes discontinued/non-live catalog entries from the candidate
    pool (a dead game is a poor peer for judging a currently-running one) —
    skipped automatically if the target itself isn't 'live', or if Status is missing."""
    if catalog.empty or "Id" not in catalog.columns:
        return set()
    cat = catalog.copy()
    cat["Id"] = pd.to_numeric(cat["Id"], errors="coerce")
    target_id = int(target_id)
    row = cat[cat["Id"] == target_id]
    if row.empty:
        return set()

    mask = pd.Series(True, index=cat.index)
    if use_platform and "Platform" in cat.columns:
        mask &= (cat["Platform"] == row["Platform"].iloc[0])
    if use_product and "Product" in cat.columns:
        mask &= (cat["Product"] == row["Product"].iloc[0])
    if use_orientation and "ScreenOrientation" in cat.columns:
        mask &= (cat["ScreenOrientation"] == row["ScreenOrientation"].iloc[0])
    if live_only and "Status" in cat.columns and str(row["Status"].iloc[0]).lower() == "live":
        mask &= (cat["Status"].astype(str).str.lower() == "live")

    ids = set(cat.loc[mask, "Id"].dropna().astype(int).tolist())
    ids.discard(target_id)
    return ids


def find_peer_pool(catalog: pd.DataFrame, df: pd.DataFrame, target_id: int,
                    scale_tolerance: float = 2.0, min_pool: int = 5) -> dict:
    """
    Same attribute hierarchy as find_peers_v2 (SkinOf > Platform+Product+Orientation >
    relax > loose), but returns the full candidate id POOL for percentile-band /
    Quick-Score comparisons rather than a DTW-ranked top-k peer list — "Is It On Track?"
    needs "who counts as a peer for this game's bands," not "which 5 games look most alike."

    Returns {"ids": set[int], "family": str, "pool_size": int, "scale_applied": bool}.
    """
    target_id = int(target_id)
    w0 = df[df["launch_week"] == 0].set_index("game_id")["bet_handle"]
    target_w0 = float(w0.get(target_id, np.nan))
    scale_ok = not (np.isnan(target_w0) or target_w0 <= 0)

    def _scale_filter(ids: set[int]) -> set[int]:
        if not scale_ok:
            return ids
        lo, hi = target_w0 / scale_tolerance, target_w0 * scale_tolerance
        return {gid for gid in ids if gid in w0.index and lo <= float(w0[gid]) <= hi}

    def _try(ids, family):
        scaled = _scale_filter(ids)
        if len(scaled) >= min_pool:
            return {"ids": scaled, "family": family, "pool_size": len(scaled), "scale_applied": True}
        if len(ids) >= min_pool:
            return {"ids": ids, "family": family, "pool_size": len(ids), "scale_applied": False}
        return None

    skin_ids = _skin_group_ids(catalog, target_id)
    if len(skin_ids) >= min_pool:
        return {"ids": skin_ids, "family": "skin family", "pool_size": len(skin_ids), "scale_applied": False}

    for use_o, fam in ((True, "platform+product+orientation"), (False, "platform+product")):
        res = _try(_attr_filtered_ids(catalog, target_id, True, True, use_o), fam)
        if res:
            return res

    res = _try(_attr_filtered_ids(catalog, target_id, True, False, False), "platform")
    if res:
        return res

    if not catalog.empty and "Id" in catalog.columns:
        all_ids = set(pd.to_numeric(catalog["Id"], errors="coerce").dropna().astype(int).tolist())
    else:
        all_ids = set(df["game_id"].unique().tolist())
    all_ids.discard(target_id)
    res = _try(all_ids, "loose (full catalog)")
    if res:
        return res
    return {"ids": all_ids, "family": "loose (full catalog, thin)", "pool_size": len(all_ids), "scale_applied": False}


def find_peers_v2(
    df: pd.DataFrame,
    target_id: int,
    catalog: pd.DataFrame,
    kpi: str = "bet_decay",
    n_weeks: int | None = None,
    top_k: int = 5,
    exclude_hr: bool = True,
    scale_tolerance: float = 2.0,
    min_candidate_weeks: int = 4,
    min_pool: int = 5,  # matches find_peer_pool()'s default — was 3 here, which let the two
                        # functions silently pick different tiers for the same target game
) -> dict:
    """
    DTW-ranks the top_k closest peers within the pool find_peer_pool() selects — a thin
    ranking layer on top of the SAME tier hierarchy (SkinOf > platform+product+orientation
    > relax > loose) used for the percentile-band peer pool. Before 2026-08-18 this function
    ran its own independent copy of the tier search with a different min_pool default (3 vs
    find_peer_pool's 5) — which could silently pick a different tier than the one used for
    "Is It On Track?"'s percentile bands for the exact same target game. Unified so the peer
    COUNT/tier shown on one page and the ranked peer LIST shown on another always agree.

    Returns a dict:
      {
        "peers":            list[dict],
        "mechanic_matched":  bool,   # True = confident tier (skin family or full attribute match)
        "fallback_reason":   str | None,
        "family":            str,    # readable label for which tier matched
      }
    """
    target_id = int(target_id)
    target_df = df[df["game_id"] == target_id].sort_values("launch_week")
    if target_df.empty:
        return {"peers": [], "mechanic_matched": False, "fallback_reason": "No data for this game", "family": "none"}
    target_vec = target_df[["launch_week", kpi]].dropna(subset=[kpi])
    n = int(target_vec["launch_week"].max()) + 1 if n_weeks is None else n_weeks
    target_vals = target_vec[target_vec["launch_week"] < n][kpi].values
    if len(target_vals) < 2:
        return {"peers": [], "mechanic_matched": False, "fallback_reason": "Not enough weeks of data for this game", "family": "none"}

    def _dtw_rank(allowed_ids: set[int], matched: bool, skin_boost: bool) -> list[dict]:
        results = []
        for gid, grp in df[df["game_id"].isin(allowed_ids)].groupby("game_id"):
            if gid == target_id:
                continue
            if exclude_hr and _is_hr(gid):
                continue
            grp = grp.sort_values("launch_week")
            cand_vec = grp[["launch_week", kpi]].dropna(subset=[kpi])
            total_weeks = int(cand_vec["launch_week"].max()) + 1 if len(cand_vec) else 0
            if total_weeks < n + min_candidate_weeks:
                continue
            cand_vals = cand_vec[cand_vec["launch_week"] < n][kpi].values
            if len(cand_vals) < 2:
                continue
            dist = dtw_distance(target_vals, cand_vals)
            results.append(dict(
                game_id=gid,
                game_name=grp["game_name"].iloc[0] if "game_name" in grp.columns else str(gid),
                codebase=grp["codebase"].iloc[0] if "codebase" in grp.columns else None,
                distance=dist,
                n_match_weeks=n,
                total_weeks=total_weeks,
                mechanic_matched=matched,
                skin_boost=skin_boost,
            ))
        results.sort(key=lambda x: x["distance"])
        return results[:top_k]

    pool = find_peer_pool(catalog, df, target_id, scale_tolerance=scale_tolerance, min_pool=min_pool)
    family, allowed_ids = pool["family"], pool["ids"]
    matched = family in ("skin family", "platform+product+orientation")
    peers = _dtw_rank(allowed_ids, matched, skin_boost=(family == "skin family"))

    if not peers and family not in ("loose (full catalog)", "loose (full catalog, thin)"):
        # The assigned tier had attribute-matched candidates but none with enough weeks of
        # history for DTW — widen to the full catalog rather than silently report zero peers.
        if not catalog.empty and "Id" in catalog.columns:
            all_ids = set(pd.to_numeric(catalog["Id"], errors="coerce").dropna().astype(int).tolist())
        else:
            all_ids = set(df["game_id"].unique().tolist())
        all_ids.discard(target_id)
        peers = _dtw_rank(all_ids, matched=False, skin_boost=False)
        if peers:
            family, matched = "loose (full catalog, widened for history)", False

    if not peers:
        return {"peers": [], "mechanic_matched": False,
                "fallback_reason": "No comparable games found with enough history", "family": "none"}

    fallback_reason = None
    if not matched:
        fallback_reason = f"Limited precedent — {len(peers)} comparable games via {family}"
    elif not pool.get("scale_applied", False) and family != "skin family":
        fallback_reason = f"Scale filter relaxed — {len(peers)} comparable games by {family}"

    return {"peers": peers, "mechanic_matched": matched, "fallback_reason": fallback_reason, "family": family}


# ── Blended forecast ──────────────────────────────────────────────────

def _theil_sen_log_slope(weeks: np.ndarray, values: np.ndarray) -> float:
    """Theil-Sen median-slope in log(value) space — robust to outliers."""
    log_vals = np.log(np.maximum(values, 1e-9))
    n = len(weeks)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dw = float(weeks[j] - weeks[i])
            if dw > 0:
                slopes.append((log_vals[j] - log_vals[i]) / dw)
    return float(np.median(slopes)) if slopes else 0.0


def fit_blended_forecast(
    df: pd.DataFrame,
    target_id: int,
    catalog: pd.DataFrame,
    backtest_mape: float | None = None,
    n_forecast_weeks: int = 13,
    kpi: str = "bet_handle",
    scale_tolerance: float = 2.0,
    min_candidate_weeks: int = 4,
    top_k: int = 5,
) -> dict:
    """
    Blended Theil-Sen self-trend + mechanic-matched peer-trend forecast.

    Confirmed edge-case matrix
    ──────────────────────────────────────────────────────────────────
    ≥4wks hist, backtest exists, ≥1 peer  → exp(-k*mape) / 1.0  normalized
    ≥4wks hist, no backtest yet, ≥1 peer  → self=0.30, peer=0.70
    <4wks hist,                  ≥1 peer  → self=0.00, peer=1.00
    any hist,                    0 peers  → self=1.00, peer=0.00, band×1.5
    <4wks hist AND               0 peers  → flat naive, band×2.0 or ±40%
    ──────────────────────────────────────────────────────────────────

    Returns dict:
        forecast        list[dict]  [{week, value, lower, upper}, ...]
        self_weight     float | None
        peer_weight     float | None
        self_slope_pct  float | None   weekly % change (Theil-Sen)
        peer_slope_pct  float | None   median peer weekly % change
        peers_used      list[int]
        message         str | None
        band_multiplier float
        family          str
    """
    _K = math.log(2) / 0.20  # MAPE=20% → self_confidence=0.50

    # ── 1. Target history ──────────────────────────────────────────────
    tgt = df[df["game_id"] == target_id].sort_values("launch_week")
    tgt_vals = tgt[kpi].dropna().values.astype(float)
    tgt_weeks = tgt.loc[tgt[kpi].notna(), "launch_week"].values.astype(float)
    n_hist = len(tgt_vals)

    # ── 2. Peer matching via find_peers_v2 ─────────────────────────────
    if not catalog.empty and "Id" in catalog.columns:
        peer_result = find_peers_v2(
            df, target_id, catalog,
            kpi="bet_decay", n_weeks=None, top_k=top_k,
            exclude_hr=True, scale_tolerance=scale_tolerance,
            min_candidate_weeks=min_candidate_weeks,
        )
    else:
        peer_result = {
            "peers": [],
            "mechanic_matched": False,
            "fallback_reason": "No catalog available — cannot determine mechanic family",
            "family": "unknown",
        }

    peers = peer_result["peers"]
    peer_ids = [p["game_id"] for p in peers]
    no_peers = len(peer_ids) == 0
    thin_history = n_hist < 4

    # ── 3. Double-edge case: no data AND no peers ──────────────────────
    if thin_history and no_peers:
        last_val = float(tgt_vals[-1]) if n_hist >= 1 else 0.0
        if n_hist >= 2:
            half_band = 2.0 * float(np.std(np.diff(tgt_vals)))
        else:
            half_band = 0.40 * last_val

        start_week = int(tgt_weeks[-1]) if n_hist >= 1 else 0
        return {
            "forecast": [
                {
                    "week": start_week + t,
                    "value": round(last_val, 2),
                    "lower": round(max(0.0, last_val - half_band), 2),
                    "upper": round(last_val + half_band, 2),
                    "directional": True,
                }
                for t in range(1, n_forecast_weeks + 1)
            ],
            "reliable_horizon": 0,
            "self_weight": None,
            "peer_weight": None,
            "self_slope_pct": None,
            "peer_slope_pct": None,
            "peers_used": [],
            "message": (
                "Insufficient data — fewer than 4 weeks of history and no comparable games. "
                "Showing flat projection with wide uncertainty band."
            ),
            "band_multiplier": 2.0,
            "family": peer_result.get("family", "unknown"),
        }

    # ── 4. Weights ─────────────────────────────────────────────────────
    if no_peers:
        self_weight = 1.0
        peer_weight = 0.0
        band_multiplier = 1.5
        message = peer_result.get("fallback_reason") or "No comparable games in database."
    else:
        if thin_history:
            self_weight_raw = 0.0
        elif backtest_mape is None:
            self_weight_raw = 0.3
        else:
            self_weight_raw = math.exp(-_K * float(backtest_mape))

        peer_weight_raw = 1.0 - self_weight_raw
        self_weight = self_weight_raw  # already sum-to-1 (complement)
        peer_weight = peer_weight_raw
        band_multiplier = 1.0
        message = peer_result.get("fallback_reason")  # may be non-None if scale was relaxed

    # ── 5. Theil-Sen self slope ────────────────────────────────────────
    if not thin_history:
        window = min(n_hist, 20)
        slope_self = _theil_sen_log_slope(tgt_weeks[-window:], tgt_vals[-window:])
        self_slope_pct = (math.exp(slope_self) - 1.0) * 100.0
    else:
        slope_self = 0.0
        self_slope_pct = None

    # ── 6. Peer slope (Theil-Sen in log space, same window length) ─────
    if not no_peers:
        window = min(n_hist, 20) if not thin_history else 20
        peer_slopes = []
        for pid in peer_ids:
            p_sub = df[df["game_id"] == pid].sort_values("launch_week")
            p_vals = p_sub[kpi].dropna().values.astype(float)
            p_wks = p_sub.loc[p_sub[kpi].notna(), "launch_week"].values.astype(float)
            pw = min(len(p_vals), window)
            if pw >= 2:
                peer_slopes.append(_theil_sen_log_slope(p_wks[-pw:], p_vals[-pw:]))

        slope_peer = float(np.median(peer_slopes)) if peer_slopes else 0.0
        peer_slope_pct = (math.exp(slope_peer) - 1.0) * 100.0
    else:
        slope_peer = 0.0
        peer_slope_pct = None

    # ── 7. Blended slope → forecast ────────────────────────────────────
    blended_slope = self_weight * slope_self + peer_weight * slope_peer

    last_val = float(tgt_vals[-1])
    last_week = int(tgt_weeks[-1])
    log_last = math.log(max(last_val, 1e-9))

    # Residual std in LOG space so uncertainty is proportional, not absolute.
    # (Linear residuals inflate base_std to the scale of early high-value weeks,
    # making week+13 bands absurdly wide relative to the point estimate.)
    log_fitted = log_last + blended_slope * (tgt_weeks - last_week)
    log_residuals = np.log(np.maximum(tgt_vals, 1e-9)) - log_fitted
    log_std = float(np.std(log_residuals)) if len(log_residuals) > 1 else 0.10

    # 80% CI (z=1.28) rather than 95% (z=1.96): actionable for planning.
    _Z80 = 1.28

    # reliable_horizon: dual constraint — whichever is MORE restrictive wins.
    #
    #   Constraint A (history): beyond 2×n_hist weeks, self-trend is extrapolating
    #     too far past its own evidence base.
    #
    #   Constraint B (band ratio): upper/value = exp(Z80 × log_std × band_mult × √t)
    #     Once this exceeds MAX_BAND_RATIO the band is not useful for planning
    #     regardless of how much history exists (e.g. a 79-week volatile game
    #     with no peers correctly gets reliable_horizon=1).
    #
    # This prevents the contradiction of directional=False on a 76× band.
    _MAX_BAND_RATIO = 5.0
    _hist_cap = max(1, 2 * n_hist)
    _scale = _Z80 * log_std * band_multiplier
    if _scale > 0:
        _t_by_band = (math.log(_MAX_BAND_RATIO) / _scale) ** 2
        _band_cap = max(1, int(math.floor(_t_by_band)))
    else:
        _band_cap = n_forecast_weeks  # perfectly flat model — no band growth

    reliable_horizon = min(n_forecast_weeks, _hist_cap, _band_cap)

    # ── 8. Yearly seasonal index (≥52 weeks history only) ─────────────────
    # Applied additively in log space: log_fval += log(idx).
    # This keeps upper/value = exp(hb_log) independent of idx, so
    # reliable_horizon computed above remains valid without recomputation.
    _seasonal_index: dict = {}
    _seasonal_active = False
    _last_cal = None
    if n_hist >= 52 and "launch_date" in tgt.columns:
        try:
            _ld_dt = pd.to_datetime(tgt["launch_date"].iloc[0])
            _hist_dates = [_ld_dt + pd.Timedelta(weeks=int(w)) for w in tgt_weeks]
            _hist_woy = pd.Series([d.isocalendar()[1] for d in _hist_dates])
            _log_fitted_h = log_last + blended_slope * (tgt_weeks - last_week)
            _fitted_h = np.exp(_log_fitted_h)
            _ratios = pd.Series(np.maximum(tgt_vals, 1e-9) / np.maximum(_fitted_h, 1e-9))
            _raw_idx = _ratios.groupby(_hist_woy).median().to_dict()
            _mean_idx = float(np.mean(list(_raw_idx.values())))
            if _mean_idx > 1e-9:
                _seasonal_index = {int(k): v / _mean_idx for k, v in _raw_idx.items()}
                _seasonal_active = len(_seasonal_index) >= 26  # need ≥half-year coverage
            _last_cal = _ld_dt + pd.Timedelta(weeks=int(last_week))
        except Exception:
            pass

    forecast = []
    for t in range(1, n_forecast_weeks + 1):
        log_fval = log_last + blended_slope * t
        _sidx = 1.0
        if _seasonal_active and _last_cal is not None:
            _fc_date = _last_cal + pd.Timedelta(weeks=t)
            _woy = int(_fc_date.isocalendar()[1])
            _sidx = _seasonal_index.get(_woy, 1.0)
            log_fval += math.log(max(_sidx, 1e-9))
        hb_log = _Z80 * log_std * band_multiplier * math.sqrt(t)
        fval = math.exp(log_fval)
        forecast.append({
            "week": last_week + t,
            "value": round(fval, 2),
            "lower": round(math.exp(log_fval - hb_log), 2),
            "upper": round(math.exp(log_fval + hb_log), 2),
            "directional": t > reliable_horizon,
            "seasonal_index": round(_sidx, 4),
        })

    return {
        "forecast": forecast,
        "reliable_horizon": reliable_horizon,
        "self_weight": round(self_weight, 4),
        "peer_weight": round(peer_weight, 4),
        "self_slope_pct": round(self_slope_pct, 3) if self_slope_pct is not None else None,
        "peer_slope_pct": round(peer_slope_pct, 3) if peer_slope_pct is not None else None,
        "peers_used": peer_ids,
        "message": message,
        "band_multiplier": band_multiplier,
        "family": peer_result.get("family", "unknown"),
        "seasonal_active": _seasonal_active,
    }


def compare_game_to_portfolio(
    df: pd.DataFrame,
    target_id: int,
    kpi: str = "bet_handle",
    lookback_weeks: int = 12,
) -> dict:
    """
    Compare a game's recent Theil-Sen slope to the portfolio-wide distribution.

    Returns dict:
        target_slope_pct     float | None  target game's weekly % change
        portfolio_median_pct float         portfolio median weekly % change
        portfolio_p25_pct    float
        portfolio_p75_pct    float
        percentile_rank      float | None  0–100, target's position in portfolio
        verdict              str           "outperforming" | "underperforming" | "in line" | "insufficient data"
        n_games              int           games contributing to portfolio distribution
        lookback_weeks       int
    """
    slopes: list = []
    target_slope = None

    for gid, gdf in df.groupby("game_id"):
        series = gdf.sort_values("launch_week")
        vals = series[kpi].dropna().values.astype(float)
        wks = series.loc[series[kpi].notna(), "launch_week"].values.astype(float)
        if len(vals) < lookback_weeks:
            continue
        s = _theil_sen_log_slope(wks[-lookback_weeks:], vals[-lookback_weeks:])
        if gid == target_id:
            target_slope = s
        slopes.append(s)

    if not slopes:
        return {
            "target_slope_pct": None,
            "portfolio_median_pct": None,
            "portfolio_p25_pct": None,
            "portfolio_p75_pct": None,
            "percentile_rank": None,
            "verdict": "insufficient data",
            "n_games": 0,
            "lookback_weeks": lookback_weeks,
        }

    slopes_arr = np.array(slopes)
    p25 = float(np.percentile(slopes_arr, 25))
    p50 = float(np.median(slopes_arr))
    p75 = float(np.percentile(slopes_arr, 75))

    if target_slope is None:
        return {
            "target_slope_pct": None,
            "portfolio_median_pct": round((math.exp(p50) - 1.0) * 100.0, 3),
            "portfolio_p25_pct": round((math.exp(p25) - 1.0) * 100.0, 3),
            "portfolio_p75_pct": round((math.exp(p75) - 1.0) * 100.0, 3),
            "percentile_rank": None,
            "verdict": "insufficient data",
            "n_games": len(slopes),
            "lookback_weeks": lookback_weeks,
        }

    target_slope_pct = (math.exp(target_slope) - 1.0) * 100.0
    pct_rank = float(np.mean(slopes_arr <= target_slope) * 100.0)

    if target_slope > p75:
        verdict = "outperforming"
    elif target_slope < p25:
        verdict = "underperforming"
    else:
        verdict = "in line"

    return {
        "target_slope_pct": round(target_slope_pct, 3),
        "portfolio_median_pct": round((math.exp(p50) - 1.0) * 100.0, 3),
        "portfolio_p25_pct": round((math.exp(p25) - 1.0) * 100.0, 3),
        "portfolio_p75_pct": round((math.exp(p75) - 1.0) * 100.0, 3),
        "percentile_rank": round(pct_rank, 1),
        "verdict": verdict,
        "n_games": len(slopes),
        "lookback_weeks": lookback_weeks,
    }


def compute_retrospective_standing(
    df: pd.DataFrame,
    target_id: int,
    kpi: str = "bet_handle",
    lookback_weeks: int = 12,
    max_history_weeks: int = 52,
) -> list:
    """
    For each calendar week in the target game's history (last max_history_weeks weeks,
    starting from week lookback_weeks to have a full window), compute the game's
    Theil-Sen slope vs the portfolio distribution as of that week.

    Returns list of dicts: [{launch_week, verdict, target_slope_pct, portfolio_median_pct}, ...]
    sorted by launch_week ascending.  Empty list if target has < lookback_weeks history.

    Performance: pre-computes per-game rolling slopes once (single groupby pass), then
    does O(1) lookups per annotatable week.  Avoids the O(annotatable × n_games) groupby
    overhead that made the naive implementation ~4s for 120 games.

    Cache key for callers: (target_id, kpi, len(df), df["game_id"].nunique())
    """
    tgt = df[df["game_id"] == target_id].sort_values("launch_week")
    tgt_valid = tgt[tgt[kpi].notna()]
    tgt_launch_weeks = tgt_valid["launch_week"].values.astype(int)

    if len(tgt_launch_weeks) < lookback_weeks:
        return []

    # Determine which weeks to annotate (last max_history_weeks of target history)
    all_wks = tgt_launch_weeks
    cutoff = all_wks[-1] - max_history_weeks
    annotatable = [
        w for w in all_wks
        if w > cutoff and np.sum(all_wks <= w) >= lookback_weeks
    ]
    if not annotatable:
        return []

    annotatable_set = set(annotatable)

    # ── Single-pass precomputation: rolling Theil-Sen slope per game per week ──
    # slope_table[gid][week_t] = slope over (week_t-lookback+1 .. week_t)
    # We only compute weeks that appear in annotatable_set for portfolio games,
    # since that's all we need for the lookup.
    slope_table: dict = {}  # {game_id: {week_t: slope}}
    for gid, gdf in df.groupby("game_id"):
        gsub = gdf[gdf[kpi].notna()].sort_values("launch_week")
        vals = gsub[kpi].values.astype(float)
        wks = gsub["launch_week"].values.astype(int)
        n = len(wks)
        if n < lookback_weeks:
            continue
        gslopes: dict = {}
        for i in range(lookback_weeks - 1, n):
            wt = wks[i]
            if wt in annotatable_set:
                sl = _theil_sen_log_slope(
                    wks[i - lookback_weeks + 1: i + 1].astype(float),
                    vals[i - lookback_weeks + 1: i + 1],
                )
                gslopes[wt] = sl
        if gslopes:
            slope_table[gid] = gslopes

    # ── Per-annotatable-week verdict lookup ────────────────────────────────────
    result = []
    for week_t in annotatable:
        target_slope = slope_table.get(target_id, {}).get(week_t)
        if target_slope is None:
            continue

        slopes_at_t = [
            s
            for gid, gslopes in slope_table.items()
            if gid != target_id and (s := gslopes.get(week_t)) is not None
        ]
        if not slopes_at_t:
            continue

        sarr = np.array(slopes_at_t)
        p25 = float(np.percentile(sarr, 25))
        p75 = float(np.percentile(sarr, 75))
        p50 = float(np.median(sarr))

        if target_slope > p75:
            verdict = "outperforming"
        elif target_slope < p25:
            verdict = "underperforming"
        else:
            verdict = "in line"

        result.append({
            "launch_week": int(week_t),
            "verdict": verdict,
            "target_slope_pct": round((math.exp(target_slope) - 1.0) * 100.0, 3),
            "portfolio_median_pct": round((math.exp(p50) - 1.0) * 100.0, 3),
        })

    return result


# ── Forecast cone ─────────────────────────────────────────────────────
def compute_forecast(
    df: pd.DataFrame,
    peer_ids: list[int],
    from_week: int,
    to_week: int,
    kpi: str = "bet_handle",
) -> pd.DataFrame:
    """
    P25/P50/P75 bands from peer games' trajectories for weeks [from_week, to_week].
    Returns DataFrame with columns: launch_week, p25, p50, p75, n_peers.
    """
    rows = []
    for gid in peer_ids:
        sub = df[(df["game_id"] == gid) &
                 (df["launch_week"] >= from_week) &
                 (df["launch_week"] <= to_week)][["launch_week", kpi]].dropna(subset=[kpi])
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=["launch_week", "p25", "p50", "p75", "n_peers"])

    combined = pd.concat(rows, ignore_index=True)
    result = (combined.groupby("launch_week")[kpi]
              .agg(p25=lambda x: np.nanpercentile(x, 25),
                   p50=lambda x: np.nanpercentile(x, 50),
                   p75=lambda x: np.nanpercentile(x, 75),
                   n_peers="count")
              .reset_index())
    return result


# ── Scale-aware peer matching ─────────────────────────────────────────
def find_peers_scaled(
    df: pd.DataFrame,
    target_id: int,
    kpi: str = "bet_decay",
    n_weeks: int | None = None,
    top_k: int = 5,
    exclude_hr: bool = True,
    scale_tolerance: float = 2.0,
    min_candidate_weeks: int = 4,
) -> list[dict]:
    """Like find_peers but only considers candidates whose week-0 bet handle is
    within `scale_tolerance`× of the target game's week-0 bet handle.
    Falls back to unconstrained if fewer than top_k candidates qualify."""
    w0 = df[df["launch_week"] == 0].set_index("game_id")["bet_handle"]
    target_w0 = float(w0.get(target_id, np.nan))
    if np.isnan(target_w0) or target_w0 <= 0:
        return find_peers(df, target_id, kpi=kpi, n_weeks=n_weeks, top_k=top_k,
                          exclude_hr=exclude_hr, min_candidate_weeks=min_candidate_weeks)
    lo, hi = target_w0 / scale_tolerance, target_w0 * scale_tolerance
    scaled_ids = set(w0[(w0 >= lo) & (w0 <= hi)].index.tolist())
    df_scaled = df[df["game_id"].isin(scaled_ids)]
    results = find_peers(df_scaled, target_id, kpi=kpi, n_weeks=n_weeks, top_k=top_k,
                         exclude_hr=exclude_hr, min_candidate_weeks=min_candidate_weeks)
    if len(results) < max(2, top_k // 2):
        results = find_peers(df, target_id, kpi=kpi, n_weeks=n_weeks, top_k=top_k,
                             exclude_hr=exclude_hr, min_candidate_weeks=min_candidate_weeks)
        for r in results:
            r["scale_constrained"] = False
    else:
        for r in results:
            r["scale_constrained"] = True
    return results


def detect_ramp(df: pd.DataFrame, target_id: int, ramp_weeks: int = 4) -> dict:
    """Detect soft-launch / ramp-up: if stores grow >50% in the first ramp_weeks,
    the week-0 baseline is unreliable."""
    gdf = df[df["game_id"] == target_id].sort_values("launch_week")
    if "stores" not in gdf.columns or gdf.empty:
        return {"is_ramping": False, "note": ""}
    early = gdf[gdf["launch_week"] < ramp_weeks][["launch_week", "stores"]].dropna()
    if len(early) < 2:
        return {"is_ramping": False, "note": ""}
    s0 = float(early.iloc[0]["stores"]); sn = float(early.iloc[-1]["stores"])
    if s0 <= 0:
        return {"is_ramping": False, "note": ""}
    growth = (sn - s0) / s0
    if growth > 0.5:
        return {
            "is_ramping": True,
            "growth_pct": round(growth * 100, 1),
            "w0_stores": int(s0),
            "peak_stores": int(sn),
            "note": (f"Soft launch detected: stores grew {growth*100:.0f}% in first {ramp_weeks} weeks "
                     f"({int(s0)} → {int(sn)}). Week-0 bet baseline may understate true launch scale."),
        }
    return {"is_ramping": False, "growth_pct": round(growth * 100, 1), "note": ""}


def revenue_milestone_bands(
    df: pd.DataFrame,
    target_id: int,
    milestones: list[int] | None = None,
    scale_tolerance: float = 2.0,
    exclude_hr: bool = True,
) -> pd.DataFrame:
    """For milestone weeks [4, 8, 13], compute P25/P50/P75 of net_rev and bet_handle
    from games whose week-0 bet is within scale_tolerance× of the target.
    Returns a DataFrame with one row per milestone week."""
    if milestones is None:
        milestones = [4, 8, 13]
    w0 = df[df["launch_week"] == 0].set_index("game_id")["bet_handle"]
    target_w0 = float(w0.get(target_id, np.nan))
    if np.isnan(target_w0) or target_w0 <= 0:
        peers_df = df
    else:
        lo, hi = target_w0 / scale_tolerance, target_w0 * scale_tolerance
        scaled_ids = set(w0[(w0 >= lo) & (w0 <= hi)].index.tolist())
        peers_df = df[df["game_id"].isin(scaled_ids)]
    if exclude_hr:
        peers_df = peers_df[~peers_df["game_id"].apply(_is_hr)]
    rows = []
    for wk in milestones:
        at_wk = peers_df[(peers_df["launch_week"] == wk) & (peers_df["game_id"] != target_id)]
        for kpi in ("net_rev", "bet_handle", "hold_pct", "bet_decay"):
            if kpi not in at_wk.columns:
                continue
            vals = at_wk[kpi].dropna().values
            if len(vals) < 3:
                continue
            rows.append({
                "week": wk, "kpi": kpi, "n": len(vals),
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90)),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Fleet bands ───────────────────────────────────────────────────────
def compute_fleet_bands(
    df: pd.DataFrame,
    kpi: str = "bet_decay",
    exclude_hr: bool = True,
    min_games_per_week: int = 3,
) -> pd.DataFrame:
    """
    P10/P25/P50/P75/P90 bands per launch_week across all games.
    Only weeks where ≥ min_games_per_week games have data are included.
    """
    src = df[~df["game_id"].apply(_is_hr)] if exclude_hr else df
    if kpi not in src.columns:
        return pd.DataFrame()

    result = (src.groupby("launch_week")[kpi]
              .agg(p10=lambda x: np.nanpercentile(x.dropna(), 10) if x.dropna().size else np.nan,
                   p25=lambda x: np.nanpercentile(x.dropna(), 25) if x.dropna().size else np.nan,
                   p50=lambda x: np.nanpercentile(x.dropna(), 50) if x.dropna().size else np.nan,
                   p75=lambda x: np.nanpercentile(x.dropna(), 75) if x.dropna().size else np.nan,
                   p90=lambda x: np.nanpercentile(x.dropna(), 90) if x.dropna().size else np.nan,
                   n_games=lambda x: x.dropna().size)
              .reset_index())
    return result[result["n_games"] >= min_games_per_week]


def load_live_player_segments(conn) -> pd.DataFrame:
    """
    Run the full player segmentation pipeline live from SQL.
    Matches Player Insights (app.py) exactly:
      - Source: BetSpinSummaryCashView3 with PlatformName='Pong' AND CasinoName='PFH'
      - Window: 2026-01-01 to MAX(Date) in DB (same fixed start as Player Insights)
    Returns a DataFrame matching the PlayerSegmentation.csv schema:
    AccountNumber, Cluster, State, PrimaryStoreId, PrimaryStoreName,
    BetVolume2026, CI30, CI30Prior, TotalSpins, LAD, AD30, Recency, LastPlayed
    """
    import datetime, numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    # Use MAX date in DB as as_of (matches Player Insights get_latest_date() call)
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(\"Date\") FROM BetSpinSummaryCashView3 "
                    "WHERE PlatformName='Pong' AND CasinoName='PFH' AND \"Date\" >= '2026-01-01'")
        row = cur.fetchone()
        as_of = str(row[0])[:10] if row and row[0] else datetime.date.today().isoformat()
    except Exception:
        as_of = datetime.date.today().isoformat()

    window = "2026-01-01"  # fixed start — same as Player Insights WINDOW_START

    extract_sql = f"""
    WITH bet_summary AS (
      SELECT AccountNumber, StoreNumber, Date,
             SUM(CAST(TotalBet AS BIGINT)) AS DailyBet_cents,
             SUM(Spins) AS DailySpins
      FROM BetSpinSummaryCashView3
      WHERE PlatformName='Pong' AND CasinoName='PFH'
        AND Date >= '{window}' AND Date <= '{as_of}'
      GROUP BY AccountNumber, StoreNumber, Date
    )
    SELECT AccountNumber,
      SUM(DailyBet_cents) / 100.0 AS LCI,
      SUM(CASE WHEN Date > DATEADD('day',-30,'{as_of}') THEN DailyBet_cents ELSE 0 END)/100.0 AS CI30,
      SUM(CASE WHEN Date > DATEADD('day',-60,'{as_of}') AND Date <= DATEADD('day',-30,'{as_of}') THEN DailyBet_cents ELSE 0 END)/100.0 AS CI30Prior,
      SUM(DailySpins) AS TotalSpins,
      COUNT(DISTINCT Date) AS LAD,
      COUNT(DISTINCT CASE WHEN Date > DATEADD('day',-30,'{as_of}') THEN Date END) AS AD30,
      DATEDIFF('day', MAX(Date), '{as_of}') AS Recency
    FROM bet_summary
    GROUP BY AccountNumber
    HAVING COUNT(DISTINCT Date) >= 3 AND SUM(DailyBet_cents) > 0
    """

    store_sql = f"""
    WITH psv AS (
      SELECT b.AccountNumber, b.StoreNumber,
             SUM(CAST(b.TotalBet AS BIGINT)) AS WindowBet,
             ROW_NUMBER() OVER (PARTITION BY b.AccountNumber ORDER BY SUM(CAST(b.TotalBet AS BIGINT)) DESC) AS rn
      FROM BetSpinSummaryCashView3 b
      WHERE b.PlatformName='Pong' AND b.CasinoName='PFH'
        AND b.Date >= '{window}' AND b.Date <= '{as_of}'
      GROUP BY b.AccountNumber, b.StoreNumber
    )
    SELECT ps.AccountNumber, ps.StoreNumber AS PrimaryStoreId,
           c.StateProv AS State, c.BusinessName AS PrimaryStoreName
    FROM psv ps
    LEFT JOIN CrmLocationView c ON CAST(ps.StoreNumber AS VARCHAR(50)) = c.LocationId
    WHERE ps.rn = 1
    """

    try:
        df       = E.query_df(conn, extract_sql)
        store_df = E.query_df(conn, store_sql)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Cast all numeric columns — Oracle returns decimal.Decimal which np.log1p can't handle
    for col in df.select_dtypes(exclude=['number']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['LCI','CI30','CI30Prior','TotalSpins','LAD','AD30','Recency']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Feature engineering
    X = pd.DataFrame(index=df.index)
    X['log_LCI']          = np.log1p(df['LCI'])
    X['log_CI30']         = np.log1p(df['CI30'])
    max_rec               = df['Recency'].max() or 1
    X['recency_score']    = 1.0 - (df['Recency'] / max_rec)
    X['recent_active_pct']= df['AD30'] / 30.0
    X['log_LAD']          = np.log1p(df['LAD'])
    avg_bet               = np.where(df['TotalSpins'] > 0, df['LCI'] / df['TotalSpins'], 0)
    X['log_avg_bet']      = np.log1p(np.minimum(avg_bet, 100))
    spins_day             = np.where(df['LAD'] > 0, df['TotalSpins'] / df['LAD'], 0)
    X['log_spins_day']    = np.log1p(spins_day)
    trend                 = np.where(df['CI30Prior'] > 0, (df['CI30'] / df['CI30Prior']) - 1.0,
                                     np.where(df['CI30'] > 0, 5.0, 0.0))
    X['trend_clipped']    = np.clip(trend, -1, 5)

    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=6, n_init=25, random_state=42, max_iter=500)
    df['cluster_raw'] = km.fit_predict(Xs)

    # Map cluster IDs to names by centroid
    centroids  = pd.DataFrame(km.cluster_centers_, columns=X.columns)
    remaining  = list(range(6))
    name_map   = {}
    hr         = centroids.loc[remaining].sort_values('log_avg_bet', ascending=False).index[0]
    name_map[hr] = 'High-Roller'; remaining.remove(hr)
    vip        = centroids.loc[remaining].sort_values('recent_active_pct', ascending=False).index[0]
    name_map[vip] = 'VIP'; remaining.remove(vip)
    surge      = centroids.loc[remaining].sort_values('trend_clipped', ascending=False).index[0]
    name_map[surge] = 'Surging Spender'; remaining.remove(surge)
    lapsed     = centroids.loc[remaining].sort_values('recency_score', ascending=True).index[0]
    name_map[lapsed] = 'Lapsed'; remaining.remove(lapsed)
    light      = centroids.loc[remaining].sort_values('log_LCI', ascending=True).index[0]
    name_map[light] = 'Light Tried'; remaining.remove(light)
    name_map[remaining[0]] = 'Steady Regular'

    df['Cluster'] = df['cluster_raw'].map(name_map)
    df = df.merge(store_df, on='AccountNumber', how='left')

    output = df[['AccountNumber','Cluster','State','PrimaryStoreId',
                 'PrimaryStoreName','LCI','CI30','CI30Prior','TotalSpins',
                 'LAD','AD30','Recency']].copy()
    output.rename(columns={'LCI':'BetVolume2026'}, inplace=True)
    output['LastPlayed'] = (
        pd.to_datetime(as_of) - pd.to_timedelta(output['Recency'], unit='D')
    ).dt.date
    return output


def load_weekly_segment_mechanic_bets(conn, window_start: str = "2025-10-01", as_of_date: str = None) -> pd.DataFrame:
    import datetime
    if as_of_date is None:
        as_of_date = datetime.date.today().isoformat()
    """
    Weekly bet volume per player per game for the last ~9 months.
    Used for SES forecasting on the Player Segments page — gives real week-by-week
    history per segment so the model has enough data points to calibrate properly.
    Returns: AccountNumber, game_name, year_week (e.g. 202601), week_start (date), BetVolume
    """
    sql = f"""
    SELECT
        b.AccountNumber,
        gc.Name                                          AS game_name,
        YEAR(b.Date) * 100 + DATEPART('week', b.Date) AS year_week,
        MIN(b.Date)                                      AS week_start,
        SUM(CAST(b.TotalBet AS BIGINT)) / 100.0         AS BetVolume
    FROM BetSpinSummaryCashView3 b
    JOIN GameCatalogView1 gc
        ON gc.Id = TRY_CAST(b.GameId AS INT)
    WHERE b.PlatformName='Pong' AND b.CasinoName='PFH'
      AND b.Date >= '{window_start}'
      AND b.Date <= '{as_of_date}'
      AND TRY_CAST(b.GameId AS INT) IS NOT NULL
    GROUP BY
        b.AccountNumber,
        gc.Name,
        YEAR(b.Date) * 100 + DATEPART('week', b.Date)
    HAVING SUM(CAST(b.TotalBet AS BIGINT)) > 0
    """
    try:
        return E.query_df(conn, sql)
    except Exception:
        return pd.DataFrame()


def load_player_game_bets(conn, window_start: str = "2026-01-01", as_of_date: str = None) -> pd.DataFrame:
    import datetime
    if as_of_date is None:
        as_of_date = datetime.date.today().isoformat()
    """
    Player x Game bet volume for V1 (PFH/Pong) only.
    Joins BetSpinSummaryCashView3Pong with GameCatalogView1 to get game names.
    Returns: AccountNumber, game_id, game_name, BetVolume
    """
    sql = f"""
    SELECT
        b.AccountNumber,
        TRY_CAST(b.GameId AS INT)          AS game_id,
        gc.Name                            AS game_name,
        SUM(CAST(b.TotalBet AS BIGINT)) / 100.0 AS BetVolume
    FROM BetSpinSummaryCashView3 b
    JOIN GameCatalogView1 gc
        ON gc.Id = TRY_CAST(b.GameId AS INT)
    WHERE b.PlatformName='Pong' AND b.CasinoName='PFH'
      AND b.Date >= '{window_start}'
      AND b.Date <= '{as_of_date}'
      AND TRY_CAST(b.GameId AS INT) IS NOT NULL
    GROUP BY b.AccountNumber, TRY_CAST(b.GameId AS INT), gc.Name
    HAVING SUM(CAST(b.TotalBet AS BIGINT)) > 0
    """
    try:
        df = E.query_df(conn, sql)
        if not df.empty:
            df["BetVolume"] = pd.to_numeric(df["BetVolume"], errors="coerce").fillna(0)
        return df
    except Exception:
        return pd.DataFrame()


# ── Utility functions extracted from dashboard ────────────────────────────────

def detect_ramp(df: pd.DataFrame, target_id: int, ramp_weeks: int = 4) -> dict:
    """Detect soft-launch ramp in first `ramp_weeks` weeks (store count growth)."""
    gdf = df[df["game_id"] == target_id].sort_values("launch_week")
    if "stores" not in gdf.columns or gdf.empty:
        return {"is_ramping": False, "note": ""}
    early = gdf[gdf["launch_week"] < ramp_weeks][["launch_week", "stores"]].dropna()
    if len(early) < 2:
        return {"is_ramping": False, "note": ""}
    s0 = float(early.iloc[0]["stores"]); sn = float(early.iloc[-1]["stores"])
    if s0 <= 0:
        return {"is_ramping": False, "note": ""}
    growth = (sn - s0) / s0
    if growth > 0.5:
        return {"is_ramping": True, "growth_pct": round(growth * 100, 1),
                "w0_stores": int(s0), "peak_stores": int(sn),
                "note": (f"Soft launch detected: stores grew {growth*100:.0f}% in first "
                         f"{ramp_weeks} weeks ({int(s0)} → {int(sn)}). "
                         f"Week-0 bet baseline may understate true launch scale.")}
    return {"is_ramping": False, "growth_pct": round(growth * 100, 1), "note": ""}


def revenue_milestone_bands(df: pd.DataFrame, target_id: int,
                             milestones=None, scale_tolerance: float = 2.0,
                             exclude_hr: bool = True) -> pd.DataFrame:
    """P10/P25/P50/P75/P90 bands for peers at fixed milestone weeks."""
    if milestones is None:
        milestones = [4, 8, 13]
    w0 = df[df["launch_week"] == 0].set_index("game_id")["bet_handle"]
    target_w0 = float(w0.get(target_id, np.nan))
    if np.isnan(target_w0) or target_w0 <= 0:
        peers_df = df
    else:
        lo, hi = target_w0 / scale_tolerance, target_w0 * scale_tolerance
        scaled_ids = set(w0[(w0 >= lo) & (w0 <= hi)].index.tolist())
        peers_df = df[df["game_id"].isin(scaled_ids)]
    if exclude_hr:
        peers_df = peers_df[~peers_df["game_id"].apply(_is_hr)]
    rows = []
    for wk in milestones:
        at_wk = peers_df[(peers_df["launch_week"] == wk) & (peers_df["game_id"] != target_id)]
        for kpi in ("net_rev", "bet_handle", "hold_pct", "bet_decay"):
            if kpi not in at_wk.columns:
                continue
            vals = at_wk[kpi].dropna().values
            if len(vals) < 3:
                continue
            rows.append({"week": wk, "kpi": kpi, "n": len(vals),
                         "p25": float(np.percentile(vals, 25)),
                         "p50": float(np.percentile(vals, 50)),
                         "p75": float(np.percentile(vals, 75)),
                         "p10": float(np.percentile(vals, 10)),
                         "p90": float(np.percentile(vals, 90))})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def bands_kpi(df: pd.DataFrame, kpi: str, min_games: int = 3) -> pd.DataFrame:
    """Per-launch-week percentile bands for a KPI across non-HR games.

    Hot path — called once per (peer pool, kpi) in the CEO Health Screen loop
    (up to 260x4 times for V2). `.apply(_is_hr)` and 5 separate per-group
    np.percentile calls (each re-sorting the same data) previously dominated
    that loop's runtime (~305s of ~311s, profiled 2026-08-21); both are
    replaced here with vectorized equivalents that return identical values.
    """
    d = df[pd.to_numeric(df["game_id"], errors="coerce") < 95000]
    d = d[d[kpi].notna()]
    _cols = ["launch_week", "p10", "p25", "p50", "p75", "p90", "n"]
    if d.empty:
        # groupby.quantile().unstack() on zero rows yields 0 columns, not 5 --
        # unlike the old .agg(p10=..., ...) form, which always named its columns
        # even with no groups. Crashed live on a young game's peer pool with no
        # non-null rows for a given kpi (ValueError: Length mismatch 0 vs 5).
        return pd.DataFrame(columns=_cols)
    g = d.groupby("launch_week")[kpi]
    n = g.size()
    bands = g.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
    bands.columns = ["p10", "p25", "p50", "p75", "p90"]
    bands.loc[n < min_games, :] = np.nan
    return bands.assign(n=n).reset_index()


def pct_rank(df: pd.DataFrame, gid: int, kpi: str, at_week: int) -> float:
    """Percentile rank (0-100) of game `gid` in the `kpi` distribution at `at_week`."""
    arr = df[(df["launch_week"] == at_week) & df[kpi].notna()][kpi].values
    gv = df[(df["game_id"] == gid) & (df["launch_week"] == at_week)][kpi]
    if gv.empty or len(arr) == 0:
        return np.nan
    return float(np.sum(arr <= float(gv.iloc[0])) / len(arr) * 100)


def compute_cluster_features(df: pd.DataFrame, games_meta: pd.DataFrame,
                              mechanic_map: dict | None = None) -> pd.DataFrame:
    """Build a per-game feature matrix for clustering from weekly launch data."""
    if mechanic_map is None:
        mechanic_map = {}
    rows = []
    for _, gmeta in games_meta.iterrows():
        gid  = gmeta["game_id"]
        name = gmeta["game_name"]
        gdf  = df[df["game_id"] == gid].sort_values("launch_week")
        if gdf.empty or len(gdf) < 3:
            continue
        early = gdf[gdf["launch_week"] <= 7]

        hp = early["hold_pct"].dropna()
        avg_hold = float(hp.mean()) if len(hp) >= 2 else np.nan

        if len(hp) >= 3:
            wks_hp = early.loc[hp.index, "launch_week"].values.astype(float)
            hold_slope = float(np.polyfit(wks_hp, hp.values, 1)[0])
        else:
            hold_slope = np.nan

        h1 = gdf[gdf["launch_week"] == 1]["bet_handle"].values
        h4 = gdf[gdf["launch_week"] == 4]["bet_handle"].values
        retention_ratio = float(h4[0] / h1[0]) if len(h1) and len(h4) and h1[0] > 0 else np.nan

        bd = early[early["launch_week"] >= 1]["bet_decay"].dropna()
        avg_decay = float(bd.mean()) if len(bd) >= 2 else np.nan

        ar = gdf[gdf["launch_week"] <= 4]["arpu"].dropna()
        avg_arpu = float(ar.mean()) if len(ar) >= 2 else np.nan

        h0 = gdf[gdf["launch_week"] == 0]["bet_handle"].values
        launch_scale = float(np.log1p(h0[0])) if len(h0) and h0[0] > 0 else np.nan

        nr0_4 = float(gdf[gdf["launch_week"].between(0, 4)]["net_rev"].sum())
        nr4_8 = float(gdf[gdf["launch_week"].between(5, 8)]["net_rev"].sum())
        rev_accel = (nr4_8 / nr0_4) if nr0_4 > 100 else np.nan

        weeks_live = int(gdf["launch_week"].max())

        _mkey = name.lower().replace(" - gen2", "").replace(" hr", "").strip()
        mechanic = mechanic_map.get(_mkey, "Unknown")
        if mechanic == "Unknown":
            for _mk, _mv in mechanic_map.items():
                if _mk in _mkey or _mkey in _mk:
                    mechanic = _mv
                    break

        rows.append({
            "game_id":         gid,
            "game_name":       name,
            "launch_date":     gmeta.get("launch_date", ""),
            "avg_hold":        avg_hold,
            "hold_slope":      hold_slope,
            "retention_ratio": retention_ratio,
            "avg_decay":       avg_decay,
            "avg_arpu":        avg_arpu,
            "launch_scale":    launch_scale,
            "rev_accel":       rev_accel,
            "weeks_live":      weeks_live,
            "mechanic":        mechanic,
        })
    return pd.DataFrame(rows)


def run_kmeans(X: np.ndarray, k: int = 4, n_iter: int = 100, seed: int = 42) -> np.ndarray:
    """Simple k-means (no sklearn dependency)."""
    rng   = np.random.default_rng(seed)
    n     = X.shape[0]
    cents = X[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dists  = np.array([np.linalg.norm(X - c, axis=1) for c in cents])
        new_lb = np.argmin(dists, axis=0)
        if np.all(new_lb == labels):
            break
        labels = new_lb
        for j in range(k):
            mask = labels == j
            if mask.any():
                cents[j] = X[mask].mean(axis=0)
    return labels


def fit_linear_trend(series: np.ndarray, last_week: int,
                     n_forecast: int = 26, ci_pct: float = 0.15) -> dict:
    """Last-8-week linear trend forecast with symmetric ±ci_pct band.

    Returns dict with keys: weeks, values, upper, lower (all lists).
    Returns empty lists when fewer than 2 points are available.
    """
    arr = np.asarray(series, float)
    if len(arr) < 2:
        return {"weeks": [], "values": [], "upper": [], "lower": []}
    tail = arr[-8:]
    x = np.arange(len(arr) - len(tail), len(arr), dtype=float)
    slope = float(np.polyfit(x, tail, 1)[0])
    last_y = float(tail[-1])
    weeks  = list(range(last_week + 1, last_week + n_forecast + 1))
    values = [max(0.0, last_y + slope * i) for i in range(1, n_forecast + 1)]
    upper  = [v * (1 + ci_pct) for v in values]
    lower  = [v * (1 - ci_pct) for v in values]
    return {"weeks": weeks, "values": values, "upper": upper, "lower": lower}


# ── Per-game diagnostic report ────────────────────────────────────────────────

def _diagnostic_triggers(
    blended: dict,
    game_portfolio: dict,
    retro: list,
    max_wk: int,
) -> list[str]:
    """
    Return list of trigger codes that fired.  Empty list → game is healthy.

    T1  portfolio_underperform  — slope below portfolio P25 (last 12w)
    T2  declining_forecast      — self-slope < −2%/wk AND 4w forecast drops >3% AND ≥12w history
    T3  sustained_retro_red     — last 4 retro weeks all "underperforming"
    T4  degraded_confidence     — established game (≥20w) with reliable_horizon ≤ 4
    """
    fired: list[str] = []

    # T1
    if game_portfolio.get("verdict") == "underperforming":
        fired.append("portfolio_underperform")

    # T2 — only fires for established games (≥12 weeks); early-lifecycle volatility
    # produces unreliable self-slopes even though thin_history gate clears at 4 weeks.
    self_slope = blended.get("self_slope_pct")
    fcast = blended.get("forecast", [])
    if (self_slope is not None and self_slope < -2.0
            and max_wk >= 12
            and len(fcast) >= 4
            and fcast[3]["value"] < fcast[0]["value"] * 0.97):
        fired.append("declining_forecast")

    # T3
    if len(retro) >= 4 and all(r["verdict"] == "underperforming" for r in retro[-4:]):
        fired.append("sustained_retro_red")

    # T4
    rh = blended.get("reliable_horizon", 13)
    if max_wk >= 20 and rh <= 4:
        fired.append("degraded_confidence")

    return fired


def generate_game_diagnostic(
    df: pd.DataFrame,
    target_id: int,
    blended: dict,
    game_portfolio: dict,
    retro: list,
    conn=None,
    platform: str = "PFH",
    lookback_weeks: int = 4,
    lapsed_count: int | None = None,
    total_players: int | None = None,
) -> dict | None:
    """
    Compute a structured diagnostic report for a game that is showing a problem signal.

    Returns None if the game is healthy (no trigger fires).
    Returns a dict with keys: triggers, what, why, fix.

    All monetary values are raw floats (USD). Caller formats for display.
    All percentage values are already ×100 (e.g. −3.2 means −3.2 %).
    """
    gdf = df[df["game_id"] == target_id].sort_values("launch_week")
    if gdf.empty:
        return None

    max_wk = int(gdf["launch_week"].max())
    triggers = _diagnostic_triggers(blended, game_portfolio, retro, max_wk)
    if not triggers:
        return None

    # ── WHAT: measure the recent move ─────────────────────────────────────────
    recent = gdf[gdf["launch_week"] > max_wk - lookback_weeks]
    prior  = gdf[(gdf["launch_week"] > max_wk - lookback_weeks * 2) &
                 (gdf["launch_week"] <= max_wk - lookback_weeks)]

    recent_avg = float(recent["bet_handle"].mean()) if not recent.empty else None
    prior_avg  = float(prior["bet_handle"].mean())  if not prior.empty  else None

    if recent_avg is not None and prior_avg is not None and prior_avg > EPS:
        what_pct_change = (recent_avg - prior_avg) / prior_avg * 100.0
    else:
        what_pct_change = None

    # Weekly trajectory: last N actual values
    recent_vals = recent["bet_handle"].dropna().tolist()
    prior_vals  = prior["bet_handle"].dropna().tolist()

    what = {
        "recent_avg_bh":   recent_avg,
        "prior_avg_bh":    prior_avg,
        "pct_change":      what_pct_change,   # negative = decline
        "lookback_weeks":  lookback_weeks,
        "max_wk":          max_wk,
        "self_slope_pct":  blended.get("self_slope_pct"),
        "reliable_horizon": blended.get("reliable_horizon"),
        "triggers":        triggers,
    }

    # ── WHY: diagnostic chain ─────────────────────────────────────────────────
    # Step 1: Is this portfolio-wide?
    port_trend = compute_portfolio_trend(df, window_weeks=lookback_weeks)
    portfolio_wide = (port_trend["direction"] == "down" and
                      port_trend["consensus"] >= 0.55)

    # Step 2: Peer-relative slope
    gp = game_portfolio  # already computed by caller
    peer_rank   = gp.get("percentile_rank")
    target_slope = gp.get("target_slope_pct")
    median_slope = gp.get("portfolio_median_pct")
    n_portfolio  = gp.get("n_games", 0)

    # Step 3: Location data (best-effort — requires live connection)
    # V2/V1 pass through directly so get_active_location_count uses TaskHandlerBetSpinSummary.
    # PFH maps to "PFH" (engine MODES key). EdgeLabs/Pong pass through as-is.
    _loc_mode = {"V1": "V1", "V2": "V2", "PFH": "PFH"}.get(platform, platform)
    location_data: dict | None = None
    if conn is not None:
        try:
            location_data = E.get_active_location_count(conn, target_id, _loc_mode)
        except Exception as _loc_err:
            import sys as _sys
            print(f"[DIAG] get_active_location_count failed (mode={_loc_mode!r}): {_loc_err!r}",
                  file=_sys.stderr)

    # Step 4: Retrospective pattern (last 8 available retro entries)
    retro_recent = retro[-8:] if len(retro) >= 8 else retro
    red_count  = sum(1 for r in retro_recent if r["verdict"] == "underperforming")
    red_pct    = red_count / len(retro_recent) * 100 if retro_recent else 0

    # Determine primary cause for WHY narrative
    # "Both" case: portfolio is down AND this game is significantly worse than peers
    game_much_worse = (peer_rank is not None and peer_rank < 20)

    _is_physical = (platform == "V2")
    if location_data is not None and location_data.get("count", 0) == 0:
        primary_cause = "location_pullback"
        if _is_physical:
            cause_detail = (
                "This game has been taken off all casino floors in the last 14 days — "
                "there are 0 active locations right now. When a game isn't available "
                "to play anywhere, the numbers will drop. The performance data during "
                "this time doesn't tell us anything about the game itself."
            )
        else:
            cause_detail = (
                "This game has had 0 active player sessions in the last 14 days — "
                "it may have been removed from the lobby or is otherwise unreachable. "
                "When a game can't be found or played, the numbers will drop. "
                "The performance data during this period doesn't reflect the game's health."
            )
    elif portfolio_wide and game_much_worse and target_slope is not None:
        primary_cause = "both"
        cause_detail  = (
            f"Most other games on the platform slowed down at the same time — "
            f"{port_trend['consensus']*100:.0f}% of them dropped by an average of "
            f"{abs(port_trend['pct_change']):.1f}%. That part is probably not this "
            f"game's fault. But this game dropped much more than the rest. "
            f"Out of {n_portfolio} games we track, this one is doing worse than "
            f"{100 - peer_rank:.0f}% of them "
            f"({target_slope:+.1f}%/wk vs the typical game's {median_slope:+.1f}%/wk). "
            f"The market slowdown alone doesn't explain that gap."
        )
    elif portfolio_wide:
        primary_cause = "external"
        cause_detail  = (
            f"Most other games on the platform slowed down at the same time — "
            f"{port_trend['consensus']*100:.0f}% of them dropped by an average of "
            f"{abs(port_trend['pct_change']):.1f}% over the last {lookback_weeks} weeks. "
            f"This looks like something that affected the whole market "
            f"(like a quiet season or an external event), not a problem "
            f"specific to this game."
        )
    elif game_much_worse and target_slope is not None:
        primary_cause = "game_specific"
        cause_detail  = (
            f"The other games are doing fine — only this one is slowing down. "
            f"Out of {n_portfolio} games we track, this one is doing worse than "
            f"{100 - peer_rank:.0f}% of them. "
            f"This game is changing at {target_slope:+.1f}%/wk while a typical "
            f"game on the platform is at {median_slope:+.1f}%/wk."
        )
    elif location_data is not None:
        primary_cause = "game_specific"
        if _is_physical:
            cause_detail = (
                f"The number of locations is stable ({location_data['count']} active "
                f"{location_data.get('unit_label','locations')}), so the drop isn't "
                f"explained by the game being moved off floors. Something about how "
                f"players are engaging with this game has changed."
            )
        else:
            cause_detail = (
                f"Access looks stable ({location_data['count']} active "
                f"{location_data.get('unit_label','locations')}), so the drop isn't "
                f"explained by the game being removed from the lobby. Something about how "
                f"players are engaging with this game has changed."
            )
    else:
        primary_cause = "unclear"
        if _is_physical:
            cause_detail = (
                "The data we have right now doesn't point clearly to one cause. "
                "The other games aren't showing the same problem, but we don't have "
                "floor location data to check. More information is needed before "
                "drawing any conclusions."
            )
        else:
            cause_detail = (
                "The data we have right now doesn't point clearly to one cause. "
                "The other games aren't showing the same problem. "
                "More information is needed before drawing any conclusions."
            )

    # Retrospective summary for WHY
    if retro_recent:
        last_retro_verdict = retro_recent[-1]["verdict"]
        retro_summary = (
            f"{red_count} of the last {len(retro_recent)} charted weeks "
            f"show this game tracking below the portfolio average."
        )
    else:
        last_retro_verdict = None
        retro_summary = "Not enough history to chart week-by-week standing."

    why = {
        "portfolio_wide":   portfolio_wide,
        "port_trend":       port_trend,
        "primary_cause":    primary_cause,
        "cause_detail":     cause_detail,
        "peer_rank":        peer_rank,
        "target_slope_pct": target_slope,
        "median_slope_pct": median_slope,
        "n_portfolio":      n_portfolio,
        "location_data":    location_data,
        "retro_red_count":  red_count,
        "retro_total":      len(retro_recent),
        "retro_summary":    retro_summary,
        "red_pct":          red_pct,
    }

    # ── DATA SIGNALS FOR DATA-DRIVEN FIX ACTIONS ─────────────────────────────
    # Hold % anomalies
    _neg_hold_wks = []
    _elevated_hold = None
    _elevated_hold_pct = None
    if "hold_pct" in gdf.columns:
        _hold_s = gdf.dropna(subset=["hold_pct"])
        _neg_hold_wks = [int(r["launch_week"]) for _, r in _hold_s.iterrows() if r["hold_pct"] < 0]
        _cur_hold = float(_hold_s.iloc[-1]["hold_pct"]) if not _hold_s.empty else None
        # Fleet p75 for hold
        _fleet_hold = df["hold_pct"].dropna() if "hold_pct" in df.columns else None
        if _cur_hold is not None and _fleet_hold is not None and len(_fleet_hold) > 10:
            _fleet_hold_p75 = float(_fleet_hold.quantile(0.75))
            if _cur_hold > _fleet_hold_p75:
                _elevated_hold = _cur_hold
                _elevated_hold_pct = _fleet_hold_p75

    # Site/location contraction
    _site_launch = None
    _site_current = None
    _site_pct_drop = None
    if "stores" in gdf.columns:
        _stores_s = gdf.dropna(subset=["stores"])
        if not _stores_s.empty:
            _w0_stores = _stores_s[_stores_s["launch_week"] == 0]
            _site_launch  = int(_w0_stores["stores"].values[0]) if not _w0_stores.empty else None
            _site_current = int(_stores_s.iloc[-1]["stores"])
            if _site_launch and _site_launch > 0:
                _site_pct_drop = (_site_launch - _site_current) / _site_launch * 100

    # Retention collapse vs fleet benchmark
    _retention_cur = None
    _retention_benchmark = None
    _retention_collapsed = False
    if "bet_decay" in gdf.columns:
        _decay_s = gdf.dropna(subset=["bet_decay"])
        if not _decay_s.empty:
            _retention_cur = float(_decay_s.iloc[-1]["bet_decay"])
            # Fleet p25 at same week
            _cur_wk_ret = int(_decay_s.iloc[-1]["launch_week"])
            _fleet_at_wk = df[df["launch_week"] == _cur_wk_ret]["bet_decay"].dropna()
            if len(_fleet_at_wk) > 5:
                _retention_benchmark = float(_fleet_at_wk.quantile(0.25))
                _retention_collapsed = _retention_cur < _retention_benchmark * 0.5

    # Week-0 peak pattern
    _wk0_was_peak = False
    if "bet_handle" in gdf.columns:
        _bh_s = gdf.dropna(subset=["bet_handle"])
        if len(_bh_s) >= 3:
            _peak_wk_idx = int(_bh_s["bet_handle"].idxmax())
            _peak_wk_num = int(_bh_s.loc[_peak_wk_idx, "launch_week"])
            _wk0_was_peak = (_peak_wk_num == 0)

    # ── HOW TO FIX ────────────────────────────────────────────────────────────
    if primary_cause == "external":
        fix_headline = "Wait and watch — this looks like a market-wide slowdown, not a game problem."
        if _is_physical:
            fix_body = (
                "Because most other games slowed down at the same time, moving or "
                "pulling this game from floors wouldn't fix anything — the cause is "
                "outside the game itself. The best move is to wait 2–4 weeks. "
                "If the other games recover and this one doesn't, that's the moment to act."
            )
            fix_actions = [
                "Don't change floor allocation yet — wait 2–4 weeks to see if "
                "the market picks back up.",
                "Check last year's same time period — if games always dip at this "
                "time of year, this is seasonal and will fix itself.",
            ]
        else:
            fix_body = (
                "Because most other games slowed down at the same time, this is "
                "likely a platform-wide or seasonal pattern — not a problem specific "
                "to this game. The best move is to wait 2–4 weeks. "
                "If the other games recover and this one doesn't, that's the moment to act."
            )
            fix_actions = [
                "Hold off on any lobby or promotion changes — wait 2–4 weeks to see "
                "if the platform-wide slowdown reverses.",
                "Check last year's same time period — if player activity always dips "
                "at this time of year, this is seasonal and will fix itself.",
            ]
        # Regardless of platform: flag payout issues and lapsed players if signals present
        if _neg_hold_wks:
            fix_actions.append(
                f"While waiting, review payout settings — the house lost money in "
                f"{len(_neg_hold_wks)} week(s) (W{', W'.join(str(w) for w in _neg_hold_wks)}). "
                f"This may recover naturally, but recurring negative hold warrants a config check."
            )
        if lapsed_count and total_players and lapsed_count > 0:
            _lapsed_pct = round(lapsed_count / total_players * 100)
            fix_actions.append(
                f"Run a reactivation offer to the {lapsed_count:,} lapsed players "
                f"({_lapsed_pct}% of the player base) — they already know the game and cost "
                f"less to bring back than acquiring new players."
            )
    elif primary_cause == "both":
        fix_headline = "The market slowdown is real, but this game has a deeper problem that won't fix itself when the market recovers."
        fix_body = (
            "Part of the drop is shared with other games and may ease over the next 2–4 weeks. "
            "But this game is falling harder than its neighbours — meaning there's something "
            "game-specific going on that the market recovery won't fix on its own."
        )
        fix_actions = [
            "Wait 2–4 weeks for the market-wide slowdown to settle, but don't expect a "
            "full recovery from that alone — the game-specific issues below need separate action."
        ]
        # Payout / hold anomalies
        if _neg_hold_wks or (_elevated_hold is not None and _elevated_hold_pct is not None):
            _hold_parts = []
            if _neg_hold_wks:
                _hold_parts.append(
                    f"the house lost money in {len(_neg_hold_wks)} week(s) "
                    f"(W{', W'.join(str(w) for w in _neg_hold_wks)})"
                )
            if _elevated_hold is not None and _elevated_hold_pct is not None:
                _hold_parts.append(
                    f"current hold % ({_elevated_hold:.1f}%) is above the top 25% of comparable games "
                    f"({_elevated_hold_pct:.1f}%) — a high take rate can reduce return visits"
                )
            fix_actions.append(
                f"Review the payout configuration: {' and '.join(_hold_parts)}. "
                f"A game that feels tight will lose players faster than one with the right win frequency."
            )
        # Site/lobby contraction
        if _site_pct_drop is not None and _site_pct_drop >= 30:
            if _is_physical:
                fix_actions.append(
                    f"Investigate the distribution contraction: this game launched at "
                    f"{_site_launch} locations and is now active at {_site_current} "
                    f"({_site_pct_drop:.0f}% reduction). Check whether this was intentional or automatic — "
                    f"if automatic, the drop is pulling the numbers down independently of game quality."
                )
            else:
                fix_actions.append(
                    f"Investigate the access contraction: this game launched on "
                    f"{_site_launch} sites/partner channels and is now active on {_site_current} "
                    f"({_site_pct_drop:.0f}% reduction). Find out if it was de-prioritised or removed "
                    f"from lobby rotations — restoring access is the fastest path to volume recovery."
                )
        # Retention collapse
        if _retention_collapsed and _retention_cur is not None and _retention_benchmark is not None:
            if _wk0_was_peak:
                fix_actions.append(
                    f"The game peaked at Week 0 and declined every week since — it never built a "
                    f"returning player base. Retention is now {_retention_cur:.1f}% of launch-week "
                    f"wagering vs the bottom-25% benchmark of {_retention_benchmark:.1f}%. "
                    f"This pattern means players tried it once and didn't come back — the game mechanic "
                    f"or payout feel may not be creating the 'one more spin' impulse."
                )
            else:
                fix_actions.append(
                    f"Player retention has collapsed: only {_retention_cur:.1f}% of launch-week "
                    f"wagering remains, vs the bottom-25% benchmark of {_retention_benchmark:.1f}%. "
                    f"Players are leaving faster than normal — a targeted re-engagement promotion "
                    f"to recent players is the fastest lever."
                )
        # Lapsed player reactivation
        if lapsed_count and total_players and lapsed_count > 0:
            _lapsed_pct = round(lapsed_count / total_players * 100)
            fix_actions.append(
                f"Run a reactivation campaign targeting the {lapsed_count:,} lapsed players "
                f"({_lapsed_pct}% of the {total_players:,} total player base). "
                f"They already know the game and cost less to bring back than acquiring new players."
            )
        # Fallback if no specific signals fired
        if len(fix_actions) == 1:
            if _is_physical:
                fix_actions.append(
                    "Once other games recover, check floor position and whether a nearby competing "
                    "title is drawing players away."
                )
            else:
                fix_actions.append(
                    "Once the market recovers, check lobby placement and featured rotation status — "
                    "if the game has been deprioritised, that's a fast fix."
                )
    elif primary_cause == "location_pullback":
        if _is_physical:
            fix_headline = "Get the game back on floors first — then we can properly read the numbers."
            fix_body = (
                "The drop in wagers is simply because the game isn't available to play "
                "anywhere right now. Performance data during a floor absence doesn't "
                "tell us anything useful about the game's health — the slots are just empty."
            )
            fix_actions = [
                "Check with operations: was removing it from floors intentional?",
                "If it was unintentional, restore it to its previous floor allocation "
                "and re-check performance after 4 weeks of live data.",
            ]
        else:
            fix_headline = "Get the game back in the lobby first — then we can properly read the numbers."
            fix_body = (
                "The drop in play is simply because the game has had no active sessions "
                "recently — it may have been delisted or removed from the lobby. "
                "Performance data during this gap doesn't tell us anything useful "
                "about the game's health."
            )
            fix_actions = [
                "Check with the platform team: was removing it from the lobby intentional?",
                "If it was unintentional, restore it to its previous lobby position "
                "and re-check performance after 4 weeks of live data.",
            ]
    elif primary_cause == "game_specific":
        if _is_physical:
            fix_headline = "The rest of the platform is fine — this game's problem is specific to it."
            fix_body = (
                "The other games are doing fine, so this isn't a market issue. "
                "Look at what changed for this game specifically."
            )
        else:
            fix_headline = "The rest of the platform is fine — this game's problem is specific to it."
            fix_body = (
                "The other games are doing fine, so this isn't a market issue. "
                "Look at what changed for this game specifically."
            )
        fix_actions = []
        # Payout anomalies
        if _neg_hold_wks or (_elevated_hold is not None and _elevated_hold_pct is not None):
            _hold_parts = []
            if _neg_hold_wks:
                _hold_parts.append(
                    f"negative hold in W{', W'.join(str(w) for w in _neg_hold_wks)}"
                )
            if _elevated_hold is not None and _elevated_hold_pct is not None:
                _hold_parts.append(
                    f"current hold {_elevated_hold:.1f}% (above 75% of comparable games at {_elevated_hold_pct:.1f}%)"
                )
            fix_actions.append(
                f"Review payout configuration — {'; '.join(_hold_parts)}. "
                f"A game that feels tight discourages return visits."
            )
        # Site contraction
        if _site_pct_drop is not None and _site_pct_drop >= 30:
            if _is_physical:
                fix_actions.append(
                    f"This game launched at {_site_launch} locations and is now at {_site_current} "
                    f"({_site_pct_drop:.0f}% reduction). Check whether the contraction was intentional — "
                    f"if automatic, restoring distribution may recover volume quickly."
                )
            else:
                fix_actions.append(
                    f"This game launched on {_site_launch} sites/channels and is now on {_site_current} "
                    f"({_site_pct_drop:.0f}% reduction). Check whether it has been removed from lobby "
                    f"rotations or partner sites — restoring access is the fastest lever."
                )
        # Retention and peak pattern
        if _retention_collapsed and _retention_cur is not None and _retention_benchmark is not None:
            if _wk0_was_peak:
                fix_actions.append(
                    f"The game peaked at Week 0 and declined every week since. Retention is "
                    f"{_retention_cur:.1f}% vs the bottom-25% benchmark of {_retention_benchmark:.1f}%. "
                    f"Players tried it once and didn't come back — look at win frequency and session feel."
                )
            else:
                fix_actions.append(
                    f"Retention has collapsed to {_retention_cur:.1f}% of launch-week wagering "
                    f"(bottom-25% benchmark: {_retention_benchmark:.1f}%). "
                    f"A targeted re-engagement offer to recent lapsed players may slow the decline."
                )
        # Lapsed reactivation
        if lapsed_count and total_players and lapsed_count > 0:
            _lapsed_pct = round(lapsed_count / total_players * 100)
            fix_actions.append(
                f"Target the {lapsed_count:,} lapsed players ({_lapsed_pct}% of the player base) "
                f"with a reactivation bonus — they're already familiar with the game."
            )
        # Generic fallback
        if not fix_actions:
            if _is_physical:
                fix_actions = [
                    "Check if a floor reshuffle or nearby competing game launch coincided with the drop.",
                    "Consider a targeted promotion to bring back players who drifted away.",
                    "If no improvement after 8 weeks, consider reallocating the floor space.",
                ]
            else:
                fix_actions = [
                    "Check if a lobby reshuffle or competing game launch coincided with the drop.",
                    "Verify this game is still in featured/recommended rotations.",
                    "Consider a targeted bonus offer to re-engage players who've stopped playing.",
                ]
    else:
        fix_headline = "Gather more information before acting."
        fix_body = (
            "The data we have right now doesn't point to one clear cause. "
            "Keep monitoring — if the drop continues for 2 more weeks, "
            "dig into player-level data to narrow it down."
        )
        fix_actions = []
        if _neg_hold_wks or (_elevated_hold is not None and _elevated_hold_pct is not None):
            _hold_parts = []
            if _neg_hold_wks:
                _hold_parts.append(f"negative hold in W{', W'.join(str(w) for w in _neg_hold_wks)}")
            if _elevated_hold is not None and _elevated_hold_pct is not None:
                _hold_parts.append(
                    f"current hold {_elevated_hold:.1f}% above the top 25% ({_elevated_hold_pct:.1f}%)"
                )
            fix_actions.append(
                f"Review payout configuration while the cause is unclear — {'; '.join(_hold_parts)}."
            )
        if lapsed_count and total_players and lapsed_count > 0:
            _lapsed_pct = round(lapsed_count / total_players * 100)
            fix_actions.append(
                f"Target the {lapsed_count:,} lapsed players ({_lapsed_pct}% of the player base) "
                f"with a reactivation offer — this is the lowest-risk action while the cause is unclear."
            )
        if _is_physical:
            if location_data is not None:
                _loc_n2   = location_data["count"]
                _loc_lbl2 = location_data.get("unit_label", "locations")
                fix_actions.append(
                    f"This game is currently active at {_loc_n2} {_loc_lbl2}. "
                    f"Check if underperformance is concentrated at a few of them or spread evenly."
                )
            else:
                fix_actions.append("Check location data manually if it's not showing here.")
        else:
            fix_actions.append(
                "Break down performance by player segment — VIPs dropping off is a different "
                "fix than casual players leaving, and points to different root causes."
            )
        fix_actions.append("Flag for a follow-up review in 2 weeks if no clear pattern emerges.")

    fix = {
        "headline":    fix_headline,
        "body":        fix_body,
        "actions":     fix_actions,
        "primary_cause": primary_cause,
    }

    return {"triggers": triggers, "what": what, "why": why, "fix": fix}

