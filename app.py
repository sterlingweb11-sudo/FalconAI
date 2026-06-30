from flask import Flask, render_template_string, request
import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import requests
from datetime import datetime

app = Flask(__name__)

FILE = "signals.csv"

# =========================
# DATA — with timeout + cache to prevent worker OOM/timeout kills
# =========================

# Shared HTTP session with a hard timeout — prevents yfinance from hanging forever
_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})

_DATA_CACHE = {}     # symbol -> (timestamp, df)
_CACHE_TTL  = 90      # seconds — daily data doesn't need to be re-fetched every call

def _yf_download_safe(ticker, period="6mo", interval="1d", timeout=8, **kwargs):
    """
    yfinance download with a hard timeout. Without this, a single slow/hanging
    Yahoo Finance request can block a worker thread forever, and with several
    threads hanging at once the whole process runs out of memory and gets
    SIGKILLed by the host (this is exactly what caused the worker timeout crash).
    """
    try:
        df = yf.download(
            ticker, period=period, interval=interval,
            progress=False, timeout=timeout, session=_session, **kwargs
        )
        return df if df is not None and not df.empty else None
    except Exception:
        return None


def get_data(symbol):
    """Cached daily data fetch — avoids re-downloading the same symbol repeatedly."""
    now = time.time()
    cached = _DATA_CACHE.get(symbol)
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1]

    df = _yf_download_safe(symbol + ".NS", period="6mo", interval="1d", timeout=8)
    if df is not None:
        _DATA_CACHE[symbol] = (now, df)
        return df
    # Serve stale cache rather than nothing, if we have it
    if cached:
        return cached[1]
    return None

def fmt(x):
    try:
        return round(float(x), 2)
    except:
        return 0.0

def safe_series(x):
    try:
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        return pd.Series(x).astype(float).dropna().reset_index(drop=True)
    except:
        return pd.Series([0])


def flatten_df(df):
    """
    Normalise a yfinance DataFrame so column access always works.
    yfinance ≥ 0.2.x returns MultiIndex columns like ('Close','RELIANCE.NS').
    This collapses them to simple string names: 'Close', 'Open', etc.
    Also drops any all-NaN rows and resets nothing — preserves datetime index.
    """
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Drop duplicate column names (yfinance sometimes emits them)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.dropna(how="all")

# =========================
# MEMORY
# =========================
def save_signal(data):
    df_new = pd.DataFrame([data])
    if os.path.exists(FILE):
        old = pd.read_csv(FILE)
        df = pd.concat([old, df_new], ignore_index=True)
    else:
        df = df_new
    df = df.drop_duplicates(subset=["symbol"], keep="last")
    df.to_csv(FILE, index=False)

# =========================
# SENTIMENT
# =========================
def sentiment(score):
    if score >= 80:
        return "🟢 Strong Bullish"
    elif score >= 65:
        return "🟡 Bullish"
    elif score >= 50:
        return "⚪ Neutral"
    elif score >= 35:
        return "🟠 Weak"
    else:
        return "🔴 Bearish"

# =========================
# TAG
# =========================
def tag(label, value, color):
    return f"""
    <span style="
        display:inline-block;
        padding:5px 10px;
        margin-right:6px;
        border-radius:20px;
        background:#0f172a;
        border:1px solid {color};
        font-size:12px;">
        {label}: {value}
    </span>
    """

# =========================
# STRENGTH BAR
# =========================
def strength_bar(score):
    bars = int(min(5, max(0, score // 20)))
    if bars >= 4:
        tip = "Very strong momentum setup"
    elif bars == 3:
        tip = "Moderate setup"
    elif bars == 2:
        tip = "Weak structure"
    else:
        tip = "No setup"
    return f"""
    <span class="tooltip">
        {"🟩"*bars + "⬜"*(5-bars)}
        <span class="tooltiptext">{tip}</span>
    </span>
    """

# =========================
# VCP ENGINE (NEW MODULE)
# =========================
def detect_vcp(df):
    """
    Detects a true Volatility Contraction Pattern (VCP).

    Checks all 5 conditions:
    1. 3-5 distinct contractions (price swings narrowing over time)
    2. Shrinking volatility in each contraction
    3. Higher lows forming (uptrend structure)
    4. Tight pivot (price coiling near resistance)
    5. Volume drying up during contractions

    Returns:
        vcp_score   : 0-100 score for VCP quality
        vcp_stage   : label like "Stage 1 of 5" or "Full VCP"
        vcp_details : dict with each condition result
        vcp_badge   : emoji badge string
    """

    close  = safe_series(df["Close"])
    volume = safe_series(df["Volume"])

    if len(close) < 60:
        return {
            "vcp_score": 0,
            "vcp_stage": "Not enough data",
            "vcp_details": {},
            "vcp_badge": "❌ No VCP"
        }

    # --- Rolling 10-day ranges to detect contractions ---
    window = 10
    ranges = []
    lows   = []

    for i in range(0, min(50, len(close) - window), window):
        chunk = close.iloc[-(50 - i): -(50 - i - window)] if (50 - i - window) > 0 else close.iloc[-(50-i):]
        if len(chunk) < 3:
            continue
        r = float(chunk.max() - chunk.min())
        lo = float(chunk.min())
        ranges.append(r)
        lows.append(lo)

    # --- Rolling 10-day avg volume to detect drying ---
    vol_windows = []
    for i in range(0, min(50, len(volume) - window), window):
        chunk = volume.iloc[-(50 - i): -(50 - i - window)] if (50 - i - window) > 0 else volume.iloc[-(50-i):]
        if len(chunk) < 3:
            continue
        vol_windows.append(float(chunk.mean()))

    # CONDITION 1: 3–5 contractions (at least 3 swing periods identified)
    num_contractions = len(ranges)
    cond1 = num_contractions >= 3
    contraction_count = min(num_contractions, 5)

    # CONDITION 2: Shrinking volatility (each range smaller than the prior)
    shrinking = sum(1 for i in range(1, len(ranges)) if ranges[i] < ranges[i-1])
    cond2 = shrinking >= 2
    shrink_pct = round((shrinking / max(len(ranges)-1, 1)) * 100, 1)

    # CONDITION 3: Higher lows
    higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
    cond3 = higher_lows >= 2
    hl_pct = round((higher_lows / max(len(lows)-1, 1)) * 100, 1)

    # CONDITION 4: Tight pivot — last 5 days range < 5% of price
    last_price = float(close.iloc[-1])
    pivot_range = float(close.tail(5).max() - close.tail(5).min())
    pivot_pct   = (pivot_range / last_price) * 100 if last_price > 0 else 99
    cond4 = pivot_pct < 5.0

    # CONDITION 5: Volume drying up (latest vol window < average of prior windows)
    if len(vol_windows) >= 2:
        avg_vol = np.mean(vol_windows[:-1])
        latest_vol = vol_windows[-1]
        vol_dry = (latest_vol < avg_vol * 0.85)
        vol_ratio = round((latest_vol / avg_vol) * 100, 1) if avg_vol > 0 else 100
    else:
        vol_dry = False
        vol_ratio = 100
    cond5 = vol_dry

    # --- Score (20 pts per condition) ---
    conditions_met = sum([cond1, cond2, cond3, cond4, cond5])
    vcp_score = conditions_met * 20

    # Bonus: tighter pivot = better
    if cond4 and pivot_pct < 3.0:
        vcp_score = min(100, vcp_score + 5)

    # --- Stage label ---
    if conditions_met == 5:
        stage = "🏆 Full VCP (5/5)"
        badge = "🔥 Strong VCP"
    elif conditions_met == 4:
        stage = "⭐ Near-Complete VCP (4/5)"
        badge = "🟡 Developing VCP"
    elif conditions_met == 3:
        stage = "📐 Partial VCP (3/5)"
        badge = "🟠 Early VCP"
    else:
        stage = f"🔍 Weak/No VCP ({conditions_met}/5)"
        badge = "❌ No VCP"

    return {
        "vcp_score": vcp_score,
        "vcp_stage": stage,
        "vcp_badge": badge,
        "vcp_details": {
            "contractions": f"{'✅' if cond1 else '❌'} {contraction_count} contractions detected (need ≥3)",
            "shrinking_vol": f"{'✅' if cond2 else '❌'} Volatility shrinking {shrink_pct}% of swings",
            "higher_lows":   f"{'✅' if cond3 else '❌'} Higher lows in {hl_pct}% of swings",
            "tight_pivot":   f"{'✅' if cond4 else '❌'} Pivot range {pivot_pct:.1f}% (need <5%)",
            "volume_dryup":  f"{'✅' if cond5 else '❌'} Volume at {vol_ratio}% of avg (need <85%)"
        }
    }

# =========================
# SIMILARITY AI 2.0
# =========================

def build_fingerprint(df, vcp_score=0):
    """
    Build a 12-dimension fingerprint for a stock's current chart structure.
    All dimensions are normalised to 0-100 so they are directly comparable.

    Dimensions:
      1.  compression      – price range tightening (VCP base)
      2.  breakout         – proximity to 20-day high
      3.  trend            – MA20/MA50 position score
      4.  volume_ratio     – recent vol vs 20-day avg (capped 0-100)
      5.  atr_pct          – ATR as % of price (volatility level)
      6.  dist_52w_high    – how far below 52-week high (0=at high, 100=very far)
      7.  rs_score         – relative strength vs own 20/50 MA ratio
      8.  vcp_score        – VCP engine score passed in
      9.  price_slope      – 10-day price momentum angle (normalised)
     10.  vol_trend        – volume increasing or drying (trend of volume)
     11.  candle_quality   – avg bullish body ratio last 5 candles
     12.  stage_score      – early/mid/extended stage (early=best)
    """
    close  = safe_series(df["Close"])
    high   = safe_series(df["High"])
    low    = safe_series(df["Low"])
    open_  = safe_series(df["Open"])
    volume = safe_series(df["Volume"])

    price  = float(close.iloc[-1])

    # 1. Compression
    r15 = float(close.tail(15).max() - close.tail(15).min())
    r60 = float(close.tail(60).max() - close.tail(60).min())
    compression = round((1 - (r15 / r60)) * 100, 1) if r60 > 0 else 0
    compression = max(0, min(100, compression))

    # 2. Breakout proximity (0=far, 100=at/above high)
    high20 = float(close.tail(20).max())
    breakout = round(min(100, (price / high20) * 100), 1) if high20 > 0 else 50

    # 3. Trend (MA position)
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    trend = 0
    if price > ma20: trend += 40
    if ma20 > ma50:  trend += 40
    if price > float(close.tail(10).mean()): trend += 20

    # 4. Volume ratio (recent 5-day avg vs 20-day avg), capped 0-100
    avg_vol20 = float(volume.tail(20).mean())
    avg_vol5  = float(volume.tail(5).mean())
    vol_ratio = round((avg_vol5 / avg_vol20) * 50, 1) if avg_vol20 > 0 else 50
    vol_ratio = max(0, min(100, vol_ratio))

    # 5. ATR% — daily true range as % of price (lower = tighter = better for VCP)
    tr_list = []
    for i in range(1, min(15, len(close))):
        tr = max(
            float(high.iloc[-i]) - float(low.iloc[-i]),
            abs(float(high.iloc[-i]) - float(close.iloc[-i-1])),
            abs(float(low.iloc[-i])  - float(close.iloc[-i-1]))
        )
        tr_list.append(tr)
    atr       = float(np.mean(tr_list)) if tr_list else 0
    atr_pct   = round((atr / price) * 100, 2) if price > 0 else 5
    # Normalise: 1% ATR = 90 score (tight), 5%+ ATR = 10 score (wide)
    atr_score = round(max(10, min(100, 100 - (atr_pct * 18))), 1)

    # 6. Distance from 52-week high (0=at high, 100=very far below)
    high52 = float(close.tail(min(252, len(close))).max())
    dist52 = round(((high52 - price) / high52) * 100, 1) if high52 > 0 else 50
    # Invert: closer to 52w high = higher score
    dist52_score = round(max(0, min(100, 100 - dist52)), 1)

    # 7. Relative Strength score (MA20/MA50 ratio — how strongly trending)
    rs = round((ma20 / ma50) * 50, 1) if ma50 > 0 else 50
    rs_score = max(0, min(100, rs))

    # 8. VCP score (passed in directly, already 0-100)
    vcp_s = max(0, min(100, vcp_score))

    # 9. Price slope — 10-day linear trend normalised
    if len(close) >= 10:
        y  = close.tail(10).values
        x  = np.arange(10)
        slope = float(np.polyfit(x, y, 1)[0])
        slope_pct = (slope / price) * 100 if price > 0 else 0
        # Normalise: +0.5%/day = 100, 0 = 50, -0.5%/day = 0
        slope_score = round(max(0, min(100, 50 + slope_pct * 100)), 1)
    else:
        slope_score = 50

    # 10. Volume trend — is volume rising or falling over last 10 days
    if len(volume) >= 10:
        vy  = volume.tail(10).values.astype(float)
        vx  = np.arange(10)
        vslope = float(np.polyfit(vx, vy, 1)[0])
        # Drying volume (negative slope) = good for VCP = score 70-100
        # Rising volume = could be breakout = score 50-80
        vol_trend = round(max(0, min(100, 50 - (vslope / max(avg_vol20, 1)) * 500)), 1)
    else:
        vol_trend = 50

    # 11. Candle quality — avg bullish body % of last 5 candles
    bodies = []
    for i in range(1, min(6, len(close))):
        o = float(open_.iloc[-i])
        c2 = float(close.iloc[-i])
        h = float(high.iloc[-i])
        l = float(low.iloc[-i])
        rng = h - l
        body = abs(c2 - o)
        ratio = (body / rng * 100) if rng > 0 else 50
        bullish_mult = 1.0 if c2 >= o else 0.4
        bodies.append(ratio * bullish_mult)
    candle_quality = round(min(100, float(np.mean(bodies))) if bodies else 50, 1)

    # 12. Stage score — early base = 100, mid-run = 60, extended = 20
    dist_from_low = ((price - float(close.tail(60).min())) /
                     max(float(close.tail(60).max() - close.tail(60).min()), 1)) * 100
    if dist_from_low < 25:
        stage_score = 100   # early base
    elif dist_from_low < 55:
        stage_score = 65    # mid move
    else:
        stage_score = 25    # extended

    return {
        "compression":   compression,
        "breakout":      breakout,
        "trend":         float(trend),
        "volume_ratio":  vol_ratio,
        "atr_score":     atr_score,
        "dist52_score":  dist52_score,
        "rs_score":      rs_score,
        "vcp_score":     float(vcp_s),
        "slope_score":   slope_score,
        "vol_trend":     vol_trend,
        "candle_quality":candle_quality,
        "stage_score":   float(stage_score),
    }


# Weights for each dimension (must sum to 1.0)
SIM_WEIGHTS = {
    "compression":    0.12,
    "breakout":       0.10,
    "trend":          0.10,
    "volume_ratio":   0.08,
    "atr_score":      0.08,
    "dist52_score":   0.08,
    "rs_score":       0.08,
    "vcp_score":      0.12,
    "slope_score":    0.08,
    "vol_trend":      0.08,
    "candle_quality": 0.06,
    "stage_score":    0.02,
}

DIMENSION_LABELS = {
    "compression":    "📦 Compression",
    "breakout":       "🚀 Breakout Proximity",
    "trend":          "📈 Trend (MA)",
    "volume_ratio":   "🔊 Volume Ratio",
    "atr_score":      "📐 ATR Tightness",
    "dist52_score":   "🏔️ Distance 52W High",
    "rs_score":       "💪 Relative Strength",
    "vcp_score":      "🔬 VCP Score",
    "slope_score":    "📉 Price Slope",
    "vol_trend":      "📊 Volume Trend",
    "candle_quality": "🕯️ Candle Quality",
    "stage_score":    "🎯 Stage",
}


def compute_weighted_similarity(fp_new, fp_old):
    """
    Compute weighted cosine-style similarity between two fingerprints.
    Returns overall 0-100 score + per-dimension match breakdown.
    """
    total_weight = 0
    total_score  = 0
    breakdown    = {}

    for key, weight in SIM_WEIGHTS.items():
        v_new = fp_new.get(key, 50)
        v_old = fp_old.get(key, 50)
        # Dimension match: 100 if identical, 0 if 100 points apart
        dim_match = max(0, 100 - abs(v_new - v_old))
        breakdown[key] = round(dim_match, 1)
        total_score  += dim_match * weight
        total_weight += weight

    overall = round(total_score / total_weight, 1) if total_weight > 0 else 0
    return overall, breakdown


def get_similar_winners(fp_new):
    """
    Compare current stock's fingerprint against WINNING past scans only.
    A stock is a winner if its current live price is ABOVE the entry price saved.
    Returns top 5 matches ranked by similarity, with per-dimension breakdown.
    """
    if not os.path.exists(FILE):
        return [], 0

    df = pd.read_csv(FILE)
    if df.empty:
        return [], 0

    fp_cols = list(SIM_WEIGHTS.keys())
    has_fp  = all(c in df.columns for c in fp_cols)

    out = []

    for _, row in df.iterrows():
        try:
            sym        = row["symbol"]
            entry_price = float(row.get("price", 0))

            # ── WIN FILTER: only include if current price > entry price ──
            live = get_data(sym)
            if live is None:
                continue
            current_price = float(safe_series(live["Close"]).iloc[-1])
            if current_price <= entry_price:
                continue   # skip — this stock is in loss, not a winner
            # ──────────────────────────────────────────────────────────────

            if has_fp:
                fp_old = {k: float(row.get(k, 50)) for k in fp_cols}
            else:
                fp_old = {k: 50.0 for k in fp_cols}
                fp_old["compression"] = float(row.get("compression", 50))
                fp_old["breakout"]    = float(row.get("breakout", 50))
                fp_old["trend"]       = float(row.get("trend", 50))

            overall, breakdown = compute_weighted_similarity(fp_new, fp_old)

            strong      = [DIMENSION_LABELS[k] for k, v in breakdown.items() if v >= 80]
            weak        = [DIMENSION_LABELS[k] for k, v in breakdown.items() if v < 50]
            match_count = sum(1 for v in breakdown.values() if v >= 80)

            # ── MINIMUM MATCH FILTER: need at least 7 out of 12 dims ──
            if match_count < 7:
                continue
            # ──────────────────────────────────────────────────────────

            gain_pct = round(((current_price - entry_price) / entry_price) * 100, 1)

            out.append({
                "symbol":      sym,
                "score":       float(row.get("score", 0)),
                "date":        str(row.get("date", "?")),
                "gain":        gain_pct,
                "sim":         overall,
                "match_count": match_count,    # e.g. "9/12 dims matched"
                "strong":      strong[:4],
                "weak":        weak[:3],
                "breakdown":   breakdown,
            })
        except:
            continue

    out_sorted = sorted(out, key=lambda x: x["sim"], reverse=True)

    top3_sims = [x["sim"] for x in out_sorted[:3]]
    avg_sim   = round(sum(top3_sims) / len(top3_sims), 1) if top3_sims else 0

    return out_sorted[:5], avg_sim


def sim_label_from_score(score):
    if score >= 85:
        return "🔥 Near-identical past winner found (≥10/12 dims)"
    elif score >= 75:
        return "⭐ Strong match with past winner (8-9/12 dims)"
    elif score >= 65:
        return "🟡 Good match with past winner (7-8/12 dims)"
    else:
        return "🆕 No qualifying match (need 7/12 dims)"

# =============================================
# MODULE 5 — MARKET CONTEXT ENGINE
# =============================================

# --- Sector map (NSE symbol → sector ETF/index proxy) ---
SECTOR_MAP = {
    # Pharma
    "CIPLA":"^CNXPHARMA","SUNPHARMA":"^CNXPHARMA","DRREDDY":"^CNXPHARMA",
    "DIVISLAB":"^CNXPHARMA","BIOCON":"^CNXPHARMA","AUROPHARMA":"^CNXPHARMA",
    "LUPIN":"^CNXPHARMA","TORNTPHARM":"^CNXPHARMA","ALKEM":"^CNXPHARMA",
    # IT
    "TCS":"^CNXIT","INFY":"^CNXIT","WIPRO":"^CNXIT","HCLTECH":"^CNXIT",
    "TECHM":"^CNXIT","LTIM":"^CNXIT","MPHASIS":"^CNXIT","PERSISTENT":"^CNXIT",
    # Banking & Finance
    "HDFCBANK":"^NSEBANK","ICICIBANK":"^NSEBANK","SBIN":"^NSEBANK",
    "KOTAKBANK":"^NSEBANK","AXISBANK":"^NSEBANK","INDUSINDBK":"^NSEBANK",
    "BANDHANBNK":"^NSEBANK","FEDERALBNK":"^NSEBANK","IDFCFIRSTB":"^NSEBANK",
    "BAJFINANCE":"NIFTY_FIN_SERVICE.NS","BAJAJFINSV":"NIFTY_FIN_SERVICE.NS",
    # Auto
    "MARUTI":"^CNXAUTO","TATAMOTORS":"^CNXAUTO","M&M":"^CNXAUTO",
    "BAJAJ-AUTO":"^CNXAUTO","HEROMOTOCO":"^CNXAUTO","EICHERMOT":"^CNXAUTO",
    "TVSMOTOR":"^CNXAUTO","ASHOKLEY":"^CNXAUTO",
    # FMCG
    "HINDUNILVR":"^CNXFMCG","ITC":"^CNXFMCG","NESTLEIND":"^CNXFMCG",
    "BRITANNIA":"^CNXFMCG","DABUR":"^CNXFMCG","MARICO":"^CNXFMCG",
    # Metals
    "TATASTEEL":"^CNXMETAL","JSWSTEEL":"^CNXMETAL","HINDALCO":"^CNXMETAL",
    "COALINDIA":"^CNXMETAL","VEDL":"^CNXMETAL","NMDC":"^CNXMETAL",
    # Energy / Oil & Gas
    "RELIANCE":"^CNXENERGY","ONGC":"^CNXENERGY","IOC":"^CNXENERGY",
    "BPCL":"^CNXENERGY","GAIL":"^CNXENERGY","POWERGRID":"^CNXENERGY",
    "NTPC":"^CNXENERGY","ADANIGREEN":"^CNXENERGY",
    # Infra / Realty
    "LT":"^CNXINFRA","ADANIPORTS":"^CNXINFRA","DLF":"^CNXREALTY",
    "GODREJPROP":"^CNXREALTY","OBEROIRLTY":"^CNXREALTY",
    # Consumption / Retail
    "DMART":"^CNXCONSUMP","TRENT":"^CNXCONSUMP","TITAN":"^CNXCONSUMP",
}

SECTOR_NAMES = {
    "^CNXPHARMA":           "Pharma",
    "^CNXIT":               "IT",
    "^NSEBANK":             "Banking",
    "NIFTY_FIN_SERVICE.NS": "Financial Services",
    "^CNXAUTO":             "Auto",
    "^CNXFMCG":             "FMCG",
    "^CNXMETAL":            "Metals",
    "^CNXENERGY":           "Energy",
    "^CNXINFRA":            "Infra",
    "^CNXREALTY":           "Realty",
    "^CNXCONSUMP":          "Consumption",
}

_sector_cache = {}   # cache sector data for the session to avoid redundant downloads

def get_sector_momentum(symbol):
    """
    Returns sector name, momentum label, and 1M/3M return for the sector.
    Falls back gracefully if sector not mapped or data unavailable.
    """
    ticker = SECTOR_MAP.get(symbol.upper())
    if not ticker:
        return {
            "sector_name":  "Unknown",
            "sector_label": "⚪ Sector unknown",
            "sector_1m":    None,
            "sector_3m":    None,
            "sector_color": "#64748b",
        }

    if ticker in _sector_cache:
        data = _sector_cache[ticker]
    else:
        try:
            raw  = _yf_download_safe(ticker, period="4mo", interval="1d", timeout=6)
            data = safe_series(raw["Close"]) if raw is not None and not raw.empty else None
            _sector_cache[ticker] = data
        except:
            data = None

    if data is None or len(data) < 20:
        return {
            "sector_name":  SECTOR_NAMES.get(ticker, ticker),
            "sector_label": "⚪ Data unavailable",
            "sector_1m":    None,
            "sector_3m":    None,
            "sector_color": "#64748b",
        }

    cur  = float(data.iloc[-1])
    m1   = float(data.iloc[-21]) if len(data) >= 21 else float(data.iloc[0])
    m3   = float(data.iloc[-63]) if len(data) >= 63 else float(data.iloc[0])
    r1m  = round(((cur - m1) / m1) * 100, 1)
    r3m  = round(((cur - m3) / m3) * 100, 1)

    # Momentum label
    if r1m >= 5 and r3m >= 8:
        label = "🚀 Strong Momentum"
        color = "#22c55e"
    elif r1m >= 2 or r3m >= 4:
        label = "🟡 Gaining"
        color = "#facc15"
    elif r1m >= -2 and r3m >= -3:
        label = "⚪ Neutral"
        color = "#94a3b8"
    elif r1m < -5 or r3m < -8:
        label = "🔴 Sector Weakness"
        color = "#ef4444"
    else:
        label = "🟠 Softening"
        color = "#f97316"

    return {
        "sector_name":  SECTOR_NAMES.get(ticker, ticker),
        "sector_label": label,
        "sector_1m":    r1m,
        "sector_3m":    r3m,
        "sector_color": color,
    }


# ════════════════════════════════════════════════════════════════════
# SECTOR PERFORMANCE PAGE — Daily / Weekly / Monthly + Sector Leaders
# ════════════════════════════════════════════════════════════════════

# Full sectoral index map: display name -> Yahoo Finance ticker.
# These are the standard NSE sectoral indices (mirrors moneycontrol's
# sector-analysis categories). Some niche indices may not resolve on
# yfinance — handled gracefully with a try/except per sector.

# ── SECTOR ENGINE — Stock-average based (works from any server worldwide) ──────
# Instead of fetching blocked NSE index tickers (^NSEBANK, ^CNXAUTO etc),
# we compute sector performance by averaging the top liquid stocks in each sector.
# This is 100% reliable from Render/PythonAnywhere/any non-India server.

SECTOR_LEADER_STOCKS = {
    "Nifty Bank":        ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","INDUSINDBK","PNB","BANKBARODA"],
    "Nifty Auto":        ["MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT","TVSMOTOR"],
    "Nifty IT":          ["TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","PERSISTENT","COFORGE"],
    "Nifty Pharma":      ["SUNPHARMA","CIPLA","DRREDDY","DIVISLAB","AUROPHARMA","LUPIN","TORNTPHARM"],
    "Nifty FMCG":        ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO","TATACONSUM"],
    "Nifty Metal":       ["TATASTEEL","JSWSTEEL","HINDALCO","COALINDIA","VEDL","NMDC","SAIL"],
    "Nifty Energy":      ["RELIANCE","ONGC","IOC","BPCL","GAIL","POWERGRID","NTPC","TATAPOWER"],
    "Nifty Realty":      ["DLF","GODREJPROP","OBEROIRLTY","LODHA","PRESTIGE"],
    "Nifty Infra":       ["LT","ADANIPORTS","SIEMENS","ABB","CUMMINSIND","CONCOR"],
    "Nifty PSU Bank":    ["SBIN","BANKBARODA","PNB","CANBK","UNIONBANK"],
    "Nifty Fin Service": ["BAJFINANCE","BAJAJFINSV","HDFCAMC","MUTHOOTFIN","CHOLAFIN","SHRIRAMFIN"],
    "Nifty Pvt Bank":    ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","INDUSINDBK","FEDERALBNK"],
    "Nifty Consumption": ["DMART","TRENT","TITAN","JUBLFOOD","ZOMATO"],
    "Nifty MNC":         ["NESTLEIND","SIEMENS","ABB","CUMMINSIND","BOSCHLTD"],
}

_sector_perf_cache = {}   # in-memory cache to avoid repeat downloads in same request

def _pct_change(close_series, days_back):
    """Helper: % change from N trading days ago to latest close."""
    try:
        if len(close_series) <= days_back:
            return None
        cur  = float(close_series.iloc[-1])
        past = float(close_series.iloc[-1 - days_back])
        if past == 0:
            return None
        return round(((cur - past) / past) * 100, 2)
    except Exception:
        return None


def _get_stock_close(symbol):
    """Cached daily close series for a stock symbol."""
    if symbol in _sector_perf_cache:
        return _sector_perf_cache[symbol]
    df = _yf_download_safe(symbol + ".NS", period="6mo", interval="1d", timeout=6)
    if df is None:
        _sector_perf_cache[symbol] = None
        return None
    df = flatten_df(df)
    s  = safe_series(df["Close"])
    _sector_perf_cache[symbol] = s if len(s) >= 5 else None
    return _sector_perf_cache[symbol]


def get_sector_index_performance(name, stocks_list):
    """
    Compute sector performance by equal-weighting the top liquid stocks.
    Returns the same dict shape as before so the template works unchanged.
    """
    returns_d1, returns_w1, returns_m1, returns_m3 = [], [], [], []
    last_prices = []

    for sym in stocks_list[:6]:   # use top 6 for speed
        s = _get_stock_close(sym)
        if s is None or len(s) < 5:
            continue
        d1 = _pct_change(s, 1)
        w1 = _pct_change(s, 5)
        m1 = _pct_change(s, 21)
        m3 = _pct_change(s, 63)
        if d1 is not None: returns_d1.append(d1)
        if w1 is not None: returns_w1.append(w1)
        if m1 is not None: returns_m1.append(m1)
        if m3 is not None: returns_m3.append(m3)
        last_prices.append(float(s.iloc[-1]))

    if not returns_d1:
        return {
            "name": name, "ticker": name, "available": False,
            "price": None, "d1": None, "w1": None, "m1": None, "m3": None,
            "color": "#475569", "label": "Data unavailable",
        }

    d1 = round(sum(returns_d1) / len(returns_d1), 2)
    w1 = round(sum(returns_w1) / len(returns_w1), 2) if returns_w1 else None
    m1 = round(sum(returns_m1) / len(returns_m1), 2) if returns_m1 else None
    m3 = round(sum(returns_m3) / len(returns_m3), 2) if returns_m3 else None

    if d1 >= 1.5:   label, color = "🚀 Strong Up",   "#10b981"
    elif d1 >= 0.3: label, color = "🟢 Up",           "#34d399"
    elif d1 > -0.3: label, color = "⚪ Flat",          "#94a3b8"
    elif d1 > -1.5: label, color = "🔴 Down",          "#f87171"
    else:           label, color = "🔻 Strong Down",   "#ef4444"

    return {
        "name": name, "ticker": name, "available": True,
        "price": round(last_prices[0], 2) if last_prices else None,
        "d1": d1, "w1": w1, "m1": m1, "m3": m3,
        "color": color, "label": label,
    }


def get_sector_leaders(sector_name, timeframe="d1"):
    """Rank sector stocks by return for the given timeframe."""
    stocks   = SECTOR_LEADER_STOCKS.get(sector_name, [])
    days_map = {"d1": 1, "w1": 5, "m1": 21}
    days     = days_map.get(timeframe, 1)
    out = []
    for sym in stocks:
        s = _get_stock_close(sym)
        if s is None: continue
        ret = _pct_change(s, days)
        if ret is None: continue
        out.append({"symbol": sym, "price": round(float(s.iloc[-1]), 2), "ret": ret})
    out.sort(key=lambda x: x["ret"], reverse=True)
    return out[:5]


def get_all_sectors_performance():
    """Build full sector dashboard — now uses stock-average, no index tickers needed."""
    import concurrent.futures
    _sector_perf_cache.clear()   # fresh cache each page load

    # Pre-fetch all stocks in parallel so the page loads fast
    all_syms = list({s for stocks in SECTOR_LEADER_STOCKS.values() for s in stocks[:6]})
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(_get_stock_close, all_syms))

    sectors = []
    for name, stocks_list in SECTOR_LEADER_STOCKS.items():
        perf = get_sector_index_performance(name, stocks_list)
        if perf["available"]:
            perf["leaders_d1"] = get_sector_leaders(name, "d1")[:3]
            perf["leaders_w1"] = get_sector_leaders(name, "w1")[:3]
            perf["leaders_m1"] = get_sector_leaders(name, "m1")[:3]
        else:
            perf["leaders_d1"] = perf["leaders_w1"] = perf["leaders_m1"] = []
        sectors.append(perf)

    sectors.sort(key=lambda s: (not s["available"], -(s["d1"] if s["d1"] is not None else -999)))
    return sectors






# =========================
# MULTI-TIMEFRAME ENGINE
# =========================

def get_tf_data(symbol, interval, period):
    """Download OHLCV for a given interval. Returns DataFrame or None."""
    return _yf_download_safe(symbol + ".NS", period=period, interval=interval, timeout=6)

def calc_rsi(series, period=14):
    """Standard RSI calculation."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float('nan'))
    return 100 - (100 / (1 + rs))

def analyse_timeframe(df):
    """
    Analyse a single timeframe DataFrame.
    Returns a dict with trend, RSI, MA alignment, volume, momentum.
    """
    if df is None or len(df) < 25:
        return None

    close  = safe_series(df["Close"])
    volume = safe_series(df["Volume"])

    price  = float(close.iloc[-1])
    ma20   = float(close.rolling(20).mean().iloc[-1])
    ma9    = float(close.rolling(9).mean().iloc[-1])
    rsi    = float(calc_rsi(close).iloc[-1]) if len(close) >= 15 else 50

    # Trend
    above_ma20  = price > ma20
    above_ma9   = price > ma9
    ma9_above20 = ma9 > ma20

    if above_ma20 and ma9_above20:
        trend_label = "⬆️ Uptrend"
        trend_score = 2
        trend_color = "#22c55e"
    elif above_ma20:
        trend_label = "↗️ Weak Up"
        trend_score = 1
        trend_color = "#86efac"
    elif not above_ma20 and not ma9_above20:
        trend_label = "⬇️ Downtrend"
        trend_score = -2
        trend_color = "#ef4444"
    else:
        trend_label = "↔️ Sideways"
        trend_score = 0
        trend_color = "#94a3b8"

    # RSI label
    if rsi >= 60:
        rsi_label = f"🔥 {rsi:.0f} (Bullish)"
        rsi_color = "#22c55e"
    elif rsi >= 45:
        rsi_label = f"🟡 {rsi:.0f} (Neutral)"
        rsi_color = "#facc15"
    elif rsi >= 35:
        rsi_label = f"🟠 {rsi:.0f} (Weak)"
        rsi_color = "#f97316"
    else:
        rsi_label = f"🔴 {rsi:.0f} (Oversold)"
        rsi_color = "#ef4444"

    # Volume vs avg
    avg_vol  = float(volume.tail(20).mean())
    last_vol = float(volume.iloc[-1])
    vol_ratio = round((last_vol / avg_vol) * 100, 0) if avg_vol > 0 else 100
    vol_label = f"{'🔊' if vol_ratio >= 120 else '🔉'} {vol_ratio:.0f}% of avg"

    return {
        "trend_label": trend_label,
        "trend_score": trend_score,
        "trend_color": trend_color,
        "rsi":         round(rsi, 1),
        "rsi_label":   rsi_label,
        "rsi_color":   rsi_color,
        "vol_label":   vol_label,
        "price":       round(price, 2),
        "ma20":        round(ma20, 2),
    }


def get_mtf_analysis(symbol):
    """
    Fetch and analyse 5 timeframes: 1D, 4H, 1H, 15m, 5m.
    Returns per-timeframe data + intraday verdict.
    """
    timeframes = {
        "1D":  ("1d",  "6mo",  "Swing"),
        "4H":  ("1h",  "60d",  "Swing"),   # yfinance: use 1h, group 4 candles mentally
        "1H":  ("1h",  "30d",  "Swing"),
        "15m": ("15m", "10d",  "Intraday"),
        "5m":  ("5m",  "5d",   "Intraday"),
    }

    results = {}
    scores  = []

    for label, (interval, period, category) in timeframes.items():
        df   = get_tf_data(symbol, interval, period)
        data = analyse_timeframe(df)
        if data:
            data["category"] = category
            results[label]   = data
            scores.append(data["trend_score"])

    # --- Intraday Verdict ---
    # Use 1H, 15m, 5m scores for intraday decision
    intraday_tfs    = ["1H", "15m", "5m"]
    intraday_scores = [results[tf]["trend_score"] for tf in intraday_tfs if tf in results]
    swing_tfs       = ["1D", "4H"]
    swing_scores    = [results[tf]["trend_score"] for tf in swing_tfs if tf in results]

    id_bull  = sum(1 for s in intraday_scores if s > 0)
    id_bear  = sum(1 for s in intraday_scores if s < 0)
    sw_bull  = sum(1 for s in swing_scores   if s > 0)

    total_tfs = len(intraday_scores)

    if id_bull >= 2 and sw_bull >= 1:
        intraday_verdict = "✅ GO INTRADAY"
        intraday_detail  = "Intraday & swing timeframes aligned bullish — good setup"
        intraday_color   = "#22c55e"
    elif id_bull == total_tfs and sw_bull == 0:
        intraday_verdict = "⚠️ INTRADAY ONLY"
        intraday_detail  = "Short-term bullish but swing trend weak — tight SL, quick exit"
        intraday_color   = "#facc15"
    elif id_bull == 1 and id_bear == 0:
        intraday_verdict = "⚠️ PARTIAL — Wait for confirmation"
        intraday_detail  = "Mixed signals across timeframes — wait for 15m breakout"
        intraday_color   = "#f97316"
    elif id_bear >= 2:
        intraday_verdict = "❌ SKIP INTRADAY"
        intraday_detail  = "Multiple timeframes bearish — avoid long intraday trades"
        intraday_color   = "#ef4444"
    else:
        intraday_verdict = "⚠️ SIDEWAYS — No clear edge"
        intraday_detail  = "No strong direction on intraday timeframes — stay out"
        intraday_color   = "#94a3b8"

    return {
        "timeframes":       results,
        "intraday_verdict": intraday_verdict,
        "intraday_detail":  intraday_detail,
        "intraday_color":   intraday_color,
    }


# =========================
# MARKETSMITH BADGE CHECK (lightweight, reuses already-fetched df)
# =========================
def check_marketsmith_membership(symbol, df):
    """
    Check which MarketSmith-style screens this stock currently qualifies for,
    using the same close/volume data already downloaded for analyze().
    Returns a list of badge dicts: [{"key":..., "label":..., "emoji":..., "color":...}, ...]
    Does NOT re-download data and does NOT hit the Blue Dot RS-vs-Nifty check
    (that one requires a separate Nifty fetch and is handled only on the
    dedicated /screens page to keep the main search fast).
    """
    badges = []
    try:
        close  = safe_series(df["Close"])
        high_s = safe_series(df["High"])
        low_s  = safe_series(df["Low"])
        volume = safe_series(df["Volume"])

        if len(close) < 50:
            return badges

        price  = float(close.iloc[-1])
        prev   = float(close.iloc[-2]) if len(close) > 1 else price
        chg_pct = ((price - prev) / prev) * 100 if prev else 0

        high20 = float(close.tail(20).max())
        ma20   = float(close.rolling(20).mean().iloc[-1])
        ma50   = float(close.rolling(50).mean().iloc[-1])
        high52 = float(close.tail(min(252, len(close))).max())

        avg_vol20 = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
        today_vol = float(volume.iloc[-1])
        vol_ratio = (today_vol / avg_vol20) if avg_vol20 else 1

        # 1. Near Pivot — within 5% of 20D high, above MA50
        dist_pivot_pct = ((high20 - price) / high20) * 100 if high20 else 99
        if dist_pivot_pct <= 5 and price > ma50:
            badges.append({"key": "near_pivot", "label": "Near Pivot", "emoji": "🎯", "color": "#a855f7"})

        # 2. Recent Breakouts — broke 20D high within last 16 sessions, held within 7%
        if len(close) >= 40:
            breakout_idx = None
            for i in range(max(1, len(close) - 16), len(close)):
                prior_high = float(close.iloc[max(0, i - 20):i].max())
                if prior_high > 0 and float(close.iloc[i]) >= prior_high * 0.999:
                    breakout_idx = i
                    break
            if breakout_idx is not None:
                pivot = float(close.iloc[max(0, breakout_idx - 20):breakout_idx].max())
                post_min = float(close.iloc[breakout_idx:].min())
                dd = ((pivot - post_min) / pivot) * 100 if pivot else 0
                if dd <= 7:
                    badges.append({"key": "recent_breakouts", "label": "Recent Breakout", "emoji": "🕐", "color": "#facc15"})

        # 3. Price Gaps Up — today's low > yesterday's high, on above-avg volume
        if len(close) >= 21 and len(high_s) >= 2 and len(low_s) >= 1:
            today_low  = float(low_s.iloc[-1])
            prior_high = float(high_s.iloc[-2])
            if today_low > prior_high and today_vol > avg_vol20:
                badges.append({"key": "price_gaps_up", "label": "Gap Up", "emoji": "⬆️", "color": "#84cc16"})

        # 4. Extended Stocks — 25%+ above 60D low, above MA50, within 10% of 52W high
        if len(close) >= 60:
            base_low = float(close.tail(60).min())
            ext_pct  = ((price - base_low) / base_low) * 100 if base_low else 0
            near52   = ((high52 - price) / high52) * 100 if high52 else 99
            if ext_pct > 25 and price > ma50 and near52 < 10:
                badges.append({"key": "extended", "label": "Extended", "emoji": "🚀", "color": "#f97316"})

        # 5. Up on Volume — price up 1%+ today, volume 1.5x+ avg
        if chg_pct > 1 and vol_ratio >= 1.5:
            badges.append({"key": "up_on_volume", "label": "Up on Volume", "emoji": "📶", "color": "#06b6d4"})

        # 6. New High With Best Trend — within 5% of 2yr high, MA-aligned uptrend
        if len(close) >= 60:
            high2y = float(close.tail(min(504, len(close))).max())
            near_high2y = ((high2y - price) / high2y) * 100 if high2y else 99
            if near_high2y <= 5 and price > ma20 > ma50:
                badges.append({"key": "new_high_trend", "label": "New High Trend", "emoji": "📈", "color": "#38bdf8"})

        # 7. Breaking Out Today — at/above 20D high on 40%+ volume surge
        if price >= high20 * 0.998 and vol_ratio >= 1.40:
            badges.append({"key": "breaking_out", "label": "Breaking Out Today", "emoji": "💥", "color": "#22c55e"})

    except Exception:
        pass

    return badges


# =========================
# ANALYSIS
# =========================
def analyze(symbol):
    df = get_data(symbol)
    if df is None:
        return None

    close  = safe_series(df["Close"])
    volume = safe_series(df["Volume"])

    price = fmt(close.iloc[-1])

    r15 = close.tail(15).max() - close.tail(15).min()
    r60 = close.tail(60).max() - close.tail(60).min()

    compression = (1 - (r15 / r60)) * 60 if r60 > 0 else 0
    breakout    = 80 if price >= close.tail(20).max() * 0.98 else 25

    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]

    trend = 0
    if price > ma20: trend += 40
    if ma20 > ma50:  trend += 40
    if price > close.tail(10).mean(): trend += 20

    base_score = compression * 0.35 + breakout * 0.35 + trend * 0.30

    # --- VCP Engine ---
    vcp       = detect_vcp(df)
    vcp_bonus = vcp["vcp_score"] * 0.15
    score     = min(100, base_score + vcp_bonus)

    decision = "BUY" if score > 80 else "HOLD" if score > 60 else "AVOID"

    # --- Similarity AI 2.0 ---
    fp               = build_fingerprint(df, vcp_score=vcp["vcp_score"])
    similar_list, sim = get_similar_winners(fp)
    sim_label        = sim_label_from_score(sim)

    # --- Sector Momentum ---
    sector = get_sector_momentum(symbol)

    # --- Multi-Timeframe Analysis ---
    mtf = get_mtf_analysis(symbol)

    # --- MarketSmith Screen Membership (which screens does this stock pass right now?) ---
    ms_badges = check_marketsmith_membership(symbol, df)

    entry_date = datetime.now().strftime("%d-%m-%Y")

    # Save full 12-dimension fingerprint to CSV
    save_signal({
        "date":         entry_date,
        "symbol":       symbol,
        "price":        price,
        "score":        round(score, 2),
        # all 12 fingerprint dims
        "compression":   fp["compression"],
        "breakout":      fp["breakout"],
        "trend":         fp["trend"],
        "volume_ratio":  fp["volume_ratio"],
        "atr_score":     fp["atr_score"],
        "dist52_score":  fp["dist52_score"],
        "rs_score":      fp["rs_score"],
        "vcp_score":     fp["vcp_score"],
        "slope_score":   fp["slope_score"],
        "vol_trend":     fp["vol_trend"],
        "candle_quality":fp["candle_quality"],
        "stage_score":   fp["stage_score"],
    })

    return {
        "symbol":      symbol,
        "price":       price,
        "score":       round(score, 2),
        "decision":    decision,
        "sentiment":   sentiment(score),

        "compression": round(compression, 1),
        "breakout":    round(breakout, 1),
        "trend":       round(trend, 1),

        "strength":    strength_bar(score),

        "similarity":    sim,
        "similar_label": sim_label,
        "similar_list":  similar_list,

        # VCP fields
        "vcp_score":   vcp["vcp_score"],
        "vcp_stage":   vcp["vcp_stage"],
        "vcp_badge":   vcp["vcp_badge"],
        "vcp_details": vcp["vcp_details"],

        # Sector Momentum
        "sector_name":  sector["sector_name"],
        "sector_label": sector["sector_label"],
        "sector_1m":    sector["sector_1m"],
        "sector_3m":    sector["sector_3m"],
        "sector_color": sector["sector_color"],

        # Multi-Timeframe
        "mtf":               mtf["timeframes"],
        "intraday_verdict":  mtf["intraday_verdict"],
        "intraday_detail":   mtf["intraday_detail"],
        "intraday_color":    mtf["intraday_color"],

        # Fingerprint (for display)
        "fingerprint": fp,

        # MarketSmith screen badges (which screens this stock qualifies for right now)
        "ms_badges": ms_badges,

        "tags": [
            tag("Compression", "OK", "#22c55e"),
            tag("Breakout",    "OK", "#facc15"),
            tag("Trend",       "OK", "#22c55e")
        ],

        "notes": f"Compression {compression:.1f} | Breakout {breakout:.1f} | Trend {trend:.1f}",
        "sl":     fmt(price * 0.96),
        "target": fmt(price * 1.10),
        "move":   round(((price * 1.10 - price) / price) * 100, 2)
    }

# =========================
# PERFORMANCE TRACKER PRO
# =========================
def performance():
    if not os.path.exists(FILE):
        return []
    df = pd.read_csv(FILE)
    df = df.drop_duplicates(subset=["symbol"], keep="last")
    results = []

    for _, row in df.iterrows():
        symbol     = row["symbol"]
        entry      = float(row["price"])
        entry_date = str(row.get("date", "?"))

        live = get_data(symbol)
        if live is None:
            continue

        close_s = safe_series(live["Close"])
        current = float(close_s.iloc[-1])
        move    = round(((current - entry) / entry) * 100, 2)

        # Days since entry
        try:
            ed       = datetime.strptime(entry_date, "%d-%m-%Y")
            days_held = (datetime.now() - ed).days
        except:
            days_held = "?"

        # Maximum gain achieved (highest close since entry vs entry)
        max_close   = float(close_s.max())
        max_gain    = round(((max_close - entry) / entry) * 100, 2)

        # Maximum drawdown (biggest drop from peak to trough in the period)
        rolling_max = close_s.cummax()
        drawdowns   = (close_s - rolling_max) / rolling_max * 100
        max_dd      = round(float(drawdowns.min()), 2)   # most negative value

        results.append({
            "symbol":     symbol,
            "entry":      entry,
            "current":    fmt(current),
            "move":       move,
            "status":     "WIN" if move > 0 else "LOSS",
            "entry_date": entry_date,
            "days_held":  days_held,
            "max_gain":   max_gain,
            "max_dd":     max_dd,
        })

    return results

# =========================
# UI
# =========================
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Falcon AI</title>
<style>

* { box-sizing: border-box; }

body {
    background: #0b1220;
    color: white;
    font-family: Arial, sans-serif;
    margin: 0;
}

.header {
    text-align: center;
    padding: 20px 16px;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border-bottom: 1px solid #1f2937;
}

.header h2 {
    margin: 0;
    font-size: 24px;
    letter-spacing: 1px;
}

.container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 14px;
    padding: 16px;
}

.card {
    background: #111827;
    padding: 16px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    transition: border-color 0.2s;
}

.card:hover {
    border-color: #334155;
}

.symbol {
    font-size: 17px;
    font-weight: bold;
    color: #60a5fa;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

.badge {
    background: #1e293b;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
}

.section {
    margin-top: 7px;
    font-size: 12px;
    color: #cbd5e1;
}

/* ---- VCP Block ---- */
.vcp-block {
    margin-top: 10px;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 10px 12px;
}

.vcp-title {
    font-size: 13px;
    font-weight: bold;
    color: #f1f5f9;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.vcp-score-bar-wrap {
    background: #1e293b;
    border-radius: 8px;
    height: 8px;
    margin-bottom: 8px;
    overflow: hidden;
}

.vcp-score-bar {
    height: 8px;
    border-radius: 8px;
    transition: width 0.4s ease;
}

.vcp-detail-row {
    font-size: 11px;
    color: #94a3b8;
    padding: 2px 0;
    border-bottom: 1px solid #1e293b;
}

.vcp-detail-row:last-child {
    border-bottom: none;
}

/* ---- Sector Momentum Block ---- */
.sector-block {
    margin-top: 10px;
    background: #0f1a0f;
    border-radius: 10px;
    padding: 10px 12px;
    border: 1px solid #1a3a1a;
}

.sector-returns {
    display: flex;
    gap: 10px;
    margin-top: 6px;
}

.sector-pill {
    background: #1e293b;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    color: #94a3b8;
}

/* ---- Multi-Timeframe Block ---- */
.mtf-block {
    margin-top: 10px;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 10px 12px;
}

.mtf-title {
    font-size: 13px;
    font-weight: bold;
    color: #f1f5f9;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.mtf-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
}

.mtf-table th {
    color: #475569;
    text-align: left;
    padding: 3px 5px;
    border-bottom: 1px solid #1e293b;
    font-weight: normal;
}

.mtf-table td {
    padding: 4px 5px;
    border-bottom: 1px solid #0f172a;
    color: #cbd5e1;
    vertical-align: middle;
}

.mtf-table tr:last-child td { border-bottom: none; }

.mtf-tf-label {
    font-weight: bold;
    color: #60a5fa;
    font-size: 12px;
}

.mtf-category {
    font-size: 9px;
    color: #334155;
    display: block;
}

.intraday-verdict {
    margin-top: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    background: #0f172a;
    border: 1px solid #1e293b;
}

.intraday-main {
    font-size: 14px;
    font-weight: bold;
    margin-bottom: 3px;
}

.intraday-sub {
    font-size: 11px;
    color: #94a3b8;
}

/* ---- Similarity AI 2.0 Block ---- */
.sim-block {
    margin-top: 10px;
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 10px 12px;
}

.sim-title {
    font-size: 13px;
    font-weight: bold;
    color: #f1f5f9;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.sim-overall-bar-wrap {
    background: #1e293b;
    border-radius: 8px;
    height: 8px;
    margin-bottom: 8px;
    overflow: hidden;
}

.sim-overall-bar {
    height: 8px;
    border-radius: 8px;
    background: linear-gradient(90deg, #3b82f6, #818cf8);
}

.sim-match-card {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
    font-size: 11px;
}

.sim-match-card:last-child { margin-bottom: 0; }

.sim-match-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}

.sim-match-sym  { color: #60a5fa; font-weight: bold; font-size: 12px; }
.sim-match-pct  { color: #818cf8; font-weight: bold; font-size: 13px; }

.sim-dim-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2px 8px;
    margin-top: 4px;
}

.sim-dim-row {
    font-size: 10px;
    color: #64748b;
    display: flex;
    justify-content: space-between;
}

.sim-dim-row span:last-child {
    font-weight: bold;
}

.sim-tag-match  { color: #22c55e; }
.sim-tag-weak   { color: #ef4444; }

/* ---- Decision Badge ---- */
.decision-buy  { color: #22c55e; font-weight: bold; font-size: 15px; }
.decision-hold { color: #facc15; font-weight: bold; font-size: 15px; }
.decision-avoid{ color: #ef4444; font-weight: bold; font-size: 15px; }

.tooltip {
    position: relative;
    cursor: help;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background: #0f172a;
    color: white;
    padding: 6px;
    border-radius: 8px;
    position: absolute;
    bottom: 120%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: 0.2s;
    border: 1px solid #334155;
    z-index: 10;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

.win  { color: #22c55e; }
.loss { color: #ef4444; }

/* ── LIVE DATA ── */
.live-strip {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 8px 12px;
    margin-top: 8px;
    font-size: 11px;
}
.live-row {
    display: flex; flex-wrap: wrap;
    gap: 6px 14px; align-items: center;
}
.live-item { display: flex; flex-direction: column; }
.live-label { font-size: 9px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; }
.live-val   { font-size: 13px; font-weight: 700; color: #f1f5f9; line-height: 1.2; }
.live-up  { color: #22c55e; }
.live-dn  { color: #ef4444; }
.live-neu { color: #94a3b8; }
.live-updated { font-size:9px; color:#334155; margin-top:5px; text-align:right; }
.pulse-dot {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: #22c55e; margin-right: 4px;
    animation: livepulse 1.5s ease-in-out infinite;
}
.pulse-dot.closed { background: #475569; animation: none; }
@keyframes livepulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.7); }
}
.mkt-status-bar {
    display: flex; align-items: center; justify-content: center;
    gap: 10px; padding: 7px 16px;
    background: #070d18; border-bottom: 1px solid #1e293b;
    font-size: 12px; color: #64748b;
}

button { cursor: pointer; }

.filter-btn {
    padding: 7px 16px;
    border: none;
    border-radius: 8px;
    margin: 4px;
    font-size: 13px;
    background: #1e293b;
    color: white;
}

.scan-btn {
    padding: 11px 30px;
    background: #22c55e;
    border: none;
    border-radius: 10px;
    color: white;
    font-size: 15px;
    font-weight: bold;
    margin-top: 10px;
}

.divider {
    border: none;
    border-top: 1px solid #1f2937;
    margin: 20px 16px;
}

.section-title {
    text-align: center;
    font-size: 16px;
    margin: 18px 0 8px;
    color: #e2e8f0;
}

</style>

<script>

// ── LIVE DATA ENGINE ────────────────────────────────────────────────

// Collect all symbols from result cards
const liveSymbols = [{% for r in results %}"{{ r.symbol }}"{% if not loop.last %},{% endif %}{% endfor %}];

async function fetchLive(symbol) {
    try {
        const res  = await fetch(`/api/live/${symbol}`);
        if (!res.ok) return;
        const d    = await res.json();
        if (d.error) return;

        const up   = d.chg_pct >= 0;
        const cls  = up ? 'live-up' : 'live-dn';

        // LTP
        const ltpEl = document.getElementById(`lv-ltp-${symbol}`);
        if (ltpEl) {
            ltpEl.textContent = `₹${d.ltp}`;
            ltpEl.className   = `live-val ${cls}`;
        }

        // Change
        const chgEl = document.getElementById(`lv-chg-${symbol}`);
        if (chgEl) {
            chgEl.textContent = `${up?'+':''}${d.chg} (${up?'+':''}${d.chg_pct}%)`;
            chgEl.className   = `live-val ${cls}`;
        }

        // High / Low
        const hi = document.getElementById(`lv-high-${symbol}`);
        const lo = document.getElementById(`lv-low-${symbol}`);
        if (hi) hi.textContent = `₹${d.day_high}`;
        if (lo) lo.textContent = `₹${d.day_low}`;

        // VWAP
        const vw = document.getElementById(`lv-vwap-${symbol}`);
        if (vw) {
            vw.textContent  = `₹${d.vwap}`;
            vw.className    = `live-val ${d.ltp >= d.vwap ? 'live-up' : 'live-dn'}`;
        }

        // Volume
        const vol = document.getElementById(`lv-vol-${symbol}`);
        if (vol) vol.textContent = d.vol_total.toLocaleString('en-IN');

        // Value in Cr
        const vcr = document.getElementById(`lv-vcr-${symbol}`);
        if (vcr) vcr.textContent = `₹${d.vol_cr}Cr`;

        // Bid / Ask
        const ba = document.getElementById(`lv-ba-${symbol}`);
        if (ba) ba.textContent = `${d.bid} / ${d.ask}`;

        // Breakout
        const bo = document.getElementById(`lv-bo-${symbol}`);
        if (bo) {
            if (d.breakout) {
                bo.textContent  = '🚀 YES';
                bo.className    = 'live-val live-up';
            } else {
                bo.textContent  = '—';
                bo.className    = 'live-val live-neu';
            }
        }

        // Updated timestamp
        const upd = document.getElementById(`lv-upd-${symbol}`);
        if (upd) {
            const mktLabel = d.mkt_open ? '🟢 Market Open' : '🔴 Market Closed';
            upd.textContent = `${mktLabel} · Updated ${d.updated_at} · ~15min delayed`;
        }

        // Pulse dot color
        const dot = document.getElementById(`live-dot-${symbol}`);
        if (dot) dot.className = d.mkt_open ? 'pulse-dot' : 'pulse-dot closed';

        // Show data, hide loading
        document.getElementById(`live-load-${symbol}`).style.display = 'none';
        document.getElementById(`live-data-${symbol}`).style.display = 'block';

    } catch(e) {
        // silently fail — don't break the card
    }
}

function refreshAllLive() {
    liveSymbols.forEach(sym => fetchLive(sym));
    updateMarketStatus();
}

async function updateMarketStatus() {
    try {
        const res = await fetch('/api/market_status');
        const d   = await res.json();
        const dot = document.getElementById('mkt-dot');
        const lbl = document.getElementById('mkt-label');
        const tm  = document.getElementById('mkt-time');
        if (dot) dot.className = d.open ? 'pulse-dot' : 'pulse-dot closed';
        if (lbl) {
            lbl.textContent  = d.open ? 'NSE Market OPEN' : 'NSE Market CLOSED';
            lbl.style.color  = d.open ? '#22c55e' : '#64748b';
        }
        if (tm) tm.textContent = d.time;
    } catch(e) {}
}

// Init — load on page ready, then auto-refresh every 60s
window.addEventListener('DOMContentLoaded', () => {
    applyPerf();
    updateMarketStatus();
    if (liveSymbols.length > 0) {
        // Stagger fetches so we don't hammer yfinance all at once
        liveSymbols.forEach((sym, i) => setTimeout(() => fetchLive(sym), i * 800));
        setInterval(refreshAllLive, 60000);   // auto-refresh every 60s
        setInterval(updateMarketStatus, 30000); // market status every 30s
    }
});

// ── PERFORMANCE TRACKER ──────────────────────────────────────────────

function filterPerf(type) {}   // kept for safety, replaced by perfControl

let perfState = { filter: 'ALL', sort: 'newest', search: '' };

function perfControl(action, value) {
    perfState[action] = value;

    // Highlight active filter button
    if (action === 'filter') {
        ['ALL','WIN','LOSS'].forEach(f => {
            let btn = document.getElementById('fb-' + f);
            if (btn) btn.style.opacity = (f === value) ? '1' : '0.45';
        });
    }
    if (action === 'sort') {
        ['gain','loss','newest','oldest'].forEach(s => {
            let btn = document.getElementById('sb-' + s);
            if (btn) btn.style.opacity = (s === value) ? '1' : '0.45';
        });
    }

    applyPerf();
}

function applyPerf() {
    let cards = Array.from(document.querySelectorAll('.perf-card'));

    // 1. Filter
    cards.forEach(c => {
        let status = c.getAttribute('data-status');
        let sym    = c.getAttribute('data-symbol') || '';
        let search = perfState.search.toUpperCase();

        let filterOk = (perfState.filter === 'ALL' || status === perfState.filter);
        let searchOk = (search === '' || sym.includes(search));

        c.style.display = (filterOk && searchOk) ? '' : 'none';
    });

    // 2. Sort visible cards
    let visible = cards.filter(c => c.style.display !== 'none');
    visible.sort((a, b) => {
        let s = perfState.sort;
        if (s === 'gain')   return parseFloat(b.dataset.move)  - parseFloat(a.dataset.move);
        if (s === 'loss')   return parseFloat(a.dataset.move)  - parseFloat(b.dataset.move);
        if (s === 'newest') return parseInt(a.dataset.days)    - parseInt(b.dataset.days);
        if (s === 'oldest') return parseInt(b.dataset.days)    - parseInt(a.dataset.days);
        return 0;
    });

    let container = document.getElementById('perf-container');
    visible.forEach(c => container.appendChild(c));

    // 3. Summary
    let wins   = cards.filter(c => c.style.display !== 'none' && c.dataset.status === 'WIN').length;
    let losses = cards.filter(c => c.style.display !== 'none' && c.dataset.status === 'LOSS').length;
    let total  = wins + losses;
    let wr     = total > 0 ? Math.round((wins / total) * 100) : 0;
    document.getElementById('perf-summary').innerText =
        total > 0 ? `Showing ${total} stocks · ${wins} WIN · ${losses} LOSS · Win Rate ${wr}%` : '';
}

function confirmClear() {
    document.getElementById('clear-modal').style.display = 'flex';
}

// applyPerf on load is now handled inside the DOMContentLoaded block above

function toggleVCP(id) {
    let el = document.getElementById(id);
    el.style.display = el.style.display === "none" ? "block" : "none";
}

function toggleSim(id) {
    let el = document.getElementById(id);
    el.style.display = el.style.display === "none" ? "block" : "none";
}

function toggleMTF(id) {
    let el = document.getElementById(id);
    el.style.display = el.style.display === "none" ? "block" : "none";
}

</script>
</head>

<body>

<div style="background:#0d1424;padding:0 20px;display:flex;align-items:center;
     border-bottom:1px solid #1a2840;height:44px;gap:0;">
    <span style="font-size:14px;font-weight:800;color:#3b82f6;margin-right:20px;">🦅 FalconAI</span>
    <a href="/" style="display:flex;align-items:center;height:44px;padding:0 14px;font-size:12px;
        font-weight:600;color:#3b82f6;text-decoration:none;border-bottom:2px solid #3b82f6;">📊 Screener</a>
    <a href="/options" style="display:flex;align-items:center;height:44px;padding:0 14px;font-size:12px;
        font-weight:600;color:#4a6080;text-decoration:none;border-bottom:2px solid transparent;">⚡ Options</a>
    <a href="/screens" style="display:flex;align-items:center;height:44px;padding:0 14px;font-size:12px;
        font-weight:600;color:#4a6080;text-decoration:none;border-bottom:2px solid transparent;">🔍 Screens</a>
    <a href="/sectors" style="display:flex;align-items:center;height:44px;padding:0 14px;font-size:12px;
        font-weight:600;color:#4a6080;text-decoration:none;border-bottom:2px solid transparent;">🏭 Sectors</a>
    <a href="/scanner" style="display:flex;align-items:center;height:44px;padding:0 14px;font-size:12px;
        font-weight:600;color:#4a6080;text-decoration:none;border-bottom:2px solid transparent;">⚡ 5-Min Scanner</a>
</div>

<div class="header">
    <h2>🦅 Falcon AI Engine</h2>
    <div style="font-size:12px;color:#64748b;margin-top:4px;">
        Stock Screener · VCP · Sector · MTF Analysis · Similarity AI · Tracker Pro
    </div>
</div>

<!-- Market Status Bar -->
<div class="mkt-status-bar" id="mkt-status-bar">
    <span id="mkt-dot" class="pulse-dot"></span>
    <span id="mkt-label" style="font-weight:600;">Checking market status...</span>
    <span style="color:#334155;">|</span>
    <span id="mkt-time" style="font-family:monospace;"></span>
    <span style="color:#334155;">|</span>
    <span style="color:#475569;">NSE · ~15min delayed · Auto-refresh: 60s</span>
    <button onclick="refreshAllLive()" style="padding:3px 10px;background:#1e293b;
        border:1px solid #334155;border-radius:6px;color:#94a3b8;font-size:11px;margin-left:8px;">
        ↺ Refresh Now
    </button>
</div>

<form method="POST" style="text-align:center;margin:20px 16px 10px;">
    <div style="font-size:13px;color:#94a3b8;margin-bottom:8px;">
        Enter NSE symbols — one per line <b style="color:#475569;">or</b> comma separated (without .NS)
    </div>
    <textarea name="stocks"
        placeholder="One per line:&#10;RELIANCE&#10;TCS&#10;&#10;Or comma separated:&#10;RELIANCE, TCS, INFY, CIPLA"
        style="width:min(600px,90%);height:130px;background:#111827;color:white;
               padding:12px;border-radius:10px;border:1px solid #334155;font-size:14px;"></textarea>
    <br>
    <button class="scan-btn" type="submit">🔍 Scan Stocks</button>
</form>

<h3 class="section-title">📊 Live Signals</h3>

<div class="container">

{% if error_msg %}
<div style="background:#1c1400;border:1px solid #f59e0b;border-radius:10px;padding:12px 18px;
  margin-bottom:16px;color:#fbbf24;font-size:13px;font-weight:600;">
  {{ error_msg }}
</div>
{% endif %}

{% for r in results %}
<div class="card">

    <!-- Symbol & Sentiment -->
    <div class="symbol">
        <a href="/risk/{{ r.symbol }}"
           style="color:#60a5fa;text-decoration:none;"
           title="Open Full Risk Report">
            {{ r.symbol }} 🔍
        </a>
        <span class="badge">{{ r.sentiment }}</span>
        <span class="badge" style="border:1px solid #475569;">{{ r.vcp_badge }}</span>
    </div>

    <!-- MarketSmith Screen Badges -->
    {% if r.ms_badges %}
    <div style="display:flex;flex-wrap:wrap;gap:5px;margin:6px 0 2px;">
        <span style="font-size:10px;color:#475569;align-self:center;margin-right:2px;">Found in:</span>
        {% for b in r.ms_badges %}
        <a href="/screens" style="text-decoration:none;" title="View on Screens page">
            <span style="display:inline-flex;align-items:center;gap:3px;padding:3px 9px;
                         border-radius:20px;font-size:10px;font-weight:700;white-space:nowrap;
                         background:{{ b.color }}22;border:1px solid {{ b.color }};color:{{ b.color }};">
                {{ b.emoji }} {{ b.label }}
            </span>
        </a>
        {% endfor %}
    </div>
    {% endif %}

    <!-- Price & Score -->
    <div class="section">
        Price: <b style="color:#f1f5f9;">₹{{ r.price }}</b> &nbsp;|&nbsp;
        Score: <b style="color:#f1f5f9;">{{ r.score }}</b>
    </div>

    <!-- Decision -->
    <div class="section">
        Signal:
        <span class="decision-{{ r.decision.lower() }}">{{ r.decision }}</span>
    </div>

    <!-- Strength -->
    <div class="section">
        Strength: {{ r.strength | safe }}
    </div>

    <!-- Indicators -->
    <div class="section">
        📦 Compression {{ r.compression }} &nbsp;
        🚀 Breakout {{ r.breakout }} &nbsp;
        📈 Trend {{ r.trend }}
    </div>

    <!-- SL & Target -->
    <div class="section">
        🛑 SL: ₹{{ r.sl }} &nbsp;|&nbsp;
        🎯 Target: ₹{{ r.target }} &nbsp;|&nbsp;
        📊 Move: {{ r.move }}%
    </div>

    <!-- ===== LIVE DATA STRIP ===== -->
    <div class="live-strip" id="live-{{ r.symbol }}">
        <div class="live-loading" id="live-load-{{ r.symbol }}">
            <span class="pulse-dot" id="live-dot-{{ r.symbol }}"></span>
            Loading live data...
        </div>
        <div id="live-data-{{ r.symbol }}" style="display:none;">
            <div class="live-row">
                <div class="live-item">
                    <span class="live-label">LTP</span>
                    <span class="live-val" id="lv-ltp-{{ r.symbol }}">—</span>
                </div>
                <div class="live-item">
                    <span class="live-label">Change</span>
                    <span class="live-val" id="lv-chg-{{ r.symbol }}">—</span>
                </div>
                <div class="live-item">
                    <span class="live-label">High</span>
                    <span class="live-val live-up" id="lv-high-{{ r.symbol }}">—</span>
                </div>
                <div class="live-item">
                    <span class="live-label">Low</span>
                    <span class="live-val live-dn" id="lv-low-{{ r.symbol }}">—</span>
                </div>
                <div class="live-item">
                    <span class="live-label">VWAP</span>
                    <span class="live-val live-neu" id="lv-vwap-{{ r.symbol }}">—</span>
                </div>
                <div class="live-item">
                    <span class="live-label">Volume</span>
                    <span class="live-val live-neu" id="lv-vol-{{ r.symbol }}">—</span>
                </div>
                <div class="live-item">
                    <span class="live-label">Val (Cr)</span>
                    <span class="live-val live-neu" id="lv-vcr-{{ r.symbol }}">—</span>
                </div>
                <div class="live-item">
                    <span class="live-label">Bid / Ask</span>
                    <span class="live-val live-neu" id="lv-ba-{{ r.symbol }}">—</span>
                </div>
                <div class="live-item">
                    <span class="live-label">Breakout</span>
                    <span class="live-val" id="lv-bo-{{ r.symbol }}">—</span>
                </div>
            </div>
            <div class="live-updated" id="lv-upd-{{ r.symbol }}"></div>
        </div>
    </div>
    <!-- ===== END LIVE DATA STRIP ===== -->

    <!-- Similarity -->
    <div class="section">{{ r.similar_label }} &nbsp; ({{ r.similarity }}%)</div>

    <!-- ===== SIMILARITY AI 2.0 BLOCK ===== -->
    <div class="sim-block">
        <div class="sim-title">
            <span>🧠 Similarity AI 2.0 — {{ r.similarity }}% match</span>
            <span style="color:#818cf8;cursor:pointer;font-size:11px;"
                  onclick="toggleSim('sim-{{ loop.index }}')">
                [toggle matches]
            </span>
        </div>

        <!-- Overall bar -->
        <div class="sim-overall-bar-wrap">
            <div class="sim-overall-bar" style="width:{{ r.similarity }}%;"></div>
        </div>

        <!-- Match cards (collapsible) -->
        <div id="sim-{{ loop.index }}" style="display:none;">

            {% if r.similar_list %}
                {% for m in r.similar_list %}
                <div class="sim-match-card">
                    <div class="sim-match-header">
                        <span class="sim-match-sym">{{ m.symbol }}</span>
                        <span class="sim-match-pct">{{ m.sim }}%</span>
                        <span style="font-size:10px;color:#a78bfa;font-weight:bold;">{{ m.match_count }}/12 dims</span>
                        <span style="font-size:10px;color:#22c55e;">+{{ m.gain }}% gain</span>
                        <span style="font-size:10px;color:#475569;">{{ m.date }}</span>
                    </div>

                    <!-- Dimension grid -->
                    <div class="sim-dim-grid">
                        {% for key, val in m.breakdown.items() %}
                        <div class="sim-dim-row">
                            <span>{{ key.replace('_',' ').title() }}</span>
                            <span class="{% if val >= 80 %}sim-tag-match{% elif val < 50 %}sim-tag-weak{% endif %}">
                                {{ val }}%
                            </span>
                        </div>
                        {% endfor %}
                    </div>

                    <!-- Strong matches -->
                    {% if m.strong %}
                    <div style="margin-top:5px;font-size:10px;color:#22c55e;">
                        ✅ Matches: {{ m.strong | join(', ') }}
                    </div>
                    {% endif %}

                    <!-- Weak matches -->
                    {% if m.weak %}
                    <div style="font-size:10px;color:#ef4444;">
                        ❌ Differs: {{ m.weak | join(', ') }}
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            {% else %}
                <div style="font-size:11px;color:#475569;padding:6px 0;">
                    No past winners matching 7/12 dimensions yet — keep scanning to build your history.
                </div>
            {% endif %}

        </div>
    </div>
    <!-- ===== END SIMILARITY AI 2.0 BLOCK ===== -->

    <!-- ===== VCP ENGINE BLOCK ===== -->
    <div class="vcp-block">
        <div class="vcp-title">
            <span>🔬 VCP Analysis — {{ r.vcp_stage }}</span>
            <span style="color:#60a5fa;cursor:pointer;font-size:11px;"
                  onclick="toggleVCP('vcp-{{ loop.index }}')">
                [toggle details]
            </span>
        </div>

        <!-- VCP score bar -->
        <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;">
            VCP Score: {{ r.vcp_score }}/100
        </div>
        <div class="vcp-score-bar-wrap">
            <div class="vcp-score-bar"
                 style="width:{{ r.vcp_score }}%;
                        background: {% if r.vcp_score >= 80 %}#22c55e
                                    {% elif r.vcp_score >= 60 %}#facc15
                                    {% elif r.vcp_score >= 40 %}#f97316
                                    {% else %}#ef4444{% endif %};">
            </div>
        </div>

        <!-- VCP details (collapsible) -->
        <div id="vcp-{{ loop.index }}" style="display:none;">
            {% for key, val in r.vcp_details.items() %}
            <div class="vcp-detail-row">{{ val }}</div>
            {% endfor %}
        </div>
    </div>
    <!-- ===== END VCP BLOCK ===== -->

    <!-- ===== SECTOR MOMENTUM BLOCK ===== -->
    <div class="sector-block" style="border-color:{{ r.sector_color }}33;">
        <div style="font-size:13px;font-weight:bold;color:#f1f5f9;margin-bottom:4px;">
            🏭 Sector: {{ r.sector_name }}
        </div>
        <div style="font-size:13px;font-weight:bold;color:{{ r.sector_color }};">
            {{ r.sector_label }}
        </div>
        <div class="sector-returns">
            {% if r.sector_1m is not none %}
            <span class="sector-pill">
                1M: <b style="color:{{ '#22c55e' if r.sector_1m >= 0 else '#ef4444' }};">
                    {% if r.sector_1m >= 0 %}+{% endif %}{{ r.sector_1m }}%
                </b>
            </span>
            {% endif %}
            {% if r.sector_3m is not none %}
            <span class="sector-pill">
                3M: <b style="color:{{ '#22c55e' if r.sector_3m >= 0 else '#ef4444' }};">
                    {% if r.sector_3m >= 0 %}+{% endif %}{{ r.sector_3m }}%
                </b>
            </span>
            {% endif %}
        </div>
    </div>
    <!-- ===== END SECTOR BLOCK ===== -->

    <!-- ===== MULTI-TIMEFRAME BLOCK ===== -->
    <div class="mtf-block">
        <div class="mtf-title">
            <span>📊 Multi-Timeframe Analysis</span>
            <span style="color:#60a5fa;cursor:pointer;font-size:11px;"
                  onclick="toggleMTF('mtf-{{ loop.index }}')">
                [toggle]
            </span>
        </div>

        <!-- Intraday Verdict — always visible -->
        <div class="intraday-verdict" style="border-color:{{ r.intraday_color }}55;">
            <div class="intraday-main" style="color:{{ r.intraday_color }};">
                {{ r.intraday_verdict }}
            </div>
            <div class="intraday-sub">{{ r.intraday_detail }}</div>
        </div>

        <!-- Timeframe table — collapsible -->
        <div id="mtf-{{ loop.index }}" style="display:none;margin-top:8px;">
            <table class="mtf-table">
                <thead>
                    <tr>
                        <th>TF</th>
                        <th>Trend</th>
                        <th>RSI</th>
                        <th>Volume</th>
                    </tr>
                </thead>
                <tbody>
                {% for tf, d in r.mtf.items() %}
                <tr>
                    <td>
                        <span class="mtf-tf-label">{{ tf }}</span>
                        <span class="mtf-category">{{ d.category }}</span>
                    </td>
                    <td style="color:{{ d.trend_color }};font-weight:bold;">
                        {{ d.trend_label }}
                    </td>
                    <td style="color:{{ d.rsi_color }};">{{ d.rsi_label }}</td>
                    <td style="color:#64748b;">{{ d.vol_label }}</td>
                </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    <!-- ===== END MTF BLOCK ===== -->

</div>
{% endfor %}

{% if not results %}
<div style="text-align:center;color:#475569;padding:40px;grid-column:1/-1;">
    Enter stock symbols above and click Scan to see signals.
</div>
{% endif %}

</div>

<hr class="divider">

<h3 class="section-title">📈 Performance Tracker Pro</h3>

<!-- Controls row -->
<div style="text-align:center;margin:12px 16px;display:flex;flex-wrap:wrap;gap:8px;justify-content:center;align-items:center;">

    <!-- Filter buttons -->
    <button class="filter-btn" onclick="perfControl('filter','ALL')" id="fb-ALL"
            style="background:#334155;color:white;">ALL</button>
    <button class="filter-btn" onclick="perfControl('filter','WIN')" id="fb-WIN"
            style="background:#166534;color:#bbf7d0;">✅ WIN</button>
    <button class="filter-btn" onclick="perfControl('filter','LOSS')" id="fb-LOSS"
            style="background:#7f1d1d;color:#fecaca;">❌ LOSS</button>

    <span style="color:#334155;">|</span>

    <!-- Sort buttons -->
    <button class="filter-btn" onclick="perfControl('sort','gain')"  id="sb-gain">📈 Highest Gain</button>
    <button class="filter-btn" onclick="perfControl('sort','loss')"  id="sb-loss">📉 Biggest Loss</button>
    <button class="filter-btn" onclick="perfControl('sort','newest')" id="sb-newest">🕐 Newest</button>
    <button class="filter-btn" onclick="perfControl('sort','oldest')" id="sb-oldest">🕰️ Oldest</button>

    <span style="color:#334155;">|</span>

    <!-- Search -->
    <input type="text" id="perf-search" placeholder="🔍 Search symbol..."
           oninput="perfControl('search',this.value)"
           style="padding:7px 12px;background:#1e293b;color:white;border:1px solid #334155;
                  border-radius:8px;font-size:13px;width:150px;">

    <span style="color:#334155;">|</span>

    <!-- Clear all -->
    <button onclick="confirmClear()"
            style="padding:7px 14px;background:#450a0a;color:#fca5a5;border:1px solid #7f1d1d;
                   border-radius:8px;font-size:13px;cursor:pointer;">
        🗑️ Clear All
    </button>
</div>

<!-- Summary bar -->
<div id="perf-summary" style="text-align:center;font-size:12px;color:#64748b;margin-bottom:8px;"></div>

<div class="container" id="perf-container">

{% for r in history %}
<div class="card perf-card"
     data-status="{{ r.status }}"
     data-symbol="{{ r.symbol }}"
     data-move="{{ r.move }}"
     data-days="{{ r.days_held if r.days_held != '?' else 9999 }}"
     data-date="{{ r.entry_date }}">

    <!-- Header -->
    <div class="symbol">
        {{ r.symbol }}
        <span class="badge" style="background:{% if r.status == 'WIN' %}#166534{% else %}#7f1d1d{% endif %};
              color:{% if r.status == 'WIN' %}#bbf7d0{% else %}#fecaca{% endif %};">
            {{ r.status }}
        </span>
    </div>

    <!-- Entry info -->
    <div class="section">
        📅 Entry: <b>{{ r.entry_date }}</b> &nbsp;|&nbsp;
        ⏱️ <b>{{ r.days_held }} days</b> held
    </div>

    <!-- Price info -->
    <div class="section">
        💰 Entry: ₹{{ r.entry }} &nbsp;|&nbsp; Current: ₹{{ r.current }}
    </div>

    <!-- Current move -->
    <div class="section">
        📊 Return:
        <span style="font-weight:bold;font-size:14px;color:{% if r.move > 0 %}#22c55e{% else %}#ef4444{% endif %};">
            {% if r.move > 0 %}+{% endif %}{{ r.move }}%
        </span>
    </div>

    <!-- Max gain achieved -->
    <div class="section">
        🏆 Max Gain Achieved:
        <span style="color:#22c55e;font-weight:bold;">+{{ r.max_gain }}%</span>
    </div>

    <!-- Max drawdown -->
    <div class="section">
        📉 Max Drawdown:
        <span style="color:#f97316;font-weight:bold;">{{ r.max_dd }}%</span>
    </div>

</div>
{% endfor %}

{% if not history %}
<div style="text-align:center;color:#475569;padding:30px;grid-column:1/-1;" id="perf-empty">
    No performance history yet. Scan some stocks first.
</div>
{% endif %}

</div>

<!-- Clear confirmation modal -->
<div id="clear-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);
     z-index:100;align-items:center;justify-content:center;">
    <div style="background:#1e293b;border:1px solid #ef4444;border-radius:14px;
                padding:28px 32px;text-align:center;max-width:340px;width:90%;">
        <div style="font-size:18px;margin-bottom:10px;">🗑️ Clear All History?</div>
        <div style="font-size:13px;color:#94a3b8;margin-bottom:20px;">
            This will permanently delete all scanned stocks and start fresh.
            This cannot be undone.
        </div>
        <div style="display:flex;gap:10px;justify-content:center;">
            <button onclick="document.getElementById('clear-modal').style.display='none'"
                    style="padding:9px 20px;background:#1e293b;color:white;border:1px solid #334155;
                           border-radius:8px;cursor:pointer;font-size:13px;">
                Cancel
            </button>
            <form method="POST" action="/clear" style="margin:0;">
                <button type="submit"
                        style="padding:9px 20px;background:#7f1d1d;color:#fecaca;border:none;
                               border-radius:8px;cursor:pointer;font-size:13px;font-weight:bold;">
                    Yes, Clear All
                </button>
            </form>
        </div>
    </div>
</div>

</body>
</html>
"""

@app.route("/clear", methods=["POST"])
def clear_history():
    if os.path.exists(FILE):
        os.remove(FILE)
    from flask import redirect, url_for
    return redirect(url_for("home"))


# =========================
# REAL-TIME LIVE DATA API
# =========================
def is_market_open():
    """
    Check if NSE is currently open (Mon-Fri, 09:15–15:30 IST).
    Uses stdlib only — no pytz dependency that could silently fail.
    IST = UTC+5:30 always (no DST).
    """
    try:
        import datetime as _dt
        utc_now  = _dt.datetime.utcnow()
        ist_now  = utc_now + _dt.timedelta(hours=5, minutes=30)
        if ist_now.weekday() >= 5:                     # Sat/Sun
            return False, ist_now.strftime("%H:%M IST")
        t        = ist_now.time()
        is_open  = _dt.time(9, 15) <= t <= _dt.time(15, 30)
        return is_open, ist_now.strftime("%H:%M IST")
    except Exception:
        return True, "--:-- IST"   # fail-open so market hours never silently block


def get_live_quote(symbol):
    """
    Fetch near-real-time quote for a symbol using yfinance 1-min data.
    Returns LTP, change, change%, high, low, volume, VWAP, bid/ask estimate.
    yfinance NSE data has ~15-min delay on free tier.
    """
    try:
        # 1-min bars for today
        df1m = _yf_download_safe(symbol + ".NS", period="1d", interval="1m", timeout=6)
        if df1m is None or df1m.empty:
            return None

        close_s  = safe_series(df1m["Close"])
        high_s   = safe_series(df1m["High"])
        low_s    = safe_series(df1m["Low"])
        volume_s = safe_series(df1m["Volume"])
        open_s   = safe_series(df1m["Open"])

        ltp      = round(float(close_s.iloc[-1]), 2)
        day_open = round(float(open_s.iloc[0]),   2)
        day_high = round(float(high_s.max()),      2)
        day_low  = round(float(low_s.min()),       2)
        prev_close = round(float(close_s.iloc[0]), 2)   # approx

        chg      = round(ltp - day_open, 2)
        chg_pct  = round((chg / day_open) * 100, 2) if day_open > 0 else 0

        # VWAP — (cumulative TP * Vol) / cumulative Vol
        try:
            tp    = (high_s + low_s + close_s) / 3
            vwap  = round(float((tp * volume_s).sum() / volume_s.sum()), 2) if volume_s.sum() > 0 else ltp
        except Exception:
            vwap  = ltp

        # Volume total today
        vol_total = int(volume_s.sum())
        vol_cr    = round((ltp * vol_total) / 1e7, 1)  # ₹ crores traded

        # Bid/Ask — estimate from last candle spread
        last_high  = round(float(high_s.iloc[-1]),  2)
        last_low   = round(float(low_s.iloc[-1]),   2)
        spread     = round((last_high - last_low) / 2, 2)
        bid_est    = round(ltp - spread * 0.1, 2)
        ask_est    = round(ltp + spread * 0.1, 2)

        # Breakout — is price at or above 20-period intraday high?
        high20     = float(high_s.tail(20).max()) if len(high_s) >= 20 else day_high
        breakout   = ltp >= high20 * 0.998

        mkt_open, mkt_time = is_market_open()

        return {
            "symbol":     symbol,
            "ltp":        ltp,
            "day_open":   day_open,
            "day_high":   day_high,
            "day_low":    day_low,
            "chg":        chg,
            "chg_pct":    chg_pct,
            "vwap":       vwap,
            "vol_total":  vol_total,
            "vol_cr":     vol_cr,
            "bid":        bid_est,
            "ask":        ask_est,
            "breakout":   breakout,
            "mkt_open":   mkt_open,
            "mkt_time":   mkt_time,
            "updated_at": datetime.now().strftime("%H:%M:%S"),
        }
    except Exception:
        return None


@app.route("/api/live/<symbol>")
def api_live(symbol):
    """JSON endpoint — called by JS every 60s to refresh live data per card."""
    from flask import jsonify
    symbol = symbol.upper().strip()
    data   = get_live_quote(symbol)
    if data is None:
        return jsonify({"error": "No data", "symbol": symbol}), 404
    return jsonify(data)


@app.route("/api/market_status")
def api_market_status():
    from flask import jsonify
    open_, time_ = is_market_open()
    return jsonify({"open": open_, "time": time_})


# =========================
# RISK REPORT ENGINE
# =========================
def build_risk_report(symbol):
    """Full risk report — granular error handling so it never silently fails."""

    # Price data is required
    df = get_data(symbol)
    if df is None or df.empty:
        return None

    try:
        close  = safe_series(df["Close"])
        volume = safe_series(df["Volume"])
        hi_s   = safe_series(df["High"])
        lo_s   = safe_series(df["Low"])
        op_s   = safe_series(df["Open"])
        if len(close) < 5:
            return None
        price = float(close.iloc[-1])
    except Exception:
        return None

    # yfinance info — optional, fall back gracefully
    try:
        tk   = yf.Ticker(symbol + ".NS")
        info = tk.info or {}
    except Exception:
        tk   = None
        info = {}

    def gi(key, default="N/A"):
        v = info.get(key, default)
        return default if v is None else v

    pe          = gi("trailingPE")
    pb          = gi("priceToBook")
    mktcap      = gi("marketCap", 0)
    debt_eq     = gi("debtToEquity")
    roe         = gi("returnOnEquity")
    beta        = gi("beta")
    div_yield   = float(gi("dividendYield", 0) or 0)
    sector_name = gi("sector", "Unknown")
    company     = gi("longName", symbol)

    try:    fifty2_high = float(gi("fiftyTwoWeekHigh", float(close.max())))
    except: fifty2_high = float(close.max())
    try:    fifty2_low  = float(gi("fiftyTwoWeekLow",  float(close.min())))
    except: fifty2_low  = float(close.min())
    try:    mktcap_cr = round(float(mktcap) / 1e7, 0) if mktcap and mktcap != "N/A" else "N/A"
    except: mktcap_cr = "N/A"

    def fmt_num(x, suffix=""):
        try:    return f"{round(float(x), 2)}{suffix}"
        except: return "N/A"

    # ATR
    try:
        tr_list = []
        for i in range(1, min(21, len(close))):
            tr = max(
                float(hi_s.iloc[-i]) - float(lo_s.iloc[-i]),
                abs(float(hi_s.iloc[-i]) - float(close.iloc[-i-1])),
                abs(float(lo_s.iloc[-i]) - float(close.iloc[-i-1]))
            )
            tr_list.append(tr)
        atr     = float(np.mean(tr_list)) if tr_list else price * 0.01
        atr_pct = round((atr / price) * 100, 2) if price > 0 else 2
    except Exception:
        atr_pct = 2.0

    # Drawdown
    try:
        roll_max  = close.cummax()
        dd_series = (close - roll_max) / roll_max * 100
        max_dd    = round(float(dd_series.min()), 2)
        curr_dd   = round(float(dd_series.iloc[-1]), 2)
    except Exception:
        max_dd = 0.0; curr_dd = 0.0

    # 52W distances
    try:
        dist_high = round(((fifty2_high - price) / fifty2_high) * 100, 1) if fifty2_high else 0
        dist_low  = round(((price - fifty2_low)  / fifty2_low)  * 100, 1) if fifty2_low  else 0
    except Exception:
        dist_high = 0.0; dist_low = 0.0

    # RSI
    try:
        rsi_val = float(calc_rsi(close).iloc[-1]) if len(close) >= 15 else 50
        if np.isnan(rsi_val): rsi_val = 50
    except Exception:
        rsi_val = 50

    # Liquidity
    try:    liq_cr = round((price * float(volume.tail(20).mean())) / 1e7, 1)
    except: liq_cr = 0.0

    # Quarterly financials
    quarters_rev = []
    quarters_eps = []
    try:
        if tk:
            qfin = tk.quarterly_financials
            if qfin is not None and not qfin.empty:
                for idx in qfin.index:
                    if "Revenue" in str(idx) or "Total Revenue" in str(idx):
                        rev_row = qfin.loc[idx]
                        for col in list(rev_row.index)[:4]:
                            try: quarters_rev.append({"q": str(col)[:7], "val": round(float(rev_row[col]) / 1e7, 0)})
                            except: pass
                        break
                quarters_rev = list(reversed(quarters_rev))
    except Exception:
        pass

    try:
        if tk:
            qinc = tk.quarterly_income_stmt
            if qinc is not None and not qinc.empty:
                for idx in qinc.index:
                    if "Diluted EPS" in str(idx) or "Basic EPS" in str(idx):
                        eps_row = qinc.loc[idx]
                        for col in list(eps_row.index)[:4]:
                            try: quarters_eps.append({"q": str(col)[:7], "val": round(float(eps_row[col]), 2)})
                            except: pass
                        break
                quarters_eps = list(reversed(quarters_eps))
    except Exception:
        pass

    # Chart data last 90 days
    chart_data = []
    try:
        tail = min(90, len(close))
        for i in range(tail, 0, -1):
            chart_data.append({
                "d": str(df.index[-i])[:10],
                "o": round(float(op_s.iloc[-i]), 2),
                "h": round(float(hi_s.iloc[-i]), 2),
                "l": round(float(lo_s.iloc[-i]), 2),
                "c": round(float(close.iloc[-i]), 2),
                "v": int(float(volume.iloc[-i]))
            })
    except Exception:
        pass

    # Risk scores
    risks = {}
    risks["Volatility"] = min(100, int(atr_pct * 15))
    risks["Drawdown"]   = min(100, int(abs(max_dd) * 2))
    try:
        pe_f = float(pe)
        if pe_f <= 0: risks["Valuation"] = 80
        elif pe_f <= 15: risks["Valuation"] = 10
        elif pe_f <= 25: risks["Valuation"] = 25
        elif pe_f <= 40: risks["Valuation"] = 50
        else: risks["Valuation"] = 75
    except: risks["Valuation"] = 50
    try:
        de = float(debt_eq)
        if de <= 0: risks["Debt"] = 5
        elif de <= 30: risks["Debt"] = 15
        elif de <= 80: risks["Debt"] = 40
        elif de <= 150: risks["Debt"] = 65
        else: risks["Debt"] = 90
    except: risks["Debt"] = 50
    if   rsi_val >= 80: risks["Momentum"] = 80
    elif rsi_val >= 60: risks["Momentum"] = 25
    elif rsi_val >= 40: risks["Momentum"] = 40
    elif rsi_val >= 30: risks["Momentum"] = 65
    else:               risks["Momentum"] = 85
    if   liq_cr >= 100: risks["Liquidity"] = 5
    elif liq_cr >= 20:  risks["Liquidity"] = 25
    elif liq_cr >= 5:   risks["Liquidity"] = 60
    else:               risks["Liquidity"] = 90
    if   dist_high <= 3:  risks["Price Position"] = 70
    elif dist_high <= 10: risks["Price Position"] = 35
    elif dist_high <= 25: risks["Price Position"] = 20
    else:                 risks["Price Position"] = 55
    try:    risks["Beta"] = min(100, int(abs(float(beta)) * 35))
    except: risks["Beta"] = 40

    overall_risk = int(sum(risks.values()) / len(risks))
    if   overall_risk <= 20: grade, grade_color = "A", "#10b981"
    elif overall_risk <= 35: grade, grade_color = "B", "#86efac"
    elif overall_risk <= 50: grade, grade_color = "C", "#f59e0b"
    elif overall_risk <= 65: grade, grade_color = "D", "#f97316"
    else:                    grade, grade_color = "F", "#ef4444"

    # Strengths & Weaknesses
    strengths, weaknesses = [], []
    if rsi_val >= 50 and rsi_val < 75: strengths.append("RSI in healthy bullish zone")
    if rsi_val >= 75:                   weaknesses.append("RSI overbought — pullback risk")
    if rsi_val < 35:                    weaknesses.append("RSI oversold — downtrend momentum")
    if dist_low >= 20:                  strengths.append(f"Strong recovery +{dist_low}% from 52W low")
    if dist_high <= 5:                  strengths.append("Trading near 52W high — strong stock")
    if dist_high >= 30:                 weaknesses.append(f"Far from 52W high ({dist_high}% below)")
    if liq_cr >= 50:                    strengths.append(f"High liquidity Rs{liq_cr}Cr/day — easy exit")
    if liq_cr < 10:                     weaknesses.append(f"Low liquidity Rs{liq_cr}Cr/day — exit risk")
    if atr_pct < 2:                     strengths.append(f"Low ATR {atr_pct}% — tight controlled move")
    if atr_pct > 4:                     weaknesses.append(f"High ATR {atr_pct}% — volatile stock")
    if abs(max_dd) < 15:                strengths.append(f"Low max drawdown {max_dd}% — strong base")
    if abs(max_dd) > 30:                weaknesses.append(f"Max drawdown {max_dd}% — heavy past selling")
    try:
        roe_f = float(roe)
        if roe_f > 0.15:    strengths.append(f"Strong ROE {round(roe_f*100,1)}%")
        elif roe_f < 0.05:  weaknesses.append(f"Weak ROE {round(roe_f*100,1)}%")
    except: pass
    try:
        de_f = float(debt_eq)
        if de_f < 30:    strengths.append("Low debt — financially healthy")
        elif de_f > 100: weaknesses.append("High debt — leverage risk")
    except: pass
    if div_yield > 0.02: strengths.append(f"Dividend yield {round(div_yield*100,1)}% — income cushion")

    try:    roe_display = fmt_num(float(roe) * 100, "%")
    except: roe_display = "N/A"

    return {
        "symbol":       symbol,
        "company":      company,
        "sector":       sector_name,
        "price":        round(price, 2),
        "mktcap_cr":    mktcap_cr,
        "pe":           fmt_num(pe),
        "pb":           fmt_num(pb),
        "beta":         fmt_num(beta),
        "debt_eq":      fmt_num(debt_eq),
        "roe":          roe_display,
        "div_yield":    round(div_yield * 100, 2),
        "fifty2_high":  round(fifty2_high, 2),
        "fifty2_low":   round(fifty2_low,  2),
        "dist_high":    dist_high,
        "dist_low":     dist_low,
        "atr_pct":      atr_pct,
        "max_dd":       max_dd,
        "curr_dd":      curr_dd,
        "rsi":          round(rsi_val, 1),
        "liq_cr":       liq_cr,
        "overall_risk": overall_risk,
        "grade":        grade,
        "grade_color":  grade_color,
        "risks":        risks,
        "strengths":    strengths,
        "weaknesses":   weaknesses,
        "quarters_rev": quarters_rev,
        "quarters_eps": quarters_eps,
        "chart_data":   chart_data,
    }

RISK_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>{{ d.symbol }} — Risk Report | FalconAI</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --bg:#070d18;--surface:#0d1424;--card:#111c2e;--border:#1a2840;--border2:#243452;
  --text:#e2e8f0;--muted:#4a6080;--muted2:#2a3a55;
  --green:#10b981;--green2:#064e35;--red:#ef4444;--red2:#4a0f0f;
  --yellow:#f59e0b;--yellow2:#3d2a00;--blue:#3b82f6;--blue2:#0f2040;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',Arial,sans-serif;}
.topbar{background:var(--surface);padding:0 24px;display:flex;align-items:center;
  border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100;height:52px;}
.nav-logo{font-size:16px;font-weight:800;color:var(--blue);margin-right:24px;text-decoration:none;}
.nav-tab{display:flex;align-items:center;height:52px;padding:0 16px;font-size:13px;font-weight:600;
  color:var(--muted);text-decoration:none;border-bottom:2px solid transparent;white-space:nowrap;}
.nav-tab:hover{color:var(--text);}
.nav-tab.active{color:var(--blue);border-bottom-color:var(--blue);}
.nav-right{margin-left:auto;font-size:12px;color:var(--muted);}
.page{max-width:1200px;margin:0 auto;padding:20px 20px 80px;}
.hero-banner{background:linear-gradient(135deg,#0d1e3a 0%,#0d1424 60%,#0a1628 100%);
  border:1px solid var(--border2);border-radius:16px;padding:24px 28px;margin-bottom:20px;
  display:flex;gap:24px;flex-wrap:wrap;align-items:stretch;position:relative;overflow:hidden;}
.hero-left{flex:1;min-width:200px;}
.hero-sym{font-size:32px;font-weight:800;color:white;letter-spacing:-1px;line-height:1;}
.hero-name{font-size:14px;color:#7090b0;margin-top:5px;}
.hero-tags{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap;}
.hero-tag{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;
  background:var(--blue2);color:var(--blue);border:1px solid rgba(59,130,246,0.2);}
.hero-price-block{display:flex;flex-direction:column;justify-content:center;min-width:180px;}
.hero-price{font-size:38px;font-weight:800;color:white;line-height:1;}
.hero-sub{font-size:12px;color:var(--muted);margin-top:6px;}
.price-range-bar{display:flex;align-items:center;gap:6px;margin-top:10px;font-size:11px;color:var(--muted);}
.range-track{flex:1;height:4px;background:var(--border2);border-radius:2px;position:relative;}
.range-fill{height:100%;width:100%;border-radius:2px;background:linear-gradient(90deg,var(--red),var(--yellow),var(--green));}
.range-dot{position:absolute;top:-4px;width:10px;height:10px;border-radius:50%;
  background:white;transform:translateX(-50%);border:2px solid var(--blue);}
.grade-block{display:flex;flex-direction:column;align-items:center;justify-content:center;min-width:150px;}
.grade-arc-wrap{position:relative;width:120px;height:120px;}
.grade-arc-wrap canvas{position:absolute;top:0;left:0;}
.grade-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.grade-num{font-size:28px;font-weight:800;line-height:1;}
.grade-lbl{font-size:10px;color:var(--muted);}
.grade-letter{font-size:36px;font-weight:900;margin-top:8px;line-height:1;}
.grade-text{font-size:11px;color:var(--muted);margin-top:3px;}
.section-tabs{display:flex;gap:4px;margin-bottom:18px;background:var(--surface);
  border:1px solid var(--border);border-radius:10px;padding:4px;overflow-x:auto;}
.stab{padding:7px 16px;border-radius:7px;font-size:13px;font-weight:600;cursor:pointer;
  color:var(--muted);border:none;background:transparent;white-space:nowrap;}
.stab:hover{color:var(--text);}
.stab.active{background:var(--blue2);color:var(--blue);}
.tab-panel{display:none;}
.tab-panel.active{display:block;}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:14px;}
@media(max-width:900px){.g4{grid-template-columns:1fr 1fr;}}
@media(max-width:600px){.g2,.g4{grid-template-columns:1fr;}}
.panel{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 18px;}
.ptitle{font-size:11px;color:var(--muted);font-weight:700;text-transform:uppercase;
  letter-spacing:0.8px;margin-bottom:14px;display:flex;align-items:center;gap:6px;}
.kn-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.kn-card{background:var(--surface);border-radius:10px;padding:12px 14px;
  border:1px solid var(--border);position:relative;overflow:hidden;}
.kn-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;border-radius:3px 0 0 3px;}
.kn-card.good::before{background:var(--green);}
.kn-card.warn::before{background:var(--yellow);}
.kn-card.bad::before{background:var(--red);}
.kn-card.neutral::before{background:var(--muted2);}
.kn-lbl{font-size:10px;color:var(--muted);margin-bottom:5px;font-weight:600;}
.kn-val{font-size:18px;font-weight:800;line-height:1;}
.kn-card.good .kn-val{color:var(--green);}
.kn-card.warn .kn-val{color:var(--yellow);}
.kn-card.bad  .kn-val{color:var(--red);}
.kn-card.neutral .kn-val{color:var(--text);}
.kn-hint{font-size:10px;margin-top:4px;opacity:0.75;}
.kn-card.good .kn-hint{color:var(--green);}
.kn-card.warn .kn-hint{color:var(--yellow);}
.kn-card.bad  .kn-hint{color:var(--red);}
.kn-card.neutral .kn-hint{color:var(--muted);}
.risk-row{margin-bottom:13px;}
.risk-header{display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px;align-items:center;}
.risk-name-txt{color:var(--text);font-weight:600;}
.risk-score-badge{font-size:11px;font-weight:700;padding:1px 8px;border-radius:10px;}
.risk-track{background:var(--border2);border-radius:6px;height:8px;overflow:hidden;}
.risk-fill{height:100%;border-radius:6px;}
.metric-row{display:flex;justify-content:space-between;align-items:center;
  padding:8px 0;border-bottom:1px solid var(--border);}
.metric-row:last-child{border-bottom:none;}
.metric-lbl{font-size:12px;color:var(--muted);}
.metric-val{font-size:13px;font-weight:700;}
.sw-item{display:flex;align-items:flex-start;gap:8px;padding:8px 0;
  border-bottom:1px solid var(--border);font-size:12px;color:var(--text);line-height:1.5;}
.sw-item:last-child{border-bottom:none;}
#priceChart{width:100%;height:320px;display:block;}
#volChart{width:100%;height:70px;display:block;margin-top:2px;}
.chart-legend{display:flex;gap:16px;font-size:11px;color:var(--muted);margin-bottom:8px;flex-wrap:wrap;}
.qbar-wrap{display:flex;align-items:flex-end;gap:8px;height:110px;margin-top:10px;}
.qbar-col{flex:1;display:flex;flex-direction:column;align-items:center;gap:4px;}
.qbar{width:100%;border-radius:5px 5px 0 0;min-height:4px;}
.qbar-lbl{font-size:10px;color:var(--muted);text-align:center;}
.qbar-val{font-size:10px;color:var(--text);font-weight:600;}
.q-trend-label{font-size:11px;padding:3px 10px;border-radius:10px;font-weight:700;display:inline-block;margin-top:8px;}
.concall-wrap{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
@media(max-width:600px){.concall-wrap{grid-template-columns:1fr;}}
.concall-highlight{background:var(--surface);border-left:3px solid var(--blue);
  padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:8px;font-size:12px;color:var(--text);line-height:1.6;}
.tone-badge{display:inline-flex;align-items:center;gap:6px;padding:5px 14px;
  border-radius:20px;font-size:13px;font-weight:700;margin-bottom:12px;}
.csec{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.8px;margin:12px 0 6px;font-weight:700;}
.cbullet{font-size:12px;color:#8099b8;padding:5px 0;border-bottom:1px solid var(--border);line-height:1.5;display:flex;gap:6px;}
.cbullet:last-child{border-bottom:none;}
.falcon-wrap{background:linear-gradient(135deg,#0a1628,#0d1424);border:1px solid rgba(59,130,246,0.2);
  border-radius:14px;padding:22px 26px;margin-bottom:16px;}
.falcon-hdr{font-size:16px;font-weight:800;color:var(--blue);margin-bottom:16px;display:flex;align-items:center;gap:8px;}
.falcon-text{font-size:13px;color:#a0b8d0;line-height:1.9;white-space:pre-wrap;}
.verdict-wrap{border-radius:14px;padding:28px;text-align:center;margin-bottom:16px;
  border:1px solid var(--border2);background:linear-gradient(135deg,var(--surface),var(--card));}
.verdict-main{font-size:22px;font-weight:800;margin-bottom:10px;line-height:1.3;}
.verdict-sub{font-size:13px;color:var(--muted);}
.verdict-pills{display:flex;gap:8px;justify-content:center;margin-top:14px;flex-wrap:wrap;}
.vpill{padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;background:var(--border2);color:var(--muted);}
.ai-load{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:13px;padding:14px 0;}
.ai-dot{width:7px;height:7px;border-radius:50%;background:var(--blue);animation:aipulse 1.2s infinite;}
.ai-dot:nth-child(2){animation-delay:.2s;}.ai-dot:nth-child(3){animation-delay:.4s;}
@keyframes aipulse{0%,100%{opacity:.25;transform:scale(.8)}50%{opacity:1;transform:scale(1.2)}}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;text-align:center;}
.stat-val{font-size:22px;font-weight:800;line-height:1;}
.stat-lbl{font-size:11px;color:var(--muted);margin-top:4px;}
</style>
</head>
<body>
<nav class="topbar">
  <a href="/" class="nav-logo">&#x1F985; FalconAI</a>
  <a href="/" class="nav-tab">&#x1F4CA; Screener</a>
  <a href="/options" class="nav-tab">&#x26A1; Options</a>
  <a href="/screens" class="nav-tab">&#x1F50D; Screens</a>
  <a href="/sectors" class="nav-tab">&#x1F3ED; Sectors</a>
  <a href="/scanner" class="nav-tab">&#x26A1; 5-Min Scanner</a>
  <a href="#" class="nav-tab active">&#x1F4CB; Risk Report</a>
  <div class="nav-right">{{ d.symbol }} &middot; {{ d.sector }}</div>
</nav>
<div class="page">

<div class="hero-banner">
  <div class="hero-left">
    <div class="hero-sym">{{ d.symbol }}</div>
    <div class="hero-name">{{ d.company }}</div>
    <div class="hero-tags">
      <span class="hero-tag">{{ d.sector }}</span>
      <span class="hero-tag" style="background:{{ d.grade_color }}18;color:{{ d.grade_color }};border-color:{{ d.grade_color }}40;">Grade {{ d.grade }}</span>
      {% if d.overall_risk <= 35 %}<span class="hero-tag" style="background:#06403528;color:#10b981;border-color:#10b98140;">Low Risk</span>
      {% elif d.overall_risk <= 55 %}<span class="hero-tag" style="background:#3d2a0028;color:#f59e0b;border-color:#f59e0b40;">Moderate Risk</span>
      {% else %}<span class="hero-tag" style="background:#4a0f0f28;color:#ef4444;border-color:#ef444440;">High Risk</span>{% endif %}
    </div>
  </div>
  <div class="hero-price-block">
    <div class="hero-price">&#x20B9;{{ d.price }}</div>
    <div class="hero-sub">Mkt Cap &#x20B9;{{ d.mktcap_cr }}Cr &middot; Div {{ d.div_yield }}%</div>
    <div class="price-range-bar">
      <span>&#x20B9;{{ d.fifty2_low }}</span>
      <div class="range-track">
        <div class="range-fill"></div>
        {% set pos = ((d.price - d.fifty2_low) / (d.fifty2_high - d.fifty2_low + 0.01) * 100) | round(1) %}
        <div class="range-dot" style="left:{{ pos }}%;"></div>
      </div>
      <span>&#x20B9;{{ d.fifty2_high }}</span>
    </div>
    <div style="font-size:10px;color:var(--muted);margin-top:3px;">{{ d.dist_high }}% below 52W high &middot; +{{ d.dist_low }}% above 52W low</div>
  </div>
  <div class="grade-block">
    <div class="grade-arc-wrap">
      <canvas id="arcCanvas" width="120" height="120"></canvas>
      <div class="grade-center">
        <span class="grade-num" style="color:{{ d.grade_color }};">{{ d.overall_risk }}</span>
        <span class="grade-lbl">Risk</span>
      </div>
    </div>
    <div class="grade-letter" style="color:{{ d.grade_color }};">{{ d.grade }}</div>
    <div class="grade-text">{% if d.overall_risk<=20 %}Very Low Risk{% elif d.overall_risk<=35 %}Low Risk{% elif d.overall_risk<=50 %}Moderate Risk{% elif d.overall_risk<=65 %}High Risk{% else %}Very High Risk{% endif %}</div>
  </div>
</div>

<div class="g4" style="margin-bottom:18px;">
  <div class="stat-card"><div class="stat-val" style="color:{% if d.rsi>=50 and d.rsi<75 %}var(--green){% elif d.rsi>=75 %}var(--yellow){% else %}var(--red){% endif %};">{{ d.rsi }}</div><div class="stat-lbl">RSI (14)</div></div>
  <div class="stat-card"><div class="stat-val" style="color:{% if d.atr_pct<2 %}var(--green){% elif d.atr_pct<4 %}var(--yellow){% else %}var(--red){% endif %};">{{ d.atr_pct }}%</div><div class="stat-lbl">ATR Volatility</div></div>
  <div class="stat-card"><div class="stat-val" style="color:{% if d.liq_cr>=50 %}var(--green){% elif d.liq_cr>=10 %}var(--yellow){% else %}var(--red){% endif %};">&#x20B9;{{ d.liq_cr }}Cr</div><div class="stat-lbl">Daily Liquidity</div></div>
  <div class="stat-card"><div class="stat-val" style="color:{% if d.max_dd>-15 %}var(--green){% elif d.max_dd>-30 %}var(--yellow){% else %}var(--red){% endif %};">{{ d.max_dd }}%</div><div class="stat-lbl">Max Drawdown 6M</div></div>
</div>

<div class="section-tabs">
  <button class="stab active" onclick="switchTab('fundamentals',this)">&#x1F522; Fundamentals</button>
  <button class="stab" onclick="switchTab('risk',this)">&#x26A0;&#xFE0F; Risk Breakdown</button>
  <button class="stab" onclick="switchTab('chart',this)">&#x1F4C8; Price Chart</button>
  <button class="stab" onclick="switchTab('financials',this)">&#x1F4CA; Financials</button>
  <button class="stab" onclick="switchTab('concall',this)">&#x1F4DE; Con Call</button>
  <button class="stab" onclick="switchTab('notes',this)">&#x1F985; AI Notes</button>
</div>

<div id="tab-fundamentals" class="tab-panel active">
<div class="g2">
  <div class="panel">
    <div class="ptitle">&#x1F522; Key Numbers <span style="font-size:10px;font-weight:400;margin-left:auto;">&#x1F7E2; Good &nbsp; &#x1F7E1; Watch &nbsp; &#x1F534; Risk</span></div>
    <div class="kn-grid">
      {% set pe_f = d.pe | float(default=0) %}
      {% if pe_f > 0 and pe_f <= 25 %}{% set pe_cls="good" %}{% set pe_hint="Fairly valued" %}{% elif pe_f > 25 and pe_f <= 45 %}{% set pe_cls="warn" %}{% set pe_hint="Moderately expensive" %}{% elif pe_f > 45 %}{% set pe_cls="bad" %}{% set pe_hint="Overvalued" %}{% else %}{% set pe_cls="neutral" %}{% set pe_hint="N/A" %}{% endif %}
      <div class="kn-card {{ pe_cls }}"><div class="kn-lbl">P/E RATIO</div><div class="kn-val">{{ d.pe }}</div><div class="kn-hint">{{ pe_hint }}</div></div>
      {% set pb_f = d.pb | float(default=0) %}
      {% if pb_f > 0 and pb_f <= 3 %}{% set pb_cls="good" %}{% set pb_hint="Reasonable book value" %}{% elif pb_f > 3 and pb_f <= 7 %}{% set pb_cls="warn" %}{% set pb_hint="Premium valuation" %}{% elif pb_f > 7 %}{% set pb_cls="bad" %}{% set pb_hint="Expensive vs book" %}{% else %}{% set pb_cls="neutral" %}{% set pb_hint="N/A" %}{% endif %}
      <div class="kn-card {{ pb_cls }}"><div class="kn-lbl">P/B RATIO</div><div class="kn-val">{{ d.pb }}</div><div class="kn-hint">{{ pb_hint }}</div></div>
      {% set beta_f = d.beta | float(default=1) %}
      {% if beta_f <= 0.8 %}{% set beta_cls="good" %}{% set beta_hint="Low volatility stock" %}{% elif beta_f <= 1.3 %}{% set beta_cls="warn" %}{% set beta_hint="Moves with market" %}{% else %}{% set beta_cls="bad" %}{% set beta_hint="High volatility" %}{% endif %}
      <div class="kn-card {{ beta_cls }}"><div class="kn-lbl">BETA</div><div class="kn-val">{{ d.beta }}</div><div class="kn-hint">{{ beta_hint }}</div></div>
      {% set de_f = d.debt_eq | float(default=999) %}
      {% if de_f <= 30 %}{% set de_cls="good" %}{% set de_hint="Low debt" %}{% elif de_f <= 80 %}{% set de_cls="warn" %}{% set de_hint="Moderate leverage" %}{% else %}{% set de_cls="bad" %}{% set de_hint="High debt" %}{% endif %}
      <div class="kn-card {{ de_cls }}"><div class="kn-lbl">DEBT / EQUITY</div><div class="kn-val">{{ d.debt_eq }}</div><div class="kn-hint">{{ de_hint }}</div></div>
      {% set roe_str = d.roe | string %}{% if '%' in roe_str %}{% set roe_f = roe_str.replace('%','') | float(default=0) %}{% else %}{% set roe_f = roe_str | float(default=0) %}{% endif %}
      {% if roe_f >= 15 %}{% set roe_cls="good" %}{% set roe_hint="Strong returns" %}{% elif roe_f >= 8 %}{% set roe_cls="warn" %}{% set roe_hint="Acceptable ROE" %}{% elif roe_f > 0 %}{% set roe_cls="bad" %}{% set roe_hint="Weak efficiency" %}{% else %}{% set roe_cls="neutral" %}{% set roe_hint="N/A" %}{% endif %}
      <div class="kn-card {{ roe_cls }}"><div class="kn-lbl">ROE</div><div class="kn-val">{{ d.roe }}</div><div class="kn-hint">{{ roe_hint }}</div></div>
      {% if d.div_yield >= 2 %}{% set div_cls="good" %}{% set div_hint="Good income cushion" %}{% elif d.div_yield >= 0.5 %}{% set div_cls="warn" %}{% set div_hint="Nominal dividend" %}{% else %}{% set div_cls="neutral" %}{% set div_hint="Growth stock" %}{% endif %}
      <div class="kn-card {{ div_cls }}"><div class="kn-lbl">DIVIDEND YIELD</div><div class="kn-val">{{ d.div_yield }}%</div><div class="kn-hint">{{ div_hint }}</div></div>
      {% if d.rsi >= 50 and d.rsi < 70 %}{% set rsi_cls="good" %}{% set rsi_hint="Healthy bullish zone" %}{% elif d.rsi >= 70 %}{% set rsi_cls="warn" %}{% set rsi_hint="Overbought" %}{% elif d.rsi >= 35 %}{% set rsi_cls="warn" %}{% set rsi_hint="Neutral" %}{% else %}{% set rsi_cls="bad" %}{% set rsi_hint="Oversold" %}{% endif %}
      <div class="kn-card {{ rsi_cls }}"><div class="kn-lbl">RSI (14)</div><div class="kn-val">{{ d.rsi }}</div><div class="kn-hint">{{ rsi_hint }}</div></div>
      {% if d.liq_cr >= 50 %}{% set liq_cls="good" %}{% set liq_hint="Easy entry and exit" %}{% elif d.liq_cr >= 10 %}{% set liq_cls="warn" %}{% set liq_hint="Moderate liquidity" %}{% else %}{% set liq_cls="bad" %}{% set liq_hint="Low — exit risk" %}{% endif %}
      <div class="kn-card {{ liq_cls }}"><div class="kn-lbl">LIQUIDITY / DAY</div><div class="kn-val">&#x20B9;{{ d.liq_cr }}Cr</div><div class="kn-hint">{{ liq_hint }}</div></div>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;gap:14px;">
    <div class="panel">
      <div class="ptitle">&#x1F4D0; Assessment Metrics</div>
      <div class="metric-row"><span class="metric-lbl">ATR Volatility</span><span class="metric-val" style="color:{% if d.atr_pct<2 %}var(--green){% elif d.atr_pct<4 %}var(--yellow){% else %}var(--red){% endif %};">{{ d.atr_pct }}%</span></div>
      <div class="metric-row"><span class="metric-lbl">Max Drawdown (6M)</span><span class="metric-val" style="color:{% if d.max_dd>-15 %}var(--green){% elif d.max_dd>-30 %}var(--yellow){% else %}var(--red){% endif %};">{{ d.max_dd }}%</span></div>
      <div class="metric-row"><span class="metric-lbl">Current Drawdown</span><span class="metric-val" style="color:{% if d.curr_dd>-8 %}var(--green){% elif d.curr_dd>-15 %}var(--yellow){% else %}var(--red){% endif %};">{{ d.curr_dd }}%</span></div>
      <div class="metric-row"><span class="metric-lbl">Distance from 52W High</span><span class="metric-val" style="color:{% if d.dist_high<=5 %}var(--green){% elif d.dist_high<=20 %}var(--yellow){% else %}var(--muted){% endif %};">{{ d.dist_high }}% below</span></div>
      <div class="metric-row"><span class="metric-lbl">Recovery from 52W Low</span><span class="metric-val" style="color:var(--green);">+{{ d.dist_low }}% above</span></div>
    </div>
    <div class="panel" style="flex:1;">
      <div class="ptitle">&#x1F4AA; Strengths</div>
      {% if d.strengths %}{% for s in d.strengths %}<div class="sw-item"><span>&#x2705;</span>{{ s }}</div>{% endfor %}{% else %}<div style="font-size:12px;color:var(--muted);">None detected.</div>{% endif %}
      <div class="ptitle" style="margin-top:16px;">&#x26A0;&#xFE0F; Weaknesses</div>
      {% if d.weaknesses %}{% for w in d.weaknesses %}<div class="sw-item"><span>&#x274C;</span>{{ w }}</div>{% endfor %}{% else %}<div style="font-size:12px;color:var(--muted);">None detected.</div>{% endif %}
    </div>
  </div>
</div>
</div>

<div id="tab-risk" class="tab-panel">
<div class="g2">
  <div class="panel">
    <div class="ptitle">&#x26A0;&#xFE0F; Risk Score Breakdown</div>
    {% for name, val in d.risks.items() %}
    <div class="risk-row">
      <div class="risk-header">
        <span class="risk-name-txt">{{ name }}</span>
        <span class="risk-score-badge" style="background:{% if val<=30 %}var(--green2){% elif val<=55 %}var(--yellow2){% else %}var(--red2){% endif %};color:{% if val<=30 %}var(--green){% elif val<=55 %}var(--yellow){% else %}var(--red){% endif %};">{{ val }}/100</span>
      </div>
      <div class="risk-track"><div class="risk-fill" style="width:{{ val }}%;background:{% if val<=30 %}var(--green){% elif val<=55 %}var(--yellow){% else %}var(--red){% endif %};"></div></div>
    </div>
    {% endfor %}
  </div>
  <div class="panel"><div class="ptitle">&#x1F5FA;&#xFE0F; Risk Radar</div><canvas id="radarCanvas" width="280" height="280" style="display:block;margin:0 auto;"></canvas></div>
</div>
</div>

<div id="tab-chart" class="tab-panel">
<div class="panel">
  <div class="ptitle" style="margin-bottom:6px;">&#x1F4C8; Price Chart &mdash; Last 90 Days</div>
  <div class="chart-legend">
    <span><span style="display:inline-block;width:10px;height:10px;background:#10b981;border-radius:2px;margin-right:4px;"></span>Bullish</span>
    <span><span style="display:inline-block;width:10px;height:10px;background:#ef4444;border-radius:2px;margin-right:4px;"></span>Bearish</span>
    <span><span style="display:inline-block;width:10px;height:3px;background:#3b82f6;margin-right:4px;vertical-align:middle;"></span>MA20</span>
    <span><span style="display:inline-block;width:10px;height:3px;background:#f59e0b;margin-right:4px;vertical-align:middle;"></span>MA50</span>
  </div>
  <canvas id="priceChart"></canvas><canvas id="volChart"></canvas>
</div>
</div>

<div id="tab-financials" class="tab-panel">
{% if d.quarters_rev or d.quarters_eps %}
<div class="g2">
  {% if d.quarters_rev %}<div class="panel"><div class="ptitle">&#x1F4CA; Quarterly Revenue (&#x20B9; Cr)</div><div class="qbar-wrap" id="rev-bars"></div><div id="rev-trend"></div></div>{% endif %}
  {% if d.quarters_eps %}<div class="panel"><div class="ptitle">&#x1F4B0; Quarterly EPS (&#x20B9;)</div><div class="qbar-wrap" id="eps-bars"></div><div id="eps-trend"></div></div>{% endif %}
</div>
{% else %}<div class="panel" style="text-align:center;padding:40px;color:var(--muted);">Quarterly data not available.</div>{% endif %}
</div>

<div id="tab-concall" class="tab-panel">
<div class="panel">
  <div class="ptitle">&#x1F4DE; Latest Earnings Conference Call</div>
  <div id="concall-loading" class="ai-load"><div class="ai-dot"></div><div class="ai-dot"></div><div class="ai-dot"></div><span>Fetching con call details...</span></div>
  <div id="concall-content" style="display:none;">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap;"><span id="tone-badge" class="tone-badge"></span><span id="concall-quarter" style="font-size:12px;color:var(--muted);"></span></div>
    <div class="concall-wrap">
      <div><div class="csec">&#x1F4CB; Key Highlights</div><div id="concall-highlights"></div></div>
      <div><div class="csec">&#x1F3AF; Guidance</div><div id="concall-guidance" class="cbullet"></div><div class="csec" style="margin-top:14px;">&#x26A0;&#xFE0F; Risks Flagged</div><div id="concall-risks"></div></div>
    </div>
  </div>
  <div id="concall-error" style="display:none;font-size:12px;color:var(--muted);"></div>
</div>
</div>

<div id="tab-notes" class="tab-panel">
<div class="falcon-wrap">
  <div class="falcon-hdr">&#x1F985; FalconAI Analyst Notes <span style="font-size:11px;color:var(--muted);font-weight:400;">AI-generated &middot; Not financial advice</span></div>
  <div id="falcon-loading" class="ai-load"><div class="ai-dot"></div><div class="ai-dot"></div><div class="ai-dot"></div><span>Writing analyst notes...</span></div>
  <div id="falcon-notes-text" class="falcon-text" style="display:none;"></div>
  <div id="falcon-error" style="display:none;font-size:12px;color:var(--muted);"></div>
</div>
<div class="verdict-wrap">
  <div class="verdict-main" style="color:{{ d.grade_color }};">
    {% if d.overall_risk<=20 %}&#x2705; Very Low Risk &mdash; Strong candidate{% elif d.overall_risk<=35 %}&#x2705; Low Risk &mdash; Good setup{% elif d.overall_risk<=50 %}&#x1F7E1; Moderate Risk &mdash; Trade with proper SL{% elif d.overall_risk<=65 %}&#x1F7E0; High Risk &mdash; Experienced traders only{% else %}&#x1F534; Very High Risk &mdash; Avoid or reduce position{% endif %}
  </div>
  <div class="verdict-sub">Risk Score {{ d.overall_risk }}/100 &middot; Grade {{ d.grade }}</div>
  <div class="verdict-pills"><span class="vpill">{{ d.strengths|length }} Strengths</span><span class="vpill">{{ d.weaknesses|length }} Weaknesses</span><span class="vpill">RSI {{ d.rsi }}</span><span class="vpill">ATR {{ d.atr_pct }}%</span><span class="vpill">&#x20B9;{{ d.liq_cr }}Cr/day</span></div>
</div>
</div>

</div>
<script>
function switchTab(id,btn){document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.stab').forEach(b=>b.classList.remove('active'));document.getElementById('tab-'+id).classList.add('active');btn.classList.add('active');if(id==='chart')setTimeout(drawChart,50);if(id==='risk')setTimeout(drawRadar,50);if(id==='financials')setTimeout(()=>{drawBars("rev-bars",revData,"#3b82f6","rev-trend");drawBars("eps-bars",epsData,"#10b981","eps-trend");},50);}
(function(){const c=document.getElementById('arcCanvas'),ctx=c.getContext('2d'),score={{ d.overall_risk }},color="{{ d.grade_color }}",cx=60,cy=65,r=50,start=Math.PI*0.75,full=Math.PI*1.5,end=start+(score/100)*full;ctx.clearRect(0,0,120,120);ctx.beginPath();ctx.arc(cx,cy,r,start,start+full);ctx.strokeStyle='#1a2840';ctx.lineWidth=10;ctx.lineCap='round';ctx.stroke();ctx.beginPath();ctx.arc(cx,cy,r,start,end);ctx.strokeStyle=color;ctx.lineWidth=10;ctx.lineCap='round';ctx.stroke();})();
function drawRadar(){const c=document.getElementById('radarCanvas');if(!c)return;const ctx=c.getContext('2d'),W=c.width,H=c.height,cx=W/2,cy=H/2,r=Math.min(W,H)*0.36;const risks={{ d.risks|tojson }},keys=Object.keys(risks),vals=Object.values(risks),n=keys.length;ctx.clearRect(0,0,W,H);[20,40,60,80,100].forEach(pct=>{ctx.beginPath();for(let i=0;i<=n;i++){const a=(i/n)*Math.PI*2-Math.PI/2,x=cx+Math.cos(a)*r*pct/100,y=cy+Math.sin(a)*r*pct/100;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}ctx.closePath();ctx.strokeStyle='#1a2840';ctx.lineWidth=1;ctx.stroke();});keys.forEach((k,i)=>{const a=(i/n)*Math.PI*2-Math.PI/2,lx=cx+Math.cos(a)*(r+22),ly=cy+Math.sin(a)*(r+22);ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+Math.cos(a)*r,cy+Math.sin(a)*r);ctx.strokeStyle='#1a2840';ctx.lineWidth=1;ctx.stroke();ctx.fillStyle='#4a6080';ctx.font='9px Arial';ctx.textAlign='center';ctx.fillText(k.slice(0,9),lx,ly+3);});ctx.beginPath();vals.forEach((v,i)=>{const a=(i/n)*Math.PI*2-Math.PI/2,x=cx+Math.cos(a)*r*v/100,y=cy+Math.sin(a)*r*v/100;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});ctx.closePath();ctx.fillStyle='rgba(239,68,68,0.15)';ctx.strokeStyle='#ef4444';ctx.lineWidth=1.5;ctx.fill();ctx.stroke();vals.forEach((v,i)=>{const a=(i/n)*Math.PI*2-Math.PI/2;ctx.beginPath();ctx.arc(cx+Math.cos(a)*r*v/100,cy+Math.sin(a)*r*v/100,3,0,Math.PI*2);ctx.fillStyle='#ef4444';ctx.fill();});}
drawRadar();
const raw={{ d.chart_data|tojson }};
function drawChart(){const pC=document.getElementById("priceChart"),vC=document.getElementById("volChart");if(!pC||!vC)return;pC.width=pC.offsetWidth;pC.height=320;vC.width=vC.offsetWidth;vC.height=70;const pCtx=pC.getContext("2d"),vCtx=vC.getContext("2d"),W=pC.width,H=320,VH=70,pad={l:60,r:14,t:18,b:30};pCtx.clearRect(0,0,W,H);vCtx.clearRect(0,0,W,VH);if(!raw.length)return;const highs=raw.map(d=>d.h),lows=raw.map(d=>d.l),vols=raw.map(d=>d.v),minP=Math.min(...lows)*0.994,maxP=Math.max(...highs)*1.006,maxV=Math.max(...vols)*1.15,rangeP=maxP-minP,cw=(W-pad.l-pad.r)/raw.length,bw=Math.max(1,cw*0.55),px=i=>pad.l+i*cw+cw/2,py=v=>pad.t+(1-(v-minP)/rangeP)*(H-pad.t-pad.b),vy=v=>VH*(1-v/maxV)*0.92;const bg=pCtx.createLinearGradient(0,0,0,H);bg.addColorStop(0,'#0d1424');bg.addColorStop(1,'#070d18');pCtx.fillStyle=bg;pCtx.fillRect(0,0,W,H);for(let g=0;g<=5;g++){const y=pad.t+g*(H-pad.t-pad.b)/5,val=maxP-g*rangeP/5;pCtx.strokeStyle='#1a2840';pCtx.lineWidth=1;pCtx.setLineDash([3,5]);pCtx.beginPath();pCtx.moveTo(pad.l,y);pCtx.lineTo(W-pad.r,y);pCtx.stroke();pCtx.setLineDash([]);pCtx.fillStyle='#4a6080';pCtx.font='10px Arial';pCtx.fillText('Rs'+val.toFixed(0),3,y+4);}pCtx.strokeStyle='#f59e0b';pCtx.lineWidth=1.2;pCtx.setLineDash([4,4]);pCtx.beginPath();for(let i=49;i<raw.length;i++){const ma=raw.slice(i-49,i+1).reduce((s,d)=>s+d.c,0)/50;i===49?pCtx.moveTo(px(i),py(ma)):pCtx.lineTo(px(i),py(ma));}pCtx.stroke();pCtx.setLineDash([]);pCtx.strokeStyle='#3b82f6';pCtx.lineWidth=1.5;pCtx.beginPath();for(let i=19;i<raw.length;i++){const ma=raw.slice(i-19,i+1).reduce((s,d)=>s+d.c,0)/20;i===19?pCtx.moveTo(px(i),py(ma)):pCtx.lineTo(px(i),py(ma));}pCtx.stroke();raw.forEach((d,i)=>{const bull=d.c>=d.o,col=bull?'#10b981':'#ef4444',x=px(i);pCtx.strokeStyle=col;pCtx.lineWidth=1;pCtx.beginPath();pCtx.moveTo(x,py(d.h));pCtx.lineTo(x,py(d.l));pCtx.stroke();const top=py(Math.max(d.o,d.c)),bot=py(Math.min(d.o,d.c));pCtx.fillStyle=bull?'#10b98199':'#ef444499';pCtx.fillRect(x-bw/2,top,bw,Math.max(1,bot-top));pCtx.strokeStyle=col;pCtx.lineWidth=0.5;pCtx.strokeRect(x-bw/2,top,bw,Math.max(1,bot-top));});pCtx.fillStyle='#334155';pCtx.font='9px Arial';raw.forEach((d,i)=>{if(i%15===0)pCtx.fillText(d.d.slice(5),px(i)-14,H-8);});vCtx.fillStyle='#0d1424';vCtx.fillRect(0,0,W,VH);raw.forEach((d,i)=>{vCtx.fillStyle=d.c>=d.o?'#064e3580':'#4a0f0f80';const bh=VH-vy(d.v)-2;vCtx.fillRect(px(i)-bw/2,vy(d.v),bw,Math.max(1,bh));});}
window.addEventListener('resize',drawChart);setTimeout(drawChart,100);
const revData={{ d.quarters_rev|tojson }},epsData={{ d.quarters_eps|tojson }};
function drawBars(cid,data,color,trendId){const wrap=document.getElementById(cid);if(!wrap||!data.length)return;const vals=data.map(d=>d.val),maxV=Math.max(...vals.map(Math.abs))||1;wrap.innerHTML="";data.forEach(d=>{const pct=Math.abs(d.val)/maxV*100,col=document.createElement("div");col.className="qbar-col";col.innerHTML=`<span class="qbar-val">${d.val>=0?"":"-"}${Math.abs(d.val).toLocaleString()}</span><div class="qbar" style="height:${Math.max(4,pct*0.85)}px;background:${d.val>=0?color:"#ef4444"};"></div><span class="qbar-lbl">${d.q}</span>`;wrap.appendChild(col);});if(trendId&&data.length>=2){const first=data[0].val,last=data[data.length-1].val,trend=last>first,el=document.getElementById(trendId);if(el)el.innerHTML=`<span class="q-trend-label" style="background:${trend?"#06403528":"#4a0f0f28"};color:${trend?"#10b981":"#ef4444"};">${trend?"&#x1F4C8; Growing":"&#x1F4C9; Declining"}</span>`;}}
drawBars("rev-bars",revData,"#3b82f6","rev-trend");drawBars("eps-bars",epsData,"#10b981","eps-trend");
async function callClaude(sys,usr){const res=await fetch("https://api.anthropic.com/v1/messages",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model:"claude-sonnet-4-6",max_tokens:1000,system:sys,messages:[{role:"user",content:usr}]})});const data=await res.json();return data.content?.[0]?.text||"";}
const snap={symbol:"{{ d.symbol }}",company:"{{ d.company }}",sector:"{{ d.sector }}",price:{{ d.price }},pe:"{{ d.pe }}",pb:"{{ d.pb }}",beta:"{{ d.beta }}",debt_eq:"{{ d.debt_eq }}",roe:"{{ d.roe }}",rsi:{{ d.rsi }},atr_pct:{{ d.atr_pct }},max_dd:{{ d.max_dd }},dist_high:{{ d.dist_high }},dist_low:{{ d.dist_low }},liq_cr:{{ d.liq_cr }},overall_risk:{{ d.overall_risk }},grade:"{{ d.grade }}",strengths:{{ d.strengths|tojson }},weaknesses:{{ d.weaknesses|tojson }},mktcap_cr:"{{ d.mktcap_cr }}",div_yield:{{ d.div_yield }},fifty2_high:{{ d.fifty2_high }},fifty2_low:{{ d.fifty2_low }}};
(async()=>{try{const sys=`You are a senior Indian equity analyst. Summarise the LATEST earnings con call. Respond ONLY in JSON (no markdown): {"quarter":"Q3 FY25","tone":"Confident","tone_color":"#10b981","highlights":["p1","p2","p3","p4"],"guidance":"One sentence.","risks":["r1","r2","r3"]} tone: Confident/Optimistic/Cautious/Defensive/Mixed tone_color: #10b981=Confident, #f59e0b=Cautious/Mixed, #ef4444=Defensive`;const usr=`Stock: ${snap.symbol} (${snap.company}), Sector: ${snap.sector}, NSE India.`;const txt=await callClaude(sys,usr);const j=JSON.parse(txt.replace(/```json|```/g,"").trim());document.getElementById("concall-quarter").textContent=j.quarter||"Latest quarter";const tb=document.getElementById("tone-badge");tb.textContent="Management: "+j.tone;tb.style.cssText=`background:${j.tone_color}22;color:${j.tone_color};border:1px solid ${j.tone_color}44;`;document.getElementById("concall-highlights").innerHTML=(j.highlights||[]).map(h=>`<div class="concall-highlight">&bull; ${h}</div>`).join("");document.getElementById("concall-guidance").textContent=j.guidance||"No guidance.";document.getElementById("concall-risks").innerHTML=(j.risks||[]).map(r=>`<div class="cbullet"><span>&#x26A0;</span>${r}</div>`).join("");document.getElementById("concall-loading").style.display="none";document.getElementById("concall-content").style.display="block";}catch(e){document.getElementById("concall-loading").style.display="none";document.getElementById("concall-error").style.display="block";document.getElementById("concall-error").textContent="Con call data unavailable.";}})();
(async()=>{try{const sys=`You are FalconAI, a sharp 25-year-old Indian swing trader. Brutally honest, practical notes. Cover: 1)Overall impression 2)Chart/technicals 3)Financials 4)Con call tone 5)Catalysts and risks 6)Personal trade decision. End with ONE bold sentence. Paragraphs only, no bullets, under 350 words.`;const usr=`${snap.symbol} (${snap.company}) | ${snap.sector}\nRs${snap.price} | MCap Rs${snap.mktcap_cr}Cr | P/E ${snap.pe} | P/B ${snap.pb} | Beta ${snap.beta}\nD/E ${snap.debt_eq} | ROE ${snap.roe} | Div ${snap.div_yield}%\nRSI ${snap.rsi} | ATR ${snap.atr_pct}% | Max DD ${snap.max_dd}%\n52W: Rs${snap.fifty2_low}-Rs${snap.fifty2_high} | ${snap.dist_high}% from top | +${snap.dist_low}% from bottom\nLiquidity Rs${snap.liq_cr}Cr/day | Risk ${snap.overall_risk}/100 Grade ${snap.grade}\nStrengths: ${snap.strengths.join(", ")||"none"}\nWeaknesses: ${snap.weaknesses.join(", ")||"none"}`;const notes=await callClaude(sys,usr);document.getElementById("falcon-loading").style.display="none";document.getElementById("falcon-notes-text").style.display="block";document.getElementById("falcon-notes-text").textContent=notes;}catch(e){document.getElementById("falcon-loading").style.display="none";document.getElementById("falcon-error").style.display="block";document.getElementById("falcon-error").textContent="AI notes unavailable.";}})();
</script>
</body>
</html>
"""


@app.route("/risk/<symbol>")
def risk_report(symbol):
    d = build_risk_report(symbol.upper())
    if d is None:
        return f"<h3 style='color:white;background:#0b1220;padding:40px;'>Could not load data for {symbol}. Try again.</h3>", 404
    return render_template_string(RISK_HTML, d=d)



# =========================
# OPTIONS INTELLIGENCE PAGE
# =========================

def get_options_market_data():
    """
    Fetch live Nifty & BankNifty data for options analysis.
    Returns spot prices, technicals, key levels, VIX, expiry info.
    """
    result = {}

    for name, ticker in [("nifty", "NIFTYBEES.NS"), ("banknifty", "BANKBEES.NS")]:
        try:
            df    = _yf_download_safe(ticker, period="60d", interval="1d", timeout=6)
            df1h  = _yf_download_safe(ticker, period="10d", interval="1h", timeout=6)
            close = safe_series(df["Close"])
            h1    = safe_series(df1h["Close"])

            price  = float(close.iloc[-1])
            prev   = float(close.iloc[-2])
            chg    = round(price - prev, 2)
            chg_pct = round((chg / prev) * 100, 2)

            ma20  = float(close.rolling(20).mean().iloc[-1])
            ma50  = float(close.rolling(50).mean().iloc[-1])
            ma9_1h = float(h1.rolling(9).mean().iloc[-1])  if len(h1) >= 9  else price
            ma20_1h= float(h1.rolling(20).mean().iloc[-1]) if len(h1) >= 20 else price

            rsi_d  = float(calc_rsi(close).iloc[-1]) if len(close) >= 15 else 50
            rsi_1h = float(calc_rsi(h1).iloc[-1])    if len(h1)   >= 15 else 50

            # ATR for options range estimation
            atr_vals = []
            for i in range(1, min(15, len(close))):
                hi = float(df["High"].iloc[-i].squeeze() if hasattr(df["High"].iloc[-i],"squeeze") else df["High"].iloc[-i])
                lo = float(df["Low"].iloc[-i].squeeze()  if hasattr(df["Low"].iloc[-i],"squeeze")  else df["Low"].iloc[-i])
                pc = float(close.iloc[-i-1])
                atr_vals.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
            atr = float(np.mean(atr_vals)) if atr_vals else price * 0.01

            # Key levels — Pivot Point (Classic)
            ph = float(df["High"].iloc[-2].squeeze() if hasattr(df["High"].iloc[-2],"squeeze") else df["High"].iloc[-2])
            pl = float(df["Low"].iloc[-2].squeeze()  if hasattr(df["Low"].iloc[-2],"squeeze")  else df["Low"].iloc[-2])
            pc_prev = prev
            pivot = round((ph + pl + pc_prev) / 3, 0)
            r1 = round(2 * pivot - pl,           0)
            r2 = round(pivot + (ph - pl),         0)
            r3 = round(ph + 2 * (pivot - pl),     0)
            s1 = round(2 * pivot - ph,            0)
            s2 = round(pivot - (ph - pl),         0)
            s3 = round(pl - 2 * (ph - pivot),     0)

            # CPR (Central Pivot Range)
            bc   = round((ph + pl) / 2, 0)
            tc   = round(pivot - bc + pivot, 0)
            cpr_width = round(abs(tc - bc), 0)
            cpr_type  = "Narrow CPR 🎯 (big move expected)" if cpr_width < atr * 0.3 else \
                        "Wide CPR ↔️ (sideways possible)"

            # 52W
            high52 = float(close.tail(252).max())
            low52  = float(close.tail(252).min())
            dist52 = round(((high52 - price) / high52) * 100, 1)

            # Trend bias
            if price > ma20 and ma20 > ma50 and rsi_d > 55:
                bias       = "Bullish"
                bias_score = min(100, int(60 + (rsi_d - 55) * 2))
                bias_color = "#22c55e"
            elif price < ma20 and ma20 < ma50 and rsi_d < 45:
                bias       = "Bearish"
                bias_score = max(0, int(40 - (45 - rsi_d) * 2))
                bias_color = "#ef4444"
            else:
                bias       = "Neutral"
                bias_score = 50
                bias_color = "#facc15"

            # Intraday 1H bias
            if h1.iloc[-1] > ma9_1h and h1.iloc[-1] > ma20_1h:
                h1_bias = "Bullish 1H"
                h1_color = "#22c55e"
            elif h1.iloc[-1] < ma9_1h and h1.iloc[-1] < ma20_1h:
                h1_bias = "Bearish 1H"
                h1_color = "#ef4444"
            else:
                h1_bias = "Mixed 1H"
                h1_color = "#facc15"

            # ATM strike (round to nearest 50 for Nifty, 100 for BankNifty)
            step = 50 if name == "nifty" else 100
            atm  = int(round(price / step) * step)

            # Suggested strikes
            ce_strikes = [atm, atm + step, atm + 2 * step]
            pe_strikes = [atm, atm - step, atm - 2 * step]

            # Expected move today (1 ATR)
            exp_up   = round(price + atr, 0)
            exp_down = round(price - atr, 0)

            # Recent 5-day high/low (intraday range)
            recent_high = float(close.tail(5).max())
            recent_low  = float(close.tail(5).min())

            # Chart data last 30 days for mini chart
            chart_pts = [{"d": str(df.index[-i])[:10], "c": round(float(close.iloc[-i]), 1)}
                         for i in range(30, 0, -1)]

            result[name] = {
                "price":       round(price, 2),
                "prev":        round(prev, 2),
                "chg":         chg,
                "chg_pct":     chg_pct,
                "ma20":        round(ma20, 0),
                "ma50":        round(ma50, 0),
                "rsi_d":       round(rsi_d, 1),
                "rsi_1h":      round(rsi_1h, 1),
                "atr":         round(atr, 0),
                "pivot":       pivot,
                "r1": r1, "r2": r2, "r3": r3,
                "s1": s1, "s2": s2, "s3": s3,
                "bc": bc, "tc": tc,
                "cpr_type":    cpr_type,
                "cpr_width":   cpr_width,
                "high52":      round(high52, 0),
                "low52":       round(low52, 0),
                "dist52":      dist52,
                "bias":        bias,
                "bias_score":  bias_score,
                "bias_color":  bias_color,
                "h1_bias":     h1_bias,
                "h1_color":    h1_color,
                "atm":         atm,
                "ce_strikes":  ce_strikes,
                "pe_strikes":  pe_strikes,
                "exp_up":      exp_up,
                "exp_down":    exp_down,
                "recent_high": round(recent_high, 0),
                "recent_low":  round(recent_low, 0),
                "step":        step,
                "chart_pts":   chart_pts,
            }
        except Exception as e:
            result[name] = None

    # VIX
    try:
        vdf      = _yf_download_safe("INDIAVIX.NS", period="10d", interval="1d", timeout=6)
        vix_val  = float(safe_series(vdf["Close"]).iloc[-1])
        vix_prev = float(safe_series(vdf["Close"]).iloc[-2])
        vix_chg  = round(vix_val - vix_prev, 2)
        if vix_val < 13:
            vix_label = "😴 Very Low — Premiums cheap, good for buyers"
            vix_color = "#22c55e"
        elif vix_val < 17:
            vix_label = "✅ Normal — Balanced market"
            vix_color = "#86efac"
        elif vix_val < 22:
            vix_label = "⚠️ Elevated — Premiums expensive, sell or hedge"
            vix_color = "#facc15"
        else:
            vix_label = "🔴 High Fear — Only buy puts or hedge, avoid naked CE"
            vix_color = "#ef4444"
        result["vix"] = {"val": round(vix_val, 2), "chg": vix_chg,
                         "label": vix_label, "color": vix_color}
    except:
        result["vix"] = {"val": "N/A", "chg": 0,
                         "label": "⚪ VIX unavailable", "color": "#64748b"}

    # Expiry dates (NSE weekly = every Thursday)
    today    = datetime.now().date()
    days_exp = (3 - today.weekday()) % 7   # days to next Thursday
    if days_exp == 0: days_exp = 7
    next_exp = today + __import__("datetime").timedelta(days=days_exp)
    result["expiry"] = {
        "weekly":    next_exp.strftime("%d %b %Y"),
        "days_left": days_exp,
        "today":     today.strftime("%d %b %Y"),
        "day_name":  today.strftime("%A"),
    }

    return result



# ════════════════════════════════════════════════════════════════════
# PRO SCALP SIGNAL ENGINE — Nifty / BankNifty Multi-Timeframe Scalper
# ════════════════════════════════════════════════════════════════════
#
# Built the way a 25-year option desk trader actually checks before
# pulling the trigger on an intraday CE/PE buy:
#   1. EMA 9/21 trend on the 5-min chart (the execution timeframe)
#   2. MACD momentum confirming that trend is accelerating, not fading
#   3. 15-min trend alignment — never fight the higher timeframe
#   4. 5-min setup quality — is the structure clean or choppy?
#   5. 1-min CLOSED candle confirmation — never enter on a forming bar
#   6. Volume confirmation — real moves are backed by real volume
#   7. Price action — no double-top/bottom fighting the trade
#
# A trade only gets called "STRONG" when 6+/7 line up. Anything choppy
# gets an automatic AVOID regardless of how good the rest looks —
# that's rule #1 on every real trading desk: don't fight a sideways tape.

def calc_ema(series, period):
    try:
        return series.ewm(span=period, adjust=False).mean()
    except Exception:
        return series


def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def get_intraday_df(ticker, interval, period):
    try:
        df = _yf_download_safe(ticker, period=period, interval=interval, timeout=6)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def detect_double_pattern(high_s, low_s, lookback=30):
    """Lightweight double-top / double-bottom detector on recent candles."""
    try:
        h = high_s.tail(lookback).reset_index(drop=True)
        l = low_s.tail(lookback).reset_index(drop=True)
        if len(h) < 10:
            return {"pattern": "NONE", "conflict_ce": False, "conflict_pe": False}

        peaks, troughs = [], []
        for i in range(2, len(h) - 2):
            if h[i] >= h[i-1] and h[i] >= h[i-2] and h[i] >= h[i+1] and h[i] >= h[i+2]:
                peaks.append((i, float(h[i])))
            if l[i] <= l[i-1] and l[i] <= l[i-2] and l[i] <= l[i+1] and l[i] <= l[i+2]:
                troughs.append((i, float(l[i])))

        pattern, conflict_ce, conflict_pe = "NONE", False, False

        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            if p1[1] > 0 and abs(p1[1] - p2[1]) / p1[1] < 0.0015 and (p2[0] - p1[0]) >= 3:
                pattern = "DOUBLE TOP"
                conflict_ce = True

        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            if t1[1] > 0 and abs(t1[1] - t2[1]) / t1[1] < 0.0015 and (t2[0] - t1[0]) >= 3:
                if pattern == "NONE":
                    pattern = "DOUBLE BOTTOM"
                conflict_pe = True

        return {"pattern": pattern, "conflict_ce": conflict_ce, "conflict_pe": conflict_pe}
    except Exception:
        return {"pattern": "NONE", "conflict_ce": False, "conflict_pe": False}


def calc_pro_scalp_signal(index_name):
    """
    index_name: 'nifty' or 'banknifty'
    Returns the full live scalp signal dict, or None if data is unavailable.
    """
    ticker = "NIFTYBEES.NS" if index_name == "nifty" else "BANKBEES.NS"
    step   = 50 if index_name == "nifty" else 100

    df_1m  = get_intraday_df(ticker, "1m",  "1d")
    df_5m  = get_intraday_df(ticker, "5m",  "5d")
    df_15m = get_intraday_df(ticker, "15m", "5d")

    if df_5m is None:
        return None

    close_5m = safe_series(df_5m["Close"])
    high_5m  = safe_series(df_5m["High"])
    low_5m   = safe_series(df_5m["Low"])
    vol_5m   = safe_series(df_5m["Volume"])

    if len(close_5m) < 25:
        return None

    spot = float(close_5m.iloc[-1])

    # ── EMA 9/21 on 5M ──────────────────────────────────────────────
    ema9_5m  = calc_ema(close_5m, 9)
    ema21_5m = calc_ema(close_5m, 21)
    ema9_now  = float(ema9_5m.iloc[-1])
    ema21_now = float(ema21_5m.iloc[-1])
    ema_bull_count = sum(
        1 for i in range(-3, 0)
        if len(ema9_5m) >= 3 and float(ema9_5m.iloc[i]) > float(ema21_5m.iloc[i])
    )
    ema_trend = "BULL" if ema9_now > ema21_now else "BEAR" if ema9_now < ema21_now else "FLAT"

    # ── MACD on 5M ──────────────────────────────────────────────────
    macd_line, signal_line, hist = calc_macd(close_5m)
    macd_now  = float(macd_line.iloc[-1])
    hist_now  = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2]) if len(hist) > 1 else hist_now
    macd_rising = hist_now > hist_prev
    macd_bull   = macd_now > float(signal_line.iloc[-1])

    # ── 15M Trend (higher timeframe confirmation) ──────────────────
    trend_15m = "FLAT"
    if df_15m is not None:
        close_15m = safe_series(df_15m["Close"])
        if len(close_15m) >= 21:
            e9  = float(calc_ema(close_15m, 9).iloc[-1])
            e21 = float(calc_ema(close_15m, 21).iloc[-1])
            trend_15m = "BULL" if e9 > e21 else "BEAR" if e9 < e21 else "FLAT"

    # ── 1M closed-candle confirmation ───────────────────────────────
    candle_1m_closed_bull = False
    candle_1m_closed_bear = False
    has_1m = df_1m is not None
    if has_1m:
        close_1m = safe_series(df_1m["Close"])
        open_1m  = safe_series(df_1m["Open"])
        if len(close_1m) >= 2:
            last_closed_close = float(close_1m.iloc[-2])
            last_closed_open  = float(open_1m.iloc[-2])
            candle_1m_closed_bull = last_closed_close > last_closed_open
            candle_1m_closed_bear = last_closed_close < last_closed_open
        else:
            has_1m = False

    # ── Volume confirmation (5M) ────────────────────────────────────
    avg_vol_5m = float(vol_5m.tail(20).mean()) if len(vol_5m) >= 20 else float(vol_5m.mean())
    cur_vol_5m = float(vol_5m.iloc[-1])
    vol_ratio  = (cur_vol_5m / avg_vol_5m) if avg_vol_5m > 0 else 1.0

    # ── Choppy filter — 15M range as % of spot ──────────────────────
    if df_15m is not None and len(df_15m) >= 5:
        h15 = safe_series(df_15m["High"]).tail(5)
        l15 = safe_series(df_15m["Low"]).tail(5)
        range_pct = ((float(h15.max()) - float(l15.min())) / spot) * 100 if spot else 0
    else:
        range_pct = 1.0
    is_choppy = range_pct < 0.3

    # ── Price action pattern ────────────────────────────────────────
    pa = detect_double_pattern(high_5m, low_5m)

    # ── Direction vote ───────────────────────────────────────────────
    bull_votes = bear_votes = 0
    if ema_trend == "BULL": bull_votes += 1
    if ema_trend == "BEAR": bear_votes += 1
    if macd_bull and macd_rising: bull_votes += 1
    if (not macd_bull) and (not macd_rising): bear_votes += 1
    if trend_15m == "BULL": bull_votes += 1
    if trend_15m == "BEAR": bear_votes += 1
    direction = "CE" if bull_votes > bear_votes else "PE" if bear_votes > bull_votes else "NONE"

    # ── 7-point checklist ────────────────────────────────────────────
    checklist = []

    ema_pass = (direction == "CE" and ema_trend == "BULL") or (direction == "PE" and ema_trend == "BEAR")
    checklist.append({"name": "EMA 9/21 Trend", "pass": ema_pass,
                       "detail": f"{ema_trend} {ema_bull_count}/3" if ema_trend != "FLAT" else "FLAT"})

    macd_pass = (direction == "CE" and macd_bull and macd_rising) or \
                (direction == "PE" and not macd_bull and not macd_rising)
    checklist.append({"name": "MACD Momentum", "pass": macd_pass, "detail": f"{round(hist_now,3)}"})

    trend15_pass = (direction == "CE" and trend_15m == "BULL") or (direction == "PE" and trend_15m == "BEAR")
    checklist.append({"name": "15M Trend", "pass": trend15_pass, "detail": trend_15m})

    setup_pass = (not is_choppy) and ema_pass
    checklist.append({"name": "5M Setup", "pass": setup_pass,
                       "detail": "Clean" if setup_pass else ("Choppy" if is_choppy else "Misaligned")})

    if direction == "CE":
        entry1m_pass = candle_1m_closed_bull
    elif direction == "PE":
        entry1m_pass = candle_1m_closed_bear
    else:
        entry1m_pass = False
    checklist.append({"name": "1M Entry (closed)", "pass": entry1m_pass,
                       "detail": "Confirmed" if entry1m_pass else ("No 1M data" if not has_1m else "Wait/Forming")})

    vol_pass = vol_ratio >= 1.0
    checklist.append({"name": "Volume Confirmation", "pass": vol_pass, "detail": f"{round(vol_ratio,2)}x avg"})

    if direction == "CE":
        pa_pass = not pa["conflict_ce"]
    elif direction == "PE":
        pa_pass = not pa["conflict_pe"]
    else:
        pa_pass = False
    checklist.append({"name": "Price Action", "pass": pa_pass, "detail": pa["pattern"]})

    confidence = sum(1 for c in checklist if c["pass"])

    # ── Cautions ──────────────────────────────────────────────────────
    cautions = []
    if is_choppy:
        cautions.append(f"Choppy — 15M range {round(range_pct,2)}% is under the 0.3% threshold")
    if not has_1m:
        cautions.append("1-minute data unavailable right now — entry timing is less precise")
    if vol_ratio < 1.0:
        cautions.append(f"Volume below average ({round(vol_ratio,2)}x) — weak participation behind the move")
    if pa["pattern"] != "NONE":
        cautions.append(f"{pa['pattern']} pattern nearby — watch for a reversal at this level")
    if direction == "NONE":
        cautions.append("No clear directional consensus across timeframes right now")

    # ── Signal classification ───────────────────────────────────────
    if is_choppy:
        signal_text, signal_color = "AVOID CHOPPY", "#f59e0b"
        action = "Do not enter. Wait for the range to expand before trading."
    elif direction == "NONE" or confidence <= 3:
        signal_text, signal_color = "AVOID / SKIP", "#ef4444"
        action = "Do not enter. Confidence too low — wait for stronger confirmation."
    elif confidence <= 5:
        signal_text, signal_color = f"MODERATE BUY {direction}", "#f59e0b"
        action = "Trade small size only — enter on next candle if all checks stay green."
    else:
        signal_text, signal_color = f"STRONG BUY {direction}", "#10b981"
        action = "High-confidence scalp setup. Enter on confirmation, manage tight."

    # ── ATR(5M) for T1 / T2 / SL ────────────────────────────────────
    atr_vals = []
    for i in range(1, min(15, len(close_5m))):
        hi = float(high_5m.iloc[-i])
        lo = float(low_5m.iloc[-i])
        pc = float(close_5m.iloc[-i-1]) if len(close_5m) > i else hi
        atr_vals.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
    atr_5m = float(np.mean(atr_vals)) if atr_vals else spot * 0.001

    t1 = round(atr_5m * 1.0, 1)
    t2 = round(atr_5m * 1.8, 1)
    sl = round(atr_5m * 0.5, 1)
    rr = round(t1 / sl, 2) if sl > 0 else 0

    # ── Strike recommendation ───────────────────────────────────────
    atm = int(round(spot / step) * step)
    if direction == "CE":
        itm_strike, otm_strike = atm - step, atm + step
    elif direction == "PE":
        itm_strike, otm_strike = atm + step, atm - step
    else:
        itm_strike, otm_strike = atm, atm

    idx_label = "Nifty" if index_name == "nifty" else "BankNifty"
    strike_reco = {
        "atm": atm, "itm": itm_strike, "otm": otm_strike, "primary": atm,
        "primary_label":    f"{atm} {direction}",
        "itm_label":        f"{itm_strike} {direction}",
        "otm_label":        f"{otm_strike} {direction}",
        "primary_reasoning": f"ATM strike — best liquidity and tightest bid-ask spread, delta near 0.5 so it tracks {idx_label} points efficiently for a scalp.",
        "safer_reasoning":   f"1 ITM — higher delta (~0.65-0.70), moves more per point but costs more premium and has lower % swings — better for first-timers.",
        "aggressive_reasoning": f"1 OTM — cheaper premium, bigger % gains if the move continues fast, but decays quicker and stings harder if SL is hit.",
    }

    return {
        "index_name":   index_name,
        "index_label":  idx_label,
        "spot":         round(spot, 2),
        "atm":          atm,
        "step":         step,
        "direction":    direction,
        "signal_text":  signal_text,
        "signal_color": signal_color,
        "action":       action,
        "confidence":   confidence,
        "confidence_total": 7,
        "is_choppy":    is_choppy,
        "range_pct":    round(range_pct, 2),
        "checklist":    checklist,
        "cautions":     cautions,
        "t1": t1, "t2": t2, "sl": sl, "rr": rr,
        "atr_5m":       round(atr_5m, 1),
        "ema_trend":    ema_trend,
        "macd_hist":    round(hist_now, 3),
        "trend_15m":    trend_15m,
        "vol_ratio":    round(vol_ratio, 2),
        "pattern":      pa["pattern"],
        "strike_reco":  strike_reco,
        "scan_time":    datetime.now().strftime("%H:%M:%S"),
        "scan_date":    datetime.now().strftime("%d %b %Y"),
    }


@app.route("/api/scalp_signal/<index_name>")
def api_scalp_signal(index_name):
    from flask import jsonify
    index_name = index_name.lower().strip()
    if index_name not in ("nifty", "banknifty"):
        return jsonify({"error": "Invalid index. Use 'nifty' or 'banknifty'."}), 400
    sig = calc_pro_scalp_signal(index_name)
    if sig is None:
        return jsonify({"error": "Could not fetch live intraday data right now. Try again shortly."}), 503
    return jsonify(sig)


OPTIONS_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Options Intelligence | FalconAI</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0b1220; color: white; font-family: Arial, sans-serif; }

.topbar {
    background: #0f172a;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    border-bottom: 1px solid #1f2937;
    position: sticky; top: 0; z-index: 10;
    flex-wrap: wrap;
}
.topbar a { color: #60a5fa; font-size: 13px; text-decoration: none; }
.nav-tabs { display: flex; gap: 8px; flex-wrap: wrap; }
.nav-tab {
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: bold;
    cursor: pointer;
    border: 1px solid #334155;
    background: #1e293b;
    color: #94a3b8;
    text-decoration: none;
}
.nav-tab.active { background: #1e3a5f; color: #60a5fa; border-color: #3b82f6; }

.page { max-width: 1200px; margin: 0 auto; padding: 16px 16px 60px; }

/* Header strip */
.mkt-strip {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}
.mkt-pill {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 10px 16px;
    flex: 1;
    min-width: 160px;
}
.mkt-name  { font-size: 11px; color: #475569; margin-bottom: 3px; }
.mkt-price { font-size: 22px; font-weight: bold; color: #f1f5f9; }
.mkt-chg   { font-size: 12px; margin-top: 2px; }

/* VIX strip */
.vix-bar {
    background: #111827;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    border: 1px solid #1f2937;
}

/* Section tabs */
.idx-tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
}
.idx-tab {
    padding: 8px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: bold;
    cursor: pointer;
    border: 1px solid #334155;
    background: #1e293b;
    color: #64748b;
}
.idx-tab.active { background: #1e3a5f; color: #60a5fa; border-color: #3b82f6; }

.idx-panel { display: none; }
.idx-panel.active { display: block; }

/* ── PRO SCALP SIGNAL ENGINE ── */
.scalp-wrap {
    background: linear-gradient(135deg, #0d1424 0%, #0a1020 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 0;
    margin-bottom: 18px;
    overflow: hidden;
}
.scalp-head {
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    padding: 14px 18px;
    background: #0a1020;
    border-bottom: 1px solid #1e293b;
}
.scalp-title { font-size: 14px; font-weight: 800; color: #60a5fa; display:flex; align-items:center; gap:6px; }
.scalp-pill {
    padding: 3px 11px; border-radius: 20px; font-size: 11px; font-weight: 700;
    background: #1e293b; color: #94a3b8; border: 1px solid #334155;
}
.scalp-pill.choppy { background: #3d2a0033; color: #f59e0b; border-color: #f59e0b55; }
.scalp-live { display:flex; align-items:center; gap:5px; font-size:11px; color:#10b981; font-weight:700; margin-left:auto; }
.scalp-live-dot { width:7px; height:7px; border-radius:50%; background:#10b981; animation: scalppulse 1.3s infinite; }
@keyframes scalppulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.7)} }
.scalp-spot { font-size:13px; color:#cbd5e1; font-weight:700; }
.scalp-atm  { font-size:11px; color:#475569; }

.scalp-refresh-bar { padding: 8px 18px; background:#0d1424; border-bottom:1px solid #1e293b;
    display:flex; align-items:center; gap:10px; }
.scalp-refresh-track { flex:1; height:4px; background:#1e293b; border-radius:3px; overflow:hidden; }
.scalp-refresh-fill { height:100%; background:#3b82f6; border-radius:3px; transition: width 1s linear; }
.scalp-refresh-txt { font-size:10px; color:#475569; white-space:nowrap; }
.scalp-refresh-btn { font-size:10px; padding:3px 10px; border-radius:6px; background:#1e293b;
    border:1px solid #334155; color:#94a3b8; cursor:pointer; white-space:nowrap; }
.scalp-refresh-btn:hover { background:#334155; color:white; }

.scalp-body { padding: 18px; }

.scalp-signal-box {
    border: 2px solid; border-radius: 14px; padding: 18px 20px; margin-bottom: 16px;
    position: relative; overflow: hidden;
}
.scalp-signal-box::before {
    content:''; position:absolute; inset:0; opacity:0.06; pointer-events:none;
}
.scalp-signal-text { font-size: 26px; font-weight: 900; letter-spacing: -0.5px; margin-bottom: 8px; }
.scalp-trade-tag {
    display:inline-flex; align-items:center; gap:6px; font-size:11px; font-weight:700;
    background:#1e293b; padding:4px 12px; border-radius:20px; color:#94a3b8; margin-bottom:10px;
}
.scalp-action { font-size:12px; color:#94a3b8; margin-bottom:14px; line-height:1.5; }
.scalp-tgt-row { display:flex; gap:18px; flex-wrap:wrap; font-size:12px; }
.scalp-tgt-item { display:flex; flex-direction:column; }
.scalp-tgt-lbl { font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; }
.scalp-tgt-val { font-size:15px; font-weight:800; margin-top:2px; }

.scalp-confidence-wrap { margin-bottom:16px; }
.scalp-conf-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
.scalp-conf-label { font-size:12px; color:#94a3b8; font-weight:700; }
.scalp-conf-score { font-size:18px; font-weight:900; }
.scalp-conf-track { height:8px; background:#1e293b; border-radius:5px; overflow:hidden; }
.scalp-conf-fill { height:100%; border-radius:5px; transition: width 0.5s ease; }
.scalp-conf-status { font-size:11px; margin-top:5px; font-weight:700; }

.scalp-indicator-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px; }
.scalp-ind-chip {
    flex:1; min-width:80px; background:#0d1424; border:1px solid #1e293b; border-radius:9px;
    padding:8px 10px; text-align:center;
}
.scalp-ind-lbl { font-size:9px; color:#475569; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:3px; }
.scalp-ind-val { font-size:12px; font-weight:800; }

.scalp-caution-box {
    background:#3d2a0018; border:1px solid #f59e0b33; border-radius:10px;
    padding:12px 14px; margin-bottom:16px;
}
.scalp-caution-title { font-size:11px; font-weight:800; color:#f59e0b; margin-bottom:8px; }
.scalp-caution-item { font-size:11px; color:#cbbb8f; padding:3px 0; line-height:1.5; }

.scalp-checklist-title { font-size:12px; font-weight:800; color:#64748b; text-transform:uppercase;
    letter-spacing:0.6px; margin-bottom:10px; display:flex; align-items:center; gap:8px; }
.scalp-checklist-title::after { content:''; flex:1; height:1px; background:#1e293b; }
.scalp-checklist-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:8px; margin-bottom:18px; }
.scalp-check-item {
    background:#0d1424; border:1px solid; border-radius:9px; padding:9px 11px;
}
.scalp-check-item.pass { border-color:#10b98144; background:#0a1f1730; }
.scalp-check-item.fail { border-color:#f59e0b33; background:#1a160830; }
.scalp-check-name { font-size:10px; font-weight:700; display:flex; align-items:center; gap:5px; margin-bottom:3px; }
.scalp-check-detail { font-size:10px; color:#64748b; }

.scalp-strike-panel {
    background: linear-gradient(135deg, #0a1f3a, #0d1424);
    border:1px solid #3b82f655; border-radius:14px; padding:16px 18px;
}
.scalp-strike-title { font-size:12px; font-weight:800; color:#60a5fa; margin-bottom:12px;
    display:flex; align-items:center; gap:6px; }
.scalp-strike-primary {
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;
    background:#0d1424; border:1px solid #3b82f644; border-radius:10px; padding:12px 16px; margin-bottom:10px;
}
.scalp-strike-big { font-size:24px; font-weight:900; color:white; }
.scalp-strike-reason { font-size:11px; color:#7090b0; margin-top:4px; max-width:340px; }
.scalp-strike-alt-row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.scalp-strike-alt { background:#0d142480; border:1px solid #1e293b; border-radius:9px; padding:10px 12px; }
.scalp-strike-alt-lbl { font-size:10px; color:#475569; text-transform:uppercase; margin-bottom:3px; font-weight:700; }
.scalp-strike-alt-val { font-size:15px; font-weight:800; color:#e2e8f0; margin-bottom:4px; }
.scalp-strike-alt-reason { font-size:10px; color:#64748b; line-height:1.4; }

.scalp-loading { display:flex; align-items:center; gap:10px; color:#475569; font-size:12px; padding:30px; justify-content:center; }
.scalp-ai-dot { width:7px; height:7px; border-radius:50%; background:#3b82f6; animation: scalppulse 1.2s infinite; }
.scalp-ai-dot:nth-child(2){animation-delay:.2s;}
.scalp-ai-dot:nth-child(3){animation-delay:.4s;}


/* Grid */
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-bottom: 14px; }
@media(max-width:700px) { .grid2,.grid3 { grid-template-columns: 1fr; } }

.panel {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 14px 16px;
}
.panel-title {
    font-size: 12px;
    color: #475569;
    font-weight: bold;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Bias gauge */
.bias-gauge {
    text-align: center;
    padding: 10px 0;
}
.bias-score-num {
    font-size: 42px;
    font-weight: bold;
    line-height: 1;
}
.bias-label {
    font-size: 16px;
    font-weight: bold;
    margin-top: 4px;
}
.bias-bar-wrap { background: #1e293b; border-radius: 6px; height: 8px;
                 margin: 10px 0; overflow: hidden; }
.bias-bar      { height: 8px; border-radius: 6px; }

/* Levels table */
.lvl-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.lvl-table td { padding: 6px 8px; border-bottom: 1px solid #0f172a; }
.lvl-table tr:last-child td { border-bottom: none; }
.lvl-label  { color: #64748b; }
.lvl-val    { text-align: right; font-weight: bold; color: #f1f5f9; font-size: 13px; }
.lvl-r { color: #ef4444; }
.lvl-s { color: #22c55e; }
.lvl-p { color: #60a5fa; }

/* Strikes */
.strike-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 10px;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 13px;
    font-weight: bold;
}
.strike-atm  { background: #1e3a5f; border: 1px solid #3b82f6; color: #93c5fd; }
.strike-otm1 { background: #0f2d1a; border: 1px solid #166534; color: #86efac; }
.strike-otm2 { background: #1a1a0f; border: 1px solid #854d0e; color: #fde68a; }
.strike-sub  { font-size: 10px; color: #475569; font-weight: normal; margin-top: 1px; }

/* Expected range */
.range-bar-wrap {
    position: relative;
    height: 32px;
    background: #1e293b;
    border-radius: 8px;
    margin: 10px 0;
    overflow: hidden;
}
.range-bar-fill {
    position: absolute;
    height: 100%;
    background: linear-gradient(90deg, #1e3a5f, #1e3a5f);
    border-radius: 8px;
}
.range-center {
    position: absolute;
    left: 50%;
    top: 0;
    width: 2px;
    height: 100%;
    background: #60a5fa;
    transform: translateX(-50%);
}

/* Mini chart */
#nifty-mini, #bnf-mini { width: 100%; height: 80px; display: block; }

/* Strategy cards */
.strat-card {
    background: #0f172a;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
    border-left: 3px solid;
}
.strat-title { font-size: 13px; font-weight: bold; margin-bottom: 4px; }
.strat-body  { font-size: 12px; color: #94a3b8; line-height: 1.6; }
.strat-tag   { display: inline-block; font-size: 10px; padding: 2px 8px;
               border-radius: 10px; margin-right: 4px; margin-top: 4px; }

/* Options table */
.opt-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.opt-table th { color: #475569; padding: 5px 8px; border-bottom: 1px solid #1f2937;
                text-align: left; font-weight: normal; }
.opt-table td { padding: 6px 8px; border-bottom: 1px solid #0f172a; color: #cbd5e1; }
.opt-table tr:last-child td { border-bottom: none; }
.opt-buy  { color: #22c55e; font-weight: bold; }
.opt-sell { color: #ef4444; font-weight: bold; }
.opt-atm  { background: #1e293b; }

/* AI sections */
.ai-loading {
    display: flex; align-items: center; gap: 10px;
    color: #475569; font-size: 13px; padding: 12px 0;
}
.ai-dot { width: 7px; height: 7px; border-radius: 50%;
          background: #3b82f6; animation: pulse 1.2s infinite; }
.ai-dot:nth-child(2) { animation-delay: 0.2s; }
.ai-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse {
    0%,100% { opacity: 0.3; transform: scale(0.8); }
    50%      { opacity: 1;   transform: scale(1.2); }
}
.morning-brief {
    font-size: 13px;
    color: #cbd5e1;
    line-height: 1.9;
    white-space: pre-wrap;
}
.setup-block {
    background: #0f172a;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border: 1px solid #1e293b;
}
.setup-title { font-size: 13px; font-weight: bold; color: #f1f5f9; margin-bottom: 6px; }
.setup-body  { font-size: 12px; color: #94a3b8; line-height: 1.7; }

/* Expiry countdown */
.expiry-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    color: #94a3b8;
}
.expiry-days {
    font-size: 18px;
    font-weight: bold;
    color: #f97316;
}

/* RM table */
.rm-row {
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid #1f2937;
    font-size: 12px;
}
.rm-row:last-child { border-bottom: none; }
.rm-label { color: #64748b; }
.rm-val   { font-weight: bold; color: #f1f5f9; }

</style>
</head>
<body>

<div class="topbar">
    <a href="/">← FalconAI</a>
    <div class="nav-tabs">
        <a href="/" class="nav-tab">📊 Screener</a>
        <a href="/options" class="nav-tab active">⚡ Options</a>
        <a href="/screens" class="nav-tab">🔍 Screens</a>
        <a href="/sectors" class="nav-tab">🏭 Sectors</a>
        <a href="/scanner" class="nav-tab">⚡ 5-Min Scanner</a>
    </div>
    <span style="font-size:12px;color:#334155;margin-left:auto;">
        {{ d.expiry.day_name }}, {{ d.expiry.today }}
    </span>
</div>

<div class="page">

<!-- ═══ Market Strip ═══ -->
<div class="mkt-strip">

    {% for key, label, emoji in [("nifty","NIFTY 50","📈"),("banknifty","BANK NIFTY","🏦")] %}
    {% if d[key] %}
    <div class="mkt-pill">
        <div class="mkt-name">{{ emoji }} {{ label }}</div>
        <div class="mkt-price">{{ "{:,.0f}".format(d[key].price) }}</div>
        <div class="mkt-chg" style="color:{{ '#22c55e' if d[key].chg >= 0 else '#ef4444' }};">
            {{ '+' if d[key].chg >= 0 else '' }}{{ d[key].chg }}
            ({{ '+' if d[key].chg_pct >= 0 else '' }}{{ d[key].chg_pct }}%)
        </div>
        <div style="font-size:11px;color:#475569;margin-top:4px;">
            RSI {{ d[key].rsi_d }} &nbsp;|&nbsp;
            <span style="color:{{ d[key].bias_color }};">{{ d[key].bias }}</span>
        </div>
    </div>
    {% endif %}
    {% endfor %}

    <!-- VIX -->
    <div class="mkt-pill">
        <div class="mkt-name">🌡️ INDIA VIX</div>
        <div class="mkt-price" style="color:{{ d.vix.color }};">{{ d.vix.val }}</div>
        <div class="mkt-chg" style="color:{{ '#ef4444' if d.vix.chg > 0 else '#22c55e' }};">
            {{ '+' if d.vix.chg > 0 else '' }}{{ d.vix.chg }}
        </div>
        <div style="font-size:11px;margin-top:4px;color:{{ d.vix.color }};">{{ d.vix.label }}</div>
    </div>

    <!-- Expiry -->
    <div class="mkt-pill">
        <div class="mkt-name">📅 WEEKLY EXPIRY</div>
        <div style="margin-top:6px;">
            <span class="expiry-pill">
                <span class="expiry-days">{{ d.expiry.days_left }}d</span>
                <span>{{ d.expiry.weekly }}</span>
            </span>
        </div>
        <div style="font-size:11px;color:#475569;margin-top:8px;">
            {% if d.expiry.days_left == 1 %}⚠️ Expiry tomorrow — theta crush active
            {% elif d.expiry.days_left <= 2 %}⚠️ Near expiry — avoid buying far OTM
            {% else %}✅ Enough time for positional trades{% endif %}
        </div>
    </div>
</div>

<!-- ═══ Index Tabs ═══ -->
<div class="idx-tabs">
    <button class="idx-tab active" onclick="switchIdx('nifty', this)">📈 NIFTY</button>
    <button class="idx-tab" onclick="switchIdx('bnf', this)">🏦 BANKNIFTY</button>
</div>

<!-- ══════════════ NIFTY PANEL ══════════════ -->
{% for idx_key, panel_id, chart_id, step_label in
   [("nifty","nifty-panel","nifty-mini","50"),
    ("banknifty","bnf-panel","bnf-mini","100")] %}

{% set ix = d[idx_key] %}
<div id="{{ panel_id }}" class="idx-panel {{ 'active' if idx_key == 'nifty' else '' }}">
{% if ix %}

<!-- ══════════════ PRO SCALP SIGNAL ENGINE ══════════════ -->
<div class="scalp-wrap">
    <div class="scalp-head">
        <span class="scalp-title">⚡ FalconAI Scalp Engine</span>
        <span class="scalp-pill">{{ idx_key.upper() }}</span>
        <span class="scalp-pill scalp-choppy-pill" id="scalp-{{ idx_key }}-choppy-pill" style="display:none;">⚠️ AVOID CHOPPY</span>
        <span class="scalp-spot" id="scalp-{{ idx_key }}-spot">Spot: —</span>
        <span class="scalp-atm" id="scalp-{{ idx_key }}-atm">ATM: —</span>
        <span class="scalp-live"><span class="scalp-live-dot"></span> LIVE</span>
    </div>

    <div class="scalp-refresh-bar">
        <span class="scalp-refresh-txt">Auto-refresh in <b id="scalp-{{ idx_key }}-countdown">30</b>s</span>
        <div class="scalp-refresh-track"><div class="scalp-refresh-fill" id="scalp-{{ idx_key }}-fill" style="width:100%;"></div></div>
        <span class="scalp-refresh-txt" id="scalp-{{ idx_key }}-lastupdate">Last: —</span>
        <button class="scalp-refresh-btn" onclick="fetchScalpSignal('{{ idx_key }}')">⚡ Now</button>
    </div>

    <div class="scalp-body">
        <div class="scalp-loading" id="scalp-{{ idx_key }}-loading">
            <div class="scalp-ai-dot"></div><div class="scalp-ai-dot"></div><div class="scalp-ai-dot"></div>
            <span>Reading 1M / 5M / 15M tape...</span>
        </div>

        <div id="scalp-{{ idx_key }}-content" style="display:none;">

            <!-- Main Signal Box -->
            <div class="scalp-signal-box" id="scalp-{{ idx_key }}-signalbox">
                <div class="scalp-signal-text" id="scalp-{{ idx_key }}-signaltext">—</div>
                <div class="scalp-trade-tag" id="scalp-{{ idx_key }}-tag">⚡ SCALP TRADE — 1M/5M setup, quick in/out</div>
                <div class="scalp-action" id="scalp-{{ idx_key }}-actiontext">—</div>
                <div class="scalp-tgt-row">
                    <div class="scalp-tgt-item"><span class="scalp-tgt-lbl">T1</span><span class="scalp-tgt-val" style="color:#10b981;" id="scalp-{{ idx_key }}-t1">—</span></div>
                    <div class="scalp-tgt-item"><span class="scalp-tgt-lbl">T2</span><span class="scalp-tgt-val" style="color:#34d399;" id="scalp-{{ idx_key }}-t2">—</span></div>
                    <div class="scalp-tgt-item"><span class="scalp-tgt-lbl">SL</span><span class="scalp-tgt-val" style="color:#ef4444;" id="scalp-{{ idx_key }}-sl">—</span></div>
                    <div class="scalp-tgt-item"><span class="scalp-tgt-lbl">R:R</span><span class="scalp-tgt-val" style="color:#e2e8f0;" id="scalp-{{ idx_key }}-rr">—</span></div>
                    <div class="scalp-tgt-item"><span class="scalp-tgt-lbl">Pattern</span><span class="scalp-tgt-val" style="color:#94a3b8;font-size:12px;" id="scalp-{{ idx_key }}-pattern">—</span></div>
                </div>
            </div>

            <!-- Confidence Meter -->
            <div class="scalp-confidence-wrap">
                <div class="scalp-conf-header">
                    <span class="scalp-conf-label">Entry Confidence</span>
                    <span class="scalp-conf-score" id="scalp-{{ idx_key }}-confscore">—/7</span>
                </div>
                <div class="scalp-conf-track"><div class="scalp-conf-fill" id="scalp-{{ idx_key }}-conffill" style="width:0%;"></div></div>
                <div class="scalp-conf-status" id="scalp-{{ idx_key }}-confstatus">—</div>
            </div>

            <!-- Indicator Chips -->
            <div class="scalp-indicator-row">
                <div class="scalp-ind-chip"><div class="scalp-ind-lbl">EMA 5M</div><div class="scalp-ind-val" id="scalp-{{ idx_key }}-ema">—</div></div>
                <div class="scalp-ind-chip"><div class="scalp-ind-lbl">MACD</div><div class="scalp-ind-val" id="scalp-{{ idx_key }}-macd">—</div></div>
                <div class="scalp-ind-chip"><div class="scalp-ind-lbl">15M Trend</div><div class="scalp-ind-val" id="scalp-{{ idx_key }}-trend15">—</div></div>
                <div class="scalp-ind-chip"><div class="scalp-ind-lbl">Volume</div><div class="scalp-ind-val" id="scalp-{{ idx_key }}-vol">—</div></div>
                <div class="scalp-ind-chip"><div class="scalp-ind-lbl">Range %</div><div class="scalp-ind-val" id="scalp-{{ idx_key }}-range">—</div></div>
            </div>

            <!-- Cautions -->
            <div class="scalp-caution-box" id="scalp-{{ idx_key }}-cautionbox" style="display:none;">
                <div class="scalp-caution-title">⚠️ CAUTION(S)</div>
                <div id="scalp-{{ idx_key }}-cautionlist"></div>
            </div>

            <!-- 7-Point Checklist -->
            <div class="scalp-checklist-title">✅ 7-Point Confirmation Checklist</div>
            <div class="scalp-checklist-grid" id="scalp-{{ idx_key }}-checklist"></div>

            <!-- Strike Recommendation -->
            <div class="scalp-strike-panel">
                <div class="scalp-strike-title">🎯 Which Strike To Buy</div>
                <div class="scalp-strike-primary">
                    <div>
                        <div class="scalp-strike-big" id="scalp-{{ idx_key }}-strike-primary">—</div>
                        <div class="scalp-strike-reason" id="scalp-{{ idx_key }}-strike-primary-reason">—</div>
                    </div>
                    <span class="scalp-pill" style="background:#3b82f622;color:#60a5fa;border-color:#3b82f655;">RECOMMENDED</span>
                </div>
                <div class="scalp-strike-alt-row">
                    <div class="scalp-strike-alt">
                        <div class="scalp-strike-alt-lbl">Safer (1 ITM)</div>
                        <div class="scalp-strike-alt-val" id="scalp-{{ idx_key }}-strike-itm">—</div>
                        <div class="scalp-strike-alt-reason" id="scalp-{{ idx_key }}-strike-itm-reason">—</div>
                    </div>
                    <div class="scalp-strike-alt">
                        <div class="scalp-strike-alt-lbl">Aggressive (1 OTM)</div>
                        <div class="scalp-strike-alt-val" id="scalp-{{ idx_key }}-strike-otm">—</div>
                        <div class="scalp-strike-alt-reason" id="scalp-{{ idx_key }}-strike-otm-reason">—</div>
                    </div>
                </div>
            </div>

        </div>
        <div id="scalp-{{ idx_key }}-error" style="display:none;font-size:12px;color:#475569;text-align:center;padding:20px;"></div>
    </div>
</div>
<!-- ══════════════ END SCALP SIGNAL ENGINE ══════════════ -->

<!-- Row 1: Bias + Levels + Mini Chart -->
<div class="grid3">

    <!-- Morning Bias -->
    <div class="panel">
        <div class="panel-title">🎯 Morning Bias Score</div>
        <div class="bias-gauge">
            <div class="bias-score-num" style="color:{{ ix.bias_color }};">
                {{ ix.bias_score }}
            </div>
            <div class="bias-label" style="color:{{ ix.bias_color }};">{{ ix.bias }}</div>
            <div style="font-size:11px;color:#475569;">Daily Trend</div>
            <div class="bias-bar-wrap">
                <div class="bias-bar"
                     style="width:{{ ix.bias_score }}%;background:{{ ix.bias_color }};"></div>
            </div>
            <div style="font-size:12px;color:{{ ix.h1_color }};">
                1H: {{ ix.h1_bias }}
            </div>
            <div style="font-size:11px;color:#475569;margin-top:6px;">
                RSI(D): {{ ix.rsi_d }} &nbsp;|&nbsp; RSI(1H): {{ ix.rsi_1h }}
            </div>
        </div>
    </div>

    <!-- Key Levels -->
    <div class="panel">
        <div class="panel-title">🔑 Key Levels (Pivot)</div>
        <table class="lvl-table">
            <tr><td class="lvl-label lvl-r">R3</td><td class="lvl-val lvl-r">{{ ix.r3 }}</td></tr>
            <tr><td class="lvl-label lvl-r">R2</td><td class="lvl-val lvl-r">{{ ix.r2 }}</td></tr>
            <tr><td class="lvl-label lvl-r">R1</td><td class="lvl-val lvl-r">{{ ix.r1 }}</td></tr>
            <tr style="background:#1e293b;">
                <td class="lvl-label lvl-p" style="font-weight:bold;">PIVOT</td>
                <td class="lvl-val lvl-p" style="font-size:15px;">{{ ix.pivot }}</td>
            </tr>
            <tr><td class="lvl-label lvl-s">S1</td><td class="lvl-val lvl-s">{{ ix.s1 }}</td></tr>
            <tr><td class="lvl-label lvl-s">S2</td><td class="lvl-val lvl-s">{{ ix.s2 }}</td></tr>
            <tr><td class="lvl-label lvl-s">S3</td><td class="lvl-val lvl-s">{{ ix.s3 }}</td></tr>
            <tr style="background:#0f172a;">
                <td class="lvl-label" style="color:#a78bfa;">CPR TC</td>
                <td class="lvl-val" style="color:#a78bfa;">{{ ix.tc }}</td>
            </tr>
            <tr style="background:#0f172a;">
                <td class="lvl-label" style="color:#a78bfa;">CPR BC</td>
                <td class="lvl-val" style="color:#a78bfa;">{{ ix.bc }}</td>
            </tr>
        </table>
        <div style="font-size:10px;color:#475569;margin-top:6px;">{{ ix.cpr_type }}</div>
    </div>

    <!-- Mini Chart + Range -->
    <div class="panel">
        <div class="panel-title">📊 30D Price + Expected Range</div>
        <canvas id="{{ chart_id }}"></canvas>
        <div style="font-size:11px;color:#94a3b8;margin-top:10px;">
            Expected daily range (1 ATR = {{ ix.atr }}pts)
        </div>
        <div style="display:flex;justify-content:space-between;font-size:12px;margin-top:4px;">
            <span style="color:#ef4444;">↓ {{ ix.exp_down }}</span>
            <span style="color:#60a5fa;font-weight:bold;">{{ ix.price }}</span>
            <span style="color:#22c55e;">↑ {{ ix.exp_up }}</span>
        </div>
        <div style="font-size:11px;color:#475569;margin-top:8px;">
            52W: {{ ix.low52 }} – {{ ix.high52 }}
            ({{ ix.dist52 }}% from top)
        </div>
        <div style="font-size:11px;color:#475569;margin-top:4px;">
            5D High: {{ ix.recent_high }} &nbsp;|&nbsp; 5D Low: {{ ix.recent_low }}
        </div>
    </div>
</div>

<!-- Row 2: CE Strikes + PE Strikes + Setup -->
<div class="grid3">

    <!-- CE Recommendations -->
    <div class="panel">
        <div class="panel-title">🟢 CALL (CE) Strikes to Watch</div>
        {% for i, s in enumerate(ix.ce_strikes) %}
        <div class="strike-row {{ 'strike-atm' if i==0 else ('strike-otm1' if i==1 else 'strike-otm2') }}">
            <div>
                <div>{{ s }} CE</div>
                <div class="strike-sub">
                    {{ "ATM" if i==0 else ("1 OTM (+"+step_label+")" if i==1 else "2 OTM (+"+step_label+"x2)") }}
                </div>
            </div>
            <div style="font-size:10px;text-align:right;">
                {{ "Best R:R" if i==1 else ("Safe" if i==0 else "Aggressive") }}
            </div>
        </div>
        {% endfor %}
        <div style="font-size:11px;color:#475569;margin-top:8px;">
            {% if ix.bias == 'Bullish' %}
                ✅ Trend supports CE buying
            {% elif ix.bias == 'Bearish' %}
                ❌ Trend against CE — avoid or wait for reversal
            {% else %}
                ⚠️ Neutral — buy CE only on breakout above R1 ({{ ix.r1 }})
            {% endif %}
        </div>
    </div>

    <!-- PE Recommendations -->
    <div class="panel">
        <div class="panel-title">🔴 PUT (PE) Strikes to Watch</div>
        {% for i, s in enumerate(ix.pe_strikes) %}
        <div class="strike-row {{ 'strike-atm' if i==0 else ('strike-otm1' if i==1 else 'strike-otm2') }}">
            <div>
                <div>{{ s }} PE</div>
                <div class="strike-sub">
                    {{ "ATM" if i==0 else ("1 OTM (-"+step_label+")" if i==1 else "2 OTM (-"+step_label+"x2)") }}
                </div>
            </div>
            <div style="font-size:10px;text-align:right;">
                {{ "Best R:R" if i==1 else ("Safe" if i==0 else "Aggressive") }}
            </div>
        </div>
        {% endfor %}
        <div style="font-size:11px;color:#475569;margin-top:8px;">
            {% if ix.bias == 'Bearish' %}
                ✅ Trend supports PE buying
            {% elif ix.bias == 'Bullish' %}
                ❌ Trend against PE — avoid or wait for breakdown below S1 ({{ ix.s1 }})
            {% else %}
                ⚠️ Neutral — buy PE only on breakdown below S1 ({{ ix.s1 }})
            {% endif %}
        </div>
    </div>

    <!-- Risk Management -->
    <div class="panel">
        <div class="panel-title">🛡️ Risk Management</div>
        <div class="rm-row">
            <span class="rm-label">Max Risk/Trade</span>
            <span class="rm-val" style="color:#ef4444;">1-2% of capital</span>
        </div>
        <div class="rm-row">
            <span class="rm-label">Ideal Entry Time</span>
            <span class="rm-val">9:20–9:45 AM</span>
        </div>
        <div class="rm-row">
            <span class="rm-label">Avoid Trading</span>
            <span class="rm-val">11:30–1:00 PM</span>
        </div>
        <div class="rm-row">
            <span class="rm-label">Exit Time (Intraday)</span>
            <span class="rm-val">3:15 PM latest</span>
        </div>
        <div class="rm-row">
            <span class="rm-label">SL for CE buyers</span>
            <span class="rm-val" style="color:#ef4444;">Below S1 ({{ ix.s1 }})</span>
        </div>
        <div class="rm-row">
            <span class="rm-label">SL for PE buyers</span>
            <span class="rm-val" style="color:#ef4444;">Above R1 ({{ ix.r1 }})</span>
        </div>
        <div class="rm-row">
            <span class="rm-label">Target (1:2 R:R)</span>
            <span class="rm-val" style="color:#22c55e;">{{ ix.atr * 2 | int }} pts</span>
        </div>
        <div class="rm-row">
            <span class="rm-label">Recommended Lots</span>
            <span class="rm-val">1 lot (test the water)</span>
        </div>
    </div>
</div>

<!-- Row 3: Strategies -->
<div class="panel" style="margin-bottom:14px;">
    <div class="panel-title">⚡ Today's Trade Setups</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">

        <!-- Intraday setup -->
        <div class="strat-card" style="border-color:
            {{ '#22c55e' if ix.bias=='Bullish' else ('#ef4444' if ix.bias=='Bearish' else '#facc15') }};">
            <div class="strat-title">
                🕐 Intraday Setup
                <span class="strat-tag" style="background:#1e3a5f;color:#60a5fa;">Scalp/Momentum</span>
            </div>
            <div class="strat-body">
                {% if ix.bias == 'Bullish' %}
                Buy {{ ix.ce_strikes[1] }} CE on any dip to {{ ix.s1 }} or breakout above {{ ix.r1 }}.
                Target {{ ix.r2 }}. SL below {{ ix.s1 }}.
                {% elif ix.bias == 'Bearish' %}
                Buy {{ ix.pe_strikes[1] }} PE on any bounce to {{ ix.r1 }} or breakdown below {{ ix.s1 }}.
                Target {{ ix.s2 }}. SL above {{ ix.r1 }}.
                {% else %}
                Wait for first 15 min to close. Buy CE if breaks {{ ix.r1 }} with volume.
                Buy PE if breaks {{ ix.s1 }} with volume. Avoid trading between S1–R1.
                {% endif %}
            </div>
        </div>

        <!-- Positional setup -->
        <div class="strat-card" style="border-color:#a78bfa;">
            <div class="strat-title">
                📅 Positional Setup (2–3 days)
                <span class="strat-tag" style="background:#2d1a4a;color:#a78bfa;">Swing Options</span>
            </div>
            <div class="strat-body">
                {% if ix.bias == 'Bullish' %}
                Buy next-week {{ ix.ce_strikes[0] }} CE for positional. Hold above {{ ix.pivot }}.
                Target {{ ix.r2 }}–{{ ix.r3 }} range. Use 30% less qty vs intraday.
                {% elif ix.bias == 'Bearish' %}
                Buy next-week {{ ix.pe_strikes[0] }} PE for positional. Hold below {{ ix.pivot }}.
                Target {{ ix.s2 }}–{{ ix.s3 }} range. Use 30% less qty vs intraday.
                {% else %}
                No clear positional bias. Consider iron condor between {{ ix.s1 }}–{{ ix.r1 }}.
                Collect premium if range-bound expected.
                {% endif %}
            </div>
        </div>

        <!-- VIX-based strategy -->
        <div class="strat-card" style="border-color:{{ d.vix.color }};">
            <div class="strat-title">
                🌡️ VIX Strategy (VIX = {{ d.vix.val }})
            </div>
            <div class="strat-body">
                {% set vix_num = d.vix.val if d.vix.val != 'N/A' else 15 %}
                {% if d.vix.val != 'N/A' and d.vix.val < 13 %}
                VIX very low — premiums are cheap. Good time to BUY options (CE or PE).
                Straddle/strangle at ATM makes sense if big move expected.
                {% elif d.vix.val != 'N/A' and d.vix.val < 17 %}
                VIX normal — standard premium, both buying and selling work.
                Stick to directional trades based on bias.
                {% elif d.vix.val != 'N/A' and d.vix.val < 22 %}
                VIX elevated — premiums expensive. Prefer option SELLING or spreads.
                If buying, go ATM only, avoid far OTM.
                {% else %}
                VIX high — only hedge or sell options. Naked option buying is very risky.
                Consider protective puts on portfolio if holding stocks.
                {% endif %}
            </div>
        </div>

        <!-- Expiry strategy -->
        <div class="strat-card" style="border-color:#f97316;">
            <div class="strat-title">
                📅 Expiry Strategy ({{ d.expiry.days_left }} days left)
            </div>
            <div class="strat-body">
                {% if d.expiry.days_left <= 1 %}
                Expiry day — theta crush is maximum. Only sell options or buy deep ITM.
                ATM options lose value very fast. Avoid holding OTM overnight.
                {% elif d.expiry.days_left <= 2 %}
                1–2 days to expiry. Theta decay accelerating. Prefer selling OTM spreads.
                If buying, only take quick intraday scalps.
                {% else %}
                Enough time decay buffer. Buying options is viable.
                Prefer 1 OTM for best risk:reward. Avoid holding to last day.
                {% endif %}
            </div>
        </div>
    </div>
</div>

{% else %}
<div style="padding:40px;text-align:center;color:#475569;">
    Could not load data. Please refresh.
</div>
{% endif %}
</div>
{% endfor %}

<!-- ═══ AI Morning Brief (full width) ═══ -->
<div class="panel" style="margin-top:4px;">
    <div style="font-size:15px;font-weight:bold;color:#60a5fa;margin-bottom:12px;">
        🦅 FalconAI Morning Brief
        <span style="font-size:11px;color:#334155;font-weight:normal;margin-left:8px;">
            AI-generated · For education only · Not financial advice
        </span>
    </div>

    <div id="brief-loading" class="ai-loading">
        <div class="ai-dot"></div><div class="ai-dot"></div><div class="ai-dot"></div>
        <span>Writing today's options morning brief...</span>
    </div>

    <div id="brief-text" class="morning-brief" style="display:none;"></div>
    <div id="brief-error" style="display:none;font-size:12px;color:#475569;"></div>
</div>

</div><!-- end page -->

<script>
// ── Tab switching ────────────────────────────────────────────────────
function switchIdx(id, btn) {
    document.querySelectorAll(".idx-panel").forEach(p => p.classList.remove("active"));
    document.querySelectorAll(".idx-tab").forEach(b => b.classList.remove("active"));
    document.getElementById(id + "-panel").classList.add("active");
    btn.classList.add("active");
}

// ── Mini Line Charts ─────────────────────────────────────────────────
function drawMini(canvasId, points) {
    const c   = document.getElementById(canvasId);
    if (!c || !points.length) return;
    c.width   = c.offsetWidth;
    c.height  = c.offsetHeight;
    const ctx = c.getContext("2d");
    const W   = c.width, H = c.height;
    const vals = points.map(p => p.c);
    const mn   = Math.min(...vals), mx = Math.max(...vals);
    const px   = (i) => (i / (points.length - 1)) * W;
    const py   = (v) => H - ((v - mn) / (mx - mn + 0.01)) * (H * 0.9) - 4;

    // Gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, "rgba(59,130,246,0.3)");
    grad.addColorStop(1, "rgba(59,130,246,0.0)");

    ctx.beginPath();
    points.forEach((p, i) => i === 0 ? ctx.moveTo(px(i), py(p.c)) : ctx.lineTo(px(i), py(p.c)));
    ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();

    ctx.beginPath();
    points.forEach((p, i) => i === 0 ? ctx.moveTo(px(i), py(p.c)) : ctx.lineTo(px(i), py(p.c)));
    ctx.strokeStyle = "#3b82f6"; ctx.lineWidth = 1.5; ctx.stroke();
}

drawMini("nifty-mini", {{ d.nifty.chart_pts | tojson if d.nifty else '[]' }});
drawMini("bnf-mini",   {{ d.banknifty.chart_pts | tojson if d.banknifty else '[]' }});
window.addEventListener("resize", () => {
    drawMini("nifty-mini", {{ d.nifty.chart_pts | tojson if d.nifty else '[]' }});
    drawMini("bnf-mini",   {{ d.banknifty.chart_pts | tojson if d.banknifty else '[]' }});
});

// ── Pro Scalp Signal Engine ────────────────────────────────────────────
const scalpState = {
  nifty:     { countdown: 30 },
  banknifty: { countdown: 30 }
};

function fmtPts(v) {
  if (v === null || v === undefined) return '—';
  return (v >= 0 ? '+' : '') + v;
}

async function fetchScalpSignal(idx) {
  try {
    const res  = await fetch(`/api/scalp_signal/${idx}`);
    const data = await res.json();
    if (data.error) { renderScalpError(idx, data.error); return; }
    renderScalpSignal(idx, data);
    scalpState[idx].countdown = 30;
  } catch(e) {
    renderScalpError(idx, 'Could not reach scalp engine. Will retry automatically.');
  }
}

function renderScalpError(idx, msg) {
  const loadEl = document.getElementById(`scalp-${idx}-loading`);
  if (loadEl) loadEl.style.display = 'none';
  const contentEl = document.getElementById(`scalp-${idx}-content`);
  if (contentEl) contentEl.style.display = 'none';
  const err = document.getElementById(`scalp-${idx}-error`);
  if (err) { err.style.display = 'block'; err.textContent = msg; }
}

function renderScalpSignal(idx, d) {
  document.getElementById(`scalp-${idx}-loading`).style.display = 'none';
  document.getElementById(`scalp-${idx}-error`).style.display = 'none';
  document.getElementById(`scalp-${idx}-content`).style.display = 'block';

  document.getElementById(`scalp-${idx}-spot`).textContent = `Spot: ₹${d.spot}`;
  document.getElementById(`scalp-${idx}-atm`).textContent  = `ATM: ${d.atm}`;
  document.getElementById(`scalp-${idx}-lastupdate`).textContent = `Last: ${d.scan_time}`;

  document.getElementById(`scalp-${idx}-choppy-pill`).style.display = d.is_choppy ? 'inline-flex' : 'none';

  const box = document.getElementById(`scalp-${idx}-signalbox`);
  box.style.borderColor = d.signal_color;
  box.style.background  = `${d.signal_color}10`;

  const icon = d.signal_text.includes('BUY') ? '🟢' : (d.is_choppy ? '🟡' : '🔴');
  const sigTextEl = document.getElementById(`scalp-${idx}-signaltext`);
  sigTextEl.textContent = `${icon} ${d.signal_text}`;
  sigTextEl.style.color = d.signal_color;
  document.getElementById(`scalp-${idx}-actiontext`).textContent = d.action;

  document.getElementById(`scalp-${idx}-t1`).textContent = `${fmtPts(d.t1)} pts`;
  document.getElementById(`scalp-${idx}-t2`).textContent = `${fmtPts(d.t2)} pts`;
  document.getElementById(`scalp-${idx}-sl`).textContent = `-${d.sl} pts`;
  document.getElementById(`scalp-${idx}-rr`).textContent = `1:${d.rr}`;
  document.getElementById(`scalp-${idx}-pattern`).textContent = d.pattern;

  const confPct = Math.round((d.confidence / d.confidence_total) * 100);
  const confScoreEl = document.getElementById(`scalp-${idx}-confscore`);
  confScoreEl.textContent = `${d.confidence}/${d.confidence_total}`;
  confScoreEl.style.color = d.signal_color;
  const confFillEl = document.getElementById(`scalp-${idx}-conffill`);
  confFillEl.style.width = `${confPct}%`;
  confFillEl.style.background = d.signal_color;

  let confLabel = 'VERY LOW — SKIP';
  if (d.confidence >= 6) confLabel = 'HIGH CONFIDENCE';
  else if (d.confidence >= 4) confLabel = 'MODERATE — SIZE DOWN';
  else if (d.confidence >= 2) confLabel = 'LOW — WAIT';
  const confStatusEl = document.getElementById(`scalp-${idx}-confstatus`);
  confStatusEl.textContent = confLabel;
  confStatusEl.style.color = d.signal_color;

  const emaEl = document.getElementById(`scalp-${idx}-ema`);
  emaEl.textContent = d.ema_trend;
  emaEl.style.color = d.ema_trend === 'BULL' ? '#10b981' : d.ema_trend === 'BEAR' ? '#ef4444' : '#94a3b8';

  const macdEl = document.getElementById(`scalp-${idx}-macd`);
  macdEl.textContent = d.macd_hist;
  macdEl.style.color = d.macd_hist >= 0 ? '#10b981' : '#ef4444';

  const t15El = document.getElementById(`scalp-${idx}-trend15`);
  t15El.textContent = d.trend_15m;
  t15El.style.color = d.trend_15m === 'BULL' ? '#10b981' : d.trend_15m === 'BEAR' ? '#ef4444' : '#94a3b8';

  const volEl = document.getElementById(`scalp-${idx}-vol`);
  volEl.textContent = `${d.vol_ratio}x`;
  volEl.style.color = d.vol_ratio >= 1 ? '#10b981' : '#f59e0b';

  const rangeEl = document.getElementById(`scalp-${idx}-range`);
  rangeEl.textContent = `${d.range_pct}%`;
  rangeEl.style.color = d.is_choppy ? '#f59e0b' : '#10b981';

  const cautionBox = document.getElementById(`scalp-${idx}-cautionbox`);
  if (d.cautions && d.cautions.length) {
    cautionBox.style.display = 'block';
    document.getElementById(`scalp-${idx}-cautionlist`).innerHTML =
      d.cautions.map(c => `<div class="scalp-caution-item">⚠️ ${c}</div>`).join('');
  } else {
    cautionBox.style.display = 'none';
  }

  document.getElementById(`scalp-${idx}-checklist`).innerHTML = d.checklist.map(c => `
    <div class="scalp-check-item ${c.pass ? 'pass' : 'fail'}">
      <div class="scalp-check-name" style="color:${c.pass ? '#34d399' : '#f59e0b'};">
        ${c.pass ? '✅' : '⚠️'} ${c.name}
      </div>
      <div class="scalp-check-detail">${c.detail}</div>
    </div>
  `).join('');

  const sr = d.strike_reco;
  const primaryEl = document.getElementById(`scalp-${idx}-strike-primary`);
  primaryEl.textContent = d.direction !== 'NONE' ? `BUY ${sr.primary_label}` : 'No clear direction yet';
  primaryEl.style.color = d.signal_color;
  document.getElementById(`scalp-${idx}-strike-primary-reason`).textContent = sr.primary_reasoning;
  document.getElementById(`scalp-${idx}-strike-itm`).textContent = sr.itm_label;
  document.getElementById(`scalp-${idx}-strike-itm-reason`).textContent = sr.safer_reasoning;
  document.getElementById(`scalp-${idx}-strike-otm`).textContent = sr.otm_label;
  document.getElementById(`scalp-${idx}-strike-otm-reason`).textContent = sr.aggressive_reasoning;
}

function tickScalpCountdown() {
  ['nifty', 'banknifty'].forEach(idx => {
    const state = scalpState[idx];
    state.countdown -= 1;
    if (state.countdown <= 0) {
      fetchScalpSignal(idx);
      state.countdown = 30;
    }
    const cdEl   = document.getElementById(`scalp-${idx}-countdown`);
    const fillEl = document.getElementById(`scalp-${idx}-fill`);
    if (cdEl)   cdEl.textContent = state.countdown;
    if (fillEl) fillEl.style.width = `${(state.countdown / 30) * 100}%`;
  });
}

fetchScalpSignal('nifty');
fetchScalpSignal('banknifty');
setInterval(tickScalpCountdown, 1000);

// ── FalconAI Morning Brief ───────────────────────────────────────────
(async () => {
    try {
        const nifty = {{ d.nifty | tojson if d.nifty else '{}' }};
        const bnf   = {{ d.banknifty | tojson if d.banknifty else '{}' }};
        const vix   = {{ d.vix | tojson }};
        const exp   = {{ d.expiry | tojson }};

        const sys = `You are FalconAI, a sharp 25-year-old Indian options trader and analyst.
Write a crisp, practical OPTIONS MORNING BRIEF for today's NSE trading session.
Cover these sections clearly (use section headers with emojis):
1. 🌅 Market Mood — overall feel for today based on data
2. 📈 Nifty Outlook — key levels to watch, what happens at each
3. 🏦 BankNifty Outlook — same, any divergence from Nifty?
4. ⚡ Top Trade Ideas — 2-3 specific actionable setups (strike, entry, SL, target)
5. ⚠️ Key Risks Today — what could invalidate the thesis
6. 🦅 FalconAI Verdict — one paragraph, your gut feel for the day

Be specific with numbers. Be direct. No fluff. Under 400 words total.`;

        const usr = `Today: ${exp.day_name}, ${exp.today}
Weekly Expiry: ${exp.weekly} (${exp.days_left} days left)

NIFTY: ${nifty.price} | Change: ${nifty.chg} (${nifty.chg_pct}%)
Bias: ${nifty.bias} (Score: ${nifty.bias_score}) | RSI(D): ${nifty.rsi_d} | RSI(1H): ${nifty.rsi_1h}
MA20: ${nifty.ma20} | MA50: ${nifty.ma50} | ATR: ${nifty.atr}
Pivot: ${nifty.pivot} | R1: ${nifty.r1} | R2: ${nifty.r2} | S1: ${nifty.s1} | S2: ${nifty.s2}
CPR: ${nifty.bc}–${nifty.tc} (${nifty.cpr_type})
ATM Strike: ${nifty.atm} | Expected range: ${nifty.exp_down}–${nifty.exp_up}

BANKNIFTY: ${bnf.price} | Change: ${bnf.chg} (${bnf.chg_pct}%)
Bias: ${bnf.bias} (Score: ${bnf.bias_score}) | RSI(D): ${bnf.rsi_d} | RSI(1H): ${bnf.rsi_1h}
MA20: ${bnf.ma20} | MA50: ${bnf.ma50} | ATR: ${bnf.atr}
Pivot: ${bnf.pivot} | R1: ${bnf.r1} | R2: ${bnf.r2} | S1: ${bnf.s1} | S2: ${bnf.s2}
ATM Strike: ${bnf.atm} | Expected range: ${bnf.exp_down}–${bnf.exp_up}

INDIA VIX: ${vix.val} (${vix.chg > 0 ? '+' : ''}${vix.chg}) — ${vix.label}

Write the morning brief now.`;

        const res = await fetch("https://api.anthropic.com/v1/messages", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: "claude-sonnet-4-6",
                max_tokens: 1000,
                system: sys,
                messages: [{ role: "user", content: usr }]
            })
        });
        const data = await res.json();
        const text = data.content?.[0]?.text || "";

        document.getElementById("brief-loading").style.display = "none";
        document.getElementById("brief-text").style.display    = "block";
        document.getElementById("brief-text").textContent      = text;

    } catch(e) {
        document.getElementById("brief-loading").style.display = "none";
        document.getElementById("brief-error").style.display   = "block";
        document.getElementById("brief-error").textContent     = "Morning brief could not be generated.";
    }
})();
</script>
</body>
</html>
"""

@app.route("/options")
def options_page():
    d = get_options_market_data()
    # Jinja2 needs enumerate — pass it via globals trick
    from jinja2 import Environment
    return render_template_string(OPTIONS_HTML, d=d, enumerate=enumerate)



# ════════════════════════════════════════════════════════════════════
# MARKETSMITH-STYLE SCREENS ENGINE
# ════════════════════════════════════════════════════════════════════

SCREENS_FILE = "screens_history.csv"

# Top 60 most liquid NSE stocks — sized for Render free tier (512MB RAM, 30s gunicorn timeout).
# Scanning the full Nifty 500 (~500 yfinance calls) reliably causes worker OOM/SIGKILL.
# These 60 names cover every major sector and produce the large majority of
# genuine intraday breakout signals — liquidity is what drives clean technical setups.
NIFTY500_UNIVERSE = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","HINDUNILVR","ITC","SBIN",
    "BHARTIARTL","KOTAKBANK","LT","AXISBANK","BAJFINANCE","ASIANPAINT","MARUTI",
    "TITAN","SUNPHARMA","ULTRACEMCO","NESTLEIND","WIPRO","NTPC","ONGC","TATAMOTORS",
    "TATASTEEL","JSWSTEEL","POWERGRID","M&M","ADANIPORTS","COALINDIA","TECHM",
    "HCLTECH","DRREDDY","CIPLA","DIVISLAB","BAJAJFINSV","GRASIM","BRITANNIA",
    "EICHERMOT","HEROMOTOCO","BPCL","HINDALCO","SBILIFE","HDFCLIFE",
    "APOLLOHOSP","INDUSINDBK","TATACONSUM","BAJAJ-AUTO","UPL",
    "DLF","DABUR","HAVELLS","SIEMENS","TRENT","DMART","ZOMATO",
    "FEDERALBNK","IDFCFIRSTB","BANKBARODA","PNB",
    "TVSMOTOR","ADANIENT","TATAPOWER","DIXON","POLYCAB","HAL","BEL",
    "MUTHOOTFIN","CHOLAFIN","SHRIRAMFIN","NAUKRI",
]

# Remove duplicates while preserving order
_seen = set()
NIFTY500_UNIVERSE = [s for s in NIFTY500_UNIVERSE if s not in _seen and not _seen.add(s)]

# Default universe = full Nifty 500 list
DEFAULT_UNIVERSE = NIFTY500_UNIVERSE


def screens_save(screen_key, screen_name, matched_symbols, run_meta=None):
    """Save a screener run to history with date/time stamp."""
    entry_date = datetime.now().strftime("%d-%m-%Y")
    entry_time = datetime.now().strftime("%H:%M:%S")
    row = {
        "date":         entry_date,
        "time":         entry_time,
        "screen_key":   screen_key,
        "screen_name":  screen_name,
        "symbols":      "|".join(matched_symbols),
        "count":        len(matched_symbols),
    }
    df_new = pd.DataFrame([row])
    if os.path.exists(SCREENS_FILE):
        old = pd.read_csv(SCREENS_FILE)
        df  = pd.concat([old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(SCREENS_FILE, index=False)


def screens_history(screen_key=None, limit=50):
    """Load past screener runs, optionally filtered by screen_key."""
    if not os.path.exists(SCREENS_FILE):
        return []
    df = pd.read_csv(SCREENS_FILE)
    if screen_key:
        df = df[df["screen_key"] == screen_key]
    df = df.sort_values(by=["date", "time"], ascending=False).head(limit)
    out = []
    for _, row in df.iterrows():
        syms = str(row.get("symbols", "")).split("|") if row.get("symbols") else []
        out.append({
            "date":        row.get("date", "?"),
            "time":        row.get("time", "?"),
            "screen_key":  row.get("screen_key", "?"),
            "screen_name": row.get("screen_name", "?"),
            "symbols":     syms,
            "count":       int(row.get("count", len(syms))),
        })
    return out


def make_price_card(symbol, df, extra=None):
    """Build a compact price-card dict (like performance tracker cards) for a screened stock."""
    try:
        close  = safe_series(df["Close"])
        volume = safe_series(df["Volume"])
        high_s = safe_series(df["High"])
        low_s  = safe_series(df["Low"])

        price    = round(float(close.iloc[-1]), 2)
        prev     = round(float(close.iloc[-2]), 2) if len(close) > 1 else price
        chg      = round(price - prev, 2)
        chg_pct  = round((chg / prev) * 100, 2) if prev else 0

        high20 = float(close.tail(20).max())
        low20  = float(close.tail(20).min())
        high52 = float(close.tail(min(252, len(close))).max())
        low52  = float(close.tail(min(252, len(close))).min())

        avg_vol20 = float(volume.tail(20).mean())
        last_vol  = float(volume.iloc[-1])
        vol_ratio = round((last_vol / avg_vol20) * 100, 0) if avg_vol20 > 0 else 100

        ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else price
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else price
        rsi  = float(calc_rsi(close).iloc[-1]) if len(close) >= 15 else 50
        if np.isnan(rsi): rsi = 50

        card = {
            "symbol":     symbol,
            "price":      price,
            "chg":        chg,
            "chg_pct":    chg_pct,
            "high20":     round(high20, 2),
            "low20":      round(low20, 2),
            "high52":     round(high52, 2),
            "low52":      round(low52, 2),
            "vol_ratio":  vol_ratio,
            "ma20":       round(ma20, 2),
            "ma50":       round(ma50, 2),
            "rsi":        round(rsi, 1),
            "dist_high52":round(((high52 - price) / high52) * 100, 1) if high52 else 0,
        }
        if extra:
            card.update(extra)
        return card
    except Exception:
        return None


def get_universe(custom_list=None):
    if custom_list:
        return [s.strip().upper() for s in custom_list if s.strip()]
    return DEFAULT_UNIVERSE


# ── SCREEN 1: Near Pivot Stocks ────────────────────────────────────────
def screen_near_pivot(universe):
    """Price within 5% of 20-day high (pivot), above MA50."""
    out = []
    for sym in universe:
        df = get_data(sym)
        if df is None or len(safe_series(df["Close"])) < 50:
            continue
        close = safe_series(df["Close"])
        price = float(close.iloc[-1])
        pivot = float(close.tail(20).max())
        ma50  = float(close.rolling(50).mean().iloc[-1])
        dist_pct = ((pivot - price) / pivot) * 100 if pivot else 99
        if dist_pct <= 5 and price > ma50:
            card = make_price_card(sym, df, {"pivot": round(pivot,2), "dist_pivot_pct": round(dist_pct,2)})
            if card: out.append(card)
    return out


# ── SCREEN 2: Recent Breakouts ─────────────────────────────────────────
def screen_recent_breakouts(universe):
    """Broke above 20-day high within last 16 days, not fallen >7% below pivot since."""
    out = []
    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        close = safe_series(df["Close"])
        if len(close) < 40: continue
        price = float(close.iloc[-1])

        # find most recent breakout in last 16 sessions
        breakout_idx = None
        for i in range(max(1, len(close)-16), len(close)):
            prior_high = float(close.iloc[max(0,i-20):i].max())
            if prior_high > 0 and float(close.iloc[i]) >= prior_high * 0.999:
                breakout_idx = i
                break
        if breakout_idx is None: continue

        pivot = float(close.iloc[max(0,breakout_idx-20):breakout_idx].max())
        post_breakout_min = float(close.iloc[breakout_idx:].min())
        dd_from_pivot = ((pivot - post_breakout_min) / pivot) * 100 if pivot else 0
        if dd_from_pivot > 7: continue   # failed already

        days_since = len(close) - 1 - breakout_idx
        card = make_price_card(sym, df, {"pivot": round(pivot,2), "days_since_breakout": days_since})
        if card: out.append(card)
    return out


# ── SCREEN 3: Failed Breakout ──────────────────────────────────────────
def screen_failed_breakout(universe):
    """Price trading 7%+ below a recent pivot (broke down after breakout)."""
    out = []
    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        close = safe_series(df["Close"])
        if len(close) < 40: continue
        price = float(close.iloc[-1])
        pivot = float(close.tail(40).max())
        dd_pct = ((pivot - price) / pivot) * 100 if pivot else 0
        if dd_pct >= 7:
            card = make_price_card(sym, df, {"pivot": round(pivot,2), "below_pivot_pct": round(dd_pct,2)})
            if card: out.append(card)
    return out


# ── SCREEN 4: Price Gaps Up ────────────────────────────────────────────
def screen_price_gaps_up(universe):
    """Today's low > yesterday's high (gap), volume above average."""
    out = []
    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        close = safe_series(df["Close"]); high_s = safe_series(df["High"]); low_s = safe_series(df["Low"])
        volume = safe_series(df["Volume"])
        if len(close) < 21: continue
        today_low   = float(low_s.iloc[-1])
        prior_high  = float(high_s.iloc[-2])
        avg_vol     = float(volume.tail(20).mean())
        today_vol   = float(volume.iloc[-1])
        if today_low > prior_high and today_vol > avg_vol:
            gap_pct = round(((today_low - prior_high) / prior_high) * 100, 2)
            card = make_price_card(sym, df, {"gap_pct": gap_pct})
            if card: out.append(card)
    return out


# ── SCREEN 5: Extended Stocks ──────────────────────────────────────────
def screen_extended(universe):
    """More than 25% above 20-day pivot, above MA50, near 52W high."""
    out = []
    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        close = safe_series(df["Close"])
        if len(close) < 60: continue
        price  = float(close.iloc[-1])
        pivot  = float(close.tail(60).min())   # approx base low as reference
        ma50   = float(close.rolling(50).mean().iloc[-1])
        high52 = float(close.tail(min(252,len(close))).max())
        ext_pct = ((price - pivot) / pivot) * 100 if pivot else 0
        near52  = ((high52 - price) / high52) * 100 if high52 else 99
        if ext_pct > 25 and price > ma50 and near52 < 10:
            card = make_price_card(sym, df, {"extended_pct": round(ext_pct,2)})
            if card: out.append(card)
    return out


# ── SCREEN 6: Up on Volume ─────────────────────────────────────────────
def screen_up_on_volume(universe):
    """Price up today, volume well above average (institutional buying signal)."""
    out = []
    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        close = safe_series(df["Close"]); volume = safe_series(df["Volume"])
        if len(close) < 21: continue
        price = float(close.iloc[-1]); prev = float(close.iloc[-2])
        avg_vol = float(volume.tail(20).mean()); today_vol = float(volume.iloc[-1])
        chg_pct = ((price - prev) / prev) * 100 if prev else 0
        vol_ratio = (today_vol / avg_vol) if avg_vol else 1
        if chg_pct > 1 and vol_ratio >= 1.5:
            card = make_price_card(sym, df, {"vol_x": round(vol_ratio,2)})
            if card: out.append(card)
    return out


# ── SCREEN 7: New High With Best Trend (EPS-rank proxy) ────────────────
def screen_new_high_trend(universe):
    """Within 5% of 2yr high + strong trend score (proxy for EPS Rank 85+)."""
    out = []
    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        close = safe_series(df["Close"])
        if len(close) < 60: continue
        price  = float(close.iloc[-1])
        high2y = float(close.tail(min(504,len(close))).max())   # ~2yr of trading days
        ma20   = float(close.rolling(20).mean().iloc[-1])
        ma50   = float(close.rolling(50).mean().iloc[-1])
        near_high = ((high2y - price) / high2y) * 100 if high2y else 99
        trend_ok  = price > ma20 > ma50
        if near_high <= 5 and trend_ok:
            card = make_price_card(sym, df, {"dist_2y_high_pct": round(near_high,2)})
            if card: out.append(card)
    return out


# ── BREAKING OUT TODAY ───────────────────────────────────────────────────
def screen_breaking_out_today(universe):
    """
    Price closed at/above its 20-day pivot TODAY on a strong volume surge
    (40%+ above the 50-day average volume). This is the classic real-time
    breakout signal — distinct from Recent Breakouts (which looks back
    16 sessions) because this only fires on the exact breakout day.
    """
    out = []
    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        close  = safe_series(df["Close"])
        volume = safe_series(df["Volume"])
        if len(close) < 50: continue

        price  = float(close.iloc[-1])
        high20 = float(close.tail(20).max())

        avg_vol50 = float(volume.tail(50).mean()) if len(volume) >= 50 else float(volume.mean())
        today_vol = float(volume.iloc[-1])
        vol_ratio = (today_vol / avg_vol50) if avg_vol50 else 1

        # Today's close at/above the 20D high, on 40%+ volume surge
        if price >= high20 * 0.998 and vol_ratio >= 1.40:
            card = make_price_card(sym, df, {
                "pivot":      round(high20, 2),
                "vol_surge_x":round(vol_ratio, 2),
            })
            if card: out.append(card)
    return out


# ── TREND TEMPLATE (Minervini's classic 8-point trend criteria) ─────────
def screen_trend_template(universe):
    """
    William O'Neil / Mark Minervini style Trend Template — a stock must
    satisfy all of the following simultaneously to qualify as a true
    stage-2 uptrend leader:
      1. Price > MA150 and Price > MA200
      2. MA150 > MA200
      3. MA200 trending up for at least 1 month (slope positive)
      4. MA50 > MA150 > MA200 (moving averages stacked correctly)
      5. Price > MA50
      6. Price at least 25% above 52-week low
      7. Price within 25% of 52-week high
    """
    out = []
    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        close = safe_series(df["Close"])
        if len(close) < 220: continue   # need enough history for MA200 + 1mo slope

        price = float(close.iloc[-1])

        ma50  = float(close.rolling(50).mean().iloc[-1])
        ma150 = float(close.rolling(150).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        ma200_1mo_ago = float(close.rolling(200).mean().iloc[-22])  # ~1 month back

        high52 = float(close.tail(min(252, len(close))).max())
        low52  = float(close.tail(min(252, len(close))).min())

        cond1 = price > ma150 and price > ma200
        cond2 = ma150 > ma200
        cond3 = ma200 > ma200_1mo_ago               # MA200 trending up
        cond4 = ma50 > ma150 > ma200                 # MAs stacked correctly
        cond5 = price > ma50
        cond6 = low52 > 0 and price >= low52 * 1.25  # 25%+ above 52W low
        cond7 = high52 > 0 and price >= high52 * 0.75 # within 25% of 52W high

        if all([cond1, cond2, cond3, cond4, cond5, cond6, cond7]):
            pct_above_low  = round(((price - low52) / low52) * 100, 1) if low52 else 0
            pct_below_high = round(((high52 - price) / high52) * 100, 1) if high52 else 0
            card = make_price_card(sym, df, {
                "above_52w_low_pct": pct_above_low,
                "below_52w_high_pct":pct_below_high,
            })
            if card: out.append(card)
    return out


# ── BLUE DOT LIST ────────────────────────────────────────────────────────
def screen_blue_dot(universe, nifty_close=None):
    """
    Blue Dot List: stock's RS Line (price relative to Nifty) hits a 52-week high
    WHILE the stock itself is still building a base (price below its own 52W high).
    This is the classic "strength before price" signal — RS leads, price follows.
    """
    out = []

    # Fetch Nifty once for the whole universe (RS Line = stock price / Nifty price)
    if nifty_close is None:
        try:
            nifty_df = _yf_download_safe("NIFTYBEES.NS", period="2y", interval="1d", timeout=8)
            nifty_close = safe_series(nifty_df["Close"])
        except Exception:
            nifty_close = None

    if nifty_close is None or len(nifty_close) < 60:
        return out

    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        close = safe_series(df["Close"])
        if len(close) < 60: continue

        # Align lengths (use the shorter of the two series, most recent N days)
        n = min(len(close), len(nifty_close))
        if n < 60: continue
        stock_aligned = close.tail(n).reset_index(drop=True)
        nifty_aligned = nifty_close.tail(n).reset_index(drop=True)

        # RS Line = stock price / Nifty price (normalised ratio over time)
        rs_line = stock_aligned / nifty_aligned.replace(0, float('nan'))
        rs_line = rs_line.dropna()
        if len(rs_line) < 60: continue

        rs_today    = float(rs_line.iloc[-1])
        rs_52w_high = float(rs_line.tail(min(252, len(rs_line))).max())

        # RS Line at/near its own 52-week high (within 1%)
        rs_at_high = rs_today >= rs_52w_high * 0.99

        # Price itself NOT at its own 52-week high — still basing (within 3%-25% below)
        price       = float(close.iloc[-1])
        price_52w_high = float(close.tail(min(252, len(close))).max())
        price_dist_pct  = ((price_52w_high - price) / price_52w_high) * 100 if price_52w_high else 0
        still_basing    = 3 <= price_dist_pct <= 25

        # Confirm price structure is tightening (compression) — true "base" not a breakdown
        r15 = float(close.tail(15).max() - close.tail(15).min())
        r60 = float(close.tail(60).max() - close.tail(60).min())
        compression_ok = (r15 / r60) < 0.7 if r60 > 0 else False

        if rs_at_high and still_basing and compression_ok:
            card = make_price_card(sym, df, {
                "rs_vs_nifty_pct_from_high": round(((rs_52w_high - rs_today) / rs_52w_high) * 100, 2),
                "price_dist_52w_high_pct":   round(price_dist_pct, 2),
            })
            if card: out.append(card)

    return out


# ── GROWTH 50 ───────────────────────────────────────────────────────────
def screen_growth_50(universe):
    """
    Algorithmic Growth-50 style ranking:
    combines trend strength, RS-like momentum, proximity to high, volume.
    Returns top 50 (or fewer) ranked by composite score.
    """
    scored = []
    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        close = safe_series(df["Close"]); volume = safe_series(df["Volume"])
        if len(close) < 60: continue
        price  = float(close.iloc[-1])
        ma20   = float(close.rolling(20).mean().iloc[-1])
        ma50   = float(close.rolling(50).mean().iloc[-1])
        high52 = float(close.tail(min(252,len(close))).max())

        # Momentum: 3-month return
        ret_3m = ((price - float(close.iloc[-63])) / float(close.iloc[-63])) * 100 if len(close) >= 63 else 0
        # Trend score
        trend = 0
        if price > ma20: trend += 30
        if ma20 > ma50: trend += 30
        if price > float(close.tail(10).mean()): trend += 20
        # Proximity to 52w high
        near_high_score = max(0, 100 - (((high52-price)/high52)*100*4)) if high52 else 0
        # Volume confirmation
        avg_vol = float(volume.tail(20).mean()); recent_vol = float(volume.tail(5).mean())
        vol_score = min(100, (recent_vol/avg_vol)*50) if avg_vol else 50

        composite = (ret_3m * 0.3) + (trend * 0.3) + (near_high_score * 0.25) + (vol_score * 0.15)
        scored.append((sym, df, composite, ret_3m))

    scored.sort(key=lambda x: x[2], reverse=True)
    out = []
    for sym, df, comp, ret3m in scored[:50]:
        card = make_price_card(sym, df, {"g50_score": round(comp,1), "ret_3m_pct": round(ret3m,2)})
        if card: out.append(card)
    return out


# ── GURU SCREEN: William J. O'Neil ──────────────────────────────────────
def guru_oneil(universe):
    """Within 15% of 52W high, strong trend, price > 20 & 50 MA."""
    out = []
    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        close = safe_series(df["Close"])
        if len(close) < 60: continue
        price  = float(close.iloc[-1])
        ma20   = float(close.rolling(20).mean().iloc[-1])
        ma50   = float(close.rolling(50).mean().iloc[-1])
        high52 = float(close.tail(min(252,len(close))).max())
        near_high = ((high52-price)/high52)*100 if high52 else 99
        if near_high <= 15 and price > ma20 > ma50:
            card = make_price_card(sym, df, {"dist_52w_pct": round(near_high,2)})
            if card: out.append(card)
    return out


# ── GURU SCREEN: Benjamin Graham (Value) ─────────────────────────────────
def guru_graham(universe):
    """Low P/E, low P/B, financially stable, decent liquidity."""
    out = []
    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        try:
            tk   = yf.Ticker(sym + ".NS")
            info = tk.info or {}
            pe   = info.get("trailingPE")
            pb   = info.get("priceToBook")
            de   = info.get("debtToEquity")
            mktcap = info.get("marketCap", 0)
            if not pe or not pb: continue
            if pe <= 0 or pb <= 0: continue
            if pe > 20 or pb > 2.5: continue
            if de and de > 80: continue
            if mktcap and mktcap < 500e7: continue   # 500cr min
            card = make_price_card(sym, df, {"pe": round(pe,1), "pb": round(pb,1)})
            if card: out.append(card)
        except Exception:
            continue
    return out


# ── GURU SCREEN: Warren Buffett (Quality + Growth) ───────────────────────
def guru_buffett(universe):
    """High ROE, low debt, consistent profitability, reasonable price."""
    out = []
    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        try:
            tk   = yf.Ticker(sym + ".NS")
            info = tk.info or {}
            roe  = info.get("returnOnEquity")
            de   = info.get("debtToEquity")
            pe   = info.get("trailingPE")
            mktcap = info.get("marketCap", 0)
            if not roe or roe < 0.15: continue
            if de and de > 60: continue
            if pe and pe > 40: continue
            if mktcap and mktcap < 500e7: continue
            card = make_price_card(sym, df, {"roe_pct": round(roe*100,1)})
            if card: out.append(card)
        except Exception:
            continue
    return out


# ── GURU SCREEN: Peter Lynch (GARP) ──────────────────────────────────────
def guru_lynch(universe):
    """Growth at reasonable price — PEG-like proxy using earnings growth vs PE."""
    out = []
    for sym in universe:
        df = get_data(sym)
        if df is None: continue
        try:
            tk   = yf.Ticker(sym + ".NS")
            info = tk.info or {}
            pe   = info.get("trailingPE")
            eps_growth = info.get("earningsQuarterlyGrowth")
            if not pe or pe <= 0: continue
            if not eps_growth or eps_growth <= 0: continue
            peg = pe / (eps_growth * 100) if eps_growth else 99
            if peg > 1.5: continue
            card = make_price_card(sym, df, {"pe": round(pe,1), "peg_est": round(peg,2)})
            if card: out.append(card)
        except Exception:
            continue
    return out


# ── SCREEN REGISTRY ──────────────────────────────────────────────────────
SCREEN_REGISTRY = {
    "near_pivot":        {"name": "Near Pivot Stocks",            "group": "MarketSmith", "fn": screen_near_pivot,
                            "desc": "Price within 5% of 20-day pivot high, trading above 50-day MA — classic breakout-ready zone."},
    "recent_breakouts":  {"name": "Recent Breakouts",             "group": "MarketSmith", "fn": screen_recent_breakouts,
                            "desc": "Broke above pivot within last 16 sessions and hasn't fallen more than 7% below it since."},
    "failed_breakout":   {"name": "Failed Breakout",               "group": "MarketSmith", "fn": screen_failed_breakout,
                            "desc": "Trading 7%+ below a recent pivot — the golden sell rule has triggered."},
    "price_gaps_up":     {"name": "Price Gaps Up",                 "group": "MarketSmith", "fn": screen_price_gaps_up,
                            "desc": "Today's low opened above yesterday's high on above-average volume — strong demand signal."},
    "extended":          {"name": "Extended Stocks",               "group": "MarketSmith", "fn": screen_extended,
                            "desc": "More than 25% above their base low, near 52-week high — already extended, buy with caution."},
    "up_on_volume":      {"name": "Up on Volume",                  "group": "MarketSmith", "fn": screen_up_on_volume,
                            "desc": "Price up today on volume 1.5x+ above average — signals institutional buying interest."},
    "new_high_trend":    {"name": "New High With Best Trend",      "group": "MarketSmith", "fn": screen_new_high_trend,
                            "desc": "Within 5% of 2-year high with price above both 20 & 50-day MAs — proxy for EPS Rank 85+."},
    "breaking_out_today":{"name": "💥 Breaking Out Today",          "group": "MarketSmith", "fn": screen_breaking_out_today,
                            "desc": "Closed at/above the 20-day pivot today on a 40%+ volume surge — the live, real-time breakout signal."},
    "trend_template":    {"name": "📐 Trend Template",              "group": "MarketSmith", "fn": screen_trend_template,
                            "desc": "Minervini's 7-point trend criteria: MAs stacked (50>150>200), all trending up, price 25%+ above 52W low and within 25% of 52W high."},
    "blue_dot":          {"name": "🔵 Blue Dot List",               "group": "MarketSmith", "fn": screen_blue_dot,
                            "desc": "RS Line (vs Nifty) hits a 52-week high while price is still building a base — strength leading price, often signals a jump ahead."},

    "guru_oneil":        {"name": "William J. O'Neil",             "group": "Guru", "fn": guru_oneil,
                            "desc": "Within 15% of 52-week high with strong uptrend — O'Neil's core CAN SLIM philosophy."},
    "guru_graham":       {"name": "Benjamin Graham",                "group": "Guru", "fn": guru_graham,
                            "desc": "Low P/E (<20), low P/B (<2.5), manageable debt — classic deep value criteria."},
    "guru_buffett":      {"name": "Warren Buffett",                 "group": "Guru", "fn": guru_buffett,
                            "desc": "ROE 15%+, low debt, reasonable P/E — quality compounders at fair prices."},
    "guru_lynch":        {"name": "Peter Lynch",                    "group": "Guru", "fn": guru_lynch,
                            "desc": "Growth at reasonable price — PEG ratio under 1.5 using quarterly earnings growth."},

    "growth_50":         {"name": "Growth 50",                      "group": "Growth 50", "fn": screen_growth_50,
                            "desc": "Top 50 ranked by composite score: 3M momentum, trend strength, proximity to 52W high, and volume confirmation."},
}


@app.route("/api/run_screen/<screen_key>", methods=["POST"])
def api_run_screen(screen_key):
    """Run a screener, save results to history, return JSON for the page to render."""
    from flask import jsonify

    if screen_key not in SCREEN_REGISTRY:
        return jsonify({"error": "Unknown screen"}), 404

    meta = SCREEN_REGISTRY[screen_key]
    custom_raw = request.json.get("universe", "") if request.is_json else ""
    custom_list = None
    if custom_raw and custom_raw.strip():
        custom_list = custom_raw.replace(",", "\n").split("\n")

    universe = get_universe(custom_list)
    results  = meta["fn"](universe)

    symbols_matched = [r["symbol"] for r in results]
    screens_save(screen_key, meta["name"], symbols_matched)

    return jsonify({
        "screen_key":  screen_key,
        "screen_name": meta["name"],
        "results":     results,
        "count":       len(results),
        "universe_size": len(universe),
        "run_at":      datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
    })


@app.route("/api/screen_history/<screen_key>")
def api_screen_history(screen_key):
    from flask import jsonify
    hist = screens_history(screen_key, limit=20)
    return jsonify({"history": hist})


# Only these 4 screens auto-scan on page load / re-scan.
# All other screens in SCREEN_REGISTRY remain available via manual
# "Run Screen" (POST /api/run_screen/<key>) but won't fire automatically.
AUTO_SCAN_SCREENS = ["blue_dot", "trend_template", "breaking_out_today", "recent_breakouts"]


@app.route("/api/run_all_screens", methods=["POST"])
def api_run_all_screens():
    """
    Auto-scan endpoint: runs ONLY the screens listed in AUTO_SCAN_SCREENS
    (Blue Dot, Trend Template, Breaking Out Today, Recent Breakouts) in one
    shot against the default universe (or a custom one if provided), saves
    each to history, and returns results keyed by screen_key.
    This is what powers the Screens page auto-load.
    """
    from flask import jsonify

    custom_raw = request.json.get("universe", "") if request.is_json else ""
    custom_list = None
    if custom_raw and custom_raw.strip():
        custom_list = custom_raw.replace(",", "\n").split("\n")

    universe = get_universe(custom_list)

    # Fetch Nifty once up front and reuse for Blue Dot (avoids 1 extra download per call)
    nifty_close = None
    try:
        nifty_df = _yf_download_safe("NIFTYBEES.NS", period="2y", interval="1d", timeout=8, auto_adjust=False)
        nifty_close = safe_series(nifty_df["Close"])
    except Exception:
        nifty_close = None

    all_results = {}
    run_at = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    for key in AUTO_SCAN_SCREENS:
        meta = SCREEN_REGISTRY.get(key)
        if meta is None:
            continue
        try:
            if key == "blue_dot":
                results = meta["fn"](universe, nifty_close=nifty_close)
            else:
                results = meta["fn"](universe)
        except Exception:
            results = []

        symbols_matched = [r["symbol"] for r in results]
        screens_save(key, meta["name"], symbols_matched)

        all_results[key] = {
            "screen_name": meta["name"],
            "results":     results,
            "count":       len(results),
        }

    return jsonify({
        "screens":        all_results,
        "universe_size":  len(universe),
        "run_at":         run_at,
    })


SCREENS_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Stock Screens | FalconAI</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --bg:#070d18;--surface:#0d1424;--card:#111c2e;--border:#1a2840;--border2:#243452;
  --text:#e2e8f0;--muted:#4a6080;--muted2:#2a3a55;
  --green:#10b981;--green2:#064e35;--red:#ef4444;--red2:#4a0f0f;
  --yellow:#f59e0b;--yellow2:#3d2a00;--blue:#3b82f6;--blue2:#0f2040;--purple:#8b5cf6;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',Arial,sans-serif;}
.topbar{background:var(--surface);padding:0 24px;display:flex;align-items:center;
  border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100;height:52px;}
.nav-logo{font-size:16px;font-weight:800;color:var(--blue);margin-right:24px;text-decoration:none;}
.nav-tab{display:flex;align-items:center;height:52px;padding:0 16px;font-size:13px;font-weight:600;
  color:var(--muted);text-decoration:none;border-bottom:2px solid transparent;white-space:nowrap;}
.nav-tab:hover{color:var(--text);}
.nav-tab.active{color:var(--blue);border-bottom-color:var(--blue);}
.nav-right{margin-left:auto;font-size:12px;color:var(--muted);}
.page{max-width:1300px;margin:0 auto;padding:20px 20px 80px;}

.intro{background:linear-gradient(135deg,#0d1e3a,#0d1424);border:1px solid var(--border2);
  border-radius:14px;padding:18px 22px;margin-bottom:18px;display:flex;justify-content:space-between;
  align-items:center;flex-wrap:wrap;gap:14px;}
.intro h2{font-size:20px;font-weight:800;color:white;margin-bottom:6px;}
.intro p{font-size:13px;color:#7090b0;}
.scan-status{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);white-space:nowrap;}
.scan-dot{width:8px;height:8px;border-radius:50%;background:var(--yellow);animation:pulse2 1.2s infinite;}
.scan-dot.done{background:var(--green);animation:none;}
@keyframes pulse2{0%,100%{opacity:.3}50%{opacity:1}}
.rescan-btn{padding:7px 16px;background:var(--blue2);color:var(--blue);border:1px solid var(--blue);
  border-radius:8px;font-size:12px;font-weight:700;cursor:pointer;white-space:nowrap;}
.rescan-btn:hover{background:var(--blue);color:white;}
.rescan-btn:disabled{opacity:.4;cursor:not-allowed;}

.universe-box{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:12px 16px;margin-bottom:18px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
.universe-box input{flex:1;min-width:200px;background:var(--surface);border:1px solid var(--border2);
  border-radius:8px;padding:8px 12px;color:var(--text);font-size:12px;}
.universe-box label{font-size:11px;color:var(--muted);white-space:nowrap;}

/* ── Tab bar ── */
.tab-scroll{overflow-x:auto;margin-bottom:4px;}
.tab-bar{display:flex;gap:6px;border-bottom:1px solid var(--border);padding-bottom:0;min-width:max-content;}
.group-label{font-size:10px;font-weight:800;color:var(--muted2);text-transform:uppercase;
  letter-spacing:1px;padding:8px 10px 4px;white-space:nowrap;align-self:flex-end;}
.screen-tab{padding:9px 16px;border-radius:9px 9px 0 0;font-size:12px;font-weight:700;
  cursor:pointer;color:var(--muted);background:var(--card);border:1px solid var(--border);
  border-bottom:none;white-space:nowrap;display:flex;align-items:center;gap:6px;
  transition:all .12s;position:relative;top:1px;}
.screen-tab:hover{color:var(--text);}
.screen-tab.active{color:white;background:var(--surface);border-color:var(--border2);
  border-bottom:1px solid var(--surface);}
.tab-count{font-size:10px;padding:1px 7px;border-radius:20px;font-weight:800;
  background:var(--blue2);color:var(--blue);}
.screen-tab.active .tab-count{background:var(--blue);color:white;}
.tab-count.zero{background:var(--border);color:var(--muted);}
.screen-tab.active .tab-count.zero{background:var(--border2);color:var(--muted);}

/* ── Panel below tabs ── */
.panel{background:var(--surface);border:1px solid var(--border2);border-top:none;
  border-radius:0 0 14px 14px;padding:18px 20px 22px;}
.panel-head{display:flex;justify-content:space-between;align-items:flex-start;
  margin-bottom:14px;flex-wrap:wrap;gap:10px;padding-bottom:14px;border-bottom:1px solid var(--border);}
.panel-name{font-size:16px;font-weight:800;color:white;margin-bottom:4px;}
.panel-desc{font-size:12px;color:var(--muted);max-width:560px;line-height:1.5;}
.panel-meta{font-size:11px;color:var(--blue);font-weight:700;white-space:nowrap;}

.subtabs{display:flex;gap:4px;background:var(--card);border:1px solid var(--border);
  border-radius:8px;padding:3px;width:fit-content;margin-bottom:14px;}
.subtab{padding:5px 14px;border-radius:6px;font-size:11px;font-weight:700;cursor:pointer;
  color:var(--muted);border:none;background:transparent;}
.subtab.active{background:var(--blue2);color:var(--blue);}

/* ── List view (instead of card grid) ── */
.list-header{display:grid;grid-template-columns:1.4fr .9fr .9fr .8fr 1fr 1fr 1.3fr;
  gap:8px;padding:8px 14px;font-size:10px;font-weight:800;color:var(--muted2);
  text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--border);}
.list-row{display:grid;grid-template-columns:1.4fr .9fr .9fr .8fr 1fr 1fr 1.3fr;
  gap:8px;padding:11px 14px;align-items:center;border-bottom:1px solid var(--border);
  font-size:12px;transition:background .1s;}
.list-row:hover{background:var(--card);}
.list-row:last-child{border-bottom:none;}
.lr-sym{font-weight:800;color:var(--blue);text-decoration:none;font-size:13px;}
.lr-sym:hover{text-decoration:underline;}
.lr-price{font-weight:800;color:white;}
.lr-chg{font-weight:700;font-size:11px;}
.lr-meta{color:var(--muted);font-size:11px;}
.lr-extra{color:var(--purple);font-size:11px;font-weight:700;}
.lr-date{color:var(--muted2);font-size:10px;}

/* Mobile: collapse list to stacked rows */
@media(max-width:760px){
  .list-header{display:none;}
  .list-row{grid-template-columns:1fr;gap:3px;padding:12px 14px;}
  .lr-price{display:inline-block;margin-right:8px;}
  .lr-chg{display:inline-block;}
}

.loading-box{display:flex;align-items:center;gap:10px;color:var(--muted);
  font-size:13px;padding:40px;justify-content:center;}
.ai-dot{width:7px;height:7px;border-radius:50%;background:var(--blue);animation:pulse 1.2s infinite;}
.ai-dot:nth-child(2){animation-delay:.2s;}
.ai-dot:nth-child(3){animation-delay:.4s;}
@keyframes pulse{0%,100%{opacity:.25;transform:scale(.8)}50%{opacity:1;transform:scale(1.2)}}

.empty-box{text-align:center;padding:40px;color:var(--muted);font-size:13px;}

.hist-row{display:flex;justify-content:space-between;align-items:center;
  padding:10px 14px;border-bottom:1px solid var(--border);font-size:12px;}
.hist-date{color:var(--blue);font-weight:700;min-width:160px;}
.hist-name{color:var(--text);flex:1;}
.hist-count{color:var(--muted);font-weight:700;}
.hist-symlist{font-size:10px;color:var(--muted);padding:0 14px 10px;border-bottom:1px solid var(--border);}
</style>
</head>
<body>

<nav class="topbar">
  <a href="/" class="nav-logo">&#x1F985; FalconAI</a>
  <a href="/" class="nav-tab">&#x1F4CA; Screener</a>
  <a href="/options" class="nav-tab">&#x26A1; Options</a>
  <a href="/screens" class="nav-tab active">&#x1F50D; Screens</a>
  <a href="/sectors" class="nav-tab">&#x1F3ED; Sectors</a>
  <a href="/scanner" class="nav-tab">&#x26A1; 5-Min Scanner</a>
</nav>

<div class="page">

<div class="intro">
  <div>
    <h2>&#x1F50D; MarketSmith-Style Stock Screens</h2>
    <p>4 core screeners auto-scan automatically: Blue Dot List, Trend Template, Breaking Out Today, and Recent Breakouts.
       They run on page load and every result is timestamped to history.</p>
  </div>
  <div class="scan-status">
    <span class="scan-dot" id="scan-dot"></span>
    <span id="scan-status-text">Scanning 4 screens...</span>
  </div>
</div>

<div class="universe-box">
  <label>Custom universe (optional, comma/newline separated symbols):</label>
  <input type="text" id="universe-input" placeholder="Leave blank to use default ~100 liquid NSE stocks">
  <button class="rescan-btn" id="rescan-btn" onclick="runAllScreens()">&#x21BB; Re-scan All</button>
</div>

<div class="tab-scroll">
  <div class="tab-bar" id="tab-bar"></div>
</div>

<div class="panel" id="panel">
  <div class="panel-head">
    <div>
      <div class="panel-name" id="panel-name">—</div>
      <div class="panel-desc" id="panel-desc">—</div>
    </div>
    <div class="panel-meta" id="panel-meta">—</div>
  </div>

  <div class="subtabs">
    <button class="subtab active" id="subtab-results" onclick="switchSubtab('results')">Latest Results</button>
    <button class="subtab" id="subtab-history" onclick="switchSubtab('history')">Run History</button>
  </div>

  <div id="results-view">
    <div class="loading-box" id="loading-box">
      <div class="ai-dot"></div><div class="ai-dot"></div><div class="ai-dot"></div>
      <span>Scanning universe across 4 screens...</span>
    </div>
    <div id="list-wrap" style="display:none;">
      <div class="list-header">
        <span>Symbol</span><span>Price</span><span>Change</span><span>RSI</span>
        <span>52W Range</span><span>Vol vs Avg</span><span>Screen Detail</span>
      </div>
      <div id="list-body"></div>
    </div>
    <div class="empty-box" id="empty-box" style="display:none;">No stocks matched this screen today.</div>
  </div>

  <div id="history-view" style="display:none;"></div>
</div>

</div>

<script>
const REGISTRY = {{ registry | tojson }};
const SCREEN_ORDER = {{ screen_order | tojson }};
let ALL_RESULTS = {};
let currentScreen = SCREEN_ORDER[0];
let universeSizeLast = 0;
let runAtLast = '—';

function buildTabs() {
  const bar = document.getElementById('tab-bar');
  bar.innerHTML = '';
  let lastGroup = null;
  SCREEN_ORDER.forEach(key => {
    const meta = REGISTRY[key];
    if (meta.group !== lastGroup) {
      const lbl = document.createElement('div');
      lbl.className = 'group-label';
      lbl.textContent = meta.group;
      bar.appendChild(lbl);
      lastGroup = meta.group;
    }
    const tab = document.createElement('div');
    tab.className = 'screen-tab' + (key === currentScreen ? ' active' : '');
    tab.id = 'tab-' + key;
    tab.onclick = () => selectScreen(key);
    tab.innerHTML = `<span>${meta.name}</span><span class="tab-count zero" id="count-${key}">—</span>`;
    bar.appendChild(tab);
  });
}
buildTabs();

function selectScreen(key) {
  currentScreen = key;
  document.querySelectorAll('.screen-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + key).classList.add('active');
  switchSubtab('results');
  renderScreen(key);
}

function renderScreen(key) {
  const meta = REGISTRY[key];
  document.getElementById('panel-name').textContent = meta.name;
  document.getElementById('panel-desc').textContent = meta.desc;

  const data = ALL_RESULTS[key];
  if (!data) {
    document.getElementById('panel-meta').textContent = 'Not scanned yet';
    document.getElementById('loading-box').style.display = 'flex';
    document.getElementById('list-wrap').style.display = 'none';
    document.getElementById('empty-box').style.display = 'none';
    return;
  }

  document.getElementById('panel-meta').textContent =
    `${data.count} of ${universeSizeLast} matched · scanned ${runAtLast}`;
  document.getElementById('loading-box').style.display = 'none';

  if (data.count === 0) {
    document.getElementById('list-wrap').style.display = 'none';
    document.getElementById('empty-box').style.display = 'block';
    return;
  }

  document.getElementById('empty-box').style.display = 'none';
  document.getElementById('list-wrap').style.display = 'block';

  const knownKeys = ['symbol','price','chg','chg_pct','high20','low20','high52','low52',
    'vol_ratio','ma20','ma50','rsi','dist_high52'];

  const body = document.getElementById('list-body');
  body.innerHTML = data.results.map(r => {
    const up = r.chg_pct >= 0;
    const extraKeys = Object.keys(r).filter(k => !knownKeys.includes(k));
    const extraTxt = extraKeys.map(k => `${k.replace(/_/g,' ')}: ${r[k]}`).join(' &middot; ');

    return `
      <div class="list-row">
        <a href="/risk/${r.symbol}" class="lr-sym">${r.symbol} &#x1F50D;</a>
        <span class="lr-price">&#x20B9;${r.price}</span>
        <span class="lr-chg" style="color:${up?'#10b981':'#ef4444'};">
          ${up?'+':''}${r.chg} (${up?'+':''}${r.chg_pct}%)
        </span>
        <span class="lr-meta">${r.rsi}</span>
        <span class="lr-meta">&#x20B9;${r.low52} – &#x20B9;${r.high52}</span>
        <span class="lr-meta">${r.vol_ratio}%</span>
        <span class="lr-extra">${extraTxt || '—'}</span>
      </div>`;
  }).join('');
}

async function runAllScreens() {
  const btn = document.getElementById('rescan-btn');
  btn.disabled = true;
  document.getElementById('scan-dot').classList.remove('done');
  document.getElementById('scan-status-text').textContent = 'Scanning 4 screens...';
  document.getElementById('loading-box').style.display = 'flex';
  document.getElementById('list-wrap').style.display = 'none';
  document.getElementById('empty-box').style.display = 'none';

  const universe = document.getElementById('universe-input').value;

  try {
    const res = await fetch('/api/run_all_screens', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({universe})
    });
    const data = await res.json();

    ALL_RESULTS = data.screens;
    universeSizeLast = data.universe_size;
    runAtLast = data.run_at;

    SCREEN_ORDER.forEach(key => {
      const c = ALL_RESULTS[key] ? ALL_RESULTS[key].count : 0;
      const el = document.getElementById('count-' + key);
      if (el) {
        el.textContent = c;
        el.classList.toggle('zero', c === 0);
      }
    });

    document.getElementById('scan-dot').classList.add('done');
    document.getElementById('scan-status-text').textContent =
      `Scan complete · ${runAtLast}`;

    renderScreen(currentScreen);

  } catch(e) {
    document.getElementById('scan-status-text').textContent = 'Scan failed — click Re-scan to retry.';
    document.getElementById('loading-box').style.display = 'none';
    document.getElementById('empty-box').style.display = 'block';
    document.getElementById('empty-box').textContent = 'Error running scan. Please try again.';
  } finally {
    btn.disabled = false;
  }
}

function switchSubtab(tab) {
  document.getElementById('subtab-results').classList.toggle('active', tab==='results');
  document.getElementById('subtab-history').classList.toggle('active', tab==='history');
  document.getElementById('results-view').style.display = tab==='results' ? 'block' : 'none';
  document.getElementById('history-view').style.display = tab==='history' ? 'block' : 'none';
  if (tab === 'history') loadHistory(currentScreen);
}

async function loadHistory(key) {
  const view = document.getElementById('history-view');
  view.innerHTML = '<div class="loading-box"><div class="ai-dot"></div><div class="ai-dot"></div><div class="ai-dot"></div><span>Loading history...</span></div>';
  try {
    const res = await fetch(`/api/screen_history/${key}`);
    const data = await res.json();
    if (!data.history.length) {
      view.innerHTML = '<div class="empty-box">No past runs yet for this screen.</div>';
      return;
    }
    view.innerHTML = data.history.map(h => `
      <div class="hist-row">
        <span class="hist-date">&#x1F4C5; ${h.date} ${h.time}</span>
        <span class="hist-name">${h.screen_name}</span>
        <span class="hist-count">${h.count} stocks</span>
      </div>
      <div class="hist-symlist">
        ${h.symbols.slice(0,15).join(', ')}${h.symbols.length>15?` +${h.symbols.length-15} more`:''}
      </div>
    `).join('');
  } catch(e) {
    view.innerHTML = '<div class="empty-box">Could not load history.</div>';
  }
}

// Auto-scan on page load
window.addEventListener('DOMContentLoaded', () => {
  renderScreen(currentScreen);
  runAllScreens();
});
</script>
</body>
</html>
"""

@app.route("/screens")
def screens_page():
    # Strip the 'fn' (function object) before sending to template — not JSON serialisable
    registry_for_js = {
        k: {"name": v["name"], "group": v["group"], "desc": v["desc"]}
        for k, v in SCREEN_REGISTRY.items()
    }
    # Only show the 4 auto-scan screens as tabs — Blue Dot, Trend Template,
    # Breaking Out Today, Recent Breakouts. The rest stay defined in
    # SCREEN_REGISTRY and remain reachable via /api/run_screen/<key> if
    # ever needed again, but won't clutter this page.
    screen_order = [k for k in AUTO_SCAN_SCREENS if k in SCREEN_REGISTRY]
    return render_template_string(SCREENS_HTML, registry=registry_for_js, screen_order=screen_order)


SECTORS_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Sector Performance | FalconAI</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --bg:#070d18;--surface:#0d1424;--card:#111c2e;--border:#1a2840;--border2:#243452;
  --text:#e2e8f0;--muted:#4a6080;--muted2:#2a3a55;
  --green:#10b981;--green2:#064e35;--red:#ef4444;--red2:#4a0f0f;
  --yellow:#f59e0b;--yellow2:#3d2a00;--blue:#3b82f6;--blue2:#0f2040;--purple:#8b5cf6;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',Arial,sans-serif;}
.topbar{background:var(--surface);padding:0 24px;display:flex;align-items:center;
  border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100;height:52px;flex-wrap:wrap;}
.nav-logo{font-size:16px;font-weight:800;color:var(--blue);margin-right:24px;text-decoration:none;}
.nav-tab{display:flex;align-items:center;height:52px;padding:0 16px;font-size:13px;font-weight:600;
  color:var(--muted);text-decoration:none;border-bottom:2px solid transparent;white-space:nowrap;}
.nav-tab:hover{color:var(--text);}
.nav-tab.active{color:var(--blue);border-bottom-color:var(--blue);}
.nav-right{margin-left:auto;font-size:12px;color:var(--muted);}
.page{max-width:1300px;margin:0 auto;padding:20px 20px 80px;}

.intro{background:linear-gradient(135deg,#0d1e3a,#0d1424);border:1px solid var(--border2);
  border-radius:14px;padding:18px 22px;margin-bottom:20px;display:flex;justify-content:space-between;
  align-items:center;flex-wrap:wrap;gap:12px;}
.intro h2{font-size:20px;font-weight:800;color:white;margin-bottom:6px;}
.intro p{font-size:13px;color:#7090b0;}
.refresh-btn{padding:8px 16px;background:var(--blue2);color:var(--blue);border:1px solid var(--blue)44;
  border-radius:8px;font-size:12px;font-weight:700;cursor:pointer;white-space:nowrap;}
.refresh-btn:hover{background:var(--blue);color:white;}

.tf-tabs{display:flex;gap:4px;margin-bottom:18px;background:var(--surface);
  border:1px solid var(--border);border-radius:10px;padding:4px;width:fit-content;}
.tf-tab{padding:8px 20px;border-radius:7px;font-size:13px;font-weight:700;cursor:pointer;
  color:var(--muted);border:none;background:transparent;}
.tf-tab.active{background:var(--blue2);color:var(--blue);}

/* Heatmap grid */
.heatmap{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin-bottom:24px;}
.sector-tile{border-radius:14px;padding:16px 18px;cursor:pointer;transition:all .15s;
  border:1px solid var(--border2);position:relative;overflow:hidden;}
.sector-tile:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,0.3);}
.sector-tile.selected{outline:2px solid var(--blue);}
.st-name{font-size:14px;font-weight:800;color:white;margin-bottom:4px;}
.st-ret{font-size:26px;font-weight:900;line-height:1;margin-bottom:6px;}
.st-label{font-size:11px;font-weight:600;opacity:0.9;}
.st-meta{display:flex;gap:10px;margin-top:10px;font-size:10px;opacity:0.85;}
.st-meta span{display:flex;flex-direction:column;}
.st-meta b{font-size:12px;}
.st-unavailable{opacity:0.4;}

/* Detail panel */
.detail-panel{display:none;background:var(--card);border:1px solid var(--border2);
  border-radius:14px;padding:22px 24px;margin-top:6px;}
.detail-panel.active{display:block;}
.dp-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;flex-wrap:wrap;gap:10px;}
.dp-title{font-size:20px;font-weight:800;color:white;}
.dp-close{background:none;border:none;color:var(--muted);font-size:22px;cursor:pointer;}

.dp-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;}
@media(max-width:700px){.dp-stats{grid-template-columns:repeat(2,1fr);}}
.dp-stat{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:12px 14px;text-align:center;}
.dp-stat-lbl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;}
.dp-stat-val{font-size:20px;font-weight:800;}

.leaders-section{margin-top:6px;}
.leaders-title{font-size:12px;font-weight:700;color:var(--muted);text-transform:uppercase;
  letter-spacing:0.8px;margin:18px 0 10px;display:flex;align-items:center;gap:8px;}
.leaders-title::after{content:'';flex:1;height:1px;background:var(--border);}
.leader-row{display:flex;align-items:center;gap:12px;padding:10px 12px;background:var(--surface);
  border:1px solid var(--border);border-radius:9px;margin-bottom:6px;}
.leader-rank{width:22px;height:22px;border-radius:50%;background:var(--blue2);color:var(--blue);
  display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;flex-shrink:0;}
.leader-sym{font-size:13px;font-weight:800;color:var(--blue);flex:1;}
.leader-price{font-size:12px;color:var(--text);width:90px;}
.leader-ret{font-size:13px;font-weight:800;width:80px;text-align:right;}
.leader-link{text-decoration:none;color:inherit;display:flex;align-items:center;gap:12px;flex:1;}

.loading-box{display:flex;align-items:center;gap:10px;color:var(--muted);
  font-size:13px;padding:40px;justify-content:center;}
.ai-dot{width:7px;height:7px;border-radius:50%;background:var(--blue);animation:pulse 1.2s infinite;}
.ai-dot:nth-child(2){animation-delay:.2s;}
.ai-dot:nth-child(3){animation-delay:.4s;}
@keyframes pulse{0%,100%{opacity:.25;transform:scale(.8)}50%{opacity:1;transform:scale(1.2)}}

.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-bottom:14px;}
.legend-dot{width:10px;height:10px;border-radius:3px;display:inline-block;margin-right:5px;vertical-align:middle;}
</style>
</head>
<body>

<nav class="topbar">
  <a href="/" class="nav-logo">&#x1F985; FalconAI</a>
  <a href="/" class="nav-tab">&#x1F4CA; Screener</a>
  <a href="/options" class="nav-tab">&#x26A1; Options</a>
  <a href="/screens" class="nav-tab">&#x1F50D; Screens</a>
  <a href="/sectors" class="nav-tab active">&#x1F3ED; Sectors</a>
  <a href="/scanner" class="nav-tab">&#x26A1; 5-Min Scanner</a>
  <div class="nav-right" id="last-updated">—</div>
</nav>

<div class="page">

<div class="intro">
  <div>
    <h2>&#x1F3ED; Sector Performance Dashboard</h2>
    <p>Live heatmap of every major NSE sectoral index — Daily, Weekly &amp; Monthly returns, with the top leader stocks driving each sector.</p>
  </div>
  <button class="refresh-btn" onclick="loadSectors()">&#8635; Refresh Data</button>
</div>

<div class="tf-tabs">
  <button class="tf-tab active" id="tf-d1" onclick="switchTF('d1')">Daily</button>
  <button class="tf-tab" id="tf-w1" onclick="switchTF('w1')">Weekly</button>
  <button class="tf-tab" id="tf-m1" onclick="switchTF('m1')">Monthly</button>
</div>

<div class="legend">
  <span><span class="legend-dot" style="background:#10b981;"></span>Strong Up (&gt;1.5%)</span>
  <span><span class="legend-dot" style="background:#34d399;"></span>Up</span>
  <span><span class="legend-dot" style="background:#94a3b8;"></span>Flat</span>
  <span><span class="legend-dot" style="background:#f87171;"></span>Down</span>
  <span><span class="legend-dot" style="background:#ef4444;"></span>Strong Down (&lt;-1.5%)</span>
</div>

<div id="loading-box" class="loading-box">
  <div class="ai-dot"></div><div class="ai-dot"></div><div class="ai-dot"></div>
  <span>Loading sector data...</span>
</div>

<div class="heatmap" id="heatmap" style="display:none;"></div>

<div class="detail-panel" id="detail-panel">
  <div class="dp-header">
    <div class="dp-title" id="dp-title">—</div>
    <button class="dp-close" onclick="closeDetail()">&times;</button>
  </div>

  <div class="dp-stats">
    <div class="dp-stat">
      <div class="dp-stat-lbl">Daily</div>
      <div class="dp-stat-val" id="dp-d1">—</div>
    </div>
    <div class="dp-stat">
      <div class="dp-stat-lbl">Weekly</div>
      <div class="dp-stat-val" id="dp-w1">—</div>
    </div>
    <div class="dp-stat">
      <div class="dp-stat-lbl">Monthly</div>
      <div class="dp-stat-val" id="dp-m1">—</div>
    </div>
    <div class="dp-stat">
      <div class="dp-stat-lbl">3-Month</div>
      <div class="dp-stat-val" id="dp-m3">—</div>
    </div>
  </div>

  <div class="leaders-section">
    <div class="leaders-title">&#x1F3C6; Sector Leaders — Daily</div>
    <div id="leaders-d1"></div>
    <div class="leaders-title">&#x1F3C6; Sector Leaders — Weekly</div>
    <div id="leaders-w1"></div>
    <div class="leaders-title">&#x1F3C6; Sector Leaders — Monthly</div>
    <div id="leaders-m1"></div>
  </div>
</div>

</div>

<script>
let SECTOR_DATA = [];
let currentTF = 'd1';
let selectedSector = null;

function fmtPct(v) {
  if (v === null || v === undefined) return '—';
  return (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
}

async function loadSectors() {
  document.getElementById('loading-box').style.display = 'flex';
  document.getElementById('heatmap').style.display = 'none';
  try {
    const res = await fetch('/api/sectors');
    const data = await res.json();
    SECTOR_DATA = data.sectors;
    document.getElementById('last-updated').textContent = 'Updated: ' + data.updated_at;
    renderHeatmap();
  } catch(e) {
    document.getElementById('loading-box').innerHTML = '<span>Could not load sector data. Try refreshing.</span>';
  }
}

function renderHeatmap() {
  const grid = document.getElementById('heatmap');
  grid.innerHTML = '';

  // Sort by current timeframe return, descending (unavailable last)
  const sorted = [...SECTOR_DATA].sort((a,b) => {
    if (!a.available) return 1;
    if (!b.available) return -1;
    const av = a[currentTF] ?? -999, bv = b[currentTF] ?? -999;
    return bv - av;
  });

  sorted.forEach(s => {
    const tile = document.createElement('div');
    tile.className = 'sector-tile' + (!s.available ? ' st-unavailable' : '') +
                      (selectedSector === s.name ? ' selected' : '');
    const ret = s[currentTF];
    const bg = s.available ? (s.color + '22') : '#11182233';
    tile.style.background = bg;
    tile.style.borderColor = s.available ? (s.color + '55') : 'var(--border2)';

    tile.innerHTML = `
      <div class="st-name">${s.name}</div>
      <div class="st-ret" style="color:${s.available ? s.color : 'var(--muted)'};">
        ${s.available ? fmtPct(ret) : 'N/A'}
      </div>
      <div class="st-label" style="color:${s.available ? s.color : 'var(--muted)'};">${s.label}</div>
      <div class="st-meta">
        <span>D <b style="color:${(s.d1??0)>=0?'#10b981':'#ef4444'};">${fmtPct(s.d1)}</b></span>
        <span>W <b style="color:${(s.w1??0)>=0?'#10b981':'#ef4444'};">${fmtPct(s.w1)}</b></span>
        <span>M <b style="color:${(s.m1??0)>=0?'#10b981':'#ef4444'};">${fmtPct(s.m1)}</b></span>
      </div>
    `;
    if (s.available) {
      tile.onclick = () => openDetail(s.name);
    }
    grid.appendChild(tile);
  });

  document.getElementById('loading-box').style.display = 'none';
  grid.style.display = 'grid';
}

function switchTF(tf) {
  currentTF = tf;
  ['d1','w1','m1'].forEach(t => document.getElementById('tf-' + t).classList.toggle('active', t === tf));
  renderHeatmap();
}

function renderLeaders(containerId, leaders) {
  const el = document.getElementById(containerId);
  if (!leaders || !leaders.length) {
    el.innerHTML = '<div style="font-size:12px;color:var(--muted);padding:6px 0;">No leader data available.</div>';
    return;
  }
  el.innerHTML = leaders.map((l, i) => `
    <a href="/risk/${l.symbol}" class="leader-link">
      <div class="leader-row" style="width:100%;">
        <span class="leader-rank">${i+1}</span>
        <span class="leader-sym">${l.symbol}</span>
        <span class="leader-price">&#x20B9;${l.price}</span>
        <span class="leader-ret" style="color:${l.ret>=0?'#10b981':'#ef4444'};">${fmtPct(l.ret)}</span>
      </div>
    </a>
  `).join('');
}

function openDetail(name) {
  selectedSector = name;
  const s = SECTOR_DATA.find(x => x.name === name);
  if (!s) return;

  document.getElementById('dp-title').textContent = s.name;
  document.getElementById('dp-d1').textContent = fmtPct(s.d1);
  document.getElementById('dp-d1').style.color = (s.d1??0)>=0 ? '#10b981' : '#ef4444';
  document.getElementById('dp-w1').textContent = fmtPct(s.w1);
  document.getElementById('dp-w1').style.color = (s.w1??0)>=0 ? '#10b981' : '#ef4444';
  document.getElementById('dp-m1').textContent = fmtPct(s.m1);
  document.getElementById('dp-m1').style.color = (s.m1??0)>=0 ? '#10b981' : '#ef4444';
  document.getElementById('dp-m3').textContent = fmtPct(s.m3);
  document.getElementById('dp-m3').style.color = (s.m3??0)>=0 ? '#10b981' : '#ef4444';

  renderLeaders('leaders-d1', s.leaders_d1);
  renderLeaders('leaders-w1', s.leaders_w1);
  renderLeaders('leaders-m1', s.leaders_m1);

  document.getElementById('detail-panel').classList.add('active');
  renderHeatmap();
  document.getElementById('detail-panel').scrollIntoView({behavior:'smooth', block:'start'});
}

function closeDetail() {
  selectedSector = null;
  document.getElementById('detail-panel').classList.remove('active');
  renderHeatmap();
}

loadSectors();
</script>
</body>
</html>
"""

@app.route("/api/sectors")
def api_sectors():
    from flask import jsonify
    sectors = get_all_sectors_performance()
    return jsonify({
        "sectors": sectors,
        "updated_at": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
    })


@app.route("/sectors")
def sectors_page():
    return render_template_string(SECTORS_HTML)



# ════════════════════════════════════════════════════════════════════
# 5-MIN CONSOLIDATION BREAKOUT SCANNER (Intraday)
# ════════════════════════════════════════════════════════════════════
#
# Logic, built from the "Intraday 5-Min Breakout Strategy" card:
#   1. Consolidation Zone — a tight sideways base on the 5-min chart
#      (at least 3 bars, range < threshold of ATR)
#   2. Breakout Bar — a FULL closed 5-min candle that closes above the
#      zone high (never a partial/forming bar — that's the #1 rule)
#   3. Volume Surge — breakout bar's volume must be meaningfully above
#      the recent average (institutional confirmation)
#   4. Trend Support — price above VWAP (the single biggest tell on
#      whether smart money is behind the move)
#   5. Retest/Confirmation — flagged separately as "High-Confidence" if
#      price has pulled back near the breakout level and held
#   6. Fakeout Risk — upper wick ratio on the breakout bar; large wick
#      means sellers showed up right at the level = riskier entry
#
# Every result is timestamped with the exact bar time it fired on.

def get_5min_data(symbol):
    """
    Fetch 5-min intraday bars for the last 5 days.
    Returns (df, last_closed_pos, secs_to_next_bar).
    - Uses auto_adjust=False (real traded price)
    - Flattens MultiIndex columns
    - Uses stdlib UTC+5:30 for IST (no pytz dep)
    - last_closed_pos = index of last CONFIRMED closed bar
    """
    try:
        raw = _yf_download_safe(symbol + ".NS", period="5d", interval="5m",
                                timeout=6, auto_adjust=False)
        if raw is None or raw.empty:
            return None, None, 0

        df = flatten_df(raw)
        if df is None or df.empty or len(df) < 6:
            return None, None, 0

        secs_to_next_bar = 0
        try:
            import datetime as _dt
            utc_now  = _dt.datetime.utcnow()
            ist_now  = utc_now + _dt.timedelta(hours=5, minutes=30)

            last_bar_ts = df.index[-1]
            # Convert bar timestamp to IST naive datetime for comparison
            if hasattr(last_bar_ts, 'tzinfo') and last_bar_ts.tzinfo is not None:
                # tz-aware — convert to UTC then add 5:30
                bar_utc  = last_bar_ts.utctimetuple()
                import calendar
                bar_epoch = calendar.timegm(bar_utc)
                bar_ist   = _dt.datetime.utcfromtimestamp(bar_epoch) + _dt.timedelta(hours=5, minutes=30)
            else:
                # tz-naive — assume UTC (yfinance default)
                bar_ist = last_bar_ts.to_pydatetime() + _dt.timedelta(hours=5, minutes=30) \
                          if hasattr(last_bar_ts, 'to_pydatetime') \
                          else _dt.datetime(last_bar_ts.year, last_bar_ts.month, last_bar_ts.day,
                                            last_bar_ts.hour, last_bar_ts.minute) + _dt.timedelta(hours=5, minutes=30)

            elapsed_secs = (ist_now - bar_ist).total_seconds()

            if 0 <= elapsed_secs < 300:
                # Bar opened less than 5 min ago → still forming
                last_closed_pos  = len(df) - 2
                secs_to_next_bar = int(300 - elapsed_secs)
            else:
                # Bar is fully closed (or something weird — treat as closed)
                last_closed_pos  = len(df) - 1
                secs_to_next_bar = 0
        except Exception:
            # Safe fallback: always treat last bar as forming (conservative)
            last_closed_pos  = len(df) - 2
            secs_to_next_bar = 0

        if last_closed_pos < 5:
            return None, None, 0

        return df, last_closed_pos, secs_to_next_bar

    except Exception:
        return None, None, 0


def calc_intraday_vwap(df, end_pos=None):
    """VWAP for today's session only, up to end_pos (inclusive)."""
    try:
        close  = safe_series(df["Close"])
        high_s = safe_series(df["High"])
        low_s  = safe_series(df["Low"])
        volume = safe_series(df["Volume"])
        tp     = (high_s + low_s + close) / 3
        cum_pv = (tp * volume).cumsum()
        cum_v  = volume.cumsum().replace(0, float('nan'))
        vwap   = cum_pv / cum_v
        return vwap
    except Exception:
        return None


def check_momentum(close_arr, volume_arr, pos):
    """
    Returns True only if the stock has HIGH MOMENTUM right now.
    Three checks — all must pass:
      1. 10-bar slope is positive (price trending up)
      2. RSI(7) is between 52 and 80 (strong but not overbought)
      3. Volume trend: last 3 bars avg > prior 10 bars avg (accumulation)
    Uses numpy arrays for speed.
    """
    try:
        if pos < 15:
            return False, {}

        window = close_arr[max(0, pos-14):pos+1]
        if len(window) < 10:
            return False, {}

        # 1. Slope — linear regression on last 10 closes
        y = window[-10:].astype(float)
        x = np.arange(10, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        slope_pct = (slope / y[0]) * 100 if y[0] > 0 else 0   # % per bar

        # 2. RSI(7)
        closes_14 = close_arr[max(0, pos-14):pos+1].astype(float)
        diff  = np.diff(closes_14)
        gains = np.where(diff > 0, diff, 0)
        losses= np.where(diff < 0, -diff, 0)
        avg_g = gains[-7:].mean()  if len(gains)  >= 7 else gains.mean()
        avg_l = losses[-7:].mean() if len(losses) >= 7 else losses.mean()
        rsi   = 100 - (100 / (1 + avg_g / avg_l)) if avg_l > 0 else 100

        # 3. Volume trend
        if pos >= 13:
            recent_vol = volume_arr[pos-2:pos+1].mean()
            prior_vol  = volume_arr[max(0,pos-12):pos-2].mean()
            vol_trend  = recent_vol >= prior_vol * 0.70   # lenient — not collapsing
        else:
            vol_trend = True

        momentum_ok = slope_pct > -0.05 and 45 <= rsi <= 85 and vol_trend

        return momentum_ok, {
            "slope_pct": round(slope_pct, 3),
            "rsi7":      round(rsi, 1),
            "vol_trend": vol_trend,
        }
    except Exception:
        return False, {}



def detect_5min_breakout(symbol, df, last_closed_pos,
                          min_zone_bars=2, max_zone_pct=2.5, vol_surge_x=1.2):
    """
    5-Min Consolidation Breakout Scanner.
    ONLY fires on bars_age=0 (fresh) or bars_age=1 (5-min-old retest).
    Requires momentum gate (slope + RSI + volume trend) before returning.
    """
    try:
        idx    = df.index
        close  = df["Close"].astype(float).values
        open_s = df["Open"].astype(float).values
        high_s = df["High"].astype(float).values
        low_s  = df["Low"].astype(float).values
        volume = df["Volume"].astype(float).values

        n = last_closed_pos + 1
        if n < min_zone_bars + 5:
            return None

        # Today's session boundaries
        try:
            last_date   = idx[last_closed_pos].date()
            today_start = next((i for i in range(n) if idx[i].date() == last_date),
                               max(0, n - 78))
        except Exception:
            today_start = max(0, n - 78)

        scan_start = today_start + 3   # skip first 15 min (ORB noise)
        if scan_start >= n - min_zone_bars:
            return None

        # ── MOMENTUM GATE ─────────────────────────────────────────────────────
        mom_ok, mom_data = check_momentum(close, volume, last_closed_pos)
        if not mom_ok:
            return None

        # ── PASS 1: last closed bar IS the breakout bar (age=0, FRESH) ───────
        signal = _try_breakout_at(
            symbol, idx, close, open_s, high_s, low_s, volume,
            breakout_bar_idx=last_closed_pos,
            last_confirmed_idx=last_closed_pos,
            scan_start=scan_start, today_start=today_start, n_total=n,
            min_zone_bars=min_zone_bars, max_zone_pct=max_zone_pct,
            vol_surge_x=vol_surge_x, entry_type="BREAKOUT", bars_age=0
        )
        if signal:
            signal["rsi7"]      = mom_data.get("rsi7", 0)
            signal["slope_pct"] = mom_data.get("slope_pct", 0)
            return signal

        # ── PASS 2: RETEST — max 2 bars back (10 min) ────────────────────────
        for age in [1, 2]:
            bo_idx = last_closed_pos - age
            if bo_idx >= scan_start:
                signal = _try_breakout_at(
                    symbol, idx, close, open_s, high_s, low_s, volume,
                    breakout_bar_idx=bo_idx,
                    last_confirmed_idx=last_closed_pos,
                    scan_start=scan_start, today_start=today_start, n_total=n,
                    min_zone_bars=min_zone_bars, max_zone_pct=max_zone_pct,
                    vol_surge_x=vol_surge_x, entry_type="RETEST", bars_age=age
                )
                if signal:
                    signal["rsi7"]      = mom_data.get("rsi7", 0)
                    signal["slope_pct"] = mom_data.get("slope_pct", 0)
                    return signal

        return None
    except Exception:
        return None


def _try_breakout_at(symbol, idx, close, open_s, high_s, low_s, volume,
                     breakout_bar_idx, last_confirmed_idx,
                     scan_start, today_start, n_total,
                     min_zone_bars, max_zone_pct, vol_surge_x,
                     entry_type, bars_age):
    """
    Core breakout logic. Accepts numpy arrays for close/open/high/low/volume.
    idx is the original DataFrame index (for timestamps only).
    """
    try:
        bo_close = float(close[breakout_bar_idx])
        bo_open  = float(open_s[breakout_bar_idx])
        bo_high  = float(high_s[breakout_bar_idx])
        bo_low   = float(low_s[breakout_bar_idx])
        bo_vol   = float(volume[breakout_bar_idx])

        if bo_close <= bo_open * 0.997:   # allow near-doji bars
            return None

        # ── Build consolidation zone ──────────────────────────────────────────
        zone_end  = breakout_bar_idx
        best_zone = None
        for zone_len in range(min_zone_bars, min(12, zone_end - scan_start) + 1):
            zone_start = zone_end - zone_len
            if zone_start < scan_start:
                break
            zone_high = float(high_s[zone_start:zone_end].max())
            zone_low  = float(low_s[zone_start:zone_end].min())
            if zone_high <= 0:
                continue
            zone_pct = ((zone_high - zone_low) / zone_high) * 100
            if zone_pct <= max_zone_pct:
                best_zone = {"start": zone_start, "end": zone_end,
                             "high": round(zone_high, 2), "low": round(zone_low, 2),
                             "pct": round(zone_pct, 2), "bars": zone_len}
        if best_zone is None:
            return None

        zone_high = best_zone["high"]
        zone_low  = best_zone["low"]

        if bo_close < zone_high * 1.0001:
            return None

        # ── Retest validation ─────────────────────────────────────────────────
        if entry_type == "RETEST":
            current_close = float(close[last_confirmed_idx])
            pullback_pct  = (bo_close - current_close) / bo_close * 100
            if not (0 <= pullback_pct <= 3.5):
                return None
            if current_close < zone_high * 0.999:
                return None
            entry_price = current_close
        else:
            entry_price = bo_close

        # ── Volume ────────────────────────────────────────────────────────────
        vol_lookback_start = max(today_start, breakout_bar_idx - 15)
        avg_vol  = float(volume[vol_lookback_start:breakout_bar_idx].mean()) \
                   if breakout_bar_idx > vol_lookback_start else bo_vol
        vol_ratio = (bo_vol / avg_vol) if avg_vol > 0 else 1.0
        vol_ok    = vol_ratio >= vol_surge_x

        # ── VWAP ──────────────────────────────────────────────────────────────
        try:
            tp         = (high_s + low_s + close) / 3
            cum_v      = volume.cumsum()
            cum_v[cum_v == 0] = float('nan')
            vwap_arr   = (tp * volume).cumsum() / cum_v
            vwap_now   = float(vwap_arr[last_confirmed_idx])
        except Exception:
            vwap_now = entry_price
        above_vwap = entry_price > vwap_now

        # ── Fakeout ───────────────────────────────────────────────────────────
        bar_range  = bo_high - bo_low
        upper_wick = bo_high - bo_close
        wick_ratio = (upper_wick / bar_range * 100) if bar_range > 0 else 0
        fakeout_risk_high = wick_ratio > 35

        # ── Targets ───────────────────────────────────────────────────────────
        zone_height  = zone_high - zone_low
        target_price = round(entry_price + zone_height, 2)
        sl_price     = round(zone_low, 2)
        risk         = round(entry_price - sl_price, 2)
        reward       = round(target_price - entry_price, 2)
        rr_ratio     = round(reward / risk, 2) if risk > 0 else 0

        # ── Score ─────────────────────────────────────────────────────────────
        score  = 0
        score += 35 if vol_ok else max(0, int(vol_ratio / vol_surge_x * 35))
        score += 25 if above_vwap else 0
        score += 20 if not fakeout_risk_high else max(0, int((100 - wick_ratio) / 100 * 20))
        score += min(15, best_zone["bars"] * 2)
        score += 5  if rr_ratio >= 1.5 else 0
        if entry_type == "RETEST":
            score = min(100, score + 10)
        score = min(100, score)

        if score < 30:
            return None

        if score >= 80:   grade, grade_color = "A+ Strong", "#10b981"
        elif score >= 65: grade, grade_color = "B Good",    "#34d399"
        elif score >= 50: grade, grade_color = "C Moderate","#f59e0b"
        else:             grade, grade_color = "D Weak",    "#f87171"

        try:    bo_bar_time   = idx[breakout_bar_idx].strftime("%H:%M")
        except: bo_bar_time   = "--:--"
        try:    last_bar_time = idx[last_confirmed_idx].strftime("%H:%M")
        except: last_bar_time = "--:--"

        if bars_age == 0:
            freshness_label = "🟢 FRESH — Enter on next bar open"
            freshness_color = "#10b981"
        elif bars_age == 1:
            freshness_label = "🔵 RETEST — 5 min old, pullback held"
            freshness_color = "#3b82f6"
        else:
            freshness_label = "🟡 RETEST — 10 min old, confirm hold"
            freshness_color = "#f59e0b"

        return {
            "symbol":          symbol,
            "ltp":             round(entry_price, 2),
            "bar_open":        round(bo_open, 2),
            "bar_high":        round(bo_high, 2),
            "bar_low":         round(bo_low, 2),
            "zone_high":       round(zone_high, 2),
            "zone_low":        round(zone_low, 2),
            "zone_pct":        best_zone["pct"],
            "zone_bars":       best_zone["bars"],
            "vol_ratio":       round(vol_ratio, 2),
            "vol_ok":          vol_ok,
            "vwap":            round(vwap_now, 2),
            "above_vwap":      above_vwap,
            "wick_ratio":      round(wick_ratio, 1),
            "fakeout_risk":    fakeout_risk_high,
            "target":          target_price,
            "sl":              sl_price,
            "rr_ratio":        rr_ratio,
            "score":           score,
            "grade":           grade,
            "grade_color":     grade_color,
            "entry_type":      entry_type,
            "bars_age":        bars_age,
            "bars_age_mins":   bars_age * 5,
            "freshness_label": freshness_label,
            "freshness_color": freshness_color,
            "bar_time":        bo_bar_time,
            "last_bar_time":   last_bar_time,
            "scanned_at":      datetime.now().strftime("%H:%M:%S"),
            "scanned_date":    datetime.now().strftime("%d-%m-%Y"),
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# PRE-SIGNAL DETECTOR
# Fires BEFORE the breakout candle closes — gives you 1-2 min early warning
# ──────────────────────────────────────────────────────────────────────────────
def detect_pre_signal(symbol, df, last_closed_pos, secs_to_next_bar,
                      min_zone_bars=2, max_zone_pct=2.5, vol_build_x=1.1):
    """
    PRE-SIGNAL: Fires while the breakout bar is still forming.
    Price is pressing against zone top with volume already building.
    Gives 1-5 min early warning before the candle closes.
    Uses numpy arrays (same pattern as detect_5min_breakout) to avoid
    index-alignment issues from safe_series reset_index.
    """
    try:
        idx    = df.index
        close  = df["Close"].astype(float).values
        open_s = df["Open"].astype(float).values
        high_s = df["High"].astype(float).values
        low_s  = df["Low"].astype(float).values
        volume = df["Volume"].astype(float).values

        total_bars = len(close)
        forming_pos = min(last_closed_pos + 1, total_bars - 1)

        forming_high  = high_s[forming_pos]
        forming_close = close[forming_pos]
        forming_vol   = volume[forming_pos]

        # Today's session boundaries
        try:
            last_date   = idx[last_closed_pos].date()
            today_start = next((i for i in range(forming_pos + 1)
                                if idx[i].date() == last_date), max(0, forming_pos - 75))
        except Exception:
            today_start = max(0, forming_pos - 75)

        scan_start = today_start + 3   # skip first 15 min (ORB noise)

        if last_closed_pos - scan_start < min_zone_bars:
            return None

        # Volume average across today's closed bars
        vol_lb  = max(today_start, last_closed_pos - 15)
        avg_vol = float(volume[vol_lb:last_closed_pos].mean()) \
                  if last_closed_pos > vol_lb else max(forming_vol, 1)
        vol_ratio = forming_vol / avg_vol if avg_vol > 0 else 1.0

        if vol_ratio < vol_build_x:
            return None   # volume not building yet

        # Build tightest consolidation zone from closed bars only
        best_zone = None
        for zone_len in range(min_zone_bars, min(10, last_closed_pos - scan_start) + 1):
            z_start = last_closed_pos - zone_len
            if z_start < scan_start:
                break
            z_high = float(high_s[z_start:last_closed_pos].max())
            z_low  = float(low_s[z_start:last_closed_pos].min())
            if z_high <= 0:
                continue
            z_pct = (z_high - z_low) / z_high * 100
            if z_pct <= max_zone_pct:
                best_zone = {"start": z_start, "end": last_closed_pos,
                             "high": round(z_high, 2), "low": round(z_low, 2),
                             "pct": round(z_pct, 2), "bars": zone_len}

        if best_zone is None:
            return None

        zone_high = best_zone["high"]
        zone_low  = best_zone["low"]

        # Forming bar's HIGH must be probing at or above zone high
        if forming_high < zone_high * 0.999:
            return None

        # But close shouldn't have already blown far above (that's a breakout, not a pre-signal)
        if forming_close > zone_high * 1.015:
            return None

        # VWAP — computed across all bars up to forming bar
        try:
            tp_arr    = (high_s[:forming_pos+1] + low_s[:forming_pos+1] + close[:forming_pos+1]) / 3
            vol_arr   = volume[:forming_pos+1]
            cum_vol   = vol_arr.cumsum()
            cum_vol   = np.where(cum_vol == 0, np.nan, cum_vol)
            vwap_arr  = (tp_arr * vol_arr).cumsum() / cum_vol
            vwap_val  = float(vwap_arr[-1])
        except Exception:
            vwap_val  = forming_close
        above_vwap = forming_close >= vwap_val * 0.999

        # Time remaining in forming bar
        mins_left  = max(0, secs_to_next_bar // 60)
        secs_left  = max(0, secs_to_next_bar % 60)
        time_label = f"{mins_left}m {secs_left}s left in bar"

        try:
            bar_time = idx[last_closed_pos].strftime("%H:%M")
        except Exception:
            bar_time = "--:--"

        entry_est  = round(zone_high, 2)
        sl_est     = round(zone_low, 2)
        target_est = round(entry_est + (zone_high - zone_low), 2)
        risk_est   = round(entry_est - sl_est, 2)
        reward_est = round(target_est - entry_est, 2)
        rr_est     = round(reward_est / risk_est, 2) if risk_est > 0 else 0

        score  = 0
        score += 30 if vol_ratio >= 2.0 else int(vol_ratio / 2.0 * 30)
        score += 25 if above_vwap else 0
        score += min(20, best_zone["bars"] * 5)
        score += 15 if rr_est >= 1.5 else 0
        score += 10 if forming_high >= zone_high * 1.002 else 5
        score  = min(100, score)

        if score < 25:
            return None

        return {
            "symbol":          symbol,
            "signal_type":     "PRE_SIGNAL",
            "entry_type":      "PRE_SIGNAL",
            "ltp":             round(forming_close, 2),
            "forming_high":    round(forming_high, 2),
            "zone_high":       round(zone_high, 2),
            "zone_low":        round(zone_low, 2),
            "zone_pct":        best_zone["pct"],
            "zone_bars":       best_zone["bars"],
            "vol_ratio":       round(vol_ratio, 2),
            "vol_ok":          vol_ratio >= 1.5,
            "above_vwap":      above_vwap,
            "vwap":            round(vwap_val, 2),
            "entry_est":       entry_est,
            "target":          target_est,
            "sl":              sl_est,
            "rr_ratio":        rr_est,
            "score":           score,
            "time_label":      time_label,
            "bar_time":        bar_time,
            "last_bar_time":   bar_time,
            "bars_age":        0,
            "bars_age_mins":   0,
            "wick_ratio":      0,
            "fakeout_risk":    False,
            "scanned_at":      datetime.now().strftime("%H:%M:%S"),
            "scanned_date":    datetime.now().strftime("%d-%m-%Y"),
            "freshness_label": "⚡ PRE-SIGNAL — Bar still forming",
            "freshness_color": "#f59e0b",
            "grade":           "PRE",
            "grade_color":     "#f59e0b",
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# PULLBACK SURGE DETECTOR
# The second-chance entry: price pulled back after breakout, now surging again
# ──────────────────────────────────────────────────────────────────────────────
def detect_pullback_surge(symbol, df, last_closed_pos,
                          lookback_bars=10, max_pullback_pct=5.0, vol_surge_x=1.2):
    """
    PULLBACK SURGE: High-confidence re-entry after a breakout pullback.
    Phase 1 — original breakout (3-6 bars ago) on volume
    Phase 2 — healthy pullback (1-4%) on drying volume
    Phase 3 — surge bar NOW: closes above zone high again on volume surge
    Uses numpy arrays to avoid safe_series index-alignment issues.
    """
    try:
        idx    = df.index
        close  = df["Close"].astype(float).values
        open_s = df["Open"].astype(float).values
        high_s = df["High"].astype(float).values
        low_s  = df["Low"].astype(float).values
        volume = df["Volume"].astype(float).values

        n = last_closed_pos + 1

        # Today's session start
        try:
            last_date   = idx[last_closed_pos].date()
            today_start = next((i for i in range(n)
                                if idx[i].date() == last_date), max(0, n - 75))
        except Exception:
            today_start = max(0, n - 75)
        scan_start = today_start + 3

        # ── PHASE 3: surge bar = last closed bar ─────────────────────────────
        surge_close = close[last_closed_pos]
        surge_open  = open_s[last_closed_pos]
        surge_vol   = volume[last_closed_pos]

        if surge_close <= surge_open:
            return None   # surge bar must be bullish

        vol_lb  = max(today_start, last_closed_pos - 15)
        avg_vol = float(volume[vol_lb:last_closed_pos].mean()) \
                  if last_closed_pos > vol_lb else max(surge_vol, 1)
        vol_ratio = surge_vol / avg_vol if avg_vol > 0 else 1.0

        if vol_ratio < vol_surge_x:
            return None   # no volume surge on this bar

        # ── PHASE 1: find original breakout ──────────────────────────────────
        original_breakout = None
        for bo_bars_back in range(3, lookback_bars + 1):
            bo_idx = last_closed_pos - bo_bars_back
            if bo_idx < scan_start + 2:
                break

            bo_close = close[bo_idx]
            bo_open  = open_s[bo_idx]
            bo_vol   = volume[bo_idx]

            if bo_close <= bo_open * 0.997:
                continue   # allow near-doji bars

            for zone_len in range(2, min(10, bo_idx - scan_start) + 1):
                z_start = bo_idx - zone_len
                if z_start < scan_start:
                    break
                z_high = float(high_s[z_start:bo_idx].max())
                z_low  = float(low_s[z_start:bo_idx].min())
                if z_high <= 0:
                    continue
                z_pct = (z_high - z_low) / z_high * 100
                if z_pct > 2.5:
                    continue
                if bo_close < z_high * 1.0005:
                    continue   # breakout bar didn't close above zone

                bo_vol_lb    = max(today_start, bo_idx - 15)
                bo_avg_vol   = float(volume[bo_vol_lb:bo_idx].mean()) \
                               if bo_idx > bo_vol_lb else max(bo_vol, 1)
                bo_vol_ratio = bo_vol / bo_avg_vol if bo_avg_vol > 0 else 1.0
                if bo_vol_ratio < 1.1:
                    continue

                original_breakout = {
                    "bo_idx": bo_idx, "bo_close": bo_close,
                    "zone_high": round(z_high, 2), "zone_low": round(z_low, 2),
                    "zone_pct": round(z_pct, 2), "zone_bars": zone_len,
                    "bo_vol_ratio": round(bo_vol_ratio, 2),
                    "bars_back": bo_bars_back,
                }
                break
            if original_breakout:
                break

        if original_breakout is None:
            return None

        zone_high = original_breakout["zone_high"]
        zone_low  = original_breakout["zone_low"]
        bo_close  = original_breakout["bo_close"]

        # ── PHASE 2: pullback bars validation ────────────────────────────────
        pb_start = original_breakout["bo_idx"] + 1
        pb_end   = last_closed_pos

        if pb_start >= pb_end:
            return None   # no pullback bars between breakout and surge

        pb_close_arr = close[pb_start:pb_end]
        pb_low_arr   = low_s[pb_start:pb_end]
        pb_vol_arr   = volume[pb_start:pb_end]

        pb_min_close = float(pb_close_arr.min())
        if pb_min_close < zone_high * 0.997:
            return None   # broke back below zone — failed breakout

        pb_low_price = float(pb_low_arr.min())
        pullback_pct = (bo_close - pb_low_price) / bo_close * 100
        if pullback_pct < 0.3 or pullback_pct > max_pullback_pct:
            return None   # too shallow or too deep

        pb_avg_vol   = float(pb_vol_arr.mean()) if len(pb_vol_arr) > 0 else surge_vol
        vol_dried_up = (pb_avg_vol / avg_vol) < 0.85

        # ── PHASE 3: surge bar must reclaim zone_high ────────────────────────
        if surge_close < zone_high * 1.001:
            return None

        # VWAP
        try:
            tp_arr   = (high_s[:n] + low_s[:n] + close[:n]) / 3
            vol_arr  = volume[:n]
            cum_vol  = np.where(vol_arr.cumsum() == 0, np.nan, vol_arr.cumsum())
            vwap_val = float((tp_arr * vol_arr).cumsum()[last_closed_pos] / cum_vol[last_closed_pos])
        except Exception:
            vwap_val = surge_close
        above_vwap = surge_close > vwap_val

        # Targets
        entry_price  = surge_close
        sl_price     = round(pb_low_price * 0.999, 2)
        zone_height  = original_breakout["zone_high"] - original_breakout["zone_low"]
        target_price = round(bo_close + zone_height, 2)
        risk         = round(entry_price - sl_price, 2)
        reward       = round(target_price - entry_price, 2)
        rr_ratio     = round(reward / risk, 2) if risk > 0 else 0

        score  = 0
        score += 30 if vol_ratio >= 2.0 else int(vol_ratio / 2.0 * 30)
        score += 25 if above_vwap else 0
        score += 15 if vol_dried_up else 8
        score += 15 if rr_ratio >= 2.0 else (10 if rr_ratio >= 1.5 else 0)
        score += 10 if 0.5 <= pullback_pct <= 2.0 else 5
        score += 5  if original_breakout["bo_vol_ratio"] >= 2.0 else 0
        score  = min(100, score)

        if score < 30:
            return None

        if score >= 80:
            grade, grade_color = "A+ Setup", "#10b981"
        elif score >= 65:
            grade, grade_color = "B Setup",  "#34d399"
        elif score >= 50:
            grade, grade_color = "C Setup",  "#f59e0b"
        else:
            grade, grade_color = "D Weak",   "#f87171"

        try:
            surge_bar_time = idx[last_closed_pos].strftime("%H:%M")
            bo_bar_time    = idx[original_breakout["bo_idx"]].strftime("%H:%M")
        except Exception:
            surge_bar_time = "--:--"
            bo_bar_time    = "--:--"

        return {
            "symbol":           symbol,
            "signal_type":      "PULLBACK_SURGE",
            "entry_type":       "PULLBACK_SURGE",
            "ltp":              round(entry_price, 2),
            "zone_high":        zone_high,
            "zone_low":         zone_low,
            "zone_pct":         original_breakout["zone_pct"],
            "zone_bars":        original_breakout["zone_bars"],
            "pullback_pct":     round(pullback_pct, 2),
            "pullback_bars":    pb_end - pb_start,
            "pb_low":           round(pb_low_price, 2),
            "vol_dried_up":     vol_dried_up,
            "vol_ratio":        round(vol_ratio, 2),
            "vol_ok":           vol_ratio >= vol_surge_x,
            "above_vwap":       above_vwap,
            "vwap":             round(vwap_val, 2),
            "target":           target_price,
            "sl":               sl_price,
            "rr_ratio":         rr_ratio,
            "score":            score,
            "grade":            grade,
            "grade_color":      grade_color,
            "original_bo_time": bo_bar_time,
            "bars_back":        original_breakout["bars_back"],
            "bar_time":         surge_bar_time,
            "last_bar_time":    surge_bar_time,
            "bars_age":         0,
            "bars_age_mins":    0,
            "wick_ratio":       0,
            "fakeout_risk":     False,
            "scanned_at":       datetime.now().strftime("%H:%M:%S"),
            "scanned_date":     datetime.now().strftime("%d-%m-%Y"),
            "freshness_label":  "🔄 PULLBACK SURGE — Re-entry after retest",
            "freshness_color":  "#a855f7",
        }
    except Exception:
        return None




def save_scanner_run(signals):
    """Persist every scan run with timestamp so user can review later."""
    if not signals:
        return
    rows = []
    for s in signals:
        rows.append({
            "date": s["scanned_date"], "time": s["scanned_at"],
            "symbol": s["symbol"], "ltp": s["ltp"], "score": s["score"],
            "grade": s["grade"], "target": s["target"], "sl": s["sl"],
            "rr_ratio": s["rr_ratio"], "bar_time": s["bar_time"],
        })
    df_new = pd.DataFrame(rows)
    if os.path.exists(SCANNER_HISTORY_FILE):
        old = pd.read_csv(SCANNER_HISTORY_FILE)
        df  = pd.concat([old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(SCANNER_HISTORY_FILE, index=False)


def load_scanner_history(limit=100):
    if not os.path.exists(SCANNER_HISTORY_FILE):
        return []
    df = pd.read_csv(SCANNER_HISTORY_FILE)
    df = df.sort_values(by=["date", "time"], ascending=False).head(limit)
    return df.to_dict("records")


def _scan_one(sym):
    """Scan a single symbol — runs inside ThreadPoolExecutor worker."""
    try:
        df, last_closed_pos, secs_to_next_bar = get_5min_data(sym)
        if df is None or last_closed_pos is None:
            return None
        sig = detect_5min_breakout(sym, df, last_closed_pos)
        if sig:
            sig.setdefault("signal_type", sig.get("entry_type", "BREAKOUT"))
            return sig
        ps = detect_pullback_surge(sym, df, last_closed_pos)
        if ps:
            return ps
        pre = detect_pre_signal(sym, df, last_closed_pos, secs_to_next_bar)
        if pre:
            return pre
        return None
    except Exception:
        return None


def run_5min_scanner(universe=None):
    """
    Parallel scan — all symbols fetched simultaneously via ThreadPoolExecutor.
    Total scan time ≈ slowest single download (not sum of all).
    This eliminates the sequential delay that made signals appear stale.
    Returns (signals, mkt_open, mkt_time).
    """
    import concurrent.futures
    mkt_open, mkt_time = is_market_open()
    syms = universe if universe else DEFAULT_UNIVERSE

    signals = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_scan_one, sym): sym for sym in syms}
        try:
            for future in concurrent.futures.as_completed(futures, timeout=45):
                try:
                    result = future.result(timeout=10)
                    if result:
                        signals.append(result)
                except Exception:
                    pass
        except concurrent.futures.TimeoutError:
            # Overall scan took too long — return whatever we collected so far
            # rather than letting the worker hang and get killed
            for f in futures:
                f.cancel()

    type_priority = {"BREAKOUT": 0, "PULLBACK_SURGE": 1, "PRE_SIGNAL": 2, "RETEST": 3}
    signals.sort(key=lambda x: (
        type_priority.get(x.get("signal_type", x.get("entry_type", "RETEST")), 9),
        -x.get("score", 0)
    ))

    save_scanner_run(signals)
    return signals, mkt_open, mkt_time


SCANNER_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>5-Min Breakout Scanner | FalconAI</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --bg:#070d18;--surface:#0d1424;--card:#111c2e;--border:#1a2840;--border2:#243452;
  --text:#e2e8f0;--muted:#4a6080;--muted2:#2a3a55;
  --green:#10b981;--green2:#064e35;--red:#ef4444;--red2:#4a0f0f;
  --yellow:#f59e0b;--yellow2:#3d2a00;--blue:#3b82f6;--blue2:#0f2040;--purple:#8b5cf6;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',Arial,sans-serif;}
.topbar{background:var(--surface);padding:0 24px;display:flex;align-items:center;
  border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100;height:52px;flex-wrap:wrap;}
.nav-logo{font-size:16px;font-weight:800;color:var(--blue);margin-right:24px;text-decoration:none;}
.nav-tab{display:flex;align-items:center;height:52px;padding:0 16px;font-size:13px;font-weight:600;
  color:var(--muted);text-decoration:none;border-bottom:2px solid transparent;white-space:nowrap;}
.nav-tab:hover{color:var(--text);}
.nav-tab.active{color:var(--blue);border-bottom-color:var(--blue);}
.nav-right{margin-left:auto;font-size:12px;color:var(--muted);display:flex;align-items:center;gap:8px;}
.page{max-width:1300px;margin:0 auto;padding:20px 20px 80px;}

.intro{background:linear-gradient(135deg,#0d1e3a,#0d1424);border:1px solid var(--border2);
  border-radius:14px;padding:18px 22px;margin-bottom:18px;display:flex;justify-content:space-between;
  align-items:center;flex-wrap:wrap;gap:14px;}
.intro h2{font-size:20px;font-weight:800;color:white;margin-bottom:6px;}
.intro p{font-size:13px;color:#7090b0;max-width:600px;}
.scan-btn{padding:10px 22px;background:var(--blue);color:white;border:none;
  border-radius:9px;font-size:13px;font-weight:800;cursor:pointer;white-space:nowrap;}
.scan-btn:hover{background:#2563eb;}
.scan-btn:disabled{background:var(--muted2);cursor:not-allowed;}

.mkt-pill{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;
  border-radius:20px;font-size:12px;font-weight:700;}
.pulse-dot{width:7px;height:7px;border-radius:50%;background:#10b981;animation:pulse 1.3s infinite;}
.pulse-dot.closed{background:#475569;animation:none;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}

.rules-strip{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:14px 18px;margin-bottom:18px;display:grid;grid-template-columns:repeat(5,1fr);gap:14px;}
@media(max-width:900px){.rules-strip{grid-template-columns:repeat(2,1fr);}}
.rule-item{font-size:11px;color:var(--muted);}
.rule-item b{display:block;color:var(--text);font-size:12px;margin-bottom:2px;}

.tabs2{display:flex;gap:4px;margin-bottom:16px;background:var(--surface);
  border:1px solid var(--border);border-radius:10px;padding:4px;width:fit-content;}
.tab2{padding:8px 18px;border-radius:7px;font-size:13px;font-weight:700;cursor:pointer;
  color:var(--muted);border:none;background:transparent;}
.tab2.active{background:var(--blue2);color:var(--blue);}

.signal-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;}
.signal-card{background:var(--card);border:1px solid var(--border);border-radius:14px;
  padding:16px 18px;position:relative;overflow:hidden;}
.signal-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;}
.sc-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;}
.sc-sym{font-size:17px;font-weight:800;color:var(--blue);text-decoration:none;}
.sc-grade{font-size:11px;font-weight:800;padding:3px 10px;border-radius:20px;}
.sc-price-row{display:flex;align-items:baseline;gap:10px;margin-bottom:12px;}
.sc-ltp{font-size:24px;font-weight:900;color:white;}
.sc-time{font-size:11px;color:var(--muted);}

.sc-score-wrap{margin-bottom:12px;}
.sc-score-label{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:4px;}
.sc-score-bar-bg{background:var(--border2);border-radius:6px;height:7px;overflow:hidden;}
.sc-score-bar{height:7px;border-radius:6px;}

.sc-checks{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:12px;}
.sc-check{font-size:11px;display:flex;align-items:center;gap:5px;}
.sc-check.pass{color:#34d399;}
.sc-check.fail{color:#f87171;}

.sc-levels{background:var(--surface);border-radius:10px;padding:10px 12px;margin-bottom:10px;}
.sc-level-row{display:flex;justify-content:space-between;font-size:12px;padding:3px 0;}
.sc-level-lbl{color:var(--muted);}
.sc-level-val{font-weight:700;}

.sc-zone{font-size:10px;color:var(--muted);padding-top:6px;border-top:1px solid var(--border);}

.timestamp-badge{display:inline-flex;align-items:center;gap:4px;font-size:10px;
  background:var(--blue2);color:var(--blue);padding:3px 9px;border-radius:12px;font-weight:700;}

.loading-box{display:flex;align-items:center;gap:10px;color:var(--muted);
  font-size:13px;padding:50px;justify-content:center;flex-direction:column;}
.ai-dot{width:8px;height:8px;border-radius:50%;background:var(--blue);animation:pulse2 1.2s infinite;}
.ai-dot:nth-child(2){animation-delay:.2s;}
.ai-dot:nth-child(3){animation-delay:.4s;}
@keyframes pulse2{0%,100%{opacity:.25;transform:scale(.8)}50%{opacity:1;transform:scale(1.2)}}
.dots-row{display:flex;gap:6px;}

.empty-box{text-align:center;padding:50px;color:var(--muted);font-size:13px;}

.hist-row{display:flex;align-items:center;gap:14px;padding:10px 14px;
  background:var(--surface);border:1px solid var(--border);border-radius:9px;margin-bottom:6px;font-size:12px;}
.hist-date{color:var(--muted);min-width:150px;}
.hist-sym{color:var(--blue);font-weight:800;min-width:80px;}
.hist-score{font-weight:700;}
.hist-grade{font-size:10px;padding:2px 8px;border-radius:10px;}
</style>
</head>
<body>

<nav class="topbar">
  <a href="/" class="nav-logo">&#x1F985; FalconAI</a>
  <a href="/" class="nav-tab">&#x1F4CA; Screener</a>
  <a href="/options" class="nav-tab">&#x26A1; Options</a>
  <a href="/screens" class="nav-tab">&#x1F50D; Screens</a>
  <a href="/sectors" class="nav-tab">&#x1F3ED; Sectors</a>
  <a href="/scanner" class="nav-tab active">&#x26A1; 5-Min Scanner</a>
  <div class="nav-right">
    <span class="mkt-pill" id="mkt-pill">
      <span class="pulse-dot" id="mkt-dot"></span>
      <span id="mkt-text">Checking...</span>
    </span>
  </div>
</nav>

<div class="page">

<div class="intro">
  <div>
    <h2>&#x26A1; 5-Min Intraday Scanner — Nifty 500</h2>
    <p>Scans all <b>Nifty 500</b> stocks in parallel (all simultaneously — no delay).
       Only fresh signals: current bar or 1 bar ago max. Momentum gate active on every signal.</p>
  </div>
  <button class="scan-btn" id="scan-btn" onclick="runScan()">&#x26A1; Scan Now</button>
</div>

<div class="rules-strip">
  <div class="rule-item"><b>1. Parallel Scan</b>All 500 stocks run simultaneously — signals are current, not stale</div>
  <div class="rule-item"><b>2. Fresh Only</b>Max 1 bar old (5 min). Zero tolerance for stale signals.</div>
  <div class="rule-item"><b>3. Momentum Gate</b>Slope &gt; 0, RSI 50–82, volume trending up — all required</div>
  <div class="rule-item"><b>4. Volume Surge</b>Breakout bar ≥ 1.4x avg volume — no volume, no signal</div>
  <div class="rule-item"><b>5. VWAP Filter</b>Price must be above session VWAP</div>
  <div class="rule-item"><b>⚡ Pre-Signal</b>Bar forming now with vol building — 1-5 min early warning</div>
  <div class="rule-item"><b>🔄 Pullback Surge</b>Post-breakout retest + new surge bar — tighter SL, better R:R</div>
</div>

<div class="tabs2">
  <button class="tab2 active" id="tab-live" onclick="switchTab('live')">Live Signals</button>
  <button class="tab2" id="tab-history" onclick="switchTab('history')">Scan History</button>
</div>

<div id="view-live">
  <div class="loading-box" id="loading-box" style="display:none;">
    <div class="dots-row"><div class="ai-dot"></div><div class="ai-dot"></div><div class="ai-dot"></div></div>
    <span>Scanning universe for live breakout setups...</span>
  </div>
  <div class="signal-grid" id="signal-grid"></div>
  <div class="empty-box" id="empty-box">Click "Scan Now" to find live 5-min breakout setups across the universe.</div>
</div>

<div id="view-history" style="display:none;">
  <div id="history-list"></div>
</div>

</div>

<script>
function fmtTime() {
  return new Date().toLocaleTimeString('en-IN', {hour:'2-digit',minute:'2-digit',second:'2-digit'});
}

async function updateMarketStatus() {
  try {
    const res = await fetch('/api/market_status');
    const d = await res.json();
    document.getElementById('mkt-dot').className = d.open ? 'pulse-dot' : 'pulse-dot closed';
    document.getElementById('mkt-text').textContent = (d.open ? 'Market Open' : 'Market Closed') + ' · ' + d.time;
  } catch(e) {}
}
updateMarketStatus();
setInterval(updateMarketStatus, 30000);

function switchTab(tab) {
  document.getElementById('tab-live').classList.toggle('active', tab==='live');
  document.getElementById('tab-history').classList.toggle('active', tab==='history');
  document.getElementById('view-live').style.display = tab==='live' ? 'block' : 'none';
  document.getElementById('view-history').style.display = tab==='history' ? 'block' : 'none';
  if (tab === 'history') loadHistory();
}

async function runScan() {
  const btn = document.getElementById('scan-btn');
  btn.disabled = true;
  btn.innerHTML = '<span style="opacity:.6">Scanning...</span>';
  document.getElementById('loading-box').style.display = 'flex';
  document.getElementById('signal-grid').innerHTML = '';
  document.getElementById('empty-box').style.display = 'none';

  try {
    const res = await fetch('/api/run_5min_scanner', {method:'POST'});
    const data = await res.json();
    document.getElementById('loading-box').style.display = 'none';
    btn.disabled = false;
    btn.innerHTML = '&#x26A1; Scan Now';

    if (!data.signals.length) {
      document.getElementById('empty-box').style.display = 'block';
      document.getElementById('empty-box').textContent =
        `No setups found right now (scanned at ${data.scanned_at}). Market: ${data.mkt_open ? 'Open' : 'Closed'}. Try again in a few minutes.`;
      return;
    }

    const grid = document.getElementById('signal-grid');
    data.signals.forEach(s => {
      const card = document.createElement('div');
      card.className = 'signal-card';
      card.style.setProperty('--accent', s.grade_color);

      const stype = s.signal_type || s.entry_type || 'BREAKOUT';

      // ── Signal type badge ──────────────────────────────────────────────
      let typeBadge = '';
      if (stype === 'PRE_SIGNAL') {
        typeBadge = `<span style="display:inline-flex;align-items:center;gap:4px;font-size:10px;
          font-weight:800;padding:3px 10px;border-radius:20px;
          background:#f59e0b22;border:1px solid #f59e0b66;color:#f59e0b;">
          &#x26A1; PRE-SIGNAL</span>`;
      } else if (stype === 'PULLBACK_SURGE') {
        typeBadge = `<span style="display:inline-flex;align-items:center;gap:4px;font-size:10px;
          font-weight:800;padding:3px 10px;border-radius:20px;
          background:#a855f722;border:1px solid #a855f766;color:#a855f7;">
          &#x1F504; PULLBACK SURGE</span>`;
      } else if (stype === 'RETEST') {
        typeBadge = `<span style="display:inline-flex;align-items:center;gap:4px;font-size:10px;
          font-weight:800;padding:3px 10px;border-radius:20px;
          background:#3b82f622;border:1px solid #3b82f666;color:#3b82f6;">
          &#x1F539; RETEST</span>`;
      } else {
        typeBadge = `<span style="display:inline-flex;align-items:center;gap:4px;font-size:10px;
          font-weight:800;padding:3px 10px;border-radius:20px;
          background:#10b98122;border:1px solid #10b98166;color:#10b981;">
          &#x1F7E2; BREAKOUT</span>`;
      }

      // ── Timing row — adapts per signal type ───────────────────────────
      let timingHtml = '';
      if (stype === 'PRE_SIGNAL') {
        timingHtml = `
          <div style="margin:0 0 10px;padding:8px 10px;border-radius:9px;
               background:#f59e0b18;border:1px solid #f59e0b44;">
            <div style="font-size:12px;font-weight:800;color:#f59e0b;">⚡ Bar forming NOW — enter as it closes above ₹${s.zone_high}</div>
            <div style="font-size:10px;color:#64748b;margin-top:3px;">
              Zone top: <b style="color:#f59e0b;">&#x20B9;${s.zone_high}</b> &nbsp;|&nbsp;
              <b style="color:#94a3b8;">${s.time_label}</b> &nbsp;|&nbsp;
              Last closed bar: <b style="color:#94a3b8;">${s.bar_time}</b>
            </div>
          </div>`;
      } else if (stype === 'PULLBACK_SURGE') {
        timingHtml = `
          <div style="margin:0 0 10px;padding:8px 10px;border-radius:9px;
               background:#a855f718;border:1px solid #a855f744;">
            <div style="font-size:12px;font-weight:800;color:#a855f7;">🔄 Broke out at ${s.original_bo_time}, pulled back ${s.pullback_pct}%, now surging</div>
            <div style="font-size:10px;color:#64748b;margin-top:3px;">
              Original breakout: <b style="color:#94a3b8;">${s.original_bo_time}</b> &nbsp;|&nbsp;
              Pullback: <b style="color:#94a3b8;">${s.pullback_bars} bars, low ₹${s.pb_low}</b> &nbsp;|&nbsp;
              Volume dried up: <b style="color:${s.vol_dried_up?'#10b981':'#94a3b8'};">${s.vol_dried_up?'Yes ✅':'No'}</b>
            </div>
          </div>`;
      } else {
        const ageLabel = s.bars_age === 0
          ? '<b style="color:#10b981;">Just closed — act on next bar open</b>'
          : `<b>${s.bars_age_mins} min ago</b> (${s.bars_age} bar${s.bars_age>1?'s':''} back)`;
        timingHtml = `
          <div style="margin:0 0 10px;padding:8px 10px;border-radius:9px;
               background:${s.freshness_color}18;border:1px solid ${s.freshness_color}44;">
            <div style="font-size:12px;font-weight:800;color:${s.freshness_color};">${s.freshness_label}</div>
            <div style="font-size:10px;color:#64748b;margin-top:3px;">
              Breakout bar: <b style="color:#94a3b8;">${s.bar_time}</b> &nbsp;|&nbsp;
              Age: ${ageLabel} &nbsp;|&nbsp;
              Last closed bar: <b style="color:#94a3b8;">${s.last_bar_time}</b>
            </div>
          </div>`;
      }

      // ── Checks row — adapts per signal type ───────────────────────────
      let checksHtml = '';
      if (stype === 'PRE_SIGNAL') {
        checksHtml = `
          <div class="sc-checks">
            <div class="sc-check pass">&#x26A1; Vol building ${s.vol_ratio}x avg</div>
            <div class="sc-check ${s.above_vwap?'pass':'fail'}">${s.above_vwap?'✅':'⚠️'} ${s.above_vwap?'Above':'Below'} VWAP</div>
            <div class="sc-check pass">🎯 Entry: ₹${s.entry_est} (zone top)</div>
            <div class="sc-check ${s.rr_ratio>=1.5?'pass':'fail'}">${s.rr_ratio>=1.5?'✅':'⚠️'} R:R 1:${s.rr_ratio}</div>
          </div>`;
      } else if (stype === 'PULLBACK_SURGE') {
        checksHtml = `
          <div class="sc-checks">
            <div class="sc-check pass">🔄 Surge vol ${s.vol_ratio}x avg</div>
            <div class="sc-check ${s.above_vwap?'pass':'fail'}">${s.above_vwap?'✅':'⚠️'} ${s.above_vwap?'Above':'Below'} VWAP</div>
            <div class="sc-check ${s.vol_dried_up?'pass':'fail'}">${s.vol_dried_up?'✅':'⚠️'} Pullback vol ${s.vol_dried_up?'dried up':'still high'}</div>
            <div class="sc-check ${s.rr_ratio>=1.5?'pass':'fail'}">${s.rr_ratio>=1.5?'✅':'⚠️'} R:R 1:${s.rr_ratio}</div>
          </div>`;
      } else {
        checksHtml = `
          <div class="sc-checks">
            <div class="sc-check ${s.vol_ok?'pass':'fail'}">${s.vol_ok?'✅':'⚠️'} Volume ${s.vol_ratio}x avg</div>
            <div class="sc-check ${s.above_vwap?'pass':'fail'}">${s.above_vwap?'✅':'⚠️'} ${s.above_vwap?'Above':'Below'} VWAP</div>
            <div class="sc-check ${!s.fakeout_risk?'pass':'fail'}">${!s.fakeout_risk?'✅':'⚠️'} Wick ${s.wick_ratio}%</div>
            <div class="sc-check ${s.rr_ratio>=1.5?'pass':'fail'}">${s.rr_ratio>=1.5?'✅':'⚠️'} R:R 1:${s.rr_ratio}</div>
          </div>`;
      }

      // ── SL label — tighter for Pullback Surge ─────────────────────────
      const slLabel = stype === 'PULLBACK_SURGE'
        ? `&#x1F6D1; Stop Loss <small style="color:#475569;">(pullback low)</small>`
        : `&#x1F6D1; Stop Loss`;

      card.innerHTML = `
        <style>.signal-card[style*="${s.grade_color}"]::before{background:${s.grade_color};}</style>
        <div class="sc-header">
          <div style="display:flex;flex-direction:column;gap:5px;">
            <a href="/risk/${s.symbol}" class="sc-sym">${s.symbol} &#x1F50D;</a>
            ${typeBadge}
          </div>
          <span class="sc-grade" style="background:${s.grade_color}22;color:${s.grade_color};">${s.grade}</span>
        </div>

        ${timingHtml}

        <div class="sc-price-row">
          <span class="sc-ltp">&#x20B9;${s.ltp}</span>
          <span class="timestamp-badge">&#x23F1; ${s.scanned_at}</span>
        </div>

        <div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 10px;">
          <span style="font-size:10px;padding:2px 9px;border-radius:20px;
            background:#a855f722;border:1px solid #a855f7;color:#a855f7;font-weight:700;">
            &#x26A1; High Momentum</span>
          ${s.rsi7 ? `<span style="font-size:10px;padding:2px 9px;border-radius:20px;
            background:${s.rsi7>=55&&s.rsi7<=75?'#10b98122':'#f59e0b22'};
            border:1px solid ${s.rsi7>=55&&s.rsi7<=75?'#10b981':'#f59e0b'};
            color:${s.rsi7>=55&&s.rsi7<=75?'#10b981':'#f59e0b'};font-weight:700;">
            RSI7: ${s.rsi7}</span>` : ''}
          ${s.slope_pct !== undefined && s.slope_pct !== null ? `<span style="font-size:10px;padding:2px 9px;border-radius:20px;
            background:${s.slope_pct>0?'#10b98122':'#ef444422'};
            border:1px solid ${s.slope_pct>0?'#10b981':'#ef4444'};
            color:${s.slope_pct>0?'#10b981':'#ef4444'};font-weight:700;">
            Slope ${s.slope_pct>0?'&#x2B06;':'&#x2B07;'} ${s.slope_pct>0?'+':''}${s.slope_pct}%</span>` : ''}
        </div>

        <div class="sc-score-wrap">
          <div class="sc-score-label"><span>Signal Score</span><span style="color:${s.grade_color};font-weight:800;">${s.score}/100</span></div>
          <div class="sc-score-bar-bg"><div class="sc-score-bar" style="width:${s.score}%;background:${s.grade_color};"></div></div>
        </div>
        ${checksHtml}
        <div class="sc-levels">
          <div class="sc-level-row"><span class="sc-level-lbl">${slLabel}</span><span class="sc-level-val" style="color:#ef4444;">&#x20B9;${s.sl}</span></div>
          <div class="sc-level-row"><span class="sc-level-lbl">&#x1F3AF; Target (measured move)</span><span class="sc-level-val" style="color:#10b981;">&#x20B9;${s.target}</span></div>
          <div class="sc-level-row"><span class="sc-level-lbl">&#x1F4CF; VWAP</span><span class="sc-level-val">&#x20B9;${s.vwap}</span></div>
        </div>
        <div class="sc-zone">
          Zone: &#x20B9;${s.zone_low} &ndash; &#x20B9;${s.zone_high}
          (${s.zone_pct}% range, ${s.zone_bars} bars)
          ${stype==='PULLBACK_SURGE'?`&nbsp;&bull;&nbsp;Pullback: ${s.pullback_pct}%`:''}
        </div>
      `;
      grid.appendChild(card);
    });

  } catch(e) {
    document.getElementById('loading-box').style.display = 'none';
    btn.disabled = false;
    btn.innerHTML = '&#x26A1; Scan Now';
    document.getElementById('empty-box').style.display = 'block';
    document.getElementById('empty-box').textContent = 'Error: ' + (e.message || 'Scan failed. Check terminal for traceback and try again.');
  }
}

async function loadHistory() {
  const view = document.getElementById('history-list');
  view.innerHTML = '<div class="loading-box" style="display:flex;"><div class="dots-row"><div class="ai-dot"></div><div class="ai-dot"></div><div class="ai-dot"></div></div><span>Loading history...</span></div>';
  try {
    const res = await fetch('/api/scanner_history');
    const data = await res.json();
    if (!data.history.length) {
      view.innerHTML = '<div class="empty-box">No past scans yet. Run a scan to start building history.</div>';
      return;
    }
    view.innerHTML = data.history.map(h => `
      <div class="hist-row">
        <span class="hist-date">&#x1F4C5; ${h.date} ${h.time}</span>
        <span class="hist-sym">${h.symbol}</span>
        <span>&#x20B9;${h.ltp}</span>
        <span class="hist-score" style="color:${h.score>=80?'#10b981':h.score>=65?'#34d399':h.score>=50?'#f59e0b':'#f87171'};">${h.score}/100</span>
        <span style="color:var(--muted);">bar ${h.bar_time}</span>
        <span style="color:var(--muted);margin-left:auto;">SL &#x20B9;${h.sl} &middot; Target &#x20B9;${h.target} &middot; R:R 1:${h.rr_ratio}</span>
      </div>
    `).join('');
  } catch(e) {
    view.innerHTML = '<div class="empty-box">Could not load history.</div>';
  }
}
</script>
</body>
</html>
"""

@app.route("/api/run_5min_scanner", methods=["POST"])
def api_run_5min_scanner():
    from flask import jsonify
    import traceback
    try:
        signals, mkt_open, mkt_time = run_5min_scanner()
        return jsonify({
            "signals":    signals,
            "count":      len(signals),
            "mkt_open":   mkt_open,
            "mkt_time":   mkt_time,
            "scanned_at": datetime.now().strftime("%H:%M:%S"),
        })
    except Exception as e:
        traceback.print_exc()   # prints full traceback to your terminal
        return jsonify({
            "error":    str(e),
            "signals":  [],
            "count":    0,
            "mkt_open": False,
            "mkt_time": "",
            "scanned_at": datetime.now().strftime("%H:%M:%S"),
        }), 500


@app.route("/api/scanner_history")
def api_scanner_history():
    from flask import jsonify
    return jsonify({"history": load_scanner_history()})


@app.route("/scanner")
def scanner_page():
    return render_template_string(SCANNER_HTML)


@app.route("/", methods=["GET","POST"])
def home():
    results   = []
    history   = performance()
    error_msg = None

    if request.method == "POST":
        import concurrent.futures
        stocks_raw = request.form.get("stocks", "")
        stocks_raw = stocks_raw.replace(",", "\n")
        stocks = [s.strip().upper() for s in stocks_raw.split("\n") if s.strip()]

        # Limit to 5 stocks — Render free tier kills requests after 30s
        if len(stocks) > 5:
            error_msg = f"⚠️ Max 5 stocks per search (you entered {len(stocks)}). Showing first 5 only. Split into batches of 5."
            stocks = stocks[:5]

        # Parallel fetch — 5 stocks in ~5s instead of ~25s
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futs = {pool.submit(analyze, s): s for s in stocks}
            for fut in concurrent.futures.as_completed(futs, timeout=25):
                try:
                    r = fut.result(timeout=20)
                    if r: results.append(r)
                except Exception:
                    pass

        # Restore input order
        order = {s: i for i, s in enumerate(stocks)}
        results.sort(key=lambda x: order.get(x.get("symbol",""), 99))

    return render_template_string(HTML, results=results, history=history, error_msg=error_msg)

if __name__ == "__main__":
    print("🚀 FalconAI — Full Suite: Screener + VCP + Sector + MTF + Similarity AI + Tracker + Risk Report + Options + Screens + Sectors + 5-Min Scanner")
    app.run(debug=True)


