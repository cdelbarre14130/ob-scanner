#!/usr/bin/env python3
"""
Rejeu HISTORIQUE : fait tourner le scanner comme s'il avait fonctionné pendant les N derniers
jours sur les paires les plus liquides, puis simule chaque trade (2 entrées, SL, TP, réduction
de moitié au prix moyen, frais) avec les mêmes règles que backtest_alerts.py.

Réglages (variables d'environnement) :
  DAYS (60)            nombre de jours rejoués
  MAX_PAIRS (60)       nombre de paires, les plus liquides d'abord (volume >= MIN_VOLUME)
  TIMEFRAMES (15m,1h)  timeframes rejoués
  EXPIRE_CANDLES (72)  un ordre non rempli après N bougies est annulé
  IMPULSE_ATR, MIN_RR, MIN_RANGE, SL_MARGIN_ATR, MAX_RANGE : pour tester d'autres réglages
                       (laisser vide = réglage actuel de ob_scanner.py)
Les autres réglages (EXCHANGE, MIN_VOLUME, RISK_PCT, SPLIT1, FEE_RATE...) sont ceux du scanner.

Filtres de tendance comparés en une seule exécution (les trades simulés sont les mêmes, seule la
sélection change) : sans filtre / timeframe supérieur / BTC / TF sup. + BTC / pente / tout.
Règles identiques à celles du scanner (voir FILTER_* dans ob_scanner.py).

Résultats : history_results.csv, history_summary.txt, et 2 messages Discord (référence + filtres).
"""
import os
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

import backtest_alerts as ba
import ob_scanner as ob

DAYS = int(os.getenv("DAYS", "60"))
MAX_PAIRS = int(os.getenv("MAX_PAIRS", "60"))
EXPIRE_CANDLES = int(os.getenv("EXPIRE_CANDLES", "72"))
OVERRIDES = {"IMPULSE_ATR": float, "MIN_RR": float, "MIN_RANGE": int, "SL_MARGIN_ATR": float, "MAX_RANGE": int}
BUCKETS = [(0, 1, "<1%"), (1, 2, "1-2%"), (2, 4, "2-4%"), (4, 1e9, ">4%")]
VARIANTS = [
    ("Sans filtre", ()),
    ("TF supérieur", ("htf",)),
    ("BTC", ("btc",)),
    ("TF sup. + BTC", ("htf", "btc")),
    ("Pente MA", ("slope",)),
    ("Tout", ("htf", "btc", "slope")),
]


def apply_overrides() -> dict:
    used = {}
    for name, cast in OVERRIDES.items():
        raw = os.getenv(name, "").strip()
        if raw:
            setattr(ob, name, cast(raw))
            used[name] = getattr(ob, name)
    return used


# ----------------------------- Pré-filtre rapide -----------------------------
def invert_df(df: pd.DataFrame) -> pd.DataFrame:
    inv = df.copy()
    inv["open"], inv["close"] = -df["open"], -df["close"]
    inv["high"], inv["low"] = -df["low"], -df["high"]
    return inv


def candidate_best(df: pd.DataFrame, side: str) -> np.ndarray:
    """Pour chaque bougie t, l'indice e de l'impulsion la plus récente dont le setup serait
    valide à t (ou -1). Reproduit les conditions de ob.detect() en version vectorisée, pour
    n'appeler la vraie fonction qu'aux moments utiles. La décision finale reste celle de
    ob.detect() : ce pré-calcul ne sert qu'à éviter des milliers d'appels inutiles."""
    d = df if side == "long" else invert_df(df)
    n = len(d)
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = ob.add_atr(d.copy())["atr"].to_numpy(float)
    best = np.full(n, -1, dtype=int)

    for e in range(ob.SWING_LOOKBACK, n):  # e croissant : la plus récente écrase les précédentes
        if np.isnan(atr[e]):
            continue
        s = None
        for length in range(1, ob.IMPULSE_MAX_CANDLES + 1):
            start = e - length + 1
            if start - ob.SWING_LOOKBACK < 0:
                break
            if c[e] - o[start] >= ob.IMPULSE_ATR * atr[e] and c[e] > h[start - ob.SWING_LOOKBACK:start].max():
                s = start
                break
        if s is None:
            continue
        ob_i = next((j for j in range(s - 1, max(s - 1 - ob.OB_SEARCH, 0), -1) if c[j] < o[j]), None)
        if ob_i is None:
            continue
        ob_low, ob_high = l[ob_i], h[ob_i]
        structure_low = l[ob_i:e + 1].min()

        t0, t1 = e + ob.MIN_RANGE, min(e + ob.MAX_RANGE - 1, n - 1)
        if t0 > t1:
            continue
        t = np.arange(t0, t1 + 1)
        j = t - e
        cmin_close = np.minimum.accumulate(c[e + 1:t1 + 1])
        cmin_low = np.minimum.accumulate(l[e + 1:t1 + 1])
        hmax = np.maximum.accumulate(h[e:t1 + 1])
        a_t = atr[t]

        entry = ob_high
        entry2 = np.minimum(ob_low, entry - ob.ENTRY2_MIN_ATR * a_t)
        sl = np.minimum.reduce([
            structure_low - ob.SL_MARGIN_ATR * a_t,
            entry - ob.MIN_RISK_ATR * a_t,
            entry2 - ob.SL_MARGIN_ATR * a_t,
        ])
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = (hmax[j] - entry) / (entry - sl)
        ok = (
            (cmin_close[j - 1] >= ob_low) & (cmin_low[j - 1] > ob_high)
            & (c[t] > ob_high) & (c[t] < hmax[j])
            & (a_t > 0) & (rr >= ob.MIN_RR - 1e-9)
        )
        best[t[ok]] = e
    return best


# ----------------------------- Rejeu -----------------------------
def regime_at(ref: dict, at_ms: int) -> int:
    """Régime (+1 / -1 / 0) d'une série de référence, avec uniquement les bougies clôturées à at_ms."""
    idx = int(np.searchsorted(ref["close_ms"], at_ms, side="right"))
    return ob.above_ma(ref["df"].iloc[max(0, idx - ob.MA_PERIOD):idx])


def make_ref(df: pd.DataFrame, tf_ms: int) -> dict:
    return {"df": df, "close_ms": df["timestamp"].to_numpy() + tf_ms}


def replay_pair(df: pd.DataFrame, tf_ms: int, sym: str, tf: str, first_idx: int,
                htf_ref=None, btc_ref=None) -> list:
    trades = []
    for side in ob.SIDES:
        best = candidate_best(df, side)
        done_e, seen = set(), set()
        for t in np.flatnonzero(best >= 0):
            if t < first_idx or best[t] in done_e:
                continue
            window = df.iloc[max(0, t - (ob.CANDLES - 1) + 1): t + 1].reset_index(drop=True)
            st = ob.detect(window, side)   # même fonction que le scanner en direct
            if not st:
                continue
            done_e.add(best[t])
            if st["ob_time"] in seen:
                continue
            seen.add(st["ob_time"])
            fut = df.iloc[t + 1:].reset_index(drop=True)
            if fut.empty:
                continue
            fut_l, st_l = ba.to_long_orientation(side, fut, st)
            alert_ms = int(df["timestamp"][t]) + tf_ms
            need = ob.MA_PERIOD + ob.SLOPE_LOOKBACK
            flags = {
                "htf": True if htf_ref is None else ob.trend_matches(side, regime_at(htf_ref, alert_ms)),
                "btc": True if btc_ref is None else ob.trend_matches(side, regime_at(btc_ref, alert_ms)),
                "slope": ob.slope_ok(df.iloc[max(0, t + 1 - need):t + 1]),
            }
            trades.append({
                "sym": sym, "tf": tf, "side": side, "flags": flags,
                "alert_utc": pd.Timestamp(alert_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M"),
                "e1": st["entry"], "e2": st["entry2"], "sl": st["sl"], "tp": st["tp"],
                "sl_pct": abs(st["entry"] - st["sl"]) / abs(st["entry"]) * 100,
                "reduce": ba.simulate(fut_l, st_l, True, expire=EXPIRE_CANDLES),
                "plain": ba.simulate(fut_l, st_l, False, expire=EXPIRE_CANDLES),
            })
    return trades


# ----------------------------- Statistiques -----------------------------
def stats(results: list) -> dict:
    closed = [x for x in results if x["f1"] and x["status"] in ("TP", "SL")]
    open_ = sum(1 for x in results if x["f1"] and x["status"] == "ouvert")
    n = len(closed)
    if n == 0:
        return {"n": 0, "open": open_, "win": 0.0, "avg_r": 0.0, "total": 0.0}
    pnl = [x["pnl_pct"] for x in closed]
    return {
        "n": n, "open": open_,
        "win": 100 * sum(p > 0 for p in pnl) / n,
        "avg_r": sum(pnl) / n / ob.RISK_PCT,
        "total": sum(pnl),
    }


def build_texts(trades: list, header: str):
    main = [header, ""]
    for tf in ob.TIMEFRAMES:
        for side in ob.SIDES:
            sub = [t for t in trades if t["tf"] == tf and t["side"] == side]
            if not sub:
                continue
            a, b = stats([t["reduce"] for t in sub]), stats([t["plain"] for t in sub])
            main.append(
                f"{tf} {side}: {a['n']} trades clos (+{a['open']} ouverts)\n"
                f"  avec réd.: {a['win']:.0f}% gagnants, {a['avg_r']:+.2f}R/trade, total {a['total']:+.1f}%\n"
                f"  sans réd.: {b['win']:.0f}% gagnants, {b['avg_r']:+.2f}R/trade, total {b['total']:+.1f}%"
            )
    bucket = ["Par distance du SL (avec réduction, long+short) :"]
    for tf in ob.TIMEFRAMES:
        parts = []
        for lo, hi, label in BUCKETS:
            sub = [t["reduce"] for t in trades if t["tf"] == tf and lo <= t["sl_pct"] < hi]
            s = stats(sub)
            if s["n"]:
                parts.append(f"{label}: {s['n']} tr., {s['avg_r']:+.2f}R")
        if parts:
            bucket.append(f"{tf} | " + " | ".join(parts))
    return "\n".join(main), "\n".join(bucket)


def passes(trade: dict, names: tuple) -> bool:
    return all(trade["flags"][n] for n in names)


def fmt_stats(s: dict) -> str:
    return f"{s['n']} tr., {s['win']:.0f}% gagn., {s['avg_r']:+.2f}R, {s['total']:+.1f}%"


def build_filter_texts(trades: list):
    """Retourne (texte Discord compact, détail long/short pour le fichier)."""
    lines = [
        f"🔬 Comparaison des filtres (méthode avec réduction ½, {ob.RISK_PCT:g}% de risque/trade)",
        f"TF sup. : {', '.join(f'{a}→{b}' for a, b in ob.HTF_MAP.items())} | BTC sur {ob.BTC_TF} | MA{ob.MA_PERIOD}",
        "",
    ]
    detail = []
    for label, names in VARIANTS:
        parts = []
        for tf in ob.TIMEFRAMES:
            sub = [t for t in trades if t["tf"] == tf and passes(t, names)]
            parts.append(f"{tf} {fmt_stats(stats([t['reduce'] for t in sub]))}")
            for side in ob.SIDES:
                sd = [t for t in sub if t["side"] == side]
                detail.append(f"{label} | {tf} {side}: {fmt_stats(stats([t['reduce'] for t in sd]))}"
                              f"  [sans réd.: {fmt_stats(stats([t['plain'] for t in sd]))}]")
        lines.append(f"{label} : " + " | ".join(parts))
    return "\n".join(lines), "\n".join(detail)


# ----------------------------- Programme -----------------------------
def main() -> None:
    overrides = apply_overrides()
    ex = getattr(ccxt, ob.EXCHANGE_ID)({"enableRateLimit": True})
    ex.load_markets()
    symbols = ob.get_symbols(ex)[:MAX_PAIRS]
    print(f"{len(symbols)} paires, {DAYS} jours, TF {ob.TIMEFRAMES}, sens {ob.SIDES}, réglages modifiés : {overrides or 'aucun'}")

    now_ms = ex.milliseconds()
    start_ms = now_ms - DAYS * 86400 * 1000
    min_idx = ob.SWING_LOOKBACK + ob.ATR_PERIOD + ob.MAX_RANGE

    def load(sym: str, tf: str) -> pd.DataFrame:
        tf_ms = ex.parse_timeframe(tf) * 1000
        since = start_ms - (ob.CANDLES + 10) * tf_ms
        df = ba.fetch_history(ex, sym, tf, since).iloc[:-1].reset_index(drop=True)  # bougie en cours retirée
        expected = (now_ms - since) // tf_ms
        if len(df) < 0.9 * expected:
            raise RuntimeError(f"historique incomplet ({len(df)}/{expected})")
        return df

    btc_tf_ms = ex.parse_timeframe(ob.BTC_TF) * 1000
    btc_ref = make_ref(load(ob.BTC_SYMBOL, ob.BTC_TF), btc_tf_ms)  # si cela échoue, on s'arrête : filtres inutilisables

    trades, skipped = [], 0
    for k, sym in enumerate(symbols, 1):
        cache = {}

        def get(tf: str) -> pd.DataFrame:
            if tf not in cache:
                cache[tf] = load(sym, tf)
            return cache[tf]

        for tf in ob.TIMEFRAMES:
            tf_ms = ex.parse_timeframe(tf) * 1000
            try:
                df = get(tf)
                htf = ob.HTF_MAP.get(tf)
                htf_ref = make_ref(get(htf), ex.parse_timeframe(htf) * 1000) if htf else None
            except Exception as err:
                print(f"  {sym} {tf}: ignoré ({err})")
                skipped += 1
                continue
            first_idx = max(int(np.searchsorted(df["timestamp"].to_numpy(), start_ms)), min_idx)
            trades.extend(replay_pair(df, tf_ms, sym, tf, first_idx, htf_ref, btc_ref))
        print(f"[{k}/{len(symbols)}] {sym}: {len(trades)} trades cumulés")

    if not trades:
        raise SystemExit("Aucun trade rejoué.")

    pd.DataFrame([{
        "paire": t["sym"], "tf": t["tf"], "sens": t["side"], "alerte_utc": t["alert_utc"],
        "E1": t["e1"], "E2": t["e2"], "SL": t["sl"], "TP": t["tp"], "SL_pct": round(t["sl_pct"], 2),
        "filtre_tf_sup": t["flags"]["htf"], "filtre_btc": t["flags"]["btc"], "filtre_pente": t["flags"]["slope"],
        "statut_avec_reduction": t["reduce"]["status"], "pnl_pct_avec_reduction": round(t["reduce"]["pnl_pct"], 3),
        "statut_sans_reduction": t["plain"]["status"], "pnl_pct_sans_reduction": round(t["plain"]["pnl_pct"], 3),
        "bougies": t["reduce"].get("bars", 0),
    } for t in trades]).sort_values("alerte_utc").to_csv("history_results.csv", index=False)

    params = f" | réglages modifiés : {overrides}" if overrides else ""
    header = (f"📊 Rejeu historique : {DAYS} j, {len(symbols)} paires, TF {','.join(ob.TIMEFRAMES)}\n"
              f"Risque {ob.RISK_PCT:g}%/trade, frais {ba.FEE_RATE*100:.2f}%/ordre, {len(trades)} setups{params}")
    main_txt, bucket_txt = build_texts(trades, header)
    filter_txt, filter_detail = build_filter_texts(trades)
    caveats = ("Limites : paires choisies sur leur volume ACTUEL (biais), une seule période de marché, "
               "pas de slippage ni de funding, total = somme théorique (trades supposés indépendants). "
               "R = 1 risque de trade. Les filtres réduisent l'échantillon : à lire avec prudence. "
               "Détail : history_results.csv")
    full = (f"{main_txt}\n\n{bucket_txt}\n\n{filter_txt}\n\nDétail des filtres par sens :\n{filter_detail}\n\n{caveats}")
    Path("history_summary.txt").write_text(full)
    print(full)

    first = f"{main_txt}\n\n{bucket_txt}"
    if len(first) > 1950:
        first = main_txt
    for msg in (first, filter_txt):
        try:
            ob.notify(msg[:1990])
        except Exception as err:
            print(f"notification échouée : {err}")


if __name__ == "__main__":
    main()
