#!/usr/bin/env python3
"""
Rejoue les alertes déjà envoyées (lues dans alerts_state.json) sur l'historique de l'exchange
et calcule ce que chaque trade aurait donné avec la méthode actuelle (2 entrées, SL, TP,
réduction de moitié au prix moyen + frais).

Seules les alertes au format récent (4 champs dans la clé : paire|timeframe|sens|OB) sont
rejouées : les toutes premières alertes (SL collé à l'OB) sont exclues automatiquement.
Option : SINCE_UTC=2026-10-04 pour ignorer aussi les alertes plus anciennes que cette date.

Hypothèses (volontairement prudentes) :
- ordres limites posés à l'alerte, exécutés quand le prix les touche (au prix de l'ordre, ou
  à l'ouverture si la bougie ouvre au-delà) ; jamais annulés ;
- dans une même bougie, ordre de traitement : entrées -> SL -> réduction -> TP
  (donc en cas de doute, le SL passe avant le TP) ;
- la réduction de moitié n'a lieu qu'à partir de la bougie suivant le remplissage de l'entrée 2 ;
- frais : FEE_RATE (0,1 % par ordre) ; ni slippage ni frais de financement.

Résultats : backtest_results.csv, backtest_summary.txt et un résumé sur Discord.
"""
import json
import os
from pathlib import Path

import ccxt
import pandas as pd

import ob_scanner as ob

FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))  # par ordre exécuté
SINCE_UTC = os.getenv("SINCE_UTC", "").strip()
COLS = ["timestamp", "open", "high", "low", "close", "volume"]


# ----------------------------- Données -----------------------------
def fetch_history(ex, sym: str, tf: str, since_ms: int) -> pd.DataFrame:
    tf_ms = ex.parse_timeframe(tf) * 1000
    rows, seen, since, now = [], set(), since_ms, ex.milliseconds()
    while since < now:
        batch = ex.fetch_ohlcv(sym, tf, since=since, limit=300)
        new = [r for r in (batch or []) if r[0] not in seen]
        if not new:
            break
        seen.update(r[0] for r in new)
        rows.extend(new)
        since = max(r[0] for r in new) + tf_ms
    df = pd.DataFrame(rows, columns=COLS).sort_values("timestamp").reset_index(drop=True)
    return df


# ----------------------------- Simulation -----------------------------
def simulate(fut: pd.DataFrame, st: dict, use_reduce: bool, expire=None) -> dict:
    """Simule un trade en orientation LONG (les shorts sont inversés avant l'appel).
    fut : bougies après l'alerte. st : entry, entry2, sl, tp (orientation long)."""
    e1, e2, sl, tp = st["entry"], st["entry2"], st["sl"], st["tp"]
    w1, w2 = ob.SPLIT1, 1 - ob.SPLIT1
    units = (ob.RISK_PCT / 100) / (w1 * abs(e1 - sl) + w2 * abs(e2 - sl))
    u1, u2 = w1 * units, w2 * units
    avg = w1 * e1 + w2 * e2
    be = avg + ob.BE_BUFFER * abs(avg)

    f1 = f2 = reduced = False
    p1 = p2 = reduce_px = exit_px = None
    status = "en attente"
    last_close = None
    bars = len(fut)

    for i, row in enumerate(fut.itertuples(index=False)):
        o, h, l, c = row.open, row.high, row.low, row.close
        last_close = c
        f2_before = f2

        if not f1 and expire is not None and i >= expire:  # ordre non rempli à temps : annulé
            return {"status": "expiré", "f1": False, "f2": False, "reduced": False, "pnl_pct": 0.0, "bars": i}
        if not f1:
            if l <= e1:
                f1, p1 = True, min(e1, o)
            elif h >= tp:
                return {"status": "manqué (TP sans entrée)", "f1": False, "f2": False,
                        "reduced": False, "pnl_pct": 0.0}
            else:
                continue
        if f1 and not f2 and l <= e2:
            f2, p2 = True, min(e2, o)

        if l <= sl:                               # SL
            exit_px, status, bars = min(sl, o), "SL", i + 1
            break
        if use_reduce and f2_before and f2 and not reduced and h >= be:
            reduced, reduce_px = True, max(be, o)
        if h >= tp:                               # TP
            exit_px, status, bars = max(tp, o), "TP", i + 1
            break

    if not f1:
        return {"status": "en attente", "f1": False, "f2": False, "reduced": False, "pnl_pct": 0.0}

    total_units = u1 + (u2 if f2 else 0)
    entries = [(u1, p1)] + ([(u2, p2)] if f2 else [])
    closing_px = exit_px if exit_px is not None else last_close  # ouvert : valorisé au dernier cours
    if status == "en attente":
        status = "ouvert"

    exits = []
    if reduced:
        exits.append((0.5 * total_units, reduce_px))
        exits.append((0.5 * total_units, closing_px))
    else:
        exits.append((total_units, closing_px))

    gross = sum(u * px for u, px in exits) - sum(u * px for u, px in entries)
    fees = FEE_RATE * (sum(u * abs(px) for u, px in entries) + sum(u * abs(px) for u, px in exits))
    return {"status": status, "f1": True, "f2": f2, "reduced": reduced,
            "pnl_pct": (gross - fees) * 100, "bars": bars}


def to_long_orientation(side: str, fut: pd.DataFrame, setup: dict):
    if side == "long":
        return fut, dict(setup)
    inv = fut.copy()
    inv["open"], inv["close"] = -fut["open"], -fut["close"]
    inv["high"], inv["low"] = -fut["low"], -fut["high"]
    st = {
        "entry": -setup["entry"], "entry2": -setup["entry2"],
        "sl": -setup["sl"], "tp": -setup["tp"],
    }
    return inv, st


def analyze_alert(df: pd.DataFrame, tf_ms: int, alert_ts: int, side: str, ob_time: int):
    """Retourne (setup, résultats) ou None si le setup n'est pas reproductible."""
    alert_ms = alert_ts * 1000
    closed = df[df["timestamp"] + tf_ms <= alert_ms].tail(ob.CANDLES).reset_index(drop=True)
    setup = ob.detect(closed, side)
    if not setup or setup["ob_time"] != ob_time:
        return None
    fut = df[df["timestamp"] + tf_ms > alert_ms].reset_index(drop=True)
    fut_l, st_l = to_long_orientation(side, fut, setup)
    return setup, {
        "reduce": simulate(fut_l, st_l, True),
        "plain": simulate(fut_l, st_l, False),
    }


# ----------------------------- Résumé -----------------------------
def summarize(rows: list, key: str) -> str:
    lines = []
    for tf in sorted({r["tf"] for r in rows}):
        for side in ("long", "short"):
            sub = [r for r in rows if r["tf"] == tf and r["side"] == side]
            if not sub:
                continue
            res = [r[key] for r in sub]
            filled = [x for x in res if x["f1"]]
            tp = sum(x["status"] == "TP" for x in filled)
            sl = sum(x["status"] == "SL" for x in filled)
            op = sum(x["status"] == "ouvert" for x in filled)
            missed = sum(x["status"].startswith("manqué") for x in res)
            wait = sum(x["status"] == "en attente" for x in res)
            realized = sum(x["pnl_pct"] for x in filled if x["status"] in ("TP", "SL"))
            latent = sum(x["pnl_pct"] for x in filled if x["status"] == "ouvert")
            lines.append(
                f"{tf} {side}: {len(sub)} alertes | {len(filled)} remplies | TP {tp} / SL {sl} / ouverts {op}"
                f" | manqués {missed}, en attente {wait}\n"
                f"   résultat clos {realized:+.2f}% du capital | latent {latent:+.2f}%"
            )
    return "\n".join(lines)


def main() -> None:
    state_path = Path(ob.STATE_FILE)
    if not state_path.exists():
        raise SystemExit("alerts_state.json introuvable : le scanner n'a encore rien enregistré.")
    state = json.loads(state_path.read_text())

    since_ts = pd.Timestamp(SINCE_UTC, tz="UTC").timestamp() if SINCE_UTC else 0
    alerts, old_format = [], 0
    for key, ts in state.items():
        parts = key.split("|")
        if len(parts) != 4:
            old_format += 1
            continue
        if ts < since_ts:
            continue
        sym, tf, side, ob_time = parts[0], parts[1], parts[2], int(parts[3])
        alerts.append({"sym": sym, "tf": tf, "side": side, "ob_time": ob_time, "ts": int(ts)})
    print(f"{len(alerts)} alertes à rejouer ({old_format} anciennes ignorées : SL très serré)")
    if not alerts:
        raise SystemExit("Aucune alerte à rejouer.")

    ex = getattr(ccxt, ob.EXCHANGE_ID)({"enableRateLimit": True})
    ex.load_markets()

    rows, not_reproduced = [], 0
    groups = {}
    for a in alerts:
        groups.setdefault((a["sym"], a["tf"]), []).append(a)

    for (sym, tf), items in groups.items():
        tf_ms = ex.parse_timeframe(tf) * 1000
        since = min(a["ts"] for a in items) * 1000 - (ob.CANDLES + 10) * tf_ms
        try:
            df = fetch_history(ex, sym, tf, since)
        except Exception as err:
            print(f"  {sym} {tf}: erreur de téléchargement ({err})")
            not_reproduced += len(items)
            continue
        for a in items:
            out = analyze_alert(df, tf_ms, a["ts"], a["side"], a["ob_time"])
            if out is None:
                not_reproduced += 1
                continue
            setup, res = out
            rows.append({
                "sym": sym, "tf": tf, "side": a["side"],
                "alert_utc": pd.Timestamp(a["ts"], unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M"),
                "e1": setup["entry"], "e2": setup["entry2"], "sl": setup["sl"], "tp": setup["tp"],
                "sl_pct": abs(setup["entry"] - setup["sl"]) / setup["entry"] * 100,
                "reduce": res["reduce"], "plain": res["plain"],
            })

    if not rows:
        raise SystemExit(f"Aucune alerte reproductible ({not_reproduced} non reproduites).")

    pd.DataFrame([{
        "paire": r["sym"], "tf": r["tf"], "sens": r["side"], "alerte_utc": r["alert_utc"],
        "E1": r["e1"], "E2": r["e2"], "SL": r["sl"], "TP": r["tp"], "SL_pct": round(r["sl_pct"], 2),
        "statut_avec_reduction": r["reduce"]["status"], "E2_remplie": r["reduce"]["f2"],
        "pnl_pct_avec_reduction": round(r["reduce"]["pnl_pct"], 3),
        "statut_sans_reduction": r["plain"]["status"],
        "pnl_pct_sans_reduction": round(r["plain"]["pnl_pct"], 3),
    } for r in rows]).to_csv("backtest_results.csv", index=False)

    header = (f"📊 Rejeu de {len(rows)} alertes (risque {ob.RISK_PCT:g}% par trade, frais {FEE_RATE*100:.2f}%/ordre)"
              f"\n{not_reproduced} non reproduites, {old_format} anciennes ignorées")
    text = (f"{header}\n\nAVEC réduction de moitié au prix moyen :\n{summarize(rows, 'reduce')}"
            f"\n\nSANS réduction :\n{summarize(rows, 'plain')}"
            f"\n\nÉchantillon réduit : à lire avec prudence. Détail : backtest_results.csv")
    Path("backtest_summary.txt").write_text(text)
    print(text)
    try:
        ob.notify(text[:1990])
    except Exception as err:
        print(f"notification échouée : {err}")


if __name__ == "__main__":
    main()
