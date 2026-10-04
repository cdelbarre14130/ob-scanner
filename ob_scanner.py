#!/usr/bin/env python3
"""
Scanner "Order Block + consolidation" sur les paires USDT d'un exchange.

LONG  : impulsion haussière -> OB (dernière bougie baissière) -> le prix consolide
        AU-DESSUS de l'OB sans l'avoir retesté -> on anticipe le retour vers l'OB.
SHORT : le miroir exact (impulsion baissière, OB = dernière bougie haussière,
        consolidation SOUS l'OB).

Deux entrées : entrée 1 = bord de l'OB touché en premier, entrée 2 = bord opposé de l'OB
(au moins 0,5 ATR plus loin). La position est répartie entre les deux (50/50 par défaut) et
dimensionnée pour que la perte au SL, une fois les 2 entrées remplies, vaille RISK_PCT % du capital.

Stop loss : sous (long) / au-dessus (short) de toute la structure de l'impulsion,
avec une marge en ATR, au moins 0,5 ATR au-delà de l'entrée 2 et 1 ATR de l'entrée 1.

Installation :  pip install ccxt pandas requests
Lancement    :  python ob_scanner.py            (un scan, pour cron / GitHub Actions)
                python ob_scanner.py --loop     (scan à chaque clôture de bougie)

Notifications (variables d'environnement, au choix) :
  DISCORD_WEBHOOK_URL
  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID

Réglages (variables d'environnement, valeurs par défaut entre parenthèses) :
  EXCHANGE (binance), TIMEFRAMES (15m,1h), MIN_VOLUME (2000000 USDT / 24h),
  SIDES (long,short), RISK_PCT (1 = % du capital risqué par trade, entrées 1+2 remplies),
  SPLIT1 (0.5 = part de la position sur l'entrée 1)
"""
import argparse
import json
import os
import time
from pathlib import Path

import ccxt
import pandas as pd
import requests

# ----------------------------- Paramètres -----------------------------
EXCHANGE_ID = os.getenv("EXCHANGE", "binance")
TIMEFRAMES = [t.strip() for t in os.getenv("TIMEFRAMES", "15m,1h").split(",") if t.strip()]
SIDES = [s.strip() for s in os.getenv("SIDES", "long,short").split(",") if s.strip()]
MIN_QUOTE_VOLUME = float(os.getenv("MIN_VOLUME", "2000000"))  # USDT sur 24h
RISK_PCT = float(os.getenv("RISK_PCT", "1"))
SPLIT1 = min(max(float(os.getenv("SPLIT1", "0.5")), 0.05), 0.95)  # part de la position sur l'entrée 1

ATR_PERIOD = 14
IMPULSE_ATR = 2.5        # taille minimale de l'impulsion, en multiples d'ATR
IMPULSE_MAX_CANDLES = 5  # durée max de l'impulsion
SWING_LOOKBACK = 20      # l'impulsion doit casser le plus haut/bas de ces N bougies
OB_SEARCH = 5            # on cherche la bougie OB sur N bougies avant l'impulsion
MIN_RANGE = 8            # bougies minimum de consolidation après l'impulsion
MAX_RANGE = 80           # au-delà, le setup est considéré périmé
SL_MARGIN_ATR = 0.5      # marge du SL au-delà de la structure, en ATR
MIN_RISK_ATR = 1.0       # distance minimale entrée -> SL, en ATR (le SL est élargi si besoin)
MIN_RR = 2.0             # ratio risque/rendement minimum (calculé sur l'entrée 1)
ENTRY2_MIN_ATR = 0.5     # écart minimal entre l'entrée 1 et l'entrée 2, en ATR
BE_BUFFER = 0.0015       # marge au-delà du prix moyen pour couvrir les frais (0,15 %)
CANDLES = 250

EXCLUDED = ("UP/", "DOWN/", "BULL/", "BEAR/")
STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "EUR", "AEUR", "USD1", "XUSD", "PYUSD"}
STATE_FILE = Path(os.getenv("STATE_FILE", "alerts_state.json"))


# ----------------------------- Détection -----------------------------
def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    return df


def detect_long(df: pd.DataFrame):
    """Setup long. df ne contient que des bougies clôturées. Retourne un dict ou None."""
    df = add_atr(df.copy()).reset_index(drop=True)
    n = len(df)
    if n < SWING_LOOKBACK + ATR_PERIOD + MAX_RANGE:
        return None

    o, h, l, c, atr = df["open"], df["high"], df["low"], df["close"], df["atr"]
    cur_atr = atr.iloc[-1]
    if pd.isna(cur_atr) or cur_atr <= 0:
        return None

    for e in range(n - 1 - MIN_RANGE, n - 1 - MAX_RANGE, -1):  # fin de l'impulsion, la plus récente d'abord
        if pd.isna(atr[e]):
            continue

        s = None
        for length in range(1, IMPULSE_MAX_CANDLES + 1):
            start = e - length + 1
            if start - SWING_LOOKBACK < 0:
                break
            move = c[e] - o[start]
            broke_swing = c[e] > h[start - SWING_LOOKBACK:start].max()
            if move >= IMPULSE_ATR * atr[e] and broke_swing:
                s = start
                break
        if s is None:
            continue

        # Order block : dernière bougie baissière avant l'impulsion
        ob = next((j for j in range(s - 1, max(s - 1 - OB_SEARCH, 0), -1) if c[j] < o[j]), None)
        if ob is None:
            continue
        ob_low, ob_high = l[ob], h[ob]

        after = df.iloc[e + 1:]
        if (after["close"] < ob_low).any():   # invalidé
            continue
        if (after["low"] <= ob_high).any():   # OB déjà retesté : on n'anticipe plus
            continue

        range_high = h[e:].max()
        price = c.iloc[-1]
        if not (ob_high < price < range_high):  # le prix doit être dans la consolidation
            continue

        # SL : sous tout le bas de la structure (de l'OB à la fin de l'impulsion),
        # avec marge, et au moins MIN_RISK_ATR d'écart avec l'entrée.
        entry = ob_high
        entry2 = min(ob_low, entry - ENTRY2_MIN_ATR * cur_atr)
        structure_low = l[ob:e + 1].min()
        sl = min(
            structure_low - SL_MARGIN_ATR * cur_atr,
            entry - MIN_RISK_ATR * cur_atr,
            entry2 - SL_MARGIN_ATR * cur_atr,
        )
        tp = range_high
        rr = (tp - entry) / (entry - sl)
        if rr < MIN_RR:
            continue

        return {
            "ob_time": int(df["timestamp"][ob]),
            "ob_low": ob_low,
            "ob_high": ob_high,
            "entry": entry,
            "entry2": entry2,
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "price": price,
            "range_candles": n - 1 - e,
        }
    return None


def detect(df: pd.DataFrame, side: str):
    """Long : détection directe. Short : on inverse les prix, on détecte un long, on réinverse."""
    if side == "long":
        return detect_long(df)
    inv = df.copy()
    inv["open"], inv["close"] = -df["open"], -df["close"]
    inv["high"], inv["low"] = -df["low"], -df["high"]
    r = detect_long(inv)
    if not r:
        return None
    return {
        "ob_time": r["ob_time"],
        "ob_low": -r["ob_high"],
        "ob_high": -r["ob_low"],
        "entry": -r["entry"],
        "entry2": -r["entry2"],
        "sl": -r["sl"],
        "tp": -r["tp"],
        "rr": r["rr"],
        "price": -r["price"],
        "range_candles": r["range_candles"],
    }


# ----------------------------- Notifications -----------------------------
def fmt(x: float) -> str:
    return f"{x:.8g}"


def notify(text: str) -> None:
    sent = False
    webhook = os.getenv("DISCORD_WEBHOOK_URL")
    if webhook:
        requests.post(webhook, json={"content": text}, timeout=15).raise_for_status()
        sent = True
    tg_token, tg_chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat:
        requests.post(
            f"https://api.telegram.org/bot{tg_token}/sendMessage",
            data={"chat_id": tg_chat, "text": text},
            timeout=15,
        )
        sent = True
    if not sent:
        print(text)


def build_message(sym: str, tf: str, side: str, st: dict) -> str:
    long = side == "long"
    head = "🟢 LONG" if long else "🔴 SHORT"
    where = "au-dessus d'un OB" if long else "sous un OB"
    e1, e2, sl, tp = st["entry"], st["entry2"], st["sl"], st["tp"]
    w1, w2 = SPLIT1, 1 - SPLIT1

    risk1, risk2 = abs(e1 - sl), abs(e2 - sl)
    # unités (par unité de capital) pour que la perte au SL, les 2 entrées remplies, = RISK_PCT %
    units = (RISK_PCT / 100) / (w1 * risk1 + w2 * risk2)
    n1 = w1 * units * e1 * 100  # tranche 1, en % du capital
    n2 = w2 * units * e2 * 100  # tranche 2, en % du capital
    total = n1 + n2
    loss_t1_only = w1 * units * risk1 * 100  # perte au SL si seule l'entrée 1 est remplie, en % du capital

    avg = w1 * e1 + w2 * e2
    be = avg * (1 + BE_BUFFER) if long else avg * (1 - BE_BUFFER)
    rr_avg = abs(tp - avg) / abs(avg - sl)
    stop_pct = abs(e1 - sl) / e1 * 100
    lever = " ⚠️ levier nécessaire" if total > 100 else ""

    return (
        f"{head} {sym} [{tf}] - consolidation {where}\n"
        f"Prix : {fmt(st['price'])} ({st['range_candles']} bougies de range)\n"
        f"Zone OB : {fmt(st['ob_low'])} - {fmt(st['ob_high'])}\n"
        f"Entrée 1 : {fmt(e1)} -> {n1:.0f}% du capital\n"
        f"Entrée 2 : {fmt(e2)} -> {n2:.0f}% du capital\n"
        f"Prix moyen (2 entrées) : {fmt(avg)}\n"
        f"SL : {fmt(sl)} ({stop_pct:.1f}% de l'entrée 1) | TP : {fmt(tp)}\n"
        f"R/R : {st['rr']:.1f} (entrée 1) / {rr_avg:.1f} (prix moyen)\n"
        f"Réduire de moitié vers : {fmt(be)} (breakeven + frais)\n"
        f"Perte au SL : {loss_t1_only:.2f}% si seule l'entrée 1 est remplie, "
        f"{RISK_PCT:g}% si les 2 le sont\n"
        f"Total engagé si les 2 entrées sont remplies : {total:.0f}% du capital{lever}"
    )


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state))


# ----------------------------- Scan -----------------------------
def get_symbols(ex) -> list:
    tickers = ex.fetch_tickers()
    symbols = []
    for sym, t in tickers.items():
        m = ex.markets.get(sym)
        if not m or not m.get("spot") or m.get("quote") != "USDT" or not m.get("active", True):
            continue
        if m["base"] in STABLES or any(sym.startswith(x) for x in EXCLUDED):
            continue
        if (t.get("quoteVolume") or 0) >= MIN_QUOTE_VOLUME:
            symbols.append((sym, t["quoteVolume"]))
    symbols.sort(key=lambda x: -x[1])
    return [s for s, _ in symbols]


def scan(ex) -> None:
    state = load_state()
    symbols = get_symbols(ex)
    print(
        f"{time.strftime('%H:%M:%S')} - {len(symbols)} paires >= {MIN_QUOTE_VOLUME:,.0f} USDT/24h, "
        f"TF {TIMEFRAMES}, sens {SIDES}"
    )

    for sym in symbols:
        for tf in TIMEFRAMES:
            try:
                raw = ex.fetch_ohlcv(sym, tf, limit=CANDLES)
            except Exception as err:
                print(f"  {sym} {tf}: erreur ({err})")
                continue
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = df.iloc[:-1]  # on retire la bougie en cours

            for side in SIDES:
                setup = detect(df, side)
                if not setup:
                    continue

                key = f"{sym}|{tf}|{side}|{setup['ob_time']}"
                if key in state:
                    continue

                try:
                    notify(build_message(sym, tf, side, setup))
                    state[key] = int(time.time())
                except Exception as err:
                    print(f"  notification échouée pour {sym} {tf} {side}: {err}")

    # purge des alertes de plus de 14 jours
    cutoff = time.time() - 14 * 86400
    save_state({k: v for k, v in state.items() if v > cutoff})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="scanner à chaque clôture de bougie")
    args = parser.parse_args()

    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    ex.load_markets()

    if not args.loop:
        scan(ex)
        return

    tf_seconds = min(ex.parse_timeframe(t) for t in TIMEFRAMES)
    while True:
        scan(ex)
        wait = tf_seconds - (time.time() % tf_seconds) + 10  # 10 s après la clôture
        time.sleep(wait)


if __name__ == "__main__":
    main()
