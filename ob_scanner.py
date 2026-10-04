#!/usr/bin/env python3
"""
Scanner "Order Block + consolidation" (setup long) sur les paires USDT d'un exchange.

Alerte quand : impulsion haussière -> OB à sa base -> le prix consolide au-dessus
de l'OB sans l'avoir encore retesté (on anticipe le retour vers l'OB).

Installation :  pip install ccxt pandas requests
Lancement    :  python ob_scanner.py            (un scan, pour cron / GitHub Actions)
                python ob_scanner.py --loop     (scan à chaque clôture de bougie)

Notifications (variables d'environnement, au choix) :
  DISCORD_WEBHOOK_URL
  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID

Réglages (variables d'environnement, valeurs par défaut entre parenthèses) :
  EXCHANGE (binance), TIMEFRAMES (15m,1h), MIN_VOLUME (2000000 USDT / 24h)
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
MIN_QUOTE_VOLUME = float(os.getenv("MIN_VOLUME", "2000000"))  # USDT sur 24h

ATR_PERIOD = 14
IMPULSE_ATR = 2.5      # taille minimale de l'impulsion, en multiples d'ATR
IMPULSE_MAX_CANDLES = 5  # durée max de l'impulsion
SWING_LOOKBACK = 20    # l'impulsion doit casser le plus haut de ces N bougies
OB_SEARCH = 5          # on cherche la dernière bougie baissière sur N bougies avant l'impulsion
MIN_RANGE = 8          # bougies minimum de consolidation après l'impulsion
MAX_RANGE = 80         # au-delà, le setup est considéré périmé
SL_MARGIN_ATR = 0.2    # marge du SL sous le bas de l'OB, en ATR
MIN_RR = 2.0           # ratio risque/rendement minimum (entrée = haut de l'OB)
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


def detect(df: pd.DataFrame):
    """Retourne un dict décrivant le setup, ou None. df ne contient que des bougies clôturées."""
    df = add_atr(df.copy()).reset_index(drop=True)
    n = len(df)
    if n < SWING_LOOKBACK + ATR_PERIOD + MAX_RANGE:
        return None

    o, h, l, c, atr = df["open"], df["high"], df["low"], df["close"], df["atr"]

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

        entry = ob_high
        sl = ob_low - SL_MARGIN_ATR * atr[e]
        tp = range_high
        if entry <= sl:
            continue
        rr = (tp - entry) / (entry - sl)
        if rr < MIN_RR:
            continue

        return {
            "ob_time": int(df["timestamp"][ob]),
            "ob_low": ob_low,
            "ob_high": ob_high,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "price": price,
            "range_candles": n - 1 - e,
        }
    return None


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
    print(f"{time.strftime('%H:%M:%S')} - {len(symbols)} paires >= {MIN_QUOTE_VOLUME:,.0f} USDT/24h, TF {TIMEFRAMES}")

    for sym in symbols:
        for tf in TIMEFRAMES:
            try:
                raw = ex.fetch_ohlcv(sym, tf, limit=CANDLES)
            except Exception as err:
                print(f"  {sym} {tf}: erreur ({err})")
                continue
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = df.iloc[:-1]  # on retire la bougie en cours

            setup = detect(df)
            if not setup:
                continue

            key = f"{sym}|{tf}|{setup['ob_time']}"
            if key in state:
                continue

            try:
                notify(
                    f"📍 {sym} [{tf}] - consolidation au-dessus d'un OB\n"
                    f"Prix : {fmt(setup['price'])} ({setup['range_candles']} bougies de range)\n"
                    f"Zone OB : {fmt(setup['ob_low'])} - {fmt(setup['ob_high'])}\n"
                    f"Entrée : {fmt(setup['entry'])} | SL : {fmt(setup['sl'])} | TP : {fmt(setup['tp'])}\n"
                    f"R/R : {setup['rr']:.1f}"
                )
                state[key] = int(time.time())
            except Exception as err:
                print(f"  notification échouée pour {sym} {tf}: {err}")

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
