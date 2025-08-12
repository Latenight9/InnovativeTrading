import time, threading, asyncio, os, json
import numpy as np, pandas as pd, torch
from datetime import datetime
from collections import deque
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
from binance import ThreadedWebsocketManager
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
from binance_keys import API_KEY, API_SECRET

from teacher_model import TeacherNet
from student_model import StudentNet
from data_preparation import prepare_data

# === PARAMETER ===
TICKERS = ["UNIUSDT", "ADAUSDT"]
PAIR_NAME = f"{TICKERS[0]}_{TICKERS[1]}"
LOOKBACK = 20
WINDOW_SIZE = 120
PATCH_SIZE = 10
STEP_SIZE = 1
ENTRY_PERCENTILE = 95               # Fallback, falls keine Threshold-Datei vorhanden
EXIT_THRESHOLD = 0.3
N_PROTOTYPES = 32
TEACHER_EMB = 768
STUDENT_EMB = 128
OUTPUT_DIM = 128
CHECK_INTERVAL = 15                 # Sekunden
THRESHOLD_MODE = "train_file"       # {"train_file","local_pct"}
THRESHOLD_FILE = os.path.join("checkpoints", f"threshold_{PAIR_NAME}.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if asyncio.get_event_loop().__class__.__name__ != 'SelectorEventLoop':
    asyncio.set_event_loop(asyncio.SelectorEventLoop())

# === Binance Testnet Setup ===
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

price_buffer = {TICKERS[0]: deque(maxlen=LOOKBACK + WINDOW_SIZE + 2000),
                TICKERS[1]: deque(maxlen=LOOKBACK + WINDOW_SIZE + 2000)}

position = 0
entry_price = None
equity = 0.0
equity_curve = []
trades = []
running = True

# === Modelle laden ===
def load_models():
    num_patches = WINDOW_SIZE // PATCH_SIZE
    teacher = TeacherNet(
        input_dim=1, patch_size=PATCH_SIZE, n_patches=num_patches,
        embedding_dim=TEACHER_EMB, output_dim=OUTPUT_DIM
    ).to(DEVICE)
    student = StudentNet(
        input_dim=1, output_dim=OUTPUT_DIM, patch_size=PATCH_SIZE,
        embedding_dim=STUDENT_EMB, n_prototypes=N_PROTOTYPES, num_patches=num_patches
    ).to(DEVICE)

    teacher.load_state_dict(torch.load(os.path.join("checkpoints", f"teacher_model_{PAIR_NAME}.pt"), map_location=DEVICE))
    student.load_state_dict(torch.load(os.path.join("checkpoints", f"student_model_{PAIR_NAME}.pt"), map_location=DEVICE))
    teacher.eval(); student.eval()
    return teacher, student

teacher, student = load_models()

def compute_spread_window():
    p1 = pd.Series(price_buffer[TICKERS[0]])
    p2 = pd.Series(price_buffer[TICKERS[1]])
    if min(len(p1), len(p2)) < LOOKBACK + WINDOW_SIZE:
        return None
    y = p1.iloc[-LOOKBACK:].reset_index(drop=True)
    x = p2.iloc[-LOOKBACK:].reset_index(drop=True)
    x = sm.add_constant(x)
    beta = sm.OLS(y, x).fit().params.iloc[1]
    spread = p1.values[-WINDOW_SIZE:] - beta * p2.values[-WINDOW_SIZE:]
    return spread, beta

def _load_train_threshold():
    if THRESHOLD_MODE != "train_file" or not os.path.exists(THRESHOLD_FILE):
        return None
    try:
        with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if "entry_score" in obj:
            return float(obj["entry_score"])
        if "entry_percentile" in obj and "scores_reference" in obj:
            ref = np.asarray(obj["scores_reference"], dtype=float)
            return float(np.percentile(ref, int(obj["entry_percentile"])))
    except Exception as e:
        print(f"⚠️ Konnte Threshold-Datei nicht verwenden: {e}")
    return None

@torch.no_grad()
def l2_score_for_window(spread_window):
    X = prepare_data(spread_window.reshape(-1, 1), WINDOW_SIZE, STEP_SIZE, PATCH_SIZE,
                     train=False, target_dim=1)
    if len(X) == 0:
        return None
    x = torch.tensor(X[-1:], dtype=torch.float32).to(DEVICE)
    c = teacher(x); z = student(x)
    return float(((z - c) ** 2).sum(dim=1).item())

@torch.no_grad()
def local_entry_threshold(spread_series, percentile=ENTRY_PERCENTILE):
    X = prepare_data(np.asarray(spread_series).reshape(-1,1), WINDOW_SIZE, STEP_SIZE, PATCH_SIZE,
                     train=False, target_dim=1)
    if len(X) == 0:
        return None
    scores = []
    for i in range(len(X)):
        x = torch.tensor(X[i:i+1], dtype=torch.float32).to(DEVICE)
        c = teacher(x); z = student(x)
        scores.append(((z - c) ** 2).sum(dim=1).item())
    scores = np.asarray(scores, dtype=float)
    return float(np.percentile(scores, percentile))

def generate_signals():
    global position
    res = compute_spread_window()
    if res is None:
        return None
    spread, beta = res

    # Richtung/Exit über Z-Score
    s = pd.Series(spread)
    z = (s - s.rolling(LOOKBACK).mean()) / s.rolling(LOOKBACK).std()
    z = float(z.fillna(0).values[-1])

    # Paper-Score
    score = l2_score_for_window(np.asarray(spread))
    if score is None:
        return None

    # Entry-Threshold: train-file bevorzugt
    entry_thresh = _load_train_threshold()
    if entry_thresh is None:
        entry_thresh = local_entry_threshold(spread)

    if position == 0 and score > entry_thresh:
        strength = np.clip(abs(z) / 2.0, 0.1, 1.0)
        direction = -np.sign(z) if z != 0 else -1
        return {"type": "entry", "direction": int(direction), "strength": float(strength)}
    elif position != 0 and abs(z) < EXIT_THRESHOLD:
        return {"type": "exit"}
    return None

def place_order(symbol, side, quantity):
    try:
        client.create_order(
            symbol=symbol,
            side=SIDE_BUY if side == 1 else SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=round(quantity, 6)
        )
        print(f"📥 Order {side}: {symbol}, Menge: {quantity}")
        return True
    except Exception as e:
        print(f"❌ Orderfehler: {e}")
        return False

def strategy_loop():
    global position, entry_price, equity, running, equity_curve

    try:
        balance_info = client.get_asset_balance(asset='USDT')
        equity = float(balance_info['free'])
    except Exception:
        equity = 10000.0
    equity_curve = [equity]

    while running:
        if min(len(price_buffer[TICKERS[0]]), len(price_buffer[TICKERS[1]])) < LOOKBACK + WINDOW_SIZE:
            time.sleep(CHECK_INTERVAL); continue

        signal = generate_signals()
        if signal:
            now = datetime.utcnow()
            if signal["type"] == "entry":
                direction = signal["direction"]
                strength  = signal["strength"]
                current_price = float(price_buffer[TICKERS[0]][-1])
                quantity = (100 * strength) / current_price
                quantity = round(quantity, 6)

                if place_order(TICKERS[0], direction, quantity):
                    entry_price = current_price
                    position = direction
                    trades.append({"time": now, "type": "entry", "side": direction,
                                   "price": entry_price, "strength": strength})
                    print(f"✅ Neue Position: {'LONG' if direction == 1 else 'SHORT'} @ {entry_price:.4f} | Stärke: {strength:.2f}")

            elif signal["type"] == "exit":
                current_price = float(price_buffer[TICKERS[0]][-1])
                quantity = 100 / current_price
                quantity = round(quantity, 6)

                if place_order(TICKERS[0], -position, quantity):
                    pnl = position * (current_price - entry_price) * quantity
                    equity += pnl
                    trades.append({"time": now, "type": "exit", "pnl": pnl, "equity": equity})
                    print(f"💼 Trade geschlossen: PnL = {pnl:.2f} USDT | Equity = {equity:.2f}")
                    entry_price = None
                    position = 0
                    equity_curve.append(equity)

        time.sleep(CHECK_INTERVAL)

def handle_socket_message(msg):
    if msg.get('e') != 'aggTrade':
        return
    symbol = msg.get('s')
    price = float(msg.get('p', 0.0))
    if symbol in price_buffer:
        price_buffer[symbol].append(price)

def main():
    global running
    print("🚀 Starte LLM-Paper-Trading (L2-Score, Eq. 8) – Threshold: train_file bevorzugt ...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
    twm.start()
    twm.start_aggtrade_socket(callback=handle_socket_message, symbol=TICKERS[0])
    twm.start_aggtrade_socket(callback=handle_socket_message, symbol=TICKERS[1])

    strategy_thread = threading.Thread(target=strategy_loop)
    strategy_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🔴 Stopp.")
        running = False
        strategy_thread.join()
    twm.join()

if __name__ == "__main__":
    main()
