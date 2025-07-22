import time
import threading
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from collections import deque
from binance.client import Client
from binance.websockets import BinanceSocketManager
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
import statsmodels.api as sm

from binance_keys import API_KEY, API_SECRET
from evaluate_anomalies import compute_anomaly_scores
from teacher_model import TeacherNet
from student_model import StudentNet
from data_preparation import prepare_data

# === KONFIGURATION ===
TICKERS = ["BTCUSDT", "ETHUSDT"]
PAIR_NAME = f"{TICKERS[0]}_{TICKERS[1]}"
LOOKBACK = 20
WINDOW_SIZE = 120
PATCH_SIZE = 10
STEP_SIZE = 1
ENTRY_PERCENTILE = 85
EXIT_THRESHOLD = 0.3
N_PROTOTYPES = 32
EMBEDDING_DIM = 768
OUTPUT_DIM = 128
CHECK_INTERVAL = 60  # Sekunden
MAX_USD_RISK = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === BINANCE CLIENT ===
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

balance_info = client.get_asset_balance(asset='USDT')
equity = float(balance_info['free'])
print(f"💰 Startkapital laut Binance: {equity:.2f} USDT")

# === LIVE-PREISPUFFER ===
price_buffer = {TICKERS[0]: deque(maxlen=LOOKBACK + 200),
                TICKERS[1]: deque(maxlen=LOOKBACK + 200)}

# === STATUSVARIABLEN ===
position = 0
entry_price = None
equity_curve = [equity]
trades = []
running = True  # Steuerung der Strategie-Loop

# === MODELLE LADEN ===
def load_models():
    num_patches = WINDOW_SIZE // PATCH_SIZE
    teacher = TeacherNet(
        input_dim=1,
        patch_size=PATCH_SIZE,
        n_patches=num_patches,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM
    ).to(DEVICE)

    student = StudentNet(
        input_dim=1,
        output_dim=OUTPUT_DIM,
        patch_size=PATCH_SIZE,
        embedding_dim=EMBEDDING_DIM,
        n_prototypes=N_PROTOTYPES,
        num_patches=num_patches
    ).to(DEVICE)

    teacher.load_state_dict(torch.load(f"teacher_model_{PAIR_NAME}.pt", map_location=DEVICE))
    student.load_state_dict(torch.load(f"student_model_{PAIR_NAME}.pt", map_location=DEVICE))
    teacher.eval()
    student.eval()
    return teacher, student

teacher, student = load_models()

# === REGRESSION für Hedge Ratio
def compute_hedge_ratio(y, x, lookback):
    y = pd.Series(y).iloc[-lookback:]
    x = pd.Series(x).iloc[-lookback:]
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    return model.params[1]

# === PERFORMANCE-BERECHNUNG
def calculate_performance(equity_series):
    daily_returns = equity_series.pct_change().dropna()
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else 0
    apr = (1 + mean_return) ** 252 - 1
    cumulative_return = (1 + daily_returns).cumprod()
    max_drawdown = (cumulative_return - cumulative_return.cummax()).min()
    return apr, sharpe, max_drawdown

# === SIGNAL-ERZEUGUNG
def generate_signals():
    p1 = pd.Series(price_buffer[TICKERS[0]])
    p2 = pd.Series(price_buffer[TICKERS[1]])

    if len(p1) < LOOKBACK + WINDOW_SIZE:
        return None

    beta = compute_hedge_ratio(p1, p2, LOOKBACK)
    spread = p1.values[-WINDOW_SIZE:] - beta * p2.values[-WINDOW_SIZE:]
    spread_series = pd.Series(spread)
    zscore = (spread_series - spread_series.rolling(LOOKBACK).mean()) / spread_series.rolling(LOOKBACK).std()
    zscore = zscore.fillna(0)

    X = prepare_data(spread_series.to_numpy().reshape(-1, 1), WINDOW_SIZE, STEP_SIZE, PATCH_SIZE, train=False, target_dim=1)
    scores = compute_anomaly_scores(teacher, student, X)
    smoothed_score = pd.Series(scores).rolling(3).mean().fillna(0).values[-1]

    entry_thresh = np.percentile(scores, ENTRY_PERCENTILE)
    last_z = zscore.values[-1]

    if position == 0 and smoothed_score > entry_thresh:
        return -np.sign(last_z)
    elif position != 0 and abs(last_z) < EXIT_THRESHOLD:
        return 0
    return None

# === ORDER-FUNKTION
def place_order(symbol, side, quantity):
    try:
        client.create_test_order(
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

# === STRATEGIE-LOOP
def strategy_loop():
    global position, entry_price, equity, running

    while running:
        if len(price_buffer[TICKERS[0]]) < LOOKBACK + WINDOW_SIZE:
            time.sleep(CHECK_INTERVAL)
            continue

        current_price = float(price_buffer[TICKERS[0]][-1])
        quantity = MAX_USD_RISK / current_price

        signal = generate_signals()
        if signal is not None and signal != position:
            success = place_order(TICKERS[0], signal, quantity)
            if success:
                now = datetime.utcnow()
                if signal == 0:
                    exit_price = current_price
                    pnl = position * (exit_price - entry_price) * quantity
                    equity += pnl
                    print(f"💼 Trade geschlossen: PnL = {pnl:.2f} USDT | Equity = {equity:.2f}")
                    trades.append({"time": now, "type": "exit", "pnl": pnl, "equity": equity})
                    entry_price = None
                    position = 0
                else:
                    entry_price = current_price
                    position = signal
                    print(f"✅ Neue Position: {'LONG' if signal == 1 else 'SHORT'} @ {entry_price}")
                    trades.append({"time": now, "type": "entry", "side": signal, "price": entry_price})

                equity_curve.append(equity)

                if len(trades) % 10 == 0:
                    equity_series = pd.Series(equity_curve)
                    apr, sharpe, max_dd = calculate_performance(equity_series)
                    print(f"\n📊 Performance nach {len(trades)} Trades:")
                    print(f"   → APR: {apr:.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}\n")

        time.sleep(CHECK_INTERVAL)

# === BINANCE WEBSOCKET HANDLER
def handle_socket_message(msg):
    if msg['e'] != 'aggTrade': return
    symbol = msg['s']
    price = float(msg['p'])
    if symbol in price_buffer:
        price_buffer[symbol].append(price)

# === START
def main():
    global running, position, equity, entry_price

    print("🚀 Starte LLM-Paper-Trading über Binance Testnet...")
    bm = BinanceSocketManager(client)
    bm.start_aggtrade_socket(TICKERS[0].lower(), handle_socket_message)
    bm.start_aggtrade_socket(TICKERS[1].lower(), handle_socket_message)
    bm.start()

    strategy_thread = threading.Thread(target=strategy_loop)
    strategy_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🔴 Manuelle Unterbrechung erkannt.")
        running = False
        strategy_thread.join()

        if position != 0:
            print("⚠️ Offene Position erkannt – versuche zu schließen...")
            current_price = float(price_buffer[TICKERS[0]][-1])
            quantity = MAX_USD_RISK / current_price
            success = place_order(TICKERS[0], -position, quantity)

            if success:
                pnl = position * (current_price - entry_price) * quantity
                equity += pnl
                print(f"✅ Position automatisch geschlossen. Finaler PnL: {pnl:.2f} USDT")
        print("🚪 Strategie sauber beendet.")

if __name__ == "__main__":
    main()
