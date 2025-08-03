import time, threading, asyncio
import numpy as np, pandas as pd, torch
from datetime import datetime
from collections import deque
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
from binance import ThreadedWebsocketManager
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')  # wichtig: kein GUI-Fenster im Thread
import matplotlib.pyplot as plt
from binance_keys import API_KEY, API_SECRET
from evaluate_anomalies import compute_anomaly_scores
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
ENTRY_PERCENTILE = 50
EXIT_THRESHOLD = 0.3
N_PROTOTYPES = 32
EMBEDDING_DIM = 768
OUTPUT_DIM = 128
CHECK_INTERVAL = 15
MAX_USD_RISK = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if asyncio.get_event_loop().__class__.__name__ != 'SelectorEventLoop':
    asyncio.set_event_loop(asyncio.SelectorEventLoop())

# === Binance Testnet Setup ===
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'
balance_info = client.get_asset_balance(asset='USDT')
equity = float(balance_info['free'])
print(f"💰 Startkapital laut Binance: {equity:.2f} USDT")

price_buffer = {TICKERS[0]: deque(maxlen=LOOKBACK + 200),
                TICKERS[1]: deque(maxlen=LOOKBACK + 200)}

position = 0
entry_price = None
equity_curve = [equity]
trades = []
running = True

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

import matplotlib.pyplot as plt

def plot_zscore(zscore_series):
    plt.figure(figsize=(10, 3))
    plt.plot(zscore_series, label='Z-Score')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axhline(EXIT_THRESHOLD, color='green', linestyle=':', label='Exit Threshold')
    plt.axhline(-EXIT_THRESHOLD, color='green', linestyle=':')
    plt.title('Z-Score der aktuellen Spread-Serie')
    plt.legend()
    plt.tight_layout()
    plt.savefig("zscore_live.png")
    plt.close()

def compute_hedge_ratio(y, x, lookback):
    y = pd.Series(y).iloc[-lookback:].reset_index(drop=True)
    x = pd.Series(x).iloc[-lookback:].reset_index(drop=True)
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    return model.params.iloc[1]

def calculate_performance(equity_series):
    daily_returns = equity_series.pct_change().dropna()
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else 0
    apr = (1 + mean_return) ** 252 - 1
    cumulative_return = (1 + daily_returns).cumprod()
    max_drawdown = (cumulative_return - cumulative_return.cummax()).min()
    return apr, sharpe, max_drawdown

# === SIGNAL-ERZEUGUNG (neu strukturiert) ===
def generate_signals():
    p1 = pd.Series(price_buffer[TICKERS[0]])
    p2 = pd.Series(price_buffer[TICKERS[1]])

    if min(len(p1), len(p2)) < LOOKBACK + WINDOW_SIZE:
        return None


    beta = compute_hedge_ratio(p1, p2, LOOKBACK)
    spread = p1.values[-WINDOW_SIZE:] - beta * p2.values[-WINDOW_SIZE:]
    spread_series = pd.Series(spread)
    zscore = (spread_series - spread_series.rolling(LOOKBACK).mean()) / spread_series.rolling(LOOKBACK).std()
    zscore = zscore.fillna(0)
    last_z = zscore.values[-1]

    X = prepare_data(spread_series.to_numpy().reshape(-1, 1), WINDOW_SIZE, STEP_SIZE, PATCH_SIZE, train=False, target_dim=1)
    scores = compute_anomaly_scores(teacher, student, X)
    smoothed_score = pd.Series(scores).rolling(3).mean().fillna(0).values[-1]
    entry_thresh = np.percentile(scores, ENTRY_PERCENTILE)

    if position == 0 and smoothed_score > entry_thresh:
        strength = np.clip(abs(last_z) / 2.0, 0.1, 1.0)
        direction = -np.sign(last_z)
        return {"type": "entry", "direction": direction, "strength": strength}

    elif position != 0 and abs(last_z) < EXIT_THRESHOLD:
        return {"type": "exit"}

    #plot_zscore(zscore)
    
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

# === STRATEGIE-LOOP ===
def strategy_loop():
    global position, entry_price, equity, running

    while running:
        if min(len(price_buffer[TICKERS[0]]), len(price_buffer[TICKERS[1]])) < LOOKBACK + WINDOW_SIZE:
            time.sleep(CHECK_INTERVAL)
            continue


        signal = generate_signals()

        if signal:
            now = datetime.utcnow()

            if signal["type"] == "entry":
                direction = signal["direction"]
                strength = signal["strength"]
                current_price = float(price_buffer[TICKERS[0]][-1])
                quantity = (MAX_USD_RISK * strength) / current_price
                quantity = round(quantity, 6)

                success = place_order(TICKERS[0], direction, quantity)
                if success:
                    entry_price = current_price
                    position = direction
                    trades.append({"time": now, "type": "entry", "side": direction, "price": entry_price, "strength": strength})
                    print(f"✅ Neue Position: {'LONG' if direction == 1 else 'SHORT'} @ {entry_price:.4f} | Stärke: {strength:.2f}")

            elif signal["type"] == "exit":
                current_price = float(price_buffer[TICKERS[0]][-1])
                quantity = MAX_USD_RISK / current_price
                quantity = round(quantity, 6)

                success = place_order(TICKERS[0], -position, quantity)
                if success:
                    pnl = position * (current_price - entry_price) * quantity
                    equity += pnl
                    trades.append({"time": now, "type": "exit", "pnl": pnl, "equity": equity})
                    print(f"💼 Trade geschlossen: PnL = {pnl:.2f} USDT | Equity = {equity:.2f}")
                    entry_price = None
                    position = 0
                    equity_curve.append(equity)

                    if len(trades) % 10 == 0:
                        equity_series = pd.Series(equity_curve)
                        apr, sharpe, max_dd = calculate_performance(equity_series)
                        print(f"\n📊 Performance nach {len(trades)} Trades:")
                        print(f"   → APR: {apr:.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}\n")

        time.sleep(CHECK_INTERVAL)

# === SOCKET HANDLER ===
def handle_socket_message(msg):
    if msg['e'] != 'aggTrade':
        return
    symbol = msg['s']
    price = float(msg['p'])
    if symbol in price_buffer:
        price_buffer[symbol].append(price)

# === MAIN ===
def main():
    global running, position, equity, entry_price

    print("🚀 Starte LLM-Paper-Trading über Binance Testnet...")
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

    twm.join()

if __name__ == "__main__":
    main()
