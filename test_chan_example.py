import os
import json
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d

from data_preparation import prepare_data
from Analysis import load_data, calculate_rolling_beta
from backtest import simulate_trades, evaluate_performance, plot_equity
from student_model import StudentNet
from teacher_model import TeacherNet
from chan_example import get_chan_example_tickers

# === Direkte Konfiguration ===
TICKERS = get_chan_example_tickers()  # Erwartet zwei Ticker
START_DATE = "2010-04-01"
END_DATE   = "2012-04-09"
INTERVAL   = "1d"
LOOKBACK_PERIOD = 20
INITIAL_CAPITAL = 10_000.0

# Anomaly LLM
WINDOW_SIZE = 120
PATCH_SIZE  = 10
STEP_SIZE   = 1
TEACHER_EMB = 768
STUDENT_EMB = 128
OUTPUT_DIM  = 128
N_PROTOTYPES = 32
ENTRY_PERCENTILE_FALLBACK = 85    # Fallback nur wenn kein train-Threshold
EXIT_THRESHOLD = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(pair_name, dim):
    num_patches = WINDOW_SIZE // PATCH_SIZE

    teacher = TeacherNet(
        input_dim=dim, embedding_dim=TEACHER_EMB, output_dim=OUTPUT_DIM,
        patch_size=PATCH_SIZE, n_patches=num_patches
    ).to(DEVICE)
    teacher.load_state_dict(torch.load(os.path.join("checkpoints", f"teacher_model_{pair_name}.pt"), map_location=DEVICE))
    teacher.eval()

    student = StudentNet(
        input_dim=dim, embedding_dim=STUDENT_EMB, output_dim=OUTPUT_DIM,
        patch_size=PATCH_SIZE, n_prototypes=N_PROTOTYPES, num_patches=num_patches
    ).to(DEVICE)
    student.load_state_dict(torch.load(os.path.join("checkpoints", f"student_model_{pair_name}.pt"), map_location=DEVICE))
    student.eval()
    return teacher, student

def load_train_threshold(pair_name):
    path = os.path.join("checkpoints", f"threshold_{pair_name}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if "entry_score" in obj:            # fester Score aus Training
            return float(obj["entry_score"])
        if "entry_percentile" in obj and "scores_reference" in obj:
            ref = np.asarray(obj["scores_reference"], dtype=float)
            return float(np.percentile(ref, int(obj["entry_percentile"])))
    except Exception as e:
        print(f"⚠️ Konnte Threshold-Datei nicht verwenden: {e}")
    return None

@torch.no_grad()
def compute_l2_scores(teacher, student, X_np):
    X = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    scores = []
    bs = 256
    for i in range(0, len(X), bs):
        xb = X[i:i+bs]
        c = teacher(xb); z = student(xb)
        s = ((z - c) ** 2).sum(dim=1)
        scores.append(s.detach().cpu().numpy())
    return np.concatenate(scores, axis=0) if scores else np.zeros(0, dtype=float)

def build_positions(spread, zscore, scores, entry_threshold=None, entry_percentile_fallback=ENTRY_PERCENTILE_FALLBACK, exit_threshold=EXIT_THRESHOLD):
    if entry_threshold is None:
        entry_threshold = float(np.percentile(scores, entry_percentile_fallback))
    in_position = False
    last_position = 0
    positions = []
    for i in range(len(scores)):
        if not in_position and scores[i] > entry_threshold:
            in_position = True
            last_position = -np.sign(zscore[i]) if zscore[i] != 0 else -1
        elif in_position and abs(zscore[i]) < exit_threshold:
            in_position = False
            last_position = 0
        positions.append(last_position)
    return pd.Series(positions, index=spread.index)

def test_chan_strategy():
    stock1, stock2 = TICKERS[0], TICKERS[1]
    pair_name = f"{stock1}_{stock2}"

    # 1) Daten laden
    prices_df = load_data([stock1, stock2], START_DATE, END_DATE, interval=INTERVAL)
    p1 = prices_df[stock1]; p2 = prices_df[stock2]

    # 2) Spread
    beta_series = calculate_rolling_beta(p1, p2, LOOKBACK_PERIOD)
    spread = (p1 - beta_series * p2).dropna()

    # 3) Z-Score
    zscore = (spread - spread.rolling(LOOKBACK_PERIOD).mean()) / spread.rolling(LOOKBACK_PERIOD).std()
    zscore = zscore.fillna(0)

    # 4) Windows/Patches
    dim = 1
    X = prepare_data(
        spread.values.reshape(-1, 1),
        window_size=WINDOW_SIZE, step_size=STEP_SIZE, patch_size=PATCH_SIZE,
        train=False, target_dim=dim,
    )

    # 5) Modelle
    teacher, student = load_models(pair_name, dim)

    # 6) Scores
    scores = compute_l2_scores(teacher, student, X)
    scores = gaussian_filter1d(scores, sigma=1)  # sanftes Glätten (optional)

    # 7) Align
    offset = len(spread) - len(scores)
    spread = spread.iloc[offset:]; zscore = zscore.iloc[offset:]; beta_series = beta_series.iloc[offset:]
    prices_df = prices_df.iloc[-len(spread):]

    # 8) Entry-Threshold (train-file bevorzugt)
    entry_threshold = load_train_threshold(pair_name)

    # 9) Positionen & Backtest
    positions = build_positions(spread, zscore, scores, entry_threshold=entry_threshold)
    trades_df = simulate_trades(prices_df, (stock1, stock2), positions, beta_series, initial_capital=INITIAL_CAPITAL)

    # 10) Evaluation
    evaluate_performance(trades_df)
    plot_equity(trades_df, (stock1, stock2))

if __name__ == "__main__":
    test_chan_strategy()
