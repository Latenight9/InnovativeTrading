import os
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from data_preparation import prepare_data
from Analysis import load_data, calculate_rolling_beta
from backtest import simulate_trades, evaluate_performance, plot_equity
from evaluate_anomalies import compute_anomaly_scores
from student_model import StudentNet
from teacher_model import TeacherNet
from chan_example import get_chan_example_tickers

# === Direkte Konfiguration ===
TICKERS = get_chan_example_tickers()  # Erwartet zwei Ticker
START_DATE = "2010-04-01"
END_DATE = "2012-04-09"
INTERVAL = "1d"
LOOKBACK_PERIOD = 20
INITIAL_CAPITAL = 10_000.0

# AnomalyLLM
WINDOW_SIZE = 120
PATCH_SIZE = 10
STEP_SIZE = 1
EMBEDDING_DIM = 768
OUTPUT_DIM = 128
N_PROTOTYPES = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(pair_name, dim):
    num_patches = WINDOW_SIZE // PATCH_SIZE

    teacher = TeacherNet(
        input_dim=dim,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        patch_size=PATCH_SIZE,
        n_patches=num_patches
    ).to(DEVICE)

    teacher_weights_path = f"teacher_model_{pair_name}.pt"
    if not os.path.exists(teacher_weights_path):
        raise FileNotFoundError(f"Lehrermodell {teacher_weights_path} nicht gefunden.")
    teacher.load_state_dict(torch.load(teacher_weights_path, map_location=DEVICE))
    teacher.eval()

    student = StudentNet(
        input_dim=dim,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        patch_size=PATCH_SIZE,
        n_prototypes=N_PROTOTYPES,
        num_patches=num_patches
    ).to(DEVICE)

    student_weights_path = f"student_model_{pair_name}.pt"
    if not os.path.exists(student_weights_path):
        raise FileNotFoundError(f"Schülermodell {student_weights_path} nicht gefunden.")
    student.load_state_dict(torch.load(student_weights_path, map_location=DEVICE))
    student.eval()

    return teacher, student


def build_positions(spread, zscore, scores, entry_percentile=85, exit_threshold=0.3):
    entry_threshold = np.percentile(scores, entry_percentile)
    in_position = False
    last_position = 0
    positions = []

    for i in range(len(scores)):
        if not in_position and scores[i] > entry_threshold:
            in_position = True
            last_position = -zscore[i]
        elif in_position and abs(zscore[i]) < exit_threshold:
            in_position = False
            last_position = 0
        positions.append(last_position)

    return pd.Series(positions, index=spread.index)


def test_chan_strategy():
    stock1, stock2 = TICKERS[0], TICKERS[1]
    pair_name = f"{stock1}_{stock2}"

    # 1. Daten laden
    prices_df = load_data([stock1, stock2], START_DATE, END_DATE, interval=INTERVAL)
    p1 = prices_df[stock1]
    p2 = prices_df[stock2]

    # 2. Spread berechnen
    beta_series = calculate_rolling_beta(p1, p2, LOOKBACK_PERIOD)
    spread = (p1 - beta_series * p2).dropna()

    # 3. Z-Score berechnen
    zscore = (spread - spread.rolling(LOOKBACK_PERIOD).mean()) / spread.rolling(LOOKBACK_PERIOD).std()
    zscore = zscore.fillna(0)

    # 4. Scaler laden
    dim = 1
    
    X = prepare_data(
        spread.values.reshape(-1, 1),
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        patch_size=PATCH_SIZE,
        train=False,
        target_dim=dim,
    )

    teacher, student = load_models(pair_name, dim)
    scores = compute_anomaly_scores(teacher, student, X)
    scores = gaussian_filter1d(scores, sigma=1)

    # 5. Align Daten (wegen Window-Verlust)
    offset = len(spread) - len(scores)
    spread = spread.iloc[offset:]
    zscore = zscore.iloc[offset:]
    beta_series = beta_series.iloc[offset:]
    prices_df = prices_df.iloc[-len(spread):]

    # 6. Positionen bestimmen
    positions = build_positions(spread, zscore, scores)

    # 7. Backtest
    trades_df = simulate_trades(prices_df, (stock1, stock2), positions, beta_series, initial_capital=INITIAL_CAPITAL)

    # 8. Evaluation
    evaluate_performance(trades_df)
    plot_equity(trades_df, (stock1, stock2))


if __name__ == "__main__":
    test_chan_strategy()
