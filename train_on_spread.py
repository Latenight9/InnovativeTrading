# train_on_spread.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from Analysis import load_data, calculate_rolling_beta
from data_preparation import prepare_data
from teacher_model import TeacherNet
from student_model import StudentNet
from losses import total_loss
from augment_time_series import augment_batch


# === Konfiguration ===
TICKERS = ["GLD", "USO"]  # oder z.B. ["BTCUSDT", "ETHUSDT"]
START_DATE = "2006-05-01"
END_DATE = "2010-04-01"
INTERVAL = "1d"
LOOKBACK_PERIOD = 20

# AnomalyLLM Parameter
WINDOW_SIZE = 120
PATCH_SIZE = 10
STEP_SIZE = 1
BATCH_SIZE = 32
EMBEDDING_DIM = 768
OUTPUT_DIM = 128
N_PROTOTYPES = 32
EPOCHS = 50
PATIENCE = 5
MIN_DELTA = 1e-4
LR = 1e-4
LR_TEACHER = 1e-4
LAMBDA_CE = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_spread_model(tickers):
    # === Daten laden ===
    prices_df = load_data(tickers, START_DATE, END_DATE, interval=INTERVAL)
    p1 = prices_df[tickers[0]]
    p2 = prices_df[tickers[1]]

    # === Rolling Beta und Spread berechnen ===
    beta_series = calculate_rolling_beta(p1, p2, LOOKBACK_PERIOD)
    spread = p1 - beta_series * p2
    spread = spread.dropna()

    print("\n📐 Rolling Beta Beispiel:")
    print(beta_series.dropna().head())
    print("\n🧮 Spread-Statistik:")
    print(spread.describe())

    # === Spread umformen für Training ===
    spread_np = spread.values.reshape(-1, 1)  # Shape: (T, 1)

    # === Fenster + Patches erzeugen ===
    patched_data = prepare_data(
        spread_np,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        patch_size=PATCH_SIZE,
        train=True,
        target_dim=1
    )

    dataset = TensorDataset(torch.tensor(patched_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # === Modell-Initialisierung ===
    num_patches = WINDOW_SIZE // PATCH_SIZE

    teacher = TeacherNet(
        input_dim=1,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        patch_size=PATCH_SIZE,
        n_patches=num_patches
    ).to(DEVICE)

    student = StudentNet(
        input_dim=1,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        n_prototypes=N_PROTOTYPES,
        patch_size=PATCH_SIZE,
        num_patches=num_patches
    ).to(DEVICE)

    optimizer_student = torch.optim.Adam(student.parameters(), lr=LR)
    optimizer_teacher = torch.optim.Adam(filter(lambda p: p.requires_grad, teacher.parameters()), lr=LR_TEACHER)

    # === Training ===
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0

        for (batch,) in loader:
            batch = batch.to(DEVICE)
            batch_aug = augment_batch(batch)

            student.train()
            teacher.train()

            z = student(batch)
            z_aug = student(batch_aug)
            c = teacher(batch)
            c_aug = teacher(batch_aug)

            loss, loss_kd, loss_ce = total_loss(z, c, z_aug, c_aug, c, c_aug, lambda_ce=LAMBDA_CE)

            optimizer_student.zero_grad()
            optimizer_teacher.zero_grad()

            loss.backward()
            optimizer_student.step()
            optimizer_teacher.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"📉 Epoch {epoch+1}: Loss={avg_loss:.4f} | λ_CE={LAMBDA_CE}")

        if avg_loss < best_loss - MIN_DELTA:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(student.state_dict(), "student_spread_model.pt")
            torch.save(teacher.state_dict(), "teacher_spread_model.pt")
            print("💾 Modelle gespeichert.")
        else:
            patience_counter += 1
            print(f"⏸️ Keine Verbesserung ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("🛑 Early stopping.")
            break

if __name__ == "__main__":
    train_spread_model(TICKERS)
