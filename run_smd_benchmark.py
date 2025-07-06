import os
import torch
import numpy as np
from tqdm import tqdm

from teacher_model import TeacherNet
from student_model import StudentNet
from augment_time_series import augment_batch
from data_preparation import prepare_data
from prototype_selection import initialize_prototypes
from losses import total_loss
from metrics import event_based_scores
from evaluate_anomalies import compute_anomaly_scores

# 🔧 Hyperparameter wie im Paper
WINDOW_SIZE = 24
STEP_SIZE = 1
PATCH_SIZE = 6
EMBEDDING_DIM = 128
OUTPUT_DIM = 128
INTERMEDIATE_DIM = 64
N_HEADS = 8
N_PROTOTYPES = 32
LAMBDA_CE = 0.5
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/SMD"

# 🔄 Hilfsfunktion zum Laden der SMD-Dateien
def load_smd_file(path):
    return np.loadtxt(path, delimiter=",")

# Alle Maschinen aus dem Datensatz
MACHINES = sorted([
    fname.replace(".txt", "")
    for fname in os.listdir(os.path.join(DATA_DIR, "train"))
    if fname.endswith(".txt")
])

def evaluate_machine(machine_name):
    print(f"🔧 Verarbeite {machine_name}...")

    # 📥 Daten laden
    train = load_smd_file(os.path.join(DATA_DIR, "train", f"{machine_name}.txt"))
    test = load_smd_file(os.path.join(DATA_DIR, "test", f"{machine_name}.txt"))
    labels = load_smd_file(os.path.join(DATA_DIR, "test_label", f"{machine_name}.txt"))

    # 📐 Fenster + Patches vorbereiten
    X_train = prepare_data(train, WINDOW_SIZE, STEP_SIZE, PATCH_SIZE)
    X_test = prepare_data(test, WINDOW_SIZE, STEP_SIZE, PATCH_SIZE)
    y_test = (prepare_data(labels, WINDOW_SIZE, STEP_SIZE, PATCH_SIZE).max(axis=1) >= 1).astype(int)

    n_channels = train.shape[1]
    input_dim = PATCH_SIZE * n_channels

    # 🧠 Modelle bauen
    teacher = TeacherNet(input_dim, EMBEDDING_DIM, OUTPUT_DIM).to(DEVICE)
    student = StudentNet(input_dim, EMBEDDING_DIM, OUTPUT_DIM,
                         PATCH_SIZE, N_HEADS, INTERMEDIATE_DIM, N_PROTOTYPES).to(DEVICE)

    initialize_prototypes(student, torch.tensor(X_train, dtype=torch.float32).to(DEVICE))

    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    teacher.eval()

    for epoch in range(EPOCHS):
        student.train()
        for start in range(0, len(X_train), BATCH_SIZE):
            batch = torch.tensor(X_train[start:start + BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            if len(batch) < BATCH_SIZE:
                break

            batch_aug = augment_batch(batch)

            with torch.no_grad():
                t_out = teacher(batch.reshape(batch.shape[0], WINDOW_SIZE, -1))

            s_out = student(batch)
            s_aug = student(batch_aug)

            labels0 = torch.zeros(batch.shape[0], device=DEVICE)
            loss, _, _ = total_loss(s_out, t_out, s_aug, labels0, lambda_ce=LAMBDA_CE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 📈 Evaluation
    student.eval()
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        z_t = teacher(x_test_tensor.reshape(len(X_test), WINDOW_SIZE, -1))
        z_s = student(x_test_tensor)

    scores = compute_anomaly_scores(z_t, z_s)
    preds = (scores > np.percentile(scores, 95)).astype(int)

    precision, recall, f1 = event_based_scores(y_test, preds)
    return precision, recall, f1

def run_full_benchmark():
    all_prec, all_rec, all_f1 = [], [], []

    print("🚀 Starte vollständige Benchmark-Auswertung über alle Maschinen...\n")

    for machine in tqdm(MACHINES):
        try:
            p, r, f = evaluate_machine(machine)
            all_prec.append(p)
            all_rec.append(r)
            all_f1.append(f)
            print(f"✅ {machine}: P={p:.4f} R={r:.4f} F1={f:.4f}")
        except Exception as e:
            print(f"❌ Fehler bei {machine}: {e}")

    print("\n📊 Durchschnittliche Performance über alle Maschinen:")
    print(f"🔹 Precision: {np.mean(all_prec):.4f}")
    print(f"🔹 Recall:    {np.mean(all_rec):.4f}")
    print(f"🔹 F1 Score:  {np.mean(all_f1):.4f}")

if __name__ == "__main__":
    run_full_benchmark()
