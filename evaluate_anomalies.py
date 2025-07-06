# evaluate_anomalies.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from teacher_model import TeacherNet
from student_model import StudentNet
from data_preparation import prepare_data, create_patches
from Analysis import load_data
from prototype_selection import select_topk_prototypes
import numpy as np

# ⚙️ Konfiguration
WINDOW_SIZE = 24
PATCH_SIZE = 6
STEP_SIZE = 1
EMBEDDING_DIM = 128
OUTPUT_DIM = 128
N_PROTOTYPES = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_anomaly_scores(teacher, student, X):
    """
    Berechnet A(w) = ‖ϕ(w) - φ(w)‖² für jedes Fenster
    """
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    anomaly_scores = []

    teacher.eval()
    student.eval()

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)  # (B, T, D)

            # Prototypenwahl
            feat = student.embed_input(x)  # (B, emb_dim)
            topk_ids = select_topk_prototypes(feat, student.prototypes, k=8)

            # Vorwärtsläufe
            t_out = teacher(x)               # φ(w)
            s_out = student(x, selected_proto_ids=topk_ids)  # ϕ(w)

            # Anomalie-Scores: ‖ϕ - φ‖²
            diff = (s_out - t_out) ** 2
            score = diff.sum(dim=1).cpu().numpy()
            anomaly_scores.extend(score)

    return np.array(anomaly_scores)


def main():
    # 🔹 Daten laden
    df = load_data(['BTC/USDT', 'ETH/USDT'], interval="1h", since_days=30)
    X = prepare_data(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE)  # (N, T, D)

    # 🔹 Modelle laden
    teacher = TeacherNet(input_dim=X.shape[2], embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM).to(DEVICE)
    student = StudentNet(input_dim=X.shape[2], embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM,
                         n_prototypes=N_PROTOTYPES).to(DEVICE)

    student.load_state_dict(torch.load("student_model.pt", map_location=DEVICE))
    teacher.eval()
    student.eval()

    # 🔹 Anomalie-Scores berechnen
    scores = compute_anomaly_scores(teacher, student, X)

    # 🔹 Ergebnisse ausgeben
    for i, s in enumerate(scores):
        print(f"Fenster {i}: Anomalie-Score = {s:.4f}")

    # Optional: Speichern
    np.save("anomaly_scores.npy", scores)

if __name__ == "__main__":
    main()
