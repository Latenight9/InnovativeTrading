import torch
from torch.utils.data import DataLoader, TensorDataset
from teacher_model import TeacherNet
from student_model import StudentNet
from data_preparation import prepare_data
from Analysis import load_data
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
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    anomaly_scores = []
    teacher.eval()
    student.eval()

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)

            student_out = student(x)  # ϕ(w)
            teacher_out = teacher(x)  # φ(w)

            diff = (student_out - teacher_out) ** 2
            score = diff.sum(dim=1).cpu().numpy()
            anomaly_scores.extend(score)

    scores = np.array(anomaly_scores)

    # 📏 [NEU] Normalisierung auf [0, 1] für stabile Percentile
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    return scores

def main():
    print("Dieses Modul stellt compute_anomaly_scores(...) bereit.")

if __name__ == "__main__":
    main()
