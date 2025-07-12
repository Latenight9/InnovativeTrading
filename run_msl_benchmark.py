import os
import argparse
import numpy as np
import torch
from teacher_model import TeacherNet
from student_model import StudentNet
from data_preparation import prepare_data
from evaluate_anomalies import compute_anomaly_scores
from metrics import event_based_scores
from transformers import GPT2Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Konfiguration
WINDOW_SIZE = 24
PATCH_SIZE = 6
STEP_SIZE = 1
EMBEDDING_DIM = 768
OUTPUT_DIM = 128
N_PROTOTYPES = 32

TEST_DIR = "data/MSL/test"
LABEL_DIR = "data/MSL/test_label"
# Ganz oben im Script (außerhalb von Funktionen)
teacher_loaded_once = set()

def evaluate_file(fname):
    test_path = os.path.join(TEST_DIR, fname)
    label_path = os.path.join(LABEL_DIR, fname)

    if not os.path.exists(label_path):
        print(f"⚠️ Kein Label für {fname} gefunden – wird übersprungen.")
        return None

    data = np.load(test_path)
    labels = np.load(label_path)
    dim = data.shape[1]
    model_path = f"student_model_{dim}.pt"
    num_patches = WINDOW_SIZE // PATCH_SIZE

    if not os.path.exists(model_path):
        print(f"⏭️ Kein Modell für Featureanzahl {dim} gefunden. Datei {fname} wird übersprungen.")
        return None

    print(f"🚀 Evaluierung: {fname} (Features: {dim})")

    # Zeitreihen vorbereiten
    X = prepare_data(data, WINDOW_SIZE, STEP_SIZE, patch_size=PATCH_SIZE)

    # Modelle laden
    teacher = TeacherNet(
        input_dim=dim,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        patch_size=PATCH_SIZE,
        n_patches=num_patches
    ).to(DEVICE)
    
    teacher_weights_path = f"teacher_{dim}.pt"
    
    if os.path.exists(teacher_weights_path):
        teacher.load_state_dict(torch.load(teacher_weights_path))
    if dim not in teacher_loaded_once:
        print("🔥 Teacher-Weights erfolgreich geladen.")
        teacher_loaded_once.add(dim)
    else:
        print("⚠️ Warnung: Keine Teacher-Weights gefunden!")

    
    teacher.eval()

    student = StudentNet(
        input_dim=dim,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        patch_size=PATCH_SIZE,
        n_prototypes=N_PROTOTYPES,
        num_patches=num_patches
    ).to(DEVICE)
    student.load_state_dict(torch.load(model_path, map_location=DEVICE))
    student.eval()

    # Anomalie-Scores berechnen
    scores = compute_anomaly_scores(teacher, student, X)
    
    # Event-basierte Auswertung
    threshold = np.percentile(scores, 95)  # Top 5% als Anomalien
    pred_events = (scores >= threshold).astype(int)
    precision, recall, f1 = event_based_scores(labels[WINDOW_SIZE-1:], pred_events)
    print(f"📊 Event-based: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Name der Testdatei (optional)")
    args = parser.parse_args()

    if args.file:
        f1 = evaluate_file(args.file)
        if f1 is not None:
            print(f"\n✅ F1 für {args.file}: {f1:.4f}")
    else:
        all_f1 = []
        for fname in sorted(os.listdir(TEST_DIR)):
            f1 = evaluate_file(fname)
            if f1 is not None:
                all_f1.append(f1)

        if all_f1:
            print(f"\n✅ Durchschnittlicher F1 über alle Dateien: {np.mean(all_f1):.4f}")
        else:
            print("⚠️ Keine gültigen Dateien oder Modelle gefunden.")

if __name__ == "__main__":
    main()
