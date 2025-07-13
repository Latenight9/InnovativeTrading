import os
import argparse
import numpy as np
import torch
from teacher_model import TeacherNet
from student_model import StudentNet
from data_preparation import prepare_data
from evaluate_anomalies import compute_anomaly_scores
from metrics_affiliation import affiliation_precision_recall_f1
from transformers import GPT2Model
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import random
from scipy.ndimage import binary_opening, binary_closing
from sklearn.preprocessing import StandardScaler
import joblib


# ⚙️ Konfiguration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 24
PATCH_SIZE = 6
STEP_SIZE = 1
EMBEDDING_DIM = 768
OUTPUT_DIM = 128
N_PROTOTYPES = 32
TEST_DIR = "data/MSL/test"
LABEL_DIR = "data/MSL/test_label"
teacher_loaded_once = set()

# 📊 Plot zur visuellen Überprüfung
def plot_alignment(scores, labels, preds, fname=""):
    plt.figure(figsize=(12, 3))
    plt.plot(scores, label="Anomaly Score", color='black')
    plt.plot(labels * max(scores), label="Label", linestyle='--', color='red', alpha=0.6)
    plt.plot(preds * max(scores), label="Prediction", linestyle=':', color='green', alpha=0.6)
    plt.title(f"Alignment Check: {fname}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 🧠 Robuste Label-Ausrichtung
def align_labels_to_scores(labels, num_scores, window_size):
    expected = len(labels) - window_size + 1
    if expected == num_scores:
        return labels[window_size - 1:]  # Standardfall
    elif len(labels) == num_scores:
        return labels
    elif len(labels) > num_scores:
        offset = len(labels) - num_scores
        print(f"⚠️ Labels ({len(labels)}) > Scores ({num_scores}) – Offset: {offset}")
        return labels[offset:]
    else:
        raise ValueError(f"❌ Labels ({len(labels)}) kürzer als Scores ({num_scores})!")


def clean_predictions(pred_array, min_event_length=5, join_gap=8):
    """
    Wandelt punktuelle binäre Predictions in echte Events um:
    - entfernt kurze Einzel-1er (Rauschen)
    - verbindet nahe 1er zu einem zusammenhängenden Event
    """
    pred_array = binary_closing(pred_array, structure=np.ones(join_gap)).astype(int)
    pred_array = binary_opening(pred_array, structure=np.ones(min_event_length)).astype(int)
    return pred_array



# 🔍 Haupt-Evaluierung für eine Datei
def evaluate_file(fname, show_plot=False):
    test_path = os.path.join(TEST_DIR, fname)
    label_path = os.path.join(LABEL_DIR, fname)

    if not os.path.exists(label_path):
        print(f"⚠️ Kein Label für {fname} gefunden – wird übersprungen.")
        return None

    data = np.load(test_path)
    scaler = joblib.load("scaler_55.pkl")
    data = scaler.transform(data)
    labels = np.load(label_path)
    dim = data.shape[1]
    if dim != 55:
        print(f"⏭️ {fname} hat {dim} Features – übersprungen (nur 55 erlaubt).")
        return None

    model_path = f"student_model_{dim}.pt"
    num_patches = WINDOW_SIZE // PATCH_SIZE

    if not os.path.exists(model_path):
        print(f"⏭️ Kein Modell für Featureanzahl {dim} gefunden. Datei {fname} wird übersprungen.")
        return None

    print(f"\n🚀 Evaluierung: {fname} (Features: {dim})")

    X = prepare_data(data, WINDOW_SIZE, STEP_SIZE, patch_size=PATCH_SIZE)

    # 🔁 Modelle laden
    teacher = TeacherNet(
        input_dim=dim,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        patch_size=PATCH_SIZE,
        n_patches=num_patches
    ).to(DEVICE)
    teacher_weights_path = os.path.join(os.path.dirname(__file__), f"teacher_{dim}.pt")
    if not os.path.exists(teacher_weights_path):
        print(f"⚠️ Keine Teacher-Weights unter '{teacher_weights_path}' gefunden!")
        return None
    teacher.load_state_dict(torch.load(teacher_weights_path))
    if dim not in teacher_loaded_once:
        print("🔥 Teacher-Weights erfolgreich geladen.")
        teacher_loaded_once.add(dim)
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

    # 🧪 Anomalie-Scores berechnen
    scores = compute_anomaly_scores(teacher, student, X)
    scores = gaussian_filter1d(scores, sigma=2)
    threshold = np.percentile(scores, 90)
    pred_events = (scores >= threshold).astype(int)
    pred_events = clean_predictions(pred_events, min_event_length=50, join_gap=50)


    # 🔁 Automatisches Label-Matching
    aligned_labels = align_labels_to_scores(labels, len(scores), WINDOW_SIZE)
    if len(aligned_labels) != len(scores):
        print(f"❌ Mismatch: Labels={len(aligned_labels)}, Scores={len(scores)}")
    else:
        print(f"✅ OK: Labels aligned ({len(scores)})")

    # 📉 Optional visualisieren
    if show_plot:
        plot_alignment(scores, aligned_labels, pred_events, fname=fname)

    precision, recall, f1 = affiliation_precision_recall_f1(aligned_labels, pred_events)
    print(f"📊 Event-based: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return f1

# 🔁 Hauptfunktion
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Name der Testdatei (optional)")
    parser.add_argument("--plot", action="store_true", help="Zeige Plots")
    args = parser.parse_args()

    if args.file:
        f1 = evaluate_file(args.file, show_plot=True)
        if f1 is not None:
            print(f"\n✅ F1 für {args.file}: {f1:.4f}")
    else:
        all_f1 = []
        to_plot = []
        for fname in sorted(os.listdir(TEST_DIR)):
            f1 = evaluate_file(fname, show_plot=False)
            if f1 is not None:
                all_f1.append(f1)
                if random.random() < 0.1:  # ca. 10% zufällig plotten
                    to_plot.append(fname)

        print(f"\n✅ Durchschnittlicher F1 über alle Dateien: {np.mean(all_f1):.4f}")

        if args.plot:
            print("\n📈 Plots für zufällige Beispiel-Dateien:")
            for fname in to_plot:
                evaluate_file(fname, show_plot=True)

if __name__ == "__main__":
    main()
