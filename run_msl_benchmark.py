import os
import argparse
import numpy as np
import torch
from teacher_model import TeacherNet
from student_model import StudentNet
from data_preparation import prepare_data
from evaluate_anomalies import compute_anomaly_scores
from metrics_affiliation import affiliation_precision_recall_f1
from scipy.ndimage import gaussian_filter1d, binary_opening, binary_closing
from sklearn.preprocessing import StandardScaler
import joblib
import random

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

def align_labels_to_scores(labels, num_scores, window_size):
    expected = len(labels) - window_size + 1
    if expected == num_scores:
        return labels[window_size - 1:]
    elif len(labels) == num_scores:
        return labels
    elif len(labels) > num_scores:
        return labels[len(labels) - num_scores:]
    else:
        raise ValueError(f"Labels ({len(labels)}) kürzer als Scores ({num_scores})")

def clean_predictions(pred_array, min_event_length=5, join_gap=8):
    pred_array = binary_closing(pred_array, structure=np.ones(join_gap)).astype(int)
    pred_array = binary_opening(pred_array, structure=np.ones(min_event_length)).astype(int)
    return pred_array

def evaluate_file(fname):
    test_path = os.path.join(TEST_DIR, fname)
    label_path = os.path.join(LABEL_DIR, fname)

    if not os.path.exists(label_path):
        return None

    data = np.load(test_path)
    dim = data.shape[1]
    
    if dim != 55:
        return None
    
    labels = np.load(label_path)
    model_path = f"student_model_{dim}.pt"
    num_patches = WINDOW_SIZE // PATCH_SIZE

    if not os.path.exists(model_path):
        return None

    X = prepare_data(data, WINDOW_SIZE, STEP_SIZE, patch_size=PATCH_SIZE, target_dim=dim, train=False)

    teacher = TeacherNet(
        input_dim=dim,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        patch_size=PATCH_SIZE,
        n_patches=num_patches
    ).to(DEVICE)

    teacher_weights_path = os.path.join(os.path.dirname(__file__), f"teacher_{dim}.pt")
    if not os.path.exists(teacher_weights_path):
        return None
    teacher.load_state_dict(torch.load(teacher_weights_path))
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

    scores = compute_anomaly_scores(teacher, student, X)
    scores = gaussian_filter1d(scores, sigma=1)
    threshold = np.percentile(scores, 85)
    pred_events = (scores >= threshold).astype(int)
    pred_events = clean_predictions(pred_events, min_event_length=5, join_gap=5)

    aligned_labels = align_labels_to_scores(labels, len(scores), WINDOW_SIZE)
    precision, recall, f1 = affiliation_precision_recall_f1(aligned_labels, pred_events)
    print(f"{fname:<15} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    return f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Name der Testdatei (optional)")
    args = parser.parse_args()

    if args.file:
        f1 = evaluate_file(args.file)
        if f1 is not None:
            print(f"\nF1 für {args.file}: {f1:.4f}")
    else:
        all_f1 = []
        for fname in sorted(os.listdir(TEST_DIR)):
            f1 = evaluate_file(fname)
            if f1 is not None:
                all_f1.append(f1)
        if all_f1:
            avg = np.mean(all_f1)
            print(f"\n✅ Durchschnittlicher F1 über {len(all_f1)} Dateien: {avg:.4f}")
        else:
            print("❌ Keine gültigen Dateien ausgewertet.")

if __name__ == "__main__":
    main()
