import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple

from teacher_model import TeacherNet
from student_model import StudentNet
from data_preparation import prepare_data
from evaluate_anomalies import l2_scores
from metrics_affiliation import affiliation_metrics  # event-wise affiliation P/R/F1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Verzeichnisse ===
TRAIN_DIR = "data/MSL/train"
TEST_DIR  = "data/MSL/test"
LABEL_DIR = "data/MSL/test_label"  # erzeugt von generate_labels.py
MODEL_DIR = "checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Hyperparameter ===
WINDOW_SIZE   = 24
PATCH_SIZE    = 6
STEP_SIZE     = 1
BATCH_SIZE    = 128
EMBEDDING_DIM = 768
OUTPUT_DIM    = 128
N_PROTOTYPES  = 32
TARGET_DIM    = 55   # wir evaluieren die 55D-Gruppe

# === Validation/Test-Split (dateiweise) ===
VAL_SPLIT = 0.4
SEED      = 42

STUDENT_WEIGHTS = os.path.join(MODEL_DIR, "student_msl.pt")
TEACHER_WEIGHTS = os.path.join(MODEL_DIR, "teacher_init.pt")


def load_train_concat(path, required_dim=None) -> np.ndarray:
    arrs, kept, skipped = [], [], []
    for fn in sorted(os.listdir(path)):
        p = os.path.join(path, fn)
        if fn.endswith(".npz"):
            data = np.load(p)
            key  = list(data.keys())[0]
            arr  = data[key]
        elif fn.endswith(".npy"):
            arr = np.load(p)
        else:
            continue
        if arr.ndim == 1: arr = arr[:, None]
        D = arr.shape[1]
        if (required_dim is None) or (D == required_dim):
            arrs.append(arr); kept.append((fn, D))
        else:
            skipped.append((fn, D))
    if skipped:
        dims = sorted(set(d for _, d in skipped))
        print(f"⚠️  Train: Überspringe {len(skipped)} Datei(en) wegen anderer D: {dims}")
    if not arrs:
        raise ValueError(f"Train: Keine passenden Dateien in {path}. required_dim={required_dim}")
    out = np.concatenate(arrs, axis=0)
    print(f"✅ Train: {len(kept)} Dateien, shape concat={out.shape}")
    return out


def load_test_items(test_dir, label_dir, input_dim) -> Tuple[List[Tuple[str, np.ndarray, np.ndarray]], List[Tuple[str,int]]]:
    items, skipped = [], []
    for fn in sorted(os.listdir(test_dir)):
        if not (fn.endswith(".npy") or fn.endswith(".npz")): continue
        p = os.path.join(test_dir, fn)
        if fn.endswith(".npz"):
            data = np.load(p)
            key  = list(data.keys())[0]
            arr  = data[key]
        else:
            arr = np.load(p)
        if arr.ndim == 1: arr = arr[:, None]
        D = arr.shape[1]
        if D != input_dim:
            skipped.append((fn, D)); continue
        stem = os.path.splitext(fn)[0]
        lab_path = os.path.join(label_dir, f"{stem}.npy")
        y_true = np.load(lab_path).astype(np.int32) if os.path.exists(lab_path) else None
        items.append((stem, arr, y_true))
    return items, skipped


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_adj = y_pred.copy()
    i = 0
    n = len(y_true)
    while i < n:
        if y_true[i] == 1:
            j = i
            while j < n and y_true[j] == 1:
                j += 1
            # GT-Event [i, j)
            if y_pred[i:j].any():
                y_adj[i:j] = 1
            i = j
        else:
            i += 1
    return y_adj


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    return prec, rec, f1


@torch.no_grad()
def raw_scores_for_windows(teacher, student, X_windows) -> np.ndarray:
    ds = TensorDataset(torch.tensor(X_windows, dtype=torch.float32))
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    vals = []
    teacher.eval(); student.eval()
    for (x,) in ld:
        x = x.to(DEVICE)
        t = teacher(x)
        s = student(x)
        d = l2_scores(s, t)  # Eq.(8)
        vals.extend(d.detach().cpu().numpy())
    return np.asarray(vals, dtype=float)


def window_preds_to_point_preds(window_preds, series_len, window_size, step_size=1):
    y_pred = np.zeros(series_len, dtype=np.int32)
    for i, p in enumerate(window_preds):
        if p:
            start = i * step_size
            end   = min(start + window_size, series_len)
            y_pred[start:end] = 1
    return y_pred


def split_items(items, val_split=0.4, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_val = int(np.floor(val_split * len(items)))
    val_idx = set(idx[:n_val].tolist())
    val_items, test_items = [], []
    for i, it in enumerate(items):
        (val_items if i in val_idx else test_items).append(it)
    return val_items, test_items


def main():
    print(f"Using TARGET_DIM={TARGET_DIM}")

    # === 1) Train laden + in Windows patchen (nur für Range/Info) ===
    Xtr_raw = load_train_concat(TRAIN_DIR, required_dim=TARGET_DIM)
    input_dim   = Xtr_raw.shape[1]
    num_patches = WINDOW_SIZE // PATCH_SIZE
    Xtr = prepare_data(Xtr_raw, WINDOW_SIZE, STEP_SIZE, PATCH_SIZE, target_dim=input_dim, train=True)
    print(f"✅ Train windows: {Xtr.shape[0]}")

    # === 2) Modelle laden ===
    assert os.path.exists(TEACHER_WEIGHTS) and os.path.exists(STUDENT_WEIGHTS), \
        "Checkpoints fehlen: erst 'train_msl.py' ausführen."
    teacher = TeacherNet(input_dim=input_dim, embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM,
                         patch_size=PATCH_SIZE, n_patches=num_patches).to(DEVICE)
    student = StudentNet(input_dim=input_dim, embedding_dim=128, output_dim=OUTPUT_DIM,
                         n_prototypes=N_PROTOTYPES, patch_size=PATCH_SIZE, num_patches=num_patches).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location=DEVICE))
    student.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=DEVICE))
    teacher.eval(); student.eval()

    # (nur Info) grobe Score-Range auf Train
    scores_tr_raw = raw_scores_for_windows(teacher, student, Xtr)
    lo, hi = float(np.percentile(scores_tr_raw, 5)), float(np.percentile(scores_tr_raw, 95))
    print(f"ℹ️  Train score range (5..95 pct): [{lo:.4f}, {hi:.4f}]")

    # === 3) Test-Items laden (D=55), dann dateiweise in VAL/TEST splitten ===
    test_items_all, skipped = load_test_items(TEST_DIR, LABEL_DIR, input_dim)
    if skipped:
        dims = sorted(set(d for _, d in skipped))
        print(f"⚠️  Test: Überspringe {len(skipped)} Datei(en) wegen anderer D: {dims}")
    test_items_all = [(stem, arr, y) for (stem, arr, y) in test_items_all if y is not None]
    print(f"✅ Test: {len(test_items_all)} Datei(en) mit D={input_dim} + Labels gefunden")
    if len(test_items_all) < 3:
        raise SystemExit("Zu wenige gelabelte Testdateien gefunden für einen sinnvollen Val/Test-Split.")

    val_items, test_items = split_items(test_items_all, VAL_SPLIT, SEED)
    print(f"🔧 Split: VAL={len(val_items)} Dateien | TEST={len(test_items)} Dateien")

    # === 4) Caches bauen (rohe Scores) für VAL & TEST ===
    def build_cache(items):
        cache = []  # (stem, sc_raw (Nw,), T, y_true)
        for stem, arr, y_true in items:
            Xw = prepare_data(arr, WINDOW_SIZE, STEP_SIZE, PATCH_SIZE, target_dim=input_dim, train=False)
            sc_raw = raw_scores_for_windows(teacher, student, Xw)
            cache.append((stem, sc_raw, arr.shape[0], y_true))
        return cache

    val_cache  = build_cache(val_items)
    test_cache = build_cache(test_items)
    print(f"✅ Val cache:  {len(val_cache)} Dateien")
    print(f"✅ Test cache: {len(test_cache)} Dateien")

    # === 5) Threshold-Sweep AUF VALIDATION (labelbasiert) ===
    val_all = np.concatenate([c[1] for c in val_cache], axis=0)
    lo_cand = float(np.percentile(val_all, 0.5))
    hi_cand = float(np.percentile(val_all, 99.5))
    cand = np.linspace(lo_cand, hi_cand, 1024)

    def eval_affiliation_at_thr(cache, thr):
        agg_p = agg_r = agg_f = 0.0
        cnt = 0
        for stem, sc_raw, T, y_true in cache:
            preds_win = (sc_raw >= thr).astype(np.int32)
            preds_ts  = window_preds_to_point_preds(preds_win, T, WINDOW_SIZE, STEP_SIZE)
            m = min(len(y_true), len(preds_ts))
            p, r, f = affiliation_metrics(y_true[:m], preds_ts[:m])
            agg_p += p; agg_r += r; agg_f += f; cnt += 1
        return (agg_f/cnt, agg_p/cnt, agg_r/cnt, cnt)

    def eval_pointadjust_at_thr(cache, thr):
        agg_p = agg_r = agg_f = 0.0
        cnt = 0
        for stem, sc_raw, T, y_true in cache:
            preds_win = (sc_raw >= thr).astype(np.int32)
            preds_ts  = window_preds_to_point_preds(preds_win, T, WINDOW_SIZE, STEP_SIZE)
            m = min(len(y_true), len(preds_ts))
            y_pa = point_adjust(y_true[:m], preds_ts[:m])
            p, r, f = point_metrics(y_true[:m], y_pa)
            agg_p += p; agg_r += r; agg_f += f; cnt += 1
        return (agg_f/cnt, agg_p/cnt, agg_r/cnt, cnt)

    # ---- VAL: Affiliation
    best_aff = (-1.0, 0.0, 0.0, 0, hi_cand)  # (F1,P,R,cnt,thr)
    for thr_c in cand:
        f1_c, p_c, r_c, n_c = eval_affiliation_at_thr(val_cache, float(thr_c))
        if f1_c > best_aff[0]:
            best_aff = (f1_c, p_c, r_c, n_c, float(thr_c))
    print(f"[VAL-AFF] Best thr={best_aff[4]:.4f} | P={best_aff[1]:.4f} R={best_aff[2]:.4f} F1={best_aff[0]:.4f} over {best_aff[3]} files")
    thr_aff = best_aff[4]

    # ---- VAL: Point-Adjusted
    best_pa = (-1.0, 0.0, 0.0, 0, hi_cand)
    for thr_c in cand:
        f1_c, p_c, r_c, n_c = eval_pointadjust_at_thr(val_cache, float(thr_c))
        if f1_c > best_pa[0]:
            best_pa = (f1_c, p_c, r_c, n_c, float(thr_c))
    print(f"[VAL-PA ] Best thr={best_pa[4]:.4f} | P={best_pa[1]:.4f} R={best_pa[2]:.4f} F1={best_pa[0]:.4f} over {best_pa[3]} files")
    thr_pa = best_pa[4]

    # === 6a) Finale Bewertung TEST (Affiliation) mit fixem thr_aff ===
    agg_p = agg_r = agg_f = 0.0
    cnt = 0
    for stem, sc_raw, T, y_true in test_cache:
        preds_win = (sc_raw >= thr_aff).astype(np.int32)
        preds_ts  = window_preds_to_point_preds(preds_win, T, WINDOW_SIZE, STEP_SIZE)
        m = min(len(y_true), len(preds_ts))
        p, r, f = affiliation_metrics(y_true[:m], preds_ts[:m])
        agg_p += p; agg_r += r; agg_f += f; cnt += 1
        print(f"[TEST-AFF {stem}] P={p:.4f} R={r:.4f} F1={f:.4f}")

    if cnt > 0:
        print(f"\n== Held-out TEST (Affiliation, over {cnt} files) ==")
        print(f"Precision={agg_p/cnt:.4f} | Recall={agg_r/cnt:.4f} | F1={agg_f/cnt:.4f}")

    # === 6b) Finale Bewertung TEST (Point-Adjusted) mit fixem thr_pa ===
    agg_p = agg_r = agg_f = 0.0
    cnt = 0
    for stem, sc_raw, T, y_true in test_cache:
        preds_win = (sc_raw >= thr_pa).astype(np.int32)
        preds_ts  = window_preds_to_point_preds(preds_win, T, WINDOW_SIZE, STEP_SIZE)
        m = min(len(y_true), len(preds_ts))
        y_pa = point_adjust(y_true[:m], preds_ts[:m])
        p, r, f = point_metrics(y_true[:m], y_pa)
        agg_p += p; agg_r += r; agg_f += f; cnt += 1
        print(f"[TEST-PA {stem}] P={p:.4f} R={r:.4f} F1={f:.4f}")

    if cnt > 0:
        print(f"\n== Held-out TEST (Point-Adjusted, over {cnt} files) ==")
        print(f"Precision={agg_p/cnt:.4f} | Recall={agg_r/cnt:.4f} | F1={agg_f/cnt:.4f}")


if __name__ == "__main__":
    main()
