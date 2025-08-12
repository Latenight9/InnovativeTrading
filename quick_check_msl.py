# quick_checks_msl.py
import os, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from student_model import StudentNet
from teacher_model import TeacherNet
from data_preparation import prepare_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Pfade / Settings (ANPASSEN falls abweichend trainiert) ===
TEST_DIR   = "data/MSL/test"
LABEL_DIR  = "data/MSL/test_label"
CKPT_DIR   = "checkpoints"
STUDENT_W  = os.path.join(CKPT_DIR, "student_msl.pt")
TEACHER_W  = os.path.join(CKPT_DIR, "teacher_init.pt")

TARGET_DIM   = 55            # wir prüfen die 55D-Gruppe
WINDOW_SIZE  = 24            # MUSS zu deinem Training passen
PATCH_SIZE   = 6             # MUSS zu deinem Training passen
STEP_SIZE    = 1
EMB_TEACHER  = 768
EMB_OUT      = 128
N_PROTOTYPES = 32            # MUSS zu deinem Training passen
MAX_WINDOWS  = 2000          # genug für schnelle Aussage

def load_any(path):
    if path.endswith(".npz"):
        z = np.load(path); key = list(z.keys())[0]; arr = z[key]
    else:
        arr = np.load(path)
    if arr.ndim == 1: arr = arr[:, None]
    return arr

def point_to_window_labels(y, window_size, step):
    n_win = (len(y) - window_size) // step + 1
    if n_win <= 0: return np.zeros(0, dtype=np.int32)
    lab = np.zeros(n_win, dtype=np.int32)
    for i in range(n_win):
        s = i*step; e = s+window_size
        lab[i] = 1 if y[s:e].any() else 0
    return lab

def build_quick_val(input_dim, max_windows=2000):
    Xs, Ys = [], []
    for fn in sorted(os.listdir(TEST_DIR)):
        if not (fn.endswith(".npy") or fn.endswith(".npz")): continue
        arr = load_any(os.path.join(TEST_DIR, fn))
        if arr.shape[1] != input_dim: continue
        stem = os.path.splitext(fn)[0]
        labp = os.path.join(LABEL_DIR, stem + ".npy")
        if not os.path.exists(labp): continue
        y = np.load(labp).astype(np.int32)
        Xw = prepare_data(arr, WINDOW_SIZE, STEP_SIZE, PATCH_SIZE, target_dim=input_dim, train=False)
        yw = point_to_window_labels(y, WINDOW_SIZE, STEP_SIZE)
        if len(Xw) and len(yw):
            Xs.append(Xw); Ys.append(yw)
        if sum(len(x) for x in Xs) >= max_windows:
            break
    if not Xs:
        return np.zeros((0, WINDOW_SIZE//PATCH_SIZE, PATCH_SIZE, input_dim), dtype=np.float32), np.zeros(0, dtype=np.int32)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(Ys, axis=0)
    if len(X) > max_windows:
        X = X[:max_windows]; y = y[:max_windows]
    return X, y

@torch.no_grad()
def proto_hist(student, Xw, batch=256):
    student.eval()
    E = student.embedding_dim
    M = student.n_prototypes
    counts = torch.zeros(M, dtype=torch.long, device=DEVICE)

    ds = DataLoader(TensorDataset(torch.tensor(Xw, dtype=torch.float32)), batch_size=batch)
    for (xb,) in ds:
        # xb: (B, N, P, D) -> (B, N, D, P)
        xb = xb.to(DEVICE).permute(0, 1, 3, 2).contiguous()
        B, N, D, P = xb.shape

        # Input-Patches -> Tokens: (B,N,D,E)
        x_tok = student.input_proj(xb.reshape(B * N * D, P)).view(B, N, D, E)
        x_tok = torch.nn.functional.normalize(x_tok, dim=-1)

        # Window-Aggregation (AMAX) -> (B,D,E)
        x_win = torch.nn.functional.normalize(x_tok.amax(dim=1), dim=-1)

        # Proto-Tokens: (D,M,E)
        p_tok = student.proto_proj(student.prototypes.view(D * M, P)).view(D, M, E)
        p_tok = torch.nn.functional.normalize(p_tok, dim=-1)

        # Ähnlichkeit & harte Auswahl je (B,D)
        sim = torch.einsum("bde,dme->bdm", x_win, p_tok)  # (B,D,M)
        idx = sim.argmax(dim=-1).reshape(-1)              # (B*D,)

        counts += torch.bincount(idx, minlength=M).to(counts.dtype)

    return counts.detach().cpu().numpy()



@torch.no_grad()
def teacher_stats(teacher, Xw, batch=256):
    outs = []
    ds = DataLoader(TensorDataset(torch.tensor(Xw, dtype=torch.float32)), batch_size=batch)
    for (xb,) in ds:
        outs.append(teacher(xb.to(DEVICE)).cpu().numpy())
    Z = np.concatenate(outs, axis=0)                         # (N,128)
    var  = float(Z.var())
    pair = float(np.mean(np.linalg.norm(Z[:,None,:]-Z[None,:,:], axis=-1)))
    return var, pair

@torch.no_grad()
def separability(teacher, student, Xw, y_win, batch=256):
    scores = []
    ds = DataLoader(TensorDataset(torch.tensor(Xw, dtype=torch.float32)), batch_size=batch)
    for (xb,) in ds:
        xb = xb.to(DEVICE)
        c = teacher(xb); z = student(xb)
        s = ((z - c)**2).sum(dim=1).cpu().numpy()
        scores.append(s)
    scores = np.concatenate(scores)
    y = y_win[:len(scores)]
    pos = scores[y==1]; neg = scores[y==0]
    if len(pos)==0 or len(neg)==0:
        return 0.0, float(scores.mean()), float(scores.std())
    mu1, mu0 = pos.mean(), neg.mean()
    s1, s0 = pos.std()+1e-12, neg.std()+1e-12
    d = (mu1 - mu0) / np.sqrt(0.5*(s1*s1 + s0*s0))
    return float(d), float(scores.mean()), float(scores.std())

def main():
    # Daten vorbereiten
    Xq, yq = build_quick_val(TARGET_DIM, MAX_WINDOWS)
    if len(Xq) == 0:
        print("Keine Quick-VAL-Fenster gefunden – prüfe Pfade/Labels.")
        return
    print(f"Quick-VAL windows: {len(Xq)} | dim={TARGET_DIM} | N_patches={WINDOW_SIZE//PATCH_SIZE}")

    # Modelle konstruieren (müssen zu Checkpoints passen!)
    n_patches = WINDOW_SIZE // PATCH_SIZE
    # korrekt
    teacher = TeacherNet(
        input_dim=TARGET_DIM,
        patch_size=PATCH_SIZE,
        n_patches=n_patches,
        embedding_dim=EMB_TEACHER,
        output_dim=EMB_OUT,
        ).to(DEVICE)

    student = StudentNet(TARGET_DIM, embedding_dim=128, output_dim=EMB_OUT,
                         n_prototypes=N_PROTOTYPES, patch_size=PATCH_SIZE, num_patches=n_patches,
                         n_heads=8, intermediate_dim=64, dropout=0.1, n_layers=2).to(DEVICE)
    # Weights laden
    assert os.path.exists(TEACHER_W) and os.path.exists(STUDENT_W), "Checkpoints fehlen."
    teacher.load_state_dict(torch.load(TEACHER_W, map_location=DEVICE))
    student.load_state_dict(torch.load(STUDENT_W, map_location=DEVICE))
    teacher.eval(); student.eval()

    # Checks
    h = proto_hist(student, Xq)
    var, pair = teacher_stats(teacher, Xq)
    d, m, s = separability(teacher, student, Xq, yq)

    # Ausgabe
    print(f"[Proto] total={h.sum()} | unique used={np.count_nonzero(h)} | top5={np.sort(h)[-5:][::-1].tolist()}")
    print(f"[Teacher] var={var:.4f} | pairwise_mean={pair:.3f}")
    print(f"[Sep] Cohen_d={d:.3f} | score_mean={m:.3f} | score_std={s:.3f}")

if __name__ == "__main__":
    main()
