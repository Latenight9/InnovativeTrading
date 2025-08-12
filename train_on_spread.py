# train_on_spread.py

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from evaluate_anomalies import l2_scores
from Analysis import load_data, calculate_rolling_beta
from data_preparation import prepare_data
from teacher_model import TeacherNet
from student_model import StudentNet
from losses import total_loss
from augment_time_series import augment_batch
from config import DATA_MODE

# ============ Konfiguration ============
TICKERS = ["GLD", "USO"]
START_DATE = "2006-04-01"
END_DATE   = "2010-04-09"
INTERVAL   = "1d"
LOOKBACK_PERIOD = 20
SINCE_DAYS = 730  # für Krypto-Modus

# Anomaly LLM Parameter (Spread: D=1)
WINDOW_SIZE   = 120
PATCH_SIZE    = 10
STEP_SIZE     = 1
BATCH_SIZE    = 32
EMBEDDING_DIM = 768   # Teacher hidden
OUTPUT_DIM    = 128   # Projektion (Teacher & Student)
N_PROTOTYPES  = 32
EPOCHS        = 50
PATIENCE      = 8
MIN_DELTA     = 1e-4

# Optimierung (nur Student)
LR               = 1e-4
WEIGHT_DECAY     = 1e-4
PROTO_LR_MULT    = 0.5     
WARMUP_EPOCHS    = 1

# Anti-Kollaps / Checks
KPP_MAX_SAMPLES     = 100_000
REINIT_AFTER_EPOCH  = 1
REINIT_MIN_UNIQUE   = 24
USAGE_CHECK_SAMPLES = 2_000
TEACHER_MIN_VAR     = 0.25
TEACHER_TRIES       = 8

# Loss-Gewichte
LAMBDA_CE = 0.0  # CE nur als Metrik, nicht optimiert

# Threshold (unlabeled)
CHECKPOINT_DIR        = "checkpoints"
ENTRY_PERCENTILE_TRAIN= 97.0
SAVE_THRESHOLD        = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ Helpers ============

@torch.no_grad()
def _compute_l2_scores(teacher, student, X_np, batch_size=512):
    if len(X_np) == 0:
        return np.zeros(0, dtype=float)

    teacher.eval(); student.eval()

    ds = TensorDataset(torch.tensor(X_np, dtype=torch.float32))
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False)

    scores = []
    for (xb,) in ld:
        xb = xb.to(DEVICE)
        z = student(xb)
        c = teacher(xb)
        s = l2_scores(z, c)            # zentrale Definition (Eq. 8)
        scores.append(s.detach().cpu().numpy())

    return np.concatenate(scores, axis=0)


def _kpp_init_per_feature_embed(X_all: np.ndarray, student: StudentNet, max_samples=KPP_MAX_SAMPLES):
    W, N, P, D = X_all.shape
    M = student.n_prototypes
    dev = student.prototypes.device

    Wt = student.proto_proj.weight.detach().cpu().numpy()
    bt = student.proto_proj.bias.detach().cpu().numpy() if student.proto_proj.bias is not None else None

    for d in range(D):
        Xd = X_all[:, :, :, d].reshape(W * N, P).astype(np.float32)
        if Xd.shape[0] > max_samples:
            idx = np.random.choice(Xd.shape[0], max_samples, replace=False)
            Xd = Xd[idx]

        # Embeddings (E) normiert
        Xe = Xd @ Wt.T
        if bt is not None: Xe += bt
        Xe /= (np.linalg.norm(Xe, axis=1, keepdims=True) + 1e-9)

        # farthest-point / k-means++ Seeding
        centers_idx = []
        j0 = np.random.randint(0, Xe.shape[0]); centers_idx.append(j0)
        dists = 1.0 - (Xe @ Xe[j0])
        for m in range(1, M):
            j = int(np.argmax(dists)); centers_idx.append(j)
            dists = np.minimum(dists, 1.0 - (Xe @ Xe[j]))

        chosen = Xd[centers_idx]
        chosen /= (np.linalg.norm(chosen, axis=1, keepdims=True) + 1e-9)
        with torch.no_grad():
            student.prototypes[d, :M].copy_(torch.tensor(chosen, device=dev))


@torch.no_grad()
def _proto_usage_embed(student: StudentNet, X_all: np.ndarray, k: int = 2_000):
    W = min(k, X_all.shape[0])
    x = torch.tensor(X_all[:W], dtype=torch.float32, device=DEVICE)  # (W,N,P,D)
    B, N, P, D = x.shape
    x_bndp = x.permute(0,1,3,2).contiguous()  # (B,N,D,P)

    E = student.embedding_dim
    M = student.n_prototypes
    x_tok = torch.nn.functional.normalize(
        student.input_proj(x_bndp.view(B*N*D, P)).view(B, N, D, E), dim=-1)
    p_tok = torch.nn.functional.normalize(
        student.proto_proj(student.prototypes.view(D*M, P)).view(D, M, E), dim=-1)

    sim = torch.einsum("bnde,dme->bndm", x_tok, p_tok)
    idx = sim.argmax(dim=-1).reshape(-1).detach().cpu().numpy()
    counts = np.bincount(idx, minlength=M)
    top5 = counts[np.argsort(counts)[::-1][:5]].tolist()
    unique_used = int((counts > 0).sum())
    return unique_used, top5


def _teacher_healthcheck_and_save(teacher: TeacherNet, sample_windows: np.ndarray, path: str,
                                  min_var: float = TEACHER_MIN_VAR, tries: int = TEACHER_TRIES):
    if os.path.exists(path):
        teacher.load_state_dict(torch.load(path, map_location=DEVICE))
        return

    import torch.nn.init as init
    ds = TensorDataset(torch.tensor(sample_windows, dtype=torch.float32))
    ld = DataLoader(ds, batch_size=128, shuffle=False)

    for _ in range(tries):
        if hasattr(teacher, "linear_embedding") and hasattr(teacher.linear_embedding, "weight"):
            init.xavier_uniform_(teacher.linear_embedding.weight)
            if teacher.linear_embedding.bias is not None:
                torch.nn.init.zeros_(teacher.linear_embedding.bias)
        if hasattr(teacher, "flatten_proj") and hasattr(teacher.flatten_proj, "weight"):
            init.xavier_uniform_(teacher.flatten_proj.weight)
            if teacher.flatten_proj.bias is not None:
                torch.nn.init.zeros_(teacher.flatten_proj.bias)

        outs = []
        teacher.eval()
        with torch.no_grad():
            for (x,) in ld:
                x = x.to(DEVICE)
                outs.append(teacher(x))
        c = torch.cat(outs, dim=0)
        v = c.var(dim=0).mean().item()
        if v >= min_var:
            torch.save(teacher.state_dict(), path)
            print(f"[teacher_init] OK (var={v:.3f}) saved to {path}")
            return

    raise RuntimeError(f"Teacher variance too low after {tries} re-inits. Check data/scale.")


# ============ Training ============

def train_spread_model(tickers):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    pair_name = f"{tickers[0]}_{tickers[1]}"
    thr_path  = os.path.join(CHECKPOINT_DIR, f"threshold_{pair_name}.json")
    teacher_path = os.path.join(CHECKPOINT_DIR, f"teacher_model_{pair_name}.pt")
    student_path = os.path.join(CHECKPOINT_DIR, f"student_model_{pair_name}.pt")

    # --- Daten laden ---
    if DATA_MODE == "crypto":
        prices_df = load_data(tickers, interval=INTERVAL, since_days=SINCE_DAYS)
    else:
        prices_df = load_data(tickers, start_date=START_DATE, end_date=END_DATE, interval=INTERVAL)

    p1 = prices_df[tickers[0]]
    p2 = prices_df[tickers[1]]

    beta_series = calculate_rolling_beta(p1, p2, LOOKBACK_PERIOD)
    spread = (p1 - beta_series * p2).dropna()
    spread_np = spread.values.reshape(-1, 1)  # (T, 1)

    # --- Fenster + Patches (Instance-Norm inside prepare_data) ---
    X_all = prepare_data(
        spread_np, window_size=WINDOW_SIZE, step_size=STEP_SIZE,
        patch_size=PATCH_SIZE, train=True, target_dim=1
    )
    if len(X_all) == 0:
        raise RuntimeError("Keine Trainingsfenster erzeugt. Prüfe WINDOW_SIZE/PATCH_SIZE/STEP_SIZE.")

    loader = DataLoader(TensorDataset(torch.tensor(X_all, dtype=torch.float32)),
                        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    num_patches = WINDOW_SIZE // PATCH_SIZE

    # --- Teacher: gesund fixieren & einfrieren ---
    teacher = TeacherNet(
        input_dim=1, embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM,
        patch_size=PATCH_SIZE, n_patches=num_patches
    ).to(DEVICE)

    sample_windows = X_all[:min(2000, len(X_all))]
    _teacher_healthcheck_and_save(teacher, sample_windows, teacher_path,
                                  min_var=TEACHER_MIN_VAR, tries=TEACHER_TRIES)
    teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
    for p in teacher.parameters(): p.requires_grad = False
    teacher.eval()

    # --- Student ---
    student = StudentNet(
        input_dim=1, embedding_dim=128, output_dim=OUTPUT_DIM,
        n_prototypes=N_PROTOTYPES, patch_size=PATCH_SIZE, num_patches=num_patches,
        n_heads=8, intermediate_dim=64, dropout=0.1, n_layers=2
    ).to(DEVICE)

    # Proto-Init im Embedding-Raum
    _kpp_init_per_feature_embed(X_all, student, max_samples=KPP_MAX_SAMPLES)

    # Optimizer (getrennte Gruppen)
    proto_params = list(student.proto_proj.parameters()) + [student.prototypes]
    base_params  = [p for n,p in student.named_parameters()
                    if (not n.startswith("proto_proj")) and (n != "prototypes")]
    opt_s = torch.optim.Adam(
        [{"params": base_params,  "lr": LR,               "weight_decay": WEIGHT_DECAY},
         {"params": proto_params, "lr": LR*PROTO_LR_MULT, "weight_decay": WEIGHT_DECAY}],
    )
    scheduler = ReduceLROnPlateau(opt_s, mode="min", factor=0.5, patience=2, min_lr=2e-5)

    best_loss = float("inf")
    patience  = 0

    for epoch in range(EPOCHS):
        # Warmup: Proto-Gruppe in Ep. 0
        if epoch < WARMUP_EPOCHS:
            opt_s.param_groups[1]["lr"] = 0.0
        else:
            opt_s.param_groups[1]["lr"] = LR * PROTO_LR_MULT

        student.train()
        sum_loss = sum_kd = sum_ce = 0.0
        nb = 0

        for (batch,) in loader:
            batch = batch.to(DEVICE)
            # leicht entschärfte Augmentierung
            batch_aug = augment_batch(
                batch,
                jitter_sigma=0.05,
                scaling_sigma=0.10,
                max_warp=0.30,
                p_jitter=0.45, p_scaling=0.35, p_warp=0.20
            ).to(DEVICE)

            # Forward
            z     = student(batch)
            z_aug = student(batch_aug)
            with torch.no_grad():
                c     = teacher(batch)
                c_aug = teacher(batch_aug)

            loss, loss_kd, loss_ce = total_loss(z, c, z_aug, c_aug, lambda_ce=LAMBDA_CE)

            opt_s.zero_grad()
            loss.backward()
            clip_grad_norm_(student.parameters(), 1.0)
            opt_s.step()

            sum_loss += loss.item(); sum_kd += loss_kd.item(); sum_ce += loss_ce.item(); nb += 1

        avg_loss = sum_loss/nb; avg_kd = sum_kd/nb; avg_ce = sum_ce/nb
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} | kd={avg_kd:.4f} | ce={avg_ce:.4f} "
              f"| lr_base={opt_s.param_groups[0]['lr']:.2e} lr_proto={opt_s.param_groups[1]['lr']:.2e}")

        scheduler.step(avg_loss)

        # einmalige Nutzungsprüfung + ggf. Re-Init
        if epoch == REINIT_AFTER_EPOCH:
            uniq, top5 = _proto_usage_embed(student, X_all, k=USAGE_CHECK_SAMPLES)
            print(f"[usage] unique used={uniq}/{N_PROTOTYPES} | top5={top5}")
            if uniq < REINIT_MIN_UNIQUE:
                print("↺ Re-Init prototypes (embedding k-means++) due to low usage …")
                _kpp_init_per_feature_embed(X_all, student, max_samples=KPP_MAX_SAMPLES)

        # Checkpoint (best loss)
        if avg_loss < best_loss - MIN_DELTA:
            best_loss = avg_loss
            patience = 0
            torch.save(student.state_dict(), student_path)
            print("💾 student checkpoint gespeichert.")
        else:
            patience += 1
            print(f"⏸️ keine Verbesserung ({patience}/{PATIENCE})")
            if patience >= PATIENCE:
                print("🛑 early stopping."); break

    # --- Threshold aus Trainingsscores ---
    if SAVE_THRESHOLD:
        teacher.eval(); student.eval()
        scores = _compute_l2_scores(teacher, student, X_all, batch_size=512)
        entry_score = float(np.percentile(scores, ENTRY_PERCENTILE_TRAIN))
        meta = {
            "mode": "l2_train_percentile",
            "entry_percentile": ENTRY_PERCENTILE_TRAIN,
            "entry_score": entry_score,
            "n_scores": int(len(scores)),
            "window_size": WINDOW_SIZE,
            "patch_size": PATCH_SIZE,
            "step_size": STEP_SIZE,
            "pair": pair_name,
        }
        with open(thr_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"✅ Threshold gespeichert → {thr_path} (Score @ {ENTRY_PERCENTILE_TRAIN}th = {entry_score:.4f})")


if __name__ == "__main__":
    train_spread_model(TICKERS)
