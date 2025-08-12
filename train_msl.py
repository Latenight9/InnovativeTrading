# train_msl.py
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from teacher_model import TeacherNet
from student_model import StudentNet
from losses import total_loss
from augment_time_series import augment_batch
from data_preparation import prepare_data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

# reproducibility
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Paths ===
DATA_DIR        = "data/MSL/train"
MODEL_DIR       = "checkpoints"
STUDENT_WEIGHTS = os.path.join(MODEL_DIR, "student_msl.pt")
TEACHER_INIT    = os.path.join(MODEL_DIR, "teacher_init.pt")  # fester Teacher-Zustand
os.makedirs(MODEL_DIR, exist_ok=True)

# === Fixed (paper) dims ===
WINDOW_SIZE   = 24
PATCH_SIZE    = 6
STEP_SIZE     = 1
EMBEDDING_DIM = 768   # Teacher hidden
OUTPUT_DIM    = 128   # Projektion (Teacher & Student)
N_PROTOTYPES  = 32    # Paper: 32

# === Hyperparameter via Umgebungsvariablen (mit Defaults) ===
def env_get(name, default, cast):
    v = os.getenv(name); return cast(v) if v is not None else default

# Training
EPOCHS        = env_get("HP_EPOCHS", 75, int)
PATIENCE      = 10
MIN_DELTA     = 1e-4

# Optimierung (nur Student)
LR            = env_get("HP_LR", 1e-4, float)   # z.B. 2e-4 via Env
WEIGHT_DECAY  = env_get("HP_WD", 1e-4, float)   # z.B. 5e-5 via Env
BATCH_SIZE    = env_get("HP_BS", 32, int)
LAMBDA_CE     = env_get("HP_LCE", 0.0, float)   # Teacher frozen => 0.0

# Anti-Kollaps
PROTO_LR_MULT       = env_get("HP_PLM", 0.5, float)   # z.B. 0.6 via Env
WARMUP_EPOCHS       = env_get("HP_WARMUP", 1, int)    # z.B. 2 via Env
KPP_MAX_SAMPLES     = env_get("HP_KPP", 100_000, int) # z.B. 120_000 / 150_000
REINIT_AFTER_EPOCH  = 1
REINIT_MIN_UNIQUE   = 24
USAGE_CHECK_SAMPLES = 2_000
TEACHER_MIN_VAR     = 0.25
TEACHER_TRIES       = 8

# Augment-Defaults (frühe Epochen sanfter steuerbar)
AUG_JIT0 = env_get("HP_JIT0", 0.03, float)
AUG_SCA0 = env_get("HP_SCA0", 0.08, float)
AUG_WAR0 = env_get("HP_WAR0", 0.20, float)

def aug_cfg(epoch: int):
    if epoch < 5:
        return dict(
            jitter_sigma=AUG_JIT0, scaling_sigma=AUG_SCA0, max_warp=AUG_WAR0,
            p_jitter=0.45, p_scaling=0.35, p_warp=0.20
        )
    else:
        return dict(
            jitter_sigma=0.05, scaling_sigma=0.10, max_warp=0.30,
            p_jitter=0.45, p_scaling=0.35, p_warp=0.20
        )

# -------------------------------------------------------------

def _load_array(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        z = np.load(path)
        key = list(z.keys())[0]
        arr = z[key]
    else:
        arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr

def _kpp_init_per_feature_embed(X_all: np.ndarray, student: StudentNet, max_samples=KPP_MAX_SAMPLES):
    """
    Diversifizierte Proto-Init je Feature per k-means++ im EMBEDDING-Raum.
    X_all: (W, N, P, D)
    """
    W, N, P, D = X_all.shape
    M = student.n_prototypes
    dev = student.prototypes.device

    # aktuelle Gewichte der Proto-Projektion
    Wt = student.proto_proj.weight.detach().cpu().numpy()
    bt = student.proto_proj.bias.detach().cpu().numpy() if student.proto_proj.bias is not None else None

    for d in range(D):
        Xd = X_all[:, :, :, d].reshape(W * N, P).astype(np.float32)
        if Xd.shape[0] > max_samples:
            idx = np.random.choice(Xd.shape[0], max_samples, replace=False)
            Xd = Xd[idx]

        # Embeddings im E-Raum
        Xe = Xd @ Wt.T
        if bt is not None: Xe += bt
        Xe /= (np.linalg.norm(Xe, axis=1, keepdims=True) + 1e-9)

        # farthest-point / k-means++ Seeding (cosine ~ 1 - dot)
        centers_idx = []
        j0 = np.random.randint(0, Xe.shape[0]); centers_idx.append(j0)
        dists = 1.0 - (Xe @ Xe[j0])
        for m in range(1, M):
            j = int(np.argmax(dists)); centers_idx.append(j)
            dists = np.minimum(dists, 1.0 - (Xe @ Xe[j]))

        # Zugehörige Patch-Vektoren (unit-norm) als Protos setzen
        chosen = Xd[centers_idx]
        chosen /= (np.linalg.norm(chosen, axis=1, keepdims=True) + 1e-9)
        with torch.no_grad():
            student.prototypes[d, :M].copy_(torch.tensor(chosen, device=dev))

@torch.no_grad()
def _proto_usage_embed(student: StudentNet, X_all: np.ndarray, k: int = 2_000):
    """Schnelle Nutzungsprüfung der Prototypen im EMBEDDING-Raum. Gibt (unique_used, top5_counts) zurück."""
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

    sim = torch.einsum("bnde,dme->bndm", x_tok, p_tok)   # (B,N,D,M)
    idx = sim.argmax(dim=-1).reshape(-1).detach().cpu().numpy()
    counts = np.bincount(idx, minlength=M)
    top5 = counts[np.argsort(counts)[::-1][:5]].tolist()
    unique_used = int((counts > 0).sum())
    return unique_used, top5

def _teacher_healthcheck_and_save(teacher: TeacherNet, sample_windows: np.ndarray, path: str,
                                  min_var: float = TEACHER_MIN_VAR, tries: int = TEACHER_TRIES):
    """
    Einmalige 'gesunde' Teacher-Init:
    - misst Varianz der Teacher-Outputs auf sample_windows
    - falls zu niedrig, re-initialisiert NUR die zwei kleinen Layer
    - speichert bei Erfolg nach 'path'
    """
    if os.path.exists(path):
        teacher.load_state_dict(torch.load(path, map_location=DEVICE))
        return

    import torch.nn.init as init
    ds = TensorDataset(torch.tensor(sample_windows, dtype=torch.float32))
    ld = DataLoader(ds, batch_size=128, shuffle=False)

    for _ in range(tries):
        # Re-Init nur der kleinen Schichten (falls vorhanden)
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
        c = torch.cat(outs, dim=0)  # (K, 128)
        v = c.var(dim=0).mean().item()
        if v >= min_var:
            torch.save(teacher.state_dict(), path)
            print(f"[teacher_init] OK (var={v:.3f}) saved to {path}")
            return

    raise RuntimeError(f"Teacher variance too low after {tries} re-inits. Check data/scale.")

# -------------------------------------------------------------

def train_on_group(file_list, input_dim):
    num_patches = WINDOW_SIZE // PATCH_SIZE

    # === Daten laden & zu Fenstern/Patches konvertieren ===
    all_windows = []
    for fname in file_list:
        arr = _load_array(os.path.join(DATA_DIR, fname))
        X = prepare_data(
            arr,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
            patch_size=PATCH_SIZE,
            target_dim=input_dim,
            train=True
        )
        if len(X) > 0:
            all_windows.append(X)
    if not all_windows:
        raise RuntimeError("Keine Trainingsfenster erzeugt – prüfe Daten & WINDOW/PATCH/STEP.")
    X_all = np.concatenate(all_windows, axis=0)  # (W, N, P, D)

    # === Teacher konstruieren & einmalig "gesund" fixieren ===
    teacher = TeacherNet(
        input_dim=input_dim, embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM,
        patch_size=PATCH_SIZE, n_patches=num_patches
    ).to(DEVICE)

    sample_windows = X_all[:min(2000, len(X_all))]  # Stichprobe
    _teacher_healthcheck_and_save(teacher, sample_windows, TEACHER_INIT,
                                  min_var=TEACHER_MIN_VAR, tries=TEACHER_TRIES)
    teacher.load_state_dict(torch.load(TEACHER_INIT, map_location=DEVICE))
    for p in teacher.parameters(): p.requires_grad = False
    teacher.eval()

    # === Student ===
    student = StudentNet(
        input_dim=input_dim, embedding_dim=128, output_dim=OUTPUT_DIM,
        n_prototypes=N_PROTOTYPES, patch_size=PATCH_SIZE, num_patches=num_patches,
        n_heads=8, intermediate_dim=64, dropout=0.1, n_layers=2
    ).to(DEVICE)

    # === k-means++-artige Proto-Init im EMBEDDING-Raum ===
    _kpp_init_per_feature_embed(X_all, student, max_samples=KPP_MAX_SAMPLES)

    # === Loader ===
    loader = DataLoader(
        TensorDataset(torch.tensor(X_all, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    # === Param-Gruppen: kleinere LR für proto_proj & prototypes (mit Warmup) ===
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
        # Warmup: Proto-Gruppe zunächst nicht updaten
        if epoch < WARMUP_EPOCHS:
            opt_s.param_groups[1]["lr"] = 0.0
        else:
            opt_s.param_groups[1]["lr"] = LR * PROTO_LR_MULT

        student.train()
        sum_loss = 0.0
        sum_kd   = 0.0
        sum_ce   = 0.0
        n_batches= 0

        for (batch,) in loader:
            batch = batch.to(DEVICE)
            cfg = aug_cfg(epoch)
            # Augmentierung gemäß cfg (keine doppelten Argumente)
            batch_aug = augment_batch(batch, **cfg).to(DEVICE)

            # Student-Forward
            z     = student(batch)
            z_aug = student(batch_aug)

            # Teacher-Targets (immer ohne Grad, Teacher ist eingefroren)
            with torch.no_grad():
                c     = teacher(batch)
                c_aug = teacher(batch_aug)

            # Verluste (KD + lambda * CE)
            loss, loss_kd, loss_ce = total_loss(z, c, z_aug, c_aug, lambda_ce=LAMBDA_CE)

            # Update (nur Student)
            opt_s.zero_grad()
            loss.backward()
            clip_grad_norm_(student.parameters(), 1.0)
            opt_s.step()

            sum_loss += loss.item()
            sum_kd   += loss_kd.item()
            sum_ce   += loss_ce.item()
            n_batches += 1

        avg_loss = sum_loss / n_batches
        avg_kd   = sum_kd   / n_batches
        avg_ce   = sum_ce   / n_batches
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} | kd={avg_kd:.4f} | ce={avg_ce:.4f} "
              f"| lr_base={opt_s.param_groups[0]['lr']:.2e} lr_proto={opt_s.param_groups[1]['lr']:.2e}")

        # Scheduler auf Gesamt-Loss
        scheduler.step(avg_loss)

        # Optional: nach Epoche 1 prüfen und ggf. re-initialisieren (einmalig)
        if epoch == REINIT_AFTER_EPOCH:
            uniq, top5 = _proto_usage_embed(student, X_all, k=USAGE_CHECK_SAMPLES)
            print(f"[usage] unique used={uniq}/{N_PROTOTYPES} | top5={top5}")
            if uniq < REINIT_MIN_UNIQUE:
                print("↺ Re-Init prototypes (embedding k-means++) due to low usage …")
                _kpp_init_per_feature_embed(X_all, student, max_samples=KPP_MAX_SAMPLES)

        # Checkpointing (best nach Loss) – NUR Student speichern
        if avg_loss < best_loss - MIN_DELTA:
            best_loss = avg_loss
            patience  = 0
            torch.save(student.state_dict(), STUDENT_WEIGHTS)
            print("💾 student checkpoint gespeichert.")
        else:
            patience += 1
            print(f"⏸️ keine Verbesserung ({patience}/{PATIENCE})")
            if patience >= PATIENCE:
                print("🛑 early stopping.")
                break

# -------------------------------------------------------------

def main():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".npy") or f.endswith(".npz"))
    grouped = {}
    for fname in files:
        arr = _load_array(os.path.join(DATA_DIR, fname))
        grouped.setdefault(arr.shape[1], []).append(fname)

    target_dim = 55
    if target_dim not in grouped:
        print(f"⚠️ Keine Daten mit Featureanzahl {target_dim} gefunden!")
        return

    print(f"🧰 Trainiere MSL-Gruppe mit D={target_dim} (Files: {len(grouped[target_dim])})")
    train_on_group(grouped[target_dim], input_dim=target_dim)

if __name__ == "__main__":
    main()
