import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from teacher_model import TeacherNet
from student_model import StudentNet
from losses import total_loss
from augment_time_series import augment_batch
from data_preparation import prepare_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚙️ Einstellungen
DATA_DIR = "data/MSL/train"
WINDOW_SIZE = 24
PATCH_SIZE = 6
STEP_SIZE = 1
BATCH_SIZE = 32
EMBEDDING_DIM = 768
OUTPUT_DIM = 128
N_PROTOTYPES = 32
EPOCHS = 50
PATIENCE = 5
MIN_DELTA = 1e-4
LR = 1e-4
LR_TEACHER = 1e-4
LAMBDA_CE = 0.5  # Anteil des Contrastive Loss

def train_on_group(file_list, input_dim, model_name):
    torch.manual_seed(42)
    np.random.seed(42)

    num_patches = WINDOW_SIZE // PATCH_SIZE

    # === Daten laden ===
    all_loaders = []
    for fname in file_list:
        data = np.load(os.path.join(DATA_DIR, fname))
        windows = prepare_data(data, WINDOW_SIZE, STEP_SIZE, patch_size=PATCH_SIZE)
        dataset = TensorDataset(torch.tensor(windows, dtype=torch.float32))
        all_loaders.append(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True))

    # === Modelle initialisieren ===
    teacher = TeacherNet(
        input_dim=input_dim,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        patch_size=PATCH_SIZE,
        n_patches=num_patches
    ).to(DEVICE)

    student = StudentNet(
        input_dim=input_dim,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        n_prototypes=N_PROTOTYPES,
        patch_size=PATCH_SIZE,
        num_patches=num_patches
    ).to(DEVICE)

    # === Optimizer ===
    optimizer_student = torch.optim.Adam(student.parameters(), lr=LR)
    optimizer_teacher = torch.optim.Adam(filter(lambda p: p.requires_grad, teacher.parameters()), lr=LR_TEACHER)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0

        for loader in all_loaders:
            for (batch,) in loader:
                batch = batch.to(DEVICE)
                batch_aug = augment_batch(batch)

                # Set training mode
                student.train()
                teacher.train()

                # Forward passes
                z = student(batch)
                z_aug = student(batch_aug)
                c = teacher(batch)
                c_aug = teacher(batch_aug)

                # Loss berechnen
                loss, loss_kd, loss_ce = total_loss(z, c, z_aug, c_aug, c, c_aug, lambda_ce=LAMBDA_CE)

                # Zero grads
                optimizer_student.zero_grad()
                optimizer_teacher.zero_grad()

                # Backward + update
                loss.backward()
                optimizer_student.step()
                optimizer_teacher.step()

                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(file_list)
        print(f"📉 Epoch {epoch+1}: Loss={avg_loss:.4f} | λ_CE={LAMBDA_CE}")

        if avg_loss < best_loss - MIN_DELTA:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(student.state_dict(), model_name)
            torch.save(teacher.state_dict(), f"teacher_{input_dim}.pt")
            print("💾 Modelle gespeichert.")
        else:
            patience_counter += 1
            print(f"⏸️ Keine Verbesserung ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("🛑 Early stopping.")
            break

def main():
    files = sorted(os.listdir(DATA_DIR))
    grouped = {}
    for fname in files:
        data = np.load(os.path.join(DATA_DIR, fname))
        grouped.setdefault(data.shape[1], []).append(fname)

    target_dim = 55
    if target_dim in grouped:
        print(f"🔧 Training für Featureanzahl {target_dim} gestartet...")
        train_on_group(grouped[target_dim], target_dim, f"student_model_{target_dim}.pt")
    else:
        print(f"⚠️ Keine Daten mit Featureanzahl {target_dim} gefunden!")

if __name__ == "__main__":
    main()
