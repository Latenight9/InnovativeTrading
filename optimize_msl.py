# optimize_msl.py
import os, subprocess, re, optuna

def run_train_and_bench(env):
    env_full = os.environ.copy(); env_full.update(env)
    subprocess.run(["python", "train_msl.py"], check=True, env=env_full)
    out = subprocess.run(["python", "run_msl_benchmark.py"], check=True,
                         capture_output=True, text=True)
    return out.stdout

def extract_val_pa_f1(text: str) -> float:
    m = re.search(r"\[VAL-PA\s*\].*F1=([0-9.]+)", text)
    return float(m.group(1)) if m else 0.0

def objective(trial: optuna.Trial):
    env = {
        # nur *deine* bestehenden HPs – keine Architekturänderung:
        "HP_LR":      f"{trial.suggest_float('lr', 1e-4, 3e-4, log=True):.8f}",
        "HP_WD":      f"{trial.suggest_float('wd', 1e-5, 2e-4, log=True):.8f}",
        "HP_BS":      str(trial.suggest_categorical('bs', [32, 64])),
        "HP_PLM":     f"{trial.suggest_float('plm', 0.5, 0.7):.6f}",
        "HP_WARMUP":  str(trial.suggest_int('warmup', 1, 3)),
        "HP_KPP":     str(trial.suggest_categorical('kpp', [100_000, 120_000, 150_000])),
        "HP_JIT0":    f"{trial.suggest_float('jit0', 0.02, 0.06):.6f}",
        "HP_SCA0":    f"{trial.suggest_float('sca0', 0.06, 0.12):.6f}",
        "HP_WAR0":    f"{trial.suggest_float('war0', 0.20, 0.40):.6f}",
        # kürzer trainieren, um Trials flott zu halten:
        "HP_EPOCHS":  "10",
    }
    text = run_train_and_bench(env)
    return extract_val_pa_f1(text)  # Ziel: VAL-PA-F1 maximieren

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="msl_pa_f1", storage="sqlite:///optuna_msl.db",
                                load_if_exists=True)
    study.optimize(objective, n_trials=20)  # z.B. 20 Versuche
    print("Best params:", study.best_trial.params, "Best VAL-PA-F1:", study.best_value)
