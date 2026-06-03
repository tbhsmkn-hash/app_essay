"""
tuning_eval.py
--------------
Hyperparameter tuning dan evaluasi model Essay Scoring Multimodal.

Mencakup:
  1. Training loop lengkap dengan early stopping
  2. Grid search / Optuna untuk hyperparameter tuning
  3. Evaluasi lengkap: QWK, RMSE, Pearson r, confusion matrix
  4. Visualisasi learning curve dan scatter plot prediksi vs aktual
  5. Integrasi Weights & Biases (opsional)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1. METRIC FUNCTIONS
# ─────────────────────────────────────────────

def compute_qwk(
    preds: np.ndarray,
    labels: np.ndarray,
    score_min: float,
    score_max: float,
) -> float:
    """
    Quadratic Weighted Kappa — metrik utama ASAP competition.
    Preds dan labels dalam skala normalized [0,1], dikonversi dulu ke integer.
    """
    # Denormalize → bulatkan ke integer
    scale   = score_max - score_min
    p_int   = np.round(preds * scale + score_min).astype(int)
    l_int   = np.round(labels * scale + score_min).astype(int)
    p_int   = np.clip(p_int, int(score_min), int(score_max))
    l_int   = np.clip(l_int, int(score_min), int(score_max))
    return cohen_kappa_score(l_int, p_int, weights="quadratic")


def compute_rmse(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(labels, preds)))


def compute_pearson(preds: np.ndarray, labels: np.ndarray) -> float:
    r, _ = pearsonr(preds, labels)
    return float(r)


def evaluate_all_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    score_min: float,
    score_max: float,
) -> dict:
    """Hitung semua metrik sekaligus."""
    return {
        "qwk"    : compute_qwk(preds, labels, score_min, score_max),
        "rmse"   : compute_rmse(preds, labels),
        "pearson": compute_pearson(preds, labels),
    }


# ─────────────────────────────────────────────
# 2. TRAINING LOOP
# ─────────────────────────────────────────────

class EarlyStopping:
    """Stop training jika val_qwk tidak meningkat selama `patience` epoch."""
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = -np.inf
        self.should_stop = False

    def __call__(self, val_qwk: float) -> bool:
        if val_qwk > self.best_score + self.min_delta:
            self.best_score = val_qwk
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  Train", leave=False):
        image          = batch["image"].to(DEVICE)
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        scores         = batch["score"].to(DEVICE)

        optimizer.zero_grad()
        preds = model(image, input_ids, attention_mask)
        loss  = criterion(preds, scores)
        loss.backward()
        # Gradient clipping untuk stabilitas
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    score_min: float,
    score_max: float,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  Eval ", leave=False):
        image          = batch["image"].to(DEVICE)
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        scores         = batch["score"].to(DEVICE)

        preds = model(image, input_ids, attention_mask)
        loss  = criterion(preds, scores)
        total_loss += loss.item()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(scores.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics    = evaluate_all_metrics(all_preds, all_labels, score_min, score_max)

    return total_loss / len(loader), metrics


def train_model(
    model: nn.Module,
    dataloaders: dict,
    config: dict,
    run_name: str = "baseline",
    use_wandb: bool = False,
) -> dict:
    """
    Full training loop dengan early stopping, scheduler, dan checkpoint.

    Args:
        model       : nn.Module (EssayScoringModel)
        dataloaders : dict dengan kunci 'train', 'val', 'test', 'config'
        config      : hyperparameter dict
        run_name    : nama eksperimen
        use_wandb   : aktifkan logging ke W&B

    Returns:
        dict hasil training: best metrics + history
    """
    score_min = dataloaders["config"]["score_min"]
    score_max = dataloaders["config"]["score_max"]

    if use_wandb:
        import wandb
        wandb.init(project="essay-scoring-multimodal", name=run_name, config=config)

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.MSELoss()
    stopper   = EarlyStopping(patience=config.get("patience", 7))
    
    # agar Streamlit tahu cara melakukan denormalisasi skor prediksi
    # simpan dalam bentuk jason sederhana
    meta_config = {
    "score_min": dataloaders["config"]["score_min"],
    "score_max": dataloaders["config"]["score_max"]
        
    }
    with open(OUTPUT_DIR / "meta_config.json", "w") as f:
        json.dump(meta_config, f)

    best_val_qwk   = -np.inf
    best_ckpt_path = OUTPUT_DIR / "checkpoints" / f"{run_name}_best.pt"
    history        = {"train_loss": [], "val_loss": [], "val_qwk": [], "val_rmse": []}

    print(f"\n{'='*55}")
    print(f"  Mulai training: {run_name}")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {config['epochs']}")
    print(f"  LR     : {config['lr']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"{'='*55}")

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(
            model, dataloaders["train"], optimizer, criterion
        )
        val_loss, val_metrics = evaluate_epoch(
            model, dataloaders["val"], criterion, score_min, score_max
        )
        scheduler.step()

        val_qwk  = val_metrics["qwk"]
        val_rmse = val_metrics["rmse"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_qwk"].append(val_qwk)
        history["val_rmse"].append(val_rmse)

        print(
            f"  Epoch {epoch:03d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val QWK: {val_qwk:.4f} | "
            f"Val RMSE: {val_rmse:.4f}"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "train_loss": train_loss, "val_loss": val_loss,
                "val_qwk": val_qwk, "val_rmse": val_rmse, "epoch": epoch
            })

        # Simpan checkpoint terbaik
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_qwk"    : val_qwk,
                "config"     : config,
            }, best_ckpt_path)

        if stopper(val_qwk):
            print(f"\n  Early stopping di epoch {epoch}.")
            break

    # Evaluasi final pada test set
    print("\n  Memuat checkpoint terbaik untuk evaluasi test...")
    ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    _, test_metrics = evaluate_epoch(
        model, dataloaders["test"], criterion, score_min, score_max
    )
    print(f"\n  TEST RESULTS ({run_name})")
    print(f"  QWK    : {test_metrics['qwk']:.4f}")
    print(f"  RMSE   : {test_metrics['rmse']:.4f}")
    print(f"  Pearson: {test_metrics['pearson']:.4f}")

    if use_wandb:
        import wandb
        wandb.log({"test_qwk": test_metrics["qwk"],
                   "test_rmse": test_metrics["rmse"]})
        wandb.finish()

    return {
        "run_name"    : run_name,
        "config"      : config,
        "history"     : history,
        "best_val_qwk": best_val_qwk,
        "test_metrics": test_metrics,
    }


# ─────────────────────────────────────────────
# 3. GRID SEARCH HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

HYPERPARAM_GRID = {
    "lr"          : [1e-5, 5e-5, 2e-4],
    "dropout"     : [0.2, 0.3, 0.5],
    "fusion_dim"  : [128, 256, 512],
    "weight_decay": [1e-2],
    "epochs"      : [50],
    "patience"    : [7],
}


def grid_search(
    model_class,
    dataloaders: dict,
    grid: dict = HYPERPARAM_GRID,
    max_trials: int = 9,
) -> pd.DataFrame:
    """
    Grid search sederhana — evaluasi kombinasi hyperparameter.

    Untuk proyek skripsi, cukup variasikan lr × dropout × fusion_dim.
    Catat semua hasil ke DataFrame untuk tabel artikel.
    """
    from itertools import product

    keys   = list(grid.keys())
    values = list(grid.values())
    combos = list(product(*values))[:max_trials]

    results = []
    print(f"\nGrid search: {len(combos)} kombinasi\n")

    for i, combo in enumerate(combos, 1):
        config   = dict(zip(keys, combo))
        run_name = f"trial_{i:02d}_lr{config['lr']}_do{config['dropout']}_fd{config['fusion_dim']}"

        print(f"[{i}/{len(combos)}] {run_name}")

        # Inisialisasi ulang model untuk setiap trial
        model = model_class(
            fusion_dim = config["fusion_dim"],
            dropout    = config["dropout"],
        )
        result = train_model(model, dataloaders, config, run_name=run_name)

        results.append({
            "run"        : run_name,
            "lr"         : config["lr"],
            "dropout"    : config["dropout"],
            "fusion_dim" : config["fusion_dim"],
            "val_qwk"    : result["best_val_qwk"],
            "test_qwk"   : result["test_metrics"]["qwk"],
            "test_rmse"  : result["test_metrics"]["rmse"],
            "test_pearson": result["test_metrics"]["pearson"],
        })

    df = pd.DataFrame(results).sort_values("val_qwk", ascending=False)
    df.to_csv(OUTPUT_DIR / "hyperparameter_results.csv", index=False)
    print("\nHasil grid search:")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────
# 4. OPTUNA — BAYESIAN OPTIMIZATION (opsional)
# ─────────────────────────────────────────────

def optuna_tune(model_class, dataloaders: dict, n_trials: int = 20):
    """
    Hyperparameter tuning dengan Optuna (lebih efisien dari grid search).
    Install: pip install optuna
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Optuna belum terinstall. Jalankan: pip install optuna")
        return

    def objective(trial):
        config = {
            "lr"          : trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "dropout"     : trial.suggest_float("dropout", 0.1, 0.6),
            "fusion_dim"  : trial.suggest_categorical("fusion_dim", [128, 256, 512]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
            "epochs"      : 30,  # Lebih pendek untuk efisiensi tuning
            "patience"    : 5,
        }
        model  = model_class(
            fusion_dim=config["fusion_dim"],
            dropout=config["dropout"]
        )
        result = train_model(
            model, dataloaders, config,
            run_name=f"optuna_trial_{trial.number}"
        )
        return result["best_val_qwk"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\nOptuna — parameter terbaik:")
    print(study.best_params)
    print(f"Best val QWK: {study.best_value:.4f}")

    # Visualisasi importance
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(OUTPUT_DIR / "figures" / "optuna_importance.html"))
    except Exception:
        pass

    return study.best_params


# ─────────────────────────────────────────────
# 5. VISUALISASI EVALUASI
# ─────────────────────────────────────────────

def plot_learning_curves(history: dict, run_name: str = "model") -> None:
    """Plot train/val loss dan val QWK per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#378ADD")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#D85A30")
    axes[0].set_xlabel("Epoch");  axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Learning Curves"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # QWK
    axes[1].plot(epochs, history["val_qwk"], color="#1D9E75", linewidth=2)
    best_epoch = np.argmax(history["val_qwk"]) + 1
    best_qwk   = max(history["val_qwk"])
    axes[1].axvline(best_epoch, linestyle="--", color="gray", alpha=0.5)
    axes[1].annotate(
        f"Best: {best_qwk:.4f} @ epoch {best_epoch}",
        xy=(best_epoch, best_qwk), xytext=(best_epoch + 1, best_qwk - 0.02),
        fontsize=9, color="gray"
    )
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("QWK")
    axes[1].set_title("Validation QWK"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "figures" / f"{run_name}_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


@torch.no_grad()
def plot_prediction_scatter(
    model: nn.Module,
    loader: DataLoader,
    score_min: float,
    score_max: float,
    run_name: str = "model",
) -> None:
    """Scatter plot prediksi vs skor aktual."""
    model.eval(); model.to(DEVICE)
    all_preds, all_labels = [], []

    for batch in loader:
        preds = model(
            batch["image"].to(DEVICE),
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        )
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch["score"].numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    # Denormalize
    scale     = score_max - score_min
    preds_raw = np.clip(preds * scale + score_min, score_min, score_max)
    labels_raw = labels * scale + score_min

    qwk  = compute_qwk(preds, labels, score_min, score_max)
    rmse = compute_rmse(preds_raw, labels_raw)
    r, _ = pearsonr(preds_raw, labels_raw)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(labels_raw, preds_raw, alpha=0.4, s=20, color="#378ADD")
    lim = [score_min - 0.5, score_max + 0.5]
    ax.plot(lim, lim, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual Score"); ax.set_ylabel("Predicted Score")
    ax.set_title(
        f"Predicted vs Actual — QWK: {qwk:.4f} | RMSE: {rmse:.3f} | r: {r:.4f}"
    )
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / "figures" / f"{run_name}_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


def plot_ablation_bar(results: dict, save: bool = True) -> None:
    """
    Bar chart untuk ablation study.

    Args:
        results: dict { 'model_name': qwk_score }
    """
    names  = list(results.keys())
    scores = list(results.values())
    colors = ["#D3D1C7"] * (len(names) - 1) + ["#1D9E75"]  # highlight model ours

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, scores, color=colors, height=0.55)
    ax.set_xlabel("Quadratic Weighted Kappa (QWK)")
    ax.set_title("Ablation Study — Kontribusi Setiap Modalitas")
    ax.set_xlim(0, 1.0)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "figures" / "ablation_bar.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.show()


def generate_results_table(results_list: list[dict]) -> pd.DataFrame:
    """
    Buat tabel hasil yang siap copy-paste ke artikel.
    """
    rows = []
    for r in results_list:
        rows.append({
            "Model"         : r["run_name"],
            "LR"            : r["config"]["lr"],
            "Dropout"       : r["config"]["dropout"],
            "Fusion Dim"    : r["config"].get("fusion_dim", "-"),
            "Val QWK"       : f"{r['best_val_qwk']:.4f}",
            "Test QWK"      : f"{r['test_metrics']['qwk']:.4f}",
            "Test RMSE"     : f"{r['test_metrics']['rmse']:.4f}",
            "Test Pearson r": f"{r['test_metrics']['pearson']:.4f}",
        })
    df = pd.DataFrame(rows)
    print("\nTabel Hasil Eksperimen:")
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "results_table.csv", index=False)
    return df


# ─────────────────────────────────────────────
# 6. QUICK DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Demo evaluasi dengan data sintetis...")

    # Simulasi preds dan labels untuk demo metrik
    np.random.seed(42)
    n = 200
    labels_norm = np.random.uniform(0, 1, n)
    # Simulasi prediksi dengan noise ± 0.1
    preds_norm  = labels_norm + np.random.normal(0, 0.1, n)
    preds_norm  = np.clip(preds_norm, 0, 1)

    score_min, score_max = 2.0, 12.0
    metrics = evaluate_all_metrics(preds_norm, labels_norm, score_min, score_max)
    print(f"\nMetrik (data sintetis):")
    print(f"  QWK    : {metrics['qwk']:.4f}")
    print(f"  RMSE   : {metrics['rmse']:.4f}")
    print(f"  Pearson: {metrics['pearson']:.4f}")

    # Demo ablation bar chart
    ablation = {
        "TF-IDF + Ridge"        : 0.612,
        "ResNet-50 (image only)": 0.538,
        "DistilBERT (text only)": 0.731,
        "Multimodal (ours)"     : 0.798,
    }
    plot_ablation_bar(ablation)
    print("\nDemo selesai. Lihat outputs/figures/")
