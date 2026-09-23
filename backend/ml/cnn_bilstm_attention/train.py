import copy
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.cnn_bilstm_attention.evaluate import evaluate_model

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_effective_class_weights(
    train_labels: List[int],
    num_classes: int = 6,
    beta: float = 0.99
) -> torch.Tensor:
    """
    Computes Class-Balanced Loss weights based on effective number of samples:
    E_n = (1 - beta^n) / (1 - beta)
    weight = 1 / E_n, normalized so mean = 1.0.
    Prevents rare classes (like Laryngozele) from destabilizing gradients.
    """
    counts = np.bincount(train_labels, minlength=num_classes)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    classes: List[str],
    device: torch.device,
    epochs: int = 35,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_class_weights: bool = False,
    label_smoothing: float = 0.0,
    patience: int = 8,
    checkpoint_dir: Optional[Path] = None,
    config_name: str = "config"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Trains the 2D CNN + BiLSTM + Attention model with early stopping.
    """
    set_seed(42)
    model.to(device)

    # Class weighting
    class_weights = None
    if use_class_weights:
        all_train_labels = []
        for _, batch_y, _, _ in train_loader:
            all_train_labels.extend(batch_y.numpy().tolist())
        weights_t = compute_effective_class_weights(all_train_labels, num_classes=len(classes)).to(device)
        class_weights = weights_t
        logger.info(f"[{config_name}] Computed Class Weights: {[round(float(w), 3) for w in weights_t.cpu().numpy()]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_model_weights = None
    best_epoch = 0
    patience_counter = 0

    history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_samples = 0

        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients through BiLSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_total += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_samples += len(batch_y)

        train_loss = train_loss_total / train_samples
        train_acc = train_correct / train_samples

        # 2. Validation Phase
        model.eval()
        val_loss_total = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch_x, batch_y, _, _ in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss_total += loss.item() * len(batch_y)
                val_samples += len(batch_y)

        val_loss = val_loss_total / val_samples
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "lr": current_lr
        })

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"[{config_name}] Epoch {epoch:2d}/{epochs:2d} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
            )

        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[{config_name}] Early stopping triggered at epoch {epoch} (Best epoch: {best_epoch}, Val Loss: {best_val_loss:.4f}).")
                break

    training_time = round(time.time() - t_start, 2)
    logger.info(f"[{config_name}] Training finished in {training_time}s across {len(history)} epochs.")

    # Load best weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Save checkpoint if requested
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"{config_name}_best.pt"
        torch.save({
            "config_name": config_name,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "classes": classes
        }, ckpt_path)
        logger.info(f"[{config_name}] Checkpoint saved to: {ckpt_path}")

    # Final evaluation on validation set using best weights
    val_eval = evaluate_model(model, val_loader, classes, device)

    summary = {
        "config_name": config_name,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "total_epochs": len(history),
        "training_time_seconds": training_time,
        "history": history,
        "validation_metrics": val_eval
    }

    return model, summary
