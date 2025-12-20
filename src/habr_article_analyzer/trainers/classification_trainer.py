# mypy: ignore-errors

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    history: dict[str, Any]

    def __init__(
        self,
        model: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: Path | str = "runs/experiment",
        train_dir: Path | str = "train/results",
        log_iters: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_iters = log_iters
        self.log_dir = log_dir
        self.train_dir = train_dir

        self.history = {
            "train_loss": [],
            "train_auc": [],
            "val_loss": [],
            "val_auc": [],
            "val_pr_auc": [],
            "learning_rate": [],
        }

        self.best_val_auc = 0
        self.global_step = 0

        print(f"TensorBoard logs will be saved to: {log_dir}")
        print(f"Run: tensorboard --logdir={log_dir}")

    def train_epoch(
        self, train_loader: Any, optimizer: Any, criterion: Any, epoch: int
    ) -> Any:
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            if batch_idx % self.log_iters == 0:
                self.writer.add_scalar(
                    "Batch/train_loss", loss.item(), self.global_step
                )
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            self.global_step += 1

        avg_loss = total_loss / len(train_loader)
        auc_score = roc_auc_score(all_labels, all_preds)

        return avg_loss, auc_score

    def validate(self, val_loader: Any, criterion: Any, epoch: int) -> Any:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        auc_score = roc_auc_score(all_labels, all_preds)

        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        pr_auc = auc(recall, precision)

        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        cm = confusion_matrix(all_labels, preds_binary)

        return avg_loss, auc_score, pr_auc, cm, all_preds, all_labels

    def log_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_auc: float,
        val_loss: float,
        val_auc: float,
        val_pr_auc: float,
        lr: float,
    ) -> None:
        self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)

        self.writer.add_scalars("AUC", {"train": train_auc, "val": val_auc}, epoch)

        self.writer.add_scalar("Metrics/val_pr_auc", val_pr_auc, epoch)
        self.writer.add_scalar("Learning_rate", lr, epoch)

    def log_confusion_matrix(self, cm: Any, epoch: int) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - Epoch {epoch}")

        self.writer.add_figure("Confusion_Matrix", fig, epoch)
        plt.close(fig)

    def log_pr_curve(self, labels: Any, preds: Any, epoch: int) -> None:
        precision, recall, _ = precision_recall_curve(labels, preds)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve - Epoch {epoch}")
        ax.grid(True, alpha=0.3)

        self.writer.add_figure("PR_Curve", fig, epoch)
        plt.close(fig)

    def log_distribution(self, preds: Any, labels: Any, epoch: int) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        pos_preds = np.array(preds)[np.array(labels) == 1]
        neg_preds = np.array(preds)[np.array(labels) == 0]

        ax.hist(pos_preds, bins=50, alpha=0.5, label="Positive", color="green")
        ax.hist(neg_preds, bins=50, alpha=0.5, label="Negative", color="red")
        ax.axvline(x=0.5, color="black", linestyle="--", label="Threshold")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.set_title(f"Prediction Distribution - Epoch {epoch}")
        ax.legend()

        self.writer.add_figure("Prediction_Distribution", fig, epoch)
        plt.close(fig)

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        epochs: int = 20,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        patience: int = 5,
    ) -> Any:

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=patience // 2, min_lr=1e-6
        )
        dummy_input = torch.zeros(1, train_loader.dataset[0][0].shape[0]).to(
            self.device
        )
        self.writer.add_graph(self.model, dummy_input)

        print("TRAINING START")

        no_improvement = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_loss, train_auc = self.train_epoch(
                train_loader, optimizer, criterion, epoch
            )

            val_loss, val_auc, val_pr_auc, cm, val_preds, val_labels = self.validate(
                val_loader, criterion, epoch
            )

            current_lr = optimizer.param_groups[0]["lr"]

            scheduler.step(val_auc)

            self.history["train_loss"].append(train_loss)
            self.history["train_auc"].append(train_auc)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)
            self.history["val_pr_auc"].append(val_pr_auc)
            self.history["learning_rate"].append(current_lr)

            self.log_metrics(
                epoch, train_loss, train_auc, val_loss, val_auc, val_pr_auc, current_lr
            )
            self.log_confusion_matrix(cm, epoch)
            self.log_pr_curve(val_labels, val_preds, epoch)
            self.log_distribution(val_preds, val_labels, epoch)

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch}/{epochs} Summary (Time: {epoch_time:.1f}s)")
            print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
            print(
                f"Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f} |"
                + "Val PR-AUC: {val_pr_auc:.4f}"
            )
            print(f"Learning Rate: {current_lr:.6f}")

            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                torch.save(self.model.state_dict(), self.train_dir / "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_auc": val_auc,
                    },
                    self.train_dir / "checkpoint.pt",
                )
                print(f"Best model saved. (AUC: {val_auc:.4f})")
                no_improvement = 0
            else:
                no_improvement += 1
                print(f"No improvement for {no_improvement} epochs")

            if no_improvement >= patience:
                print(
                    f"\nEarly stopping triggered after {epoch} epochs"
                    + "with no improvement"
                )
                break

        total_time = time.time() - start_time

        print("TRAINING COMPLETE")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best Val AUC: {self.best_val_auc:.4f}")
        print(f"View results: tensorboard --logdir={self.log_dir}")

        self.writer.close()

        return self.history
