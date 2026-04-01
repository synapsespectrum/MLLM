import math
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

plt.switch_backend("agg")


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: args.learning_rate
            / 2
            * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")


class MetricsTracker:
    def __init__(
        self,
        writer: SummaryWriter = None,
        log_iteration_freq=100,
        tracking_mlflow=False,
    ):
        """
        Args:
            writer: TensorBoard SummaryWriter
            log_iteration_freq: Log iteration metrics mỗi bao nhiêu steps
        """
        self.writer = writer
        self.mlflow_bool = tracking_mlflow
        self.log_iteration_freq = log_iteration_freq

        # Storage cho metrics
        self.iteration_metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)

        # Tracking variables
        self.current_epoch = 0
        self.global_step = 0
        self.epoch_start_time = None
        self.iteration_start_time = None

        # Best metrics tracking
        self.best_metrics = {}

        # Attention logging counter và index mapping
        self.attention_batch_count = 0

    def start_epoch(self, epoch):
        """Bắt đầu epoch mới"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.iteration_metrics.clear()  # Reset iteration metrics cho epoch mới
        print(f"📊 Starting Epoch {epoch + 1}")

    def start_iteration(self):
        """Bắt đầu iteration mới"""
        self.iteration_start_time = time.time()

    def log_iteration_metrics(self, metrics_dict, force_log=False):
        """
        Log metrics cho iteration hiện tại

        Recommended 'metrics_dict':
        - train_loss: Loss của batch hiện tại
        - learning_rate: Learning rate hiện tại
        - gradient_norm: Gradient norm (optional)

        """
        self.global_step += 1

        # Lưu metrics vào memory
        for key, value in metrics_dict.items():
            self.iteration_metrics[key].append(value)

        # Calculate iteration time nếu có
        if self.iteration_start_time:
            iteration_time = time.time() - self.iteration_start_time
            self.iteration_metrics["iteration_time"].append(iteration_time)

        # Log to TensorBoard và MLflow theo frequency
        should_log = (self.global_step % self.log_iteration_freq == 0) or force_log

        if should_log:
            self._log_to_tensorboard_iteration(metrics_dict)
            self._log_to_mlflow_iteration(metrics_dict)

    def _log_to_tensorboard_iteration(self, metrics_dict):
        """Log iteration metrics to TensorBoard"""
        if not self.writer:
            return

        for key, value in metrics_dict.items():
            self.writer.add_scalar(f"Iteration/{key}", value, self.global_step)

        # Log system metrics
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            self.writer.add_scalar(
                "System/gpu_memory_used_gb", memory_used, self.global_step
            )
            self.writer.add_scalar(
                "System/gpu_memory_cached_gb", memory_cached, self.global_step
            )

    def _log_to_mlflow_iteration(self, metrics_dict):
        """Log iteration metrics to MLflow"""
        if not self.mlflow_bool:
            return
        for key, value in metrics_dict.items():
            mlflow.log_metric(f"iter_{key}", value, step=self.global_step)

    def _calculate_epoch_averages(self):
        """Tính average metrics của epoch từ iteration metrics"""
        epoch_averages = {}

        for key, values in self.iteration_metrics.items():
            if values:
                epoch_averages[f"avg_{key}"] = np.mean(values)
                epoch_averages[f"std_{key}"] = np.std(values)
                epoch_averages[f"min_{key}"] = np.min(values)
                epoch_averages[f"max_{key}"] = np.max(values)

        return epoch_averages

    def _update_best_metrics(self, epoch_metrics):
        """Update best metrics"""
        for key, value in epoch_metrics.items():
            if "loss" in key.lower():
                # For loss metrics, lower is better
                if (
                    key not in self.best_metrics
                    or value < self.best_metrics[key]["value"]
                ):
                    self.best_metrics[key] = {
                        "value": value,
                        "epoch": self.current_epoch,
                    }
            else:
                # For other metrics, higher might be better (depends on metric)
                if (
                    key not in self.best_metrics
                    or value > self.best_metrics[key]["value"]
                ):
                    self.best_metrics[key] = {
                        "value": value,
                        "epoch": self.current_epoch,
                    }

    def log_epoch_metrics(self, epoch_metrics_dict=None):
        """
        Log metrics cho epoch hiện tại
        Args:
            epoch_metrics_dict: Dict chứa epoch-level metrics (validation loss, etc.)
                - train_loss: Average training loss của epoch
                - validation_loss: Validation loss
                - test_loss: Test loss
                - epoch_time: Thời gian training epoch

        """
        # Tính toán average của iteration metrics trong epoch
        epoch_averages = self._calculate_epoch_averages()

        # Thêm epoch time
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            epoch_averages["epoch_time"] = epoch_time

        # Thêm epoch-specific metrics
        if epoch_metrics_dict:
            epoch_averages.update(epoch_metrics_dict)

        # Update best metrics
        self._update_best_metrics(epoch_averages)

        # Store epoch metrics
        for key, value in epoch_averages.items():
            self.epoch_metrics[key].append(value)

        # Log to external systems
        self._log_to_tensorboard_epoch(epoch_averages)
        self._log_to_mlflow_epoch(epoch_averages)

    def _log_to_tensorboard_epoch(self, epoch_metrics):
        """Log epoch metrics to TensorBoard"""
        if not self.writer:
            return

        for key, value in epoch_metrics.items():
            self.writer.add_scalar(f"Epoch/{key}", value, self.current_epoch)

        # Special learning curves section
        if "train_loss" in epoch_metrics:
            self.writer.add_scalar(
                "Learning_Curves/Train_Loss",
                epoch_metrics["train_loss"],
                self.current_epoch,
            )
        if "validation_loss" in epoch_metrics:
            self.writer.add_scalar(
                "Learning_Curves/Validation_Loss",
                epoch_metrics["validation_loss"],
                self.current_epoch,
            )
        if "test_loss" in epoch_metrics:
            self.writer.add_scalar(
                "Learning_Curves/Test_Loss",
                epoch_metrics["test_loss"],
                self.current_epoch,
            )

        # 🔥 COMBINED LOSS CHART for TensorBoard - same chart with different series
        losses = {}
        if "train_loss" in epoch_metrics:
            losses["Train"] = epoch_metrics["train_loss"]
        if "validation_loss" in epoch_metrics:
            losses["Validation"] = epoch_metrics["validation_loss"]
        if "test_loss" in epoch_metrics:
            losses["Test"] = epoch_metrics["test_loss"]

        if losses:
            self.writer.add_scalars("Losses", losses, self.current_epoch)

    def _log_to_mlflow_epoch(self, epoch_metrics):
        """Log epoch metrics to MLflow"""
        if not self.mlflow_bool:
            return
        for key, value in epoch_metrics.items():
            mlflow.log_metric(f"epoch_{key}", value, step=self.current_epoch)
        # 🔥 COMBINED LOSS CHART: Log with same prefix for grouping
        if "train_loss" in epoch_metrics:
            mlflow.log_metric(
                "loss/train", epoch_metrics["train_loss"], step=self.current_epoch
            )
        if "validation_loss" in epoch_metrics:
            mlflow.log_metric(
                "loss/validation",
                epoch_metrics["validation_loss"],
                step=self.current_epoch,
            )
        if "test_loss" in epoch_metrics:
            mlflow.log_metric(
                "loss/test", epoch_metrics["test_loss"], step=self.current_epoch
            )

    def _print_epoch_summary(self, epoch_metrics):
        """In summary của epoch"""
        print(f"📈 Epoch {self.current_epoch + 1} Summary:")

        # Key metrics
        key_metrics = ["train_loss", "validation_loss", "test_loss", "epoch_time"]
        for metric in key_metrics:
            if metric in epoch_metrics:
                if "time" in metric:
                    print(f"   {metric}: {epoch_metrics[metric]:.2f}s")
                else:
                    print(f"   {metric}: {epoch_metrics[metric]:.6f}")

        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"   GPU Memory: {memory_used:.2f}GB")

        print("-" * 50)

    def log_gradient_norms(self, model):
        """Log gradient norms"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)

        if self.writer:
            self.writer.add_scalar("Gradients/total_norm", total_norm, self.global_step)

        return total_norm

    def log_test_metrics(self, mae, mse, rmse, mape, mspe):
        """
        Log final test metrics to MLflow and TensorBoard
        Args:
            mae: Mean Absolute Error
            mse: Mean Squared Error
            rmse: Root Mean Squared Error
            mape: Mean Absolute Percentage Error
            mspe: Mean Squared Percentage Error
        """
        # MLflow logging
        if self.mlflow_bool:
            try:
                # Log grouped test metrics for comparison
                mlflow.log_metric("test_metrics/mae", mae)
                mlflow.log_metric("test_metrics/mse", mse)
                mlflow.log_metric("test_metrics/rmse", rmse)
                mlflow.log_metric("test_metrics/mape", mape)
                mlflow.log_metric("test_metrics/mspe", mspe)

            except Exception as e:
                print(f"⚠️  Error logging test metrics to MLflow: {e}")

        # TensorBoard logging
        if self.writer:
            try:
                # Log individual test metrics (step=0 for final test results)
                self.writer.add_scalar("test_metrics/MAE", mae, 0)
                self.writer.add_scalar("test_metrics/MSE", mse, 0)
                self.writer.add_scalar("test_metrics/RMSE", rmse, 0)
                self.writer.add_scalar("test_metrics/MAPE", mape, 0)
                self.writer.add_scalar("test_metrics/MSPE", mspe, 0)

            except Exception as e:
                print(f"⚠️  Error logging test metrics to TensorBoard: {e}")

    def log_attention_maps(
        self,
        attention_weights,
        batch_idx,
        sample_indices,
        prefix_name="Test",
        batch_text=None,
    ):
        """
        Log attention maps to TensorBoard
        Args:
            :param attention_weights: Tensor of shape [batch_size, seq_ts_len, seq_txt_len]
            :param batch_idx: Current step for logging
            :param sample_indices: List of dataset indices for each sample in the batch
            :param prefix_name:
            :param batch_text: List of texts for each sample in the batch (optional)
        """
        batch_size, seq_ts_len, seq_txt_len = attention_weights.shape

        num_samples_to_log = batch_size

        # Store index mapping for this batch
        batch_mapping = {}

        # 1. LOG INDIVIDUAL SAMPLES với Index Information
        for sample_idx in range(num_samples_to_log):
            original_index = (
                sample_indices[sample_idx].item()
                if torch.is_tensor(sample_indices[sample_idx])
                else sample_indices[sample_idx]
            )
            sample_attention = attention_weights[
                sample_idx
            ]  # [seq_ts_len, seq_txt_len]

            # Store mapping
            logged_key = f"batch_{batch_idx:03d}_sample_{sample_idx:02d}"
            batch_mapping[logged_key] = {
                "original_index": original_index,
                "batch_idx": batch_idx,
                "sample_idx_in_batch": sample_idx,
                "attention_shape": [seq_ts_len, seq_txt_len],
            }

            # Convert to numpy
            attention_matrix = sample_attention.cpu().numpy()

            # Create detailed heatmap cho từng sample với index info
            fig, ax = plt.subplots(figsize=(14, 10))

            # Use more detailed colormap
            im = ax.imshow(
                attention_matrix, cmap="plasma", aspect="auto", interpolation="nearest"
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Attention Weight", rotation=270, labelpad=20)

            # Set labels and title với index information
            ax.set_xlabel(f"Text Token Positions (Length: {seq_txt_len})")
            ax.set_ylabel(f"Time Series Positions (Length: {seq_ts_len})")
            ax.set_title(
                f"Attention Map - Dataset Index: {original_index}\n"
                f"Batch {batch_idx}, Sample {sample_idx} | TS→Text Cross-Attention"
            )

            # Add comprehensive information box
            info_text = (
                f"Dataset Index: {original_index}\n"
                f"Batch: {batch_idx}, Sample: {sample_idx}\n"
                f"Shape: {seq_ts_len}×{seq_txt_len}\n"
                f"Max: {torch.max(sample_attention):.4f}\n"
                f"Mean: {torch.mean(sample_attention):.4f}"
            )

            # Add text information if available
            if batch_text and sample_idx < len(batch_text):
                text_preview = (
                    str(batch_text[sample_idx])[:150] + "..."
                    if len(str(batch_text[sample_idx])) > 150
                    else str(batch_text[sample_idx])
                )
                info_text += f"\n\nText Preview:\n{text_preview}"

            ax.text(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
            )

            # LOG TO TENSORBOARD
            if self.writer:
                tag_name = f"{prefix_name}_Attention_Maps/Batch_{batch_idx:03d}/Index_{original_index:06d}_Sample_{sample_idx:02d}"
                self.writer.add_figure(
                    tag_name, fig, global_step=self.attention_batch_count
                )
            # LOG TO MLFLOW
            if self.mlflow_bool:
                # Log figure
                artifact_name = f"{prefix_name.lower()}_attention_batch{batch_idx:03d}_idx{original_index:06d}_sample{sample_idx:02d}.png"
                mlflow.log_figure(fig, artifact_name)

            plt.close(fig)

        # 2. LOG SAMPLE COMPARISON với indices
        if num_samples_to_log > 1:  # make the group figure
            self._log_attention_samples_comparison(
                attention_weights[:num_samples_to_log],
                batch_idx,
                sample_indices[:num_samples_to_log],
                prefix_name,
            )

        # 3. LOG INDEX MAPPING TABLE
        self._log_index_mapping_table(batch_idx, batch_mapping, prefix_name)

        # Increment counter
        self.attention_batch_count += 1

        print(
            f"✅ Logged attention maps for Batch {batch_idx} ({num_samples_to_log}/{batch_size} samples)"
        )
        print(
            f"   📍 Dataset indices: {[sample_indices[i].item() if torch.is_tensor(sample_indices[i]) else sample_indices[i] for i in range(num_samples_to_log)]}"
        )

    def _log_attention_samples_comparison(
        self, attention_weights, batch_idx, sample_indices, prefix_name
    ):
        """Log comparison between samples trong cùng batch với index info"""
        num_samples = attention_weights.shape[0]

        # Create subplot grid để compare samples
        cols = min(4, num_samples)
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i in range(num_samples):
            ax = axes[i] if len(axes) > 1 else axes
            attention_matrix = attention_weights[i].cpu().numpy()
            original_idx = (
                sample_indices[i].item()
                if torch.is_tensor(sample_indices[i])
                else sample_indices[i]
            )

            im = ax.imshow(attention_matrix, cmap="viridis", aspect="auto")
            ax.set_title(f"Index {original_idx}\n(Sample {i})", fontsize=10)
            ax.set_xlabel("Text Tokens")
            ax.set_ylabel("TS Steps")

            # Add index info
            ax.text(
                0.02,
                0.98,
                f"Idx: {original_idx}\n",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        indices_str = [
            sample_indices[i].item()
            if torch.is_tensor(sample_indices[i])
            else sample_indices[i]
            for i in range(num_samples)
        ]
        plt.suptitle(
            f"Batch {batch_idx} - Samples Comparison\nDataset Indices: {indices_str}",
            y=1.02,
            fontsize=12,
        )

        self.writer.add_figure(
            f"{prefix_name}_Attention_Sample_Comparisons/Batch_{batch_idx:03d}_Comparison",
            fig,
            self.attention_batch_count,
        )
        plt.close(fig)

    def _log_index_mapping_table(self, batch_idx, batch_mapping, prefix_name):
        """Log bảng mapping giữa logged samples và dataset indices"""
        mapping_text = f"## Batch {batch_idx} - Index Mapping\n\n"
        mapping_text += "| Logged Key | Dataset Index |\n"
        mapping_text += "|------------|---------------|\n"

        for logged_key, info in batch_mapping.items():
            mapping_text += f"| {logged_key} | {info['original_index']} | {info['sample_idx_in_batch']} |\n"

        mapping_text += (
            f"\n**Total samples logged in this batch**: {len(batch_mapping)}\n"
        )
        mapping_text += (
            f"**TensorBoard Path**: `Attention_Maps/Batch_{batch_idx:03d}/`\n"
        )

        self.writer.add_text(
            f"{prefix_name}_Index_Mappings/Batch_{batch_idx:03d}_Mapping",
            mapping_text,
            self.attention_batch_count,
        )

    def get_epoch_summary(self):
        """Trả về summary của epoch hiện tại"""
        summary = {}
        for key, values in self.iteration_metrics.items():
            if values:
                summary[f"avg_{key}"] = np.mean(values)
                summary[f"min_{key}"] = np.min(values)
                summary[f"max_{key}"] = np.max(values)
        return summary

    def get_best_metrics_summary(self):
        """Trả về summary của best metrics"""
        print("\n🏆 Best Metrics Summary:")
        for metric, info in self.best_metrics.items():
            print(f"   {metric}: {info['value']:.6f} (Epoch {info['epoch'] + 1})")
        print("-" * 50)

        return self.best_metrics


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
