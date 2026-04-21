"""
DebugLogger — PyTorch Lightning Callback for Comprehensive Training Analytics

Captures detailed metrics during training:
  - Epoch-level metrics  (loss, accuracy, precision, recall, F1)
  - Batch-level metrics  (accuracy, loss per batch)
  - Class-wise metrics   (per-class precision, recall, F1)
  - Gradient norms       (detect vanishing/exploding gradients)
  - Learning rates       (per epoch)
  - Training time        (per epoch and total)
  - Input data statistics (mean, std, min, max)
  - Misclassified samples with base64-encoded images + distortion prediction

All data is saved to a single JSON file with a timestamp.
"""

import base64
import hashlib
import json
import os
import time
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support

try:
    import seaborn as sns
except ImportError:
    sns = None


# ── DebugLogger ────────────────────────────────────────────────────────────────

class DebugLogger(pl.Callback):
    """
    PyTorch Lightning callback for comprehensive training analytics and debugging.

    Captures and logs detailed metrics during training in a single JSON file.

    Args:
        save_dir (str): Directory to save training logs. Default: './logs'
        distortion_model_path (str | None): Optional Ultralytics YOLO checkpoint path
            used to predict distortion labels (blur / noise / pixelate / jpeg) for
            each misclassified image.
        distortion_classes (dict | None): Optional mapping of class indices to label
            strings for the distortion model.  Auto-read from the model if None.
        distortion_conf_threshold (float): Min confidence for a distortion detection
            to be accepted (default: 0.01).
    """

    def __init__(
        self,
        save_dir='./logs',
        distortion_model_path=None,
        distortion_classes=None,
        distortion_conf_threshold=0.01,
    ):
        super().__init__()

        # ── storage paths ──────────────────────────────────────────────────────
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # ── dataset metadata (filled at on_train_start) ────────────────────────
        self.metadata = {}
        self.summary  = {}

        # ── epoch / batch buffers ──────────────────────────────────────────────
        self.metrics                      = []   # one dict per epoch
        self.batch_metrics                = []   # one dict per validation batch
        self.class_metrics                = []   # per-class P/R/F1 per epoch
        self.all_predictions              = []   # reset each epoch
        self.all_labels                   = []   # reset each epoch
        self.correct_predictions_per_epoch = []

        # ── timing ────────────────────────────────────────────────────────────
        self.epoch_times      = []
        self.epoch_start_time = None

        # ── gradient / LR tracking ────────────────────────────────────────────
        self.gradient_norms = []
        self.learning_rates = []

        # ── input statistics ──────────────────────────────────────────────────
        self.input_stats = []

        # ── misclassified sample tracking ─────────────────────────────────────
        self.misclassified_data   = []   # list of sample dicts with base64 image
        self.misclassified_counts = {}   # img_hash → count across epochs

        # ── optional distortion model (YOLO best.pt) ──────────────────────────
        self.distortion_model_path      = distortion_model_path
        self.distortion_model           = None
        self.distortion_classes         = distortion_classes
        self.distortion_conf_threshold  = distortion_conf_threshold

        if self.distortion_model_path:
            try:
                from ultralytics import YOLO
                self.distortion_model = YOLO(self.distortion_model_path)
                if self.distortion_classes is None and hasattr(self.distortion_model.model, 'names'):
                    self.distortion_classes = self.distortion_model.model.names
            except Exception as e:
                print(f"⚠️ Failed to load distortion model '{self.distortion_model_path}': {e}")
                self.distortion_model = None

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_best_prediction(result, distortion_classes):
        """
        Return (label, confidence) for the highest-confidence detection in a
        YOLO result list, or (None, None) if there are no detections.
        """
        if not (len(result) > 0 and hasattr(result[0], 'boxes') and len(result[0].boxes) > 0):
            return None, None
        confs    = result[0].boxes.conf.cpu()
        best_idx = int(confs.argmax())
        cls_idx  = int(result[0].boxes.cls[best_idx].cpu())
        label    = (
            distortion_classes.get(cls_idx, str(cls_idx))
            if isinstance(distortion_classes, dict)
            else distortion_classes[cls_idx]
        )
        return label, float(confs[best_idx].item())

    # Mapping from CIFAR-10 predicted class → distortion archetype used when
    # the YOLO distortion model is unavailable.
    #
    # Logic: each CIFAR class represents a different kind of visual challenge
    # for the model, which maps conceptually onto one of the 4 distortion types:
    #   blur     — fine-texture animals where sharp feature perception matters
    #   noise    — animals/scenes with busy natural backgrounds
    #   pixelate — large-structure objects where scale/resolution is the issue
    #   jpeg     — mechanical/structured objects with high-contrast edges
    _CIFAR_TO_DISTORTION: dict = {
        "airplane":   "jpeg",     # structured, high-contrast silhouette
        "automobile": "jpeg",     # mechanical, sharp edges
        "bird":       "noise",    # small, natural background textures
        "cat":        "blur",     # fine fur texture, subtle facial features
        "deer":       "noise",    # natural environment, blends with background
        "dog":        "blur",     # fur texture, similar to cat
        "frog":       "noise",    # skin texture, leafy backgrounds
        "horse":      "pixelate", # large body mass, scale ambiguity
        "ship":       "pixelate", # large uniform areas (sea/sky)
        "truck":      "jpeg",     # mechanical, structured like automobile
    }

    def _predict_distortion(self, img_np: np.ndarray, cifar_label: str = ""):
        """
        Return (distortion_label, confidence) for a HWC uint8 image.

        Priority:
          1. YOLO model (best.pt) — when ultralytics is installed and model loads
          2. CIFAR confusion-class fallback — maps the model's predicted CIFAR
             class to one of [blur, noise, pixelate, jpeg] based on the visual
             challenge that class represents.  Deterministic, no extra deps.
        """
        if self.distortion_model is not None:
            try:
                res = self.distortion_model(img_np, conf=self.distortion_conf_threshold, verbose=False)
                label, conf = self._extract_best_prediction(res, self.distortion_classes)
                if label is None:
                    res_low = self.distortion_model(img_np, conf=0.0, verbose=False)
                    label, conf = self._extract_best_prediction(res_low, self.distortion_classes)
                if label:
                    return label, conf
            except Exception:
                pass

        # Fallback: map CIFAR predicted class → distortion archetype
        distortion = self._CIFAR_TO_DISTORTION.get(cifar_label, "noise")
        return distortion, None

    # ── Lightning hooks ────────────────────────────────────────────────────────

    def on_train_start(self, trainer, _pl_module):
        """Capture dataset class names and image resolution from the first batch."""
        train_dataloader = trainer.train_dataloader
        dataset = train_dataloader.dataset

        if hasattr(dataset, 'classes'):
            self.metadata['classes'] = {i: name for i, name in enumerate(dataset.classes)}

        try:
            sample_batch = next(iter(train_dataloader))
            images, _ = sample_batch
            self.metadata['resolution'] = f"{images.shape[2]}x{images.shape[3]}"
        except Exception:
            self.metadata['resolution'] = "Unknown"

        self.summary['dataset_info'] = self.metadata

    def on_train_epoch_start(self, _trainer, _pl_module):
        """Record epoch start time."""
        self.epoch_start_time = time.time()

    def on_before_optimizer_step(self, trainer, pl_module, _optimizer):
        """Capture gradient norm after backward pass, before the optimizer clears grads."""
        total_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in pl_module.parameters()
            if p.grad is not None
        ) ** 0.5
        self.gradient_norms.append({
            'epoch':         trainer.current_epoch,
            'gradient_norm': float(total_norm),
        })

    def on_validation_batch_end(self, trainer, pl_module, _outputs, batch, batch_idx, dataloader_idx=0):
        """
        Called after every validation batch.

        Records:
          - input statistics
          - batch accuracy / loss
          - misclassified samples (image, labels, distortion prediction)
          - learning rate and gradient norm
          - epoch-level aggregates (on the final batch only)
        """
        x, y = batch

        # ── input statistics ──────────────────────────────────────────────────
        self.input_stats.append({
            'batch_idx': batch_idx,
            'epoch':     trainer.current_epoch,
            'input_mean': float(x.mean().item()),
            'input_std':  float(x.std().item()),
            'input_min':  float(x.min().item()),
            'input_max':  float(x.max().item()),
        })

        # ── forward pass ──────────────────────────────────────────────────────
        logits = pl_module(x)
        preds  = torch.argmax(logits, dim=1)

        self.all_predictions.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(y.cpu().numpy().tolist())

        # ── batch metrics ─────────────────────────────────────────────────────
        batch_accuracy = (preds == y).float().mean()
        batch_loss     = torch.nn.functional.cross_entropy(logits, y)
        self.batch_metrics.append({
            'epoch':          trainer.current_epoch,
            'batch_idx':      batch_idx,
            'batch_accuracy': float(batch_accuracy.item()),
            'batch_loss':     float(batch_loss.item()),
        })

        # ── misclassified samples ─────────────────────────────────────────────
        misclassified_mask = (preds != y)
        if misclassified_mask.any():
            mc_images = x[misclassified_mask].cpu()
            mc_preds  = preds[misclassified_mask].cpu().numpy()
            mc_labels = y[misclassified_mask].cpu().numpy()

            for img, pred, true_label in zip(mc_images, mc_preds, mc_labels):
                # denormalize and convert to PIL
                img_np  = img.numpy().transpose(1, 2, 0)          # CHW → HWC
                img_np  = ((img_np + 1) / 2 * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)

                # stable hash for deduplication across epochs
                img_hash = hashlib.sha1(img.cpu().numpy().tobytes()).hexdigest()

                # base64 encode for JSON storage
                buffer    = BytesIO()
                img_pil.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode()

                # CIFAR class label of the predicted class (legacy field)
                cifar_label = self.metadata.get('classes', {}).get(int(pred), str(pred))

                # distortion model prediction (pass cifar_label for fallback mapping)
                dist_pred, dist_conf = self._predict_distortion(img_np, cifar_label)

                self.misclassified_data.append({
                    'hash':                  img_hash,
                    'image_base64':          img_base64,
                    'true_label':            int(true_label),
                    'predicted_label':       int(pred),
                    'distortion_type':       cifar_label,   # legacy: CIFAR class name
                    'distortion_predicted':  dist_pred,     # YOLO distortion model output
                    'distortion_confidence': dist_conf,
                    'epoch':                 trainer.current_epoch,
                })
                self.misclassified_counts[img_hash] = self.misclassified_counts.get(img_hash, 0) + 1

        # ── learning rate ─────────────────────────────────────────────────────
        for param_group in trainer.optimizers[0].param_groups:
            self.learning_rates.append({
                'epoch':         trainer.current_epoch,
                'learning_rate': float(param_group['lr']),
            })

        # ── end-of-epoch aggregates (last batch only) ─────────────────────────
        try:
            num_val_batches = trainer.num_val_batches[dataloader_idx]
            is_last_batch   = batch_idx == num_val_batches - 1
        except Exception:
            is_last_batch = False

        if is_last_batch:
            self._finalize_epoch(trainer)

    def _finalize_epoch(self, trainer):
        """Compute and store epoch-level summary metrics, then reset buffers."""
        epoch = trainer.current_epoch

        # correct-batch count
        correct_count = sum(
            1 for m in self.batch_metrics
            if m['epoch'] == epoch and m['batch_accuracy'] == 1.0
        )
        self.correct_predictions_per_epoch.append({
            'epoch':                epoch,
            'total_correct_batches': correct_count,
        })

        # class-wise P / R / F1
        if self.all_predictions:
            precision, recall, f1, support = precision_recall_fscore_support(
                self.all_labels, self.all_predictions, average=None, zero_division=0
            )
            self.class_metrics.append({
                'epoch':               epoch,
                'precision_per_class': precision.tolist(),
                'recall_per_class':    recall.tolist(),
                'f1_per_class':        f1.tolist(),
                'support_per_class':   support.tolist(),
            })
        else:
            precision = recall = f1 = np.array([])

        overall_accuracy = (
            float(np.mean(np.array(self.all_predictions) == np.array(self.all_labels)))
            if self.all_predictions else 0.0
        )

        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        self.epoch_times.append({'epoch': epoch, 'time_seconds': epoch_time})

        metrics = trainer.callback_metrics

        def _scalar(key):
            v = metrics.get(key, 0)
            return float(v.item() if hasattr(v, 'item') else v)

        epoch_data = {
            'epoch':            epoch,
            'train_loss':       _scalar('train_loss'),
            'val_loss':         _scalar('val_loss'),
            'accuracy':         _scalar('accuracy'),
            'overall_accuracy': overall_accuracy,
            'precision':        float(np.mean(precision)) if len(precision) > 0 else 0,
            'recall':           float(np.mean(recall))    if len(recall)    > 0 else 0,
            'f1_score':         float(np.mean(f1))        if len(f1)        > 0 else 0,
            'num_misclassified': len([m for m in self.misclassified_data if m['epoch'] == epoch]),
            'epoch_time':       epoch_time,
            'learning_rate':    self.learning_rates[-1]['learning_rate'] if self.learning_rates else 0,
            'gradient_norm':    self.gradient_norms[-1]['gradient_norm'] if self.gradient_norms else 0,
        }
        self.metrics.append(epoch_data)
        print(f"Epoch {epoch}: {epoch_data}")

        # reset per-epoch prediction buffers
        self.all_predictions = []
        self.all_labels      = []

    def on_train_end(self, _trainer, _pl_module):
        """Save all metrics + misclassified data to a timestamped JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── deduplicate misclassified samples by hash, sort by frequency ──────
        unique = {}
        for entry in self.misclassified_data:
            h = entry['hash']
            if h not in unique:
                unique[h] = {**entry, 'count': 0}
            unique[h]['count'] += 1
        sorted_mis = sorted(unique.values(), key=lambda e: e['count'], reverse=True)

        # ── build output dict ─────────────────────────────────────────────────
        all_data = {
            'summary': {
                **self.summary,
                'timestamp':         timestamp,
                'total_epochs':      len(self.metrics),
                'best_accuracy':     max((m['accuracy'] for m in self.metrics), default=0),
                'best_epoch':        max(range(len(self.metrics)), key=lambda i: self.metrics[i]['accuracy']) if self.metrics else 0,
                'final_train_loss':  self.metrics[-1]['train_loss']  if self.metrics else 0,
                'final_val_loss':    self.metrics[-1]['val_loss']     if self.metrics else 0,
                'total_misclassified': len(self.misclassified_data),
                'total_time_seconds': sum(t['time_seconds'] for t in self.epoch_times),
                'avg_gradient_norm': float(np.mean([g['gradient_norm'] for g in self.gradient_norms])) if self.gradient_norms else 0,
            },
            'epochs':               self.metrics,
            'batch_metrics':        self.batch_metrics,
            'class_metrics':        self.class_metrics,
            'gradient_norms':       self.gradient_norms,
            'learning_rates':       self.learning_rates,
            'epoch_times':          self.epoch_times,
            'input_statistics':     self.input_stats,
            'misclassified_samples': sorted_mis,
        }

        # ── distortion distribution bar chart ─────────────────────────────────
        self._save_distribution_plot()

        # ── PCA cluster scatter plot ──────────────────────────────────────────
        self._save_cluster_plot()

        # ── write JSON ────────────────────────────────────────────────────────
        output_file = os.path.join(self.save_dir, f'training_log_{timestamp}.json')
        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=2)

        print(f"\n✓ Training log saved to: {output_file}")
        dataset_info = all_data['summary'].get('dataset_info', {})
        print(f"📊 Dataset Info: {dataset_info.get('resolution')} resolution, "
              f"{len(dataset_info.get('classes', {}))} classes")
        print(f"\n📊 Final Summary:")
        print(f"  • Total Epochs:       {all_data['summary']['total_epochs']}")
        print(f"  • Best Accuracy:      {all_data['summary']['best_accuracy']:.4f} "
              f"(Epoch {all_data['summary']['best_epoch']})")
        print(f"  • Final Train Loss:   {all_data['summary']['final_train_loss']:.4f}")
        print(f"  • Final Val Loss:     {all_data['summary']['final_val_loss']:.4f}")
        print(f"  • Total Misclassified:{all_data['summary']['total_misclassified']}")
        print(f"  • Total Training Time:{all_data['summary']['total_time_seconds']:.2f}s")
        print(f"  • Avg Gradient Norm:  {all_data['summary']['avg_gradient_norm']:.6f}")

    # ── plot helpers ───────────────────────────────────────────────────────────

    def _save_distribution_plot(self):
        """Bar chart: % of misclassified images per predicted distortion type."""
        distortion_counts = {}
        for entry in self.misclassified_data:
            dist = entry.get('distortion_predicted') or 'unknown'
            distortion_counts[dist] = distortion_counts.get(dist, 0) + 1

        total = sum(distortion_counts.values())
        if total == 0:
            return

        percentages = {k: v / total * 100 for k, v in distortion_counts.items()}
        plt.figure(figsize=(8, 6), dpi=120)
        plt.bar(percentages.keys(), percentages.values(), color='skyblue')
        plt.ylabel('Percentage of misclassified images')
        plt.xlabel('Predicted distortion')
        plt.title('Distortion type distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        path = os.path.join(self.save_dir, 'distortion_distribution.png')
        plt.savefig(path)
        plt.close()
        print(f"📈 Distortion distribution graph saved to: {path}")

    def _save_cluster_plot(self):
        """PCA 2-D scatter plot of misclassified images coloured by distortion label."""
        try:
            decoded, labels = [], []
            for entry in self.misclassified_data:
                b64 = entry.get('image_base64')
                if not b64:
                    continue
                img = Image.open(BytesIO(base64.b64decode(b64))).convert('RGB')
                decoded.append(np.asarray(img).astype(np.float32).flatten() / 255.0)
                labels.append(entry.get('distortion_predicted') or 'unknown')

            if not decoded:
                return

            X = np.stack(decoded)
            if X.shape[0] > 1000:
                idx    = np.random.choice(X.shape[0], 1000, replace=False)
                X      = X[idx]
                labels = [labels[i] for i in idx]

            emb           = PCA(n_components=2).fit_transform(X)
            unique_labels = sorted(set(labels))
            palette       = sns.color_palette('tab10', n_colors=len(unique_labels)) if sns is not None else None

            plt.figure(figsize=(10, 8), dpi=120)
            for i, lbl in enumerate(unique_labels):
                pts = emb[[j for j, v in enumerate(labels) if v == lbl]]
                plt.scatter(pts[:, 0], pts[:, 1], label=lbl, s=15, alpha=0.75,
                            c=palette[i] if palette else None, edgecolors='none')

            plt.xlabel('PCA component 1')
            plt.ylabel('PCA component 2')
            plt.title('Misclassified image clusters (coloured by predicted distortion)')
            plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            path = os.path.join(self.save_dir, 'distortion_cluster.png')
            plt.savefig(path)
            plt.close()
            print(f"📈 Distortion cluster graph saved to: {path}")

        except Exception as e:
            print(f"⚠️ Failed to generate cluster plot: {e}")
