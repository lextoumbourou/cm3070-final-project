"""
Multi-class trainer for 5-class patch-based classification.

Classes:
  0: Background
  1: Benign mass
  2: Malignant mass
  3: Benign calcification
  4: Malignant calcification

Sources:
    - https://github.com/ml-explore/mlx-examples
    - https://ml-explore.github.io/mlx/build/html/index.html
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlxim.data import DataLoader
from mlxim.model import create_model

import wandb
from src.datasets import CSVDataset
from src.display import print_epoch_header
from src.model_utils import freeze_backbone, freeze_backbone_except_top_n
from src.transforms import get_patch_inference_transform, get_patch_train_transform

NUM_CLASSES = 5
CLASS_NAMES = ["Background", "Benign mass", "Malignant mass", "Benign calc", "Malignant calc"]


@dataclass
class ValidationMetrics:
    val_loss: float
    val_accuracy: float
    per_class_accuracy: dict[str, float]
    class_totals: list[int]
    predictions: list[int]
    targets: list[int]


def compute_class_weights(dataset: CSVDataset) -> mx.array:
    """
    Compute inverse frequency weights for class balancing.

    Background will be heavily represented, ROI classes less frequent.
    Uses inverse frequency: weight = total / (num_classes * class_count)
    """
    label_counts = Counter(label for _, label in dataset.samples)
    total = sum(label_counts.values())

    # Ensure all classes have at least 1 count to avoid division by zero
    weights = []
    for i in range(NUM_CLASSES):
        count = label_counts.get(i, 1)
        weight = total / (NUM_CLASSES * count)
        weights.append(weight)

    return mx.array(weights, dtype=mx.float32)


class MultiClassTrainer:
    """Trainer for 5-class patch-based classification."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        class_weights: mx.array | None = None,
        num_classes: int = NUM_CLASSES
    ):
        self.model = model
        self.optimizer = optimizer
        self.class_weights = class_weights
        self.num_classes = num_classes

    def loss_fn(self, model, inputs, targets):
        logits = model(inputs)
        if self.class_weights is not None:
            # Weighted cross-entropy
            loss = nn.losses.cross_entropy(logits, targets)
            weights = self.class_weights[targets]
            loss = mx.mean(loss * weights)
        else:
            loss = mx.mean(nn.losses.cross_entropy(logits, targets))
        return loss

    def train_step(self, inputs, targets):
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        loss, grads = loss_and_grad_fn(self.model, inputs, targets)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        return loss

    def compute_metrics(self, logits, targets):
        """Compute accuracy and per-class metrics."""
        predictions = mx.argmax(logits, axis=1)
        correct = mx.sum(predictions == targets)
        accuracy = correct / len(targets)
        return accuracy, predictions

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            loss = self.train_step(inputs, targets)

            batch_size = len(targets)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / total_samples
                print(f"  Batch {batch_idx + 1}: Loss = {avg_loss:.4f}")
                wandb.log({
                    "batch": epoch * len(train_loader) + batch_idx,
                    "train_loss_batch": avg_loss
                })

        avg_loss = total_loss / total_samples
        return {"loss": avg_loss}

    def validate(self, val_loader) -> ValidationMetrics:
        """Validate and compute per-class metrics."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Per-class tracking
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes

        all_predictions = []
        all_targets = []

        for inputs, targets in val_loader:
            logits = self.model(inputs)
            loss = mx.mean(nn.losses.cross_entropy(logits, targets))

            predictions = mx.argmax(logits, axis=1)
            correct = mx.sum(predictions == targets)

            batch_size = len(targets)
            total_loss += loss.item() * batch_size
            total_correct += correct.item()
            total_samples += batch_size

            # Collect for per-class metrics
            preds_np = np.array(predictions)
            targets_np = np.array(targets)
            all_predictions.extend(preds_np.tolist())
            all_targets.extend(targets_np.tolist())

            for pred, target in zip(preds_np, targets_np, strict=False):
                class_total[target] += 1
                if pred == target:
                    class_correct[target] += 1

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        # Per-class accuracy
        per_class_acc = {}
        for i in range(self.num_classes):
            if class_total[i] > 0:
                per_class_acc[CLASS_NAMES[i]] = class_correct[i] / class_total[i]
            else:
                per_class_acc[CLASS_NAMES[i]] = 0.0

        return ValidationMetrics(
            val_loss=avg_loss,
            val_accuracy=accuracy,
            per_class_accuracy=per_class_acc,
            class_totals=class_total,
            predictions=all_predictions,
            targets=all_targets,
        )

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        checkpoint_dir: Path | None = None,
        stage2_epoch: int | None = None,
        stage2_lr: float | None = None,
        stage2_unfreeze_layers: int | None = None,
        stage3_epoch: int | None = None,
        stage3_lr: float | None = None
    ):
        """
        Train the model with 3-stage progressive unfreezing (following Shen et al.).

        Stage 1: Train head only (frozen backbone)
        Stage 2: Unfreeze top N layers of backbone
        Stage 3: Unfreeze all layers
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print_epoch_header(epoch + 1, num_epochs)

            if stage2_epoch is not None and epoch == stage2_epoch:
                assert stage2_unfreeze_layers is not None
                assert stage2_lr is not None
                print(
                    f"\n>>> Stage 2: Unfreezing top {stage2_unfreeze_layers} layers, LR={stage2_lr}"
                )
                freeze_backbone_except_top_n(self.model, n_layers=stage2_unfreeze_layers)
                self.optimizer = optim.Adam(learning_rate=stage2_lr)

            if stage3_epoch is not None and epoch == stage3_epoch:
                assert stage3_lr is not None
                print(f"\n>>> Stage 3: Unfreezing all layers, LR={stage3_lr}")
                self.model.unfreeze()
                self.optimizer = optim.Adam(learning_rate=stage3_lr)

            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"Training Loss: {train_metrics['loss']:.4f}")

            # Validation
            val_metrics = self.validate(val_loader)
            print(f"Validation Loss: {val_metrics.val_loss:.4f}")
            print(f"Validation Accuracy: {val_metrics.val_accuracy:.4f}")
            print("Per-class accuracy:")
            for class_name, acc in val_metrics.per_class_accuracy.items():
                print(f"  {class_name}: {acc:.4f}")

            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_metrics['loss'],
                "val_loss": val_metrics.val_loss,
                "val_accuracy": val_metrics.val_accuracy,
                "learning_rate": (
                    self.optimizer.learning_rate.item()
                    if hasattr(self.optimizer.learning_rate, 'item')
                    else self.optimizer.learning_rate
                )
            }
            for class_name, acc in val_metrics.per_class_accuracy.items():
                log_dict[f"val_acc_{class_name.replace(' ', '_')}"] = acc

            wandb.log(log_dict)

            if checkpoint_dir and val_metrics.val_loss < best_val_loss:
                best_val_loss = val_metrics.val_loss
                checkpoint_path = checkpoint_dir / "best_model.npz"
                self.model.save_weights(str(checkpoint_path))
                print(f"Saved best model to {checkpoint_path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train 5-class patch-based model on CBIS-DDSM patches"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name of the model to use",
        required=True
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name for this training run (used for checkpoint directory)",
        required=True
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to patch data directory (datasets/prep/cbis-ddsm-patches)",
        required=True
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Path to pretrained weights (.npz) to load before training",
        default=None
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting for loss function"
    )
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    IMG_DIR = DATA_DIR / "img"
    TRAIN_CSV = DATA_DIR / "train.csv"
    VAL_CSV = DATA_DIR / "val.csv"
    TEST_CSV = DATA_DIR / "test.csv"

    # Shen et al. 2019 training schedule for S10 patch dataset
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    STAGE1_LR = 1e-3
    STAGE1_EPOCHS = 3
    STAGE2_LR = 1e-4
    STAGE2_EPOCHS = 10
    STAGE2_UNFREEZE_LAYERS = 46  # Top 46 layers for ResNet50
    STAGE3_LR = 1e-5
    IMAGE_SIZE = 224
    MODEL_NAME = args.model_name

    # Initialize wandb
    wandb.init(
        project="cm3070-mammography",
        entity="lex",
        name=args.run_name,
        config={
            "task": "5-class-patch-classification",
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "stage1_lr": STAGE1_LR,
            "stage1_epochs": STAGE1_EPOCHS,
            "stage2_lr": STAGE2_LR,
            "stage2_epochs": STAGE2_EPOCHS,
            "stage2_unfreeze_layers": STAGE2_UNFREEZE_LAYERS,
            "stage3_lr": STAGE3_LR,
            "image_size": IMAGE_SIZE,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "model": MODEL_NAME,
            "optimizer": "adam",
            "dataset": args.data_dir,
            "class_weighting": not args.no_class_weights,
            "pretrained_weights": args.weights,
            "run_name": args.run_name
        }
    )

    print("Loading datasets...")
    train_transform = get_patch_train_transform(output_size=IMAGE_SIZE)
    val_transform = get_patch_inference_transform(output_size=IMAGE_SIZE)

    train_dataset = CSVDataset(
        csv_path=str(TRAIN_CSV),
        img_dir=str(IMG_DIR),
        transform=train_transform,
    )
    val_dataset = CSVDataset(
        csv_path=str(VAL_CSV),
        img_dir=str(IMG_DIR),
        transform=val_transform,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Print class distribution
    train_labels = [label for _, label in train_dataset.samples]
    label_counts = Counter(train_labels)
    print("\nTraining class distribution:")
    for i in range(NUM_CLASSES):
        count = label_counts.get(i, 0)
        pct = 100 * count / len(train_labels)
        print(f"  {CLASS_NAMES[i]}: {count} ({pct:.1f}%)")

    # Compute class weights
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_dataset)
        print("\nClass weights:")
        for i in range(NUM_CLASSES):
            print(f"  {CLASS_NAMES[i]}: {class_weights[i].item():.4f}")
    else:
        class_weights = None
        print("\nClass weighting disabled")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    model = create_model(MODEL_NAME, num_classes=NUM_CLASSES)
    if args.weights:
        print(f"Loading weights from {args.weights}")
        model.load_weights(args.weights)
    freeze_backbone(model)

    optimizer = optim.Adam(learning_rate=STAGE1_LR)

    trainer = MultiClassTrainer(
        model=model,
        optimizer=optimizer,
        class_weights=class_weights,
        num_classes=NUM_CLASSES
    )

    stage2_start = STAGE1_EPOCHS
    stage3_start = STAGE1_EPOCHS + STAGE2_EPOCHS

    print()
    print("Starting 3-stage training (Shen et al. 2019)...")
    print(f"Stage 1 (epochs 1-{STAGE1_EPOCHS}): Head only, LR={STAGE1_LR}")
    print(
        f"Stage 2 (epochs {stage2_start+1}-{stage3_start}): "
        f"Top {STAGE2_UNFREEZE_LAYERS} layers, LR={STAGE2_LR}"
    )
    print(f"Stage 3 (epochs {stage3_start+1}-{NUM_EPOCHS}): All layers, LR={STAGE3_LR}")

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=Path("checkpoints") / args.run_name,
        stage2_epoch=stage2_start,
        stage2_lr=STAGE2_LR,
        stage2_unfreeze_layers=STAGE2_UNFREEZE_LAYERS,
        stage3_epoch=stage3_start,
        stage3_lr=STAGE3_LR
    )

    print()
    print("Training complete!")

    # Test set evaluation
    print()
    print("Running inference on test set...")
    test_dataset = CSVDataset(
        csv_path=str(TEST_CSV),
        img_dir=str(IMG_DIR),
        transform=val_transform,
    )
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    best_checkpoint = Path("checkpoints") / args.run_name / "best_model.npz"
    if best_checkpoint.exists():
        print(f"Loading best model from {best_checkpoint}")
        model.load_weights(str(best_checkpoint))

    test_metrics = trainer.validate(test_loader)
    print()
    print("Test Results:")
    print(f"  Test Loss: {test_metrics.val_loss:.4f}")
    print(f"  Test Accuracy: {test_metrics.val_accuracy:.4f}")
    print("  Per-class accuracy:")
    for class_name, acc in test_metrics.per_class_accuracy.items():
        print(f"    {class_name}: {acc:.4f}")

    # Log test metrics
    test_log = {
        "test_loss": test_metrics.val_loss,
        "test_accuracy": test_metrics.val_accuracy
    }
    for class_name, acc in test_metrics.per_class_accuracy.items():
        test_log[f"test_acc_{class_name.replace(' ', '_')}"] = acc

    wandb.log(test_log)
    wandb.finish()


if __name__ == "__main__":
    main()
