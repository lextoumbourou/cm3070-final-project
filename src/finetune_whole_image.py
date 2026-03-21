"""
Fine-tune a pre-trained whole image classifier on a new dataset.

Todo: comment and refactor.
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import wandb
from mlxim.data import DataLoader

from src.datasets import CSVDataset
from src.models.whole_image_classifier import create_whole_image_classifier
from src.transforms import get_inference_transform, get_train_transform


class FineTuner:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def loss_fn(self, model, inputs, targets):
        logits = model(inputs)
        return mx.mean(nn.losses.cross_entropy(logits, targets))

    def train_step(self, inputs, targets):
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        loss, grads = loss_and_grad_fn(self.model, inputs, targets)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        return loss

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            loss = self.train_step(inputs, targets)

            logits = self.model(inputs)
            predictions = mx.argmax(logits, axis=1)
            correct = mx.sum(predictions == targets).item()

            batch_size = len(targets)
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / total_samples
                avg_acc = total_correct / total_samples
                print(f"  Batch {batch_idx + 1}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.4f}")
                wandb.log({
                    "batch": epoch * len(train_loader) + batch_idx,
                    "train_loss_batch": avg_loss,
                    "train_acc_batch": avg_acc
                })

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        throughput = total_samples / epoch_time
        return {
            "loss": avg_loss, "accuracy": avg_acc,
            "epoch_time": epoch_time, "throughput": throughput
        }

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_probs = []
        all_targets = []

        for inputs, targets in val_loader:
            logits = self.model(inputs)
            loss = mx.mean(nn.losses.cross_entropy(logits, targets))

            predictions = mx.argmax(logits, axis=1)
            correct = mx.sum(predictions == targets)
            probs = mx.softmax(logits, axis=1)[:, 1]

            batch_size = len(targets)
            total_loss += loss.item() * batch_size
            total_correct += correct.item()
            total_samples += batch_size

            all_probs.extend(np.array(probs).tolist())
            all_targets.extend(np.array(targets).tolist())

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_targets, all_probs)
        except Exception:
            auc = 0.0

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_auc": auc,
        }

    def fit(self, train_loader, val_loader, num_epochs, checkpoint_dir=None,
            unfreeze_epoch=None, unfreeze_lr=None, unfreeze_wd=None):
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_auc = 0.0
        training_start = time.time()
        epoch_times = []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            if unfreeze_epoch is not None and epoch == unfreeze_epoch:
                print(f"\n>>> Unfreezing backbone, LR={unfreeze_lr}, WD={unfreeze_wd}")
                self.model.unfreeze_all()
                wd = unfreeze_wd or 0.0
                self.optimizer = optim.AdamW(learning_rate=unfreeze_lr, weight_decay=wd)

            train_metrics = self.train_epoch(train_loader, epoch)
            epoch_times.append(train_metrics['epoch_time'])
            loss, acc = train_metrics['loss'], train_metrics['accuracy']
            print(f"Training - Loss: {loss:.4f}, Acc: {acc:.4f}")
            t, tp = train_metrics['epoch_time'], train_metrics['throughput']
            print(f"  Epoch time: {t:.1f}s, Throughput: {tp:.2f} img/s")

            val_metrics = self.validate(val_loader)
            v_loss = val_metrics['val_loss']
            v_acc, v_auc = val_metrics['val_accuracy'], val_metrics['val_auc']
            print(f"Validation - Loss: {v_loss:.4f}, Acc: {v_acc:.4f}, AUC: {v_auc:.4f}")

            lr = self.optimizer.learning_rate
            if hasattr(lr, 'item'):
                lr = lr.item()

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_metrics['loss'],
                "train_accuracy": train_metrics['accuracy'],
                "val_loss": val_metrics['val_loss'],
                "val_accuracy": val_metrics['val_accuracy'],
                "val_auc": val_metrics['val_auc'],
                "learning_rate": lr,
                "epoch_time_sec": train_metrics['epoch_time'],
            })

            if checkpoint_dir and val_metrics['val_auc'] > best_val_auc:
                best_val_auc = val_metrics['val_auc']
                checkpoint_path = checkpoint_dir / "best_model.safetensors"
                self.model.save_weights(str(checkpoint_path))
                print(f"Saved best model (AUC={best_val_auc:.4f}) to {checkpoint_path}")

        total_training_time = time.time() - training_start
        return {
            "total_time_sec": total_training_time,
            "epoch_times": epoch_times,
            "avg_epoch_time": sum(epoch_times) / len(epoch_times) if epoch_times else 0,
            "best_val_auc": best_val_auc
        }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune whole image classifier on new dataset")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True, help="Pre-trained weights")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Epochs with frozen backbone")
    parser.add_argument("--stage1-lr", type=float, default=1e-4)
    parser.add_argument("--stage1-wd", type=float, default=0.001)
    parser.add_argument("--stage2-lr", type=float, default=1e-5)
    parser.add_argument("--stage2-wd", type=float, default=0.01)
    parser.add_argument("--target-height", type=int, default=896)
    parser.add_argument("--target-width", type=int, default=1152)
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    IMG_DIR = DATA_DIR / "img"
    TRAIN_CSV = DATA_DIR / "train.csv"
    VAL_CSV = DATA_DIR / "val.csv"

    wandb.init(
        project="cm3070-mammography",
        entity="lex",
        name=args.run_name,
        config={
            "task": "whole-image-finetune",
            "source_weights": args.weights,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "stage1_epochs": args.stage1_epochs,
            "stage1_lr": args.stage1_lr,
            "stage1_wd": args.stage1_wd,
            "stage2_lr": args.stage2_lr,
            "stage2_wd": args.stage2_wd,
            "target_height": args.target_height,
            "target_width": args.target_width,
            "backbone": args.backbone,
            "dataset": args.data_dir,
        }
    )

    print("Loading datasets...")
    train_transform = get_train_transform(args.target_height, args.target_width)
    val_transform = get_inference_transform(args.target_height, args.target_width)

    train_dataset = CSVDataset(str(TRAIN_CSV), str(IMG_DIR), train_transform)
    val_dataset = CSVDataset(str(VAL_CSV), str(IMG_DIR), val_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_labels = [label for _, label in train_dataset.samples]
    n_benign = sum(1 for label in train_labels if label == 0)
    n_malignant = sum(1 for label in train_labels if label == 1)
    print(f"Training distribution: Benign={n_benign}, Malignant={n_malignant}")

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("\nCreating whole image classifier...")
    model = create_whole_image_classifier(
        patch_weights_path=None,
        backbone_name=args.backbone,
        num_classes=2
    )

    print(f"Loading pre-trained weights from: {args.weights}")
    model.load_weights(args.weights)

    print("\nStage 1: Fine-tuning top layers only (backbone frozen)")
    model.freeze_backbone()

    optimizer = optim.AdamW(learning_rate=args.stage1_lr, weight_decay=args.stage1_wd)
    trainer = FineTuner(model, optimizer)

    print(f"Stage 1: Epochs 1-{args.stage1_epochs}, LR={args.stage1_lr}")
    print(f"Stage 2: Epochs {args.stage1_epochs+1}-{args.epochs}, LR={args.stage2_lr}")

    training_stats = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        checkpoint_dir=Path("checkpoints") / args.run_name,
        unfreeze_epoch=args.stage1_epochs,
        unfreeze_lr=args.stage2_lr,
        unfreeze_wd=args.stage2_wd
    )

    print("\nFine-tuning complete!")
    print("\n" + "=" * 50)
    print("TRAINING METRICS")
    print("=" * 50)
    total_hours = training_stats['total_time_sec'] / 3600
    print(f"Total time: {training_stats['total_time_sec']:.1f}s ({total_hours:.2f} hours)")
    print(f"Best validation AUC: {training_stats['best_val_auc']:.4f}")
    print("=" * 50)

    print("\nEvaluating on test set...")
    TEST_CSV = DATA_DIR / "test.csv"
    test_dataset = CSVDataset(str(TEST_CSV), str(IMG_DIR), val_transform)
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    best_checkpoint = Path("checkpoints") / args.run_name / "best_model.safetensors"
    if best_checkpoint.exists():
        print(f"Loading best model from {best_checkpoint}")
        model.load_weights(str(best_checkpoint))

    test_metrics = trainer.validate(test_loader)
    print("\nTest Results:")
    print(f"  Loss: {test_metrics['val_loss']:.4f}")
    print(f"  Accuracy: {test_metrics['val_accuracy']:.4f}")
    print(f"  AUC: {test_metrics['val_auc']:.4f}")

    wandb.log({
        "test_loss": test_metrics['val_loss'],
        "test_accuracy": test_metrics['val_accuracy'],
        "test_auc": test_metrics['val_auc'],
    })

    wandb.finish()


if __name__ == "__main__":
    main()
