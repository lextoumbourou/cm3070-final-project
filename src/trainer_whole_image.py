"""Whole image trainer following Shen et al. (2019) two-stage approach."""

import argparse
import csv
import time
from pathlib import Path

import albumentations as A
import cv2
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlxim.data import DataLoader
from mlxim.data._base import Dataset
from PIL import Image

import wandb
from src.models.whole_image_classifier import create_whole_image_classifier

NUM_CLASSES = 2


def get_train_transform(target_height=896, target_width=1152):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=25, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.2, p=0.5),
        A.Resize(height=target_height, width=target_width),
    ])


def get_val_transform(target_height=896, target_width=1152):
    return A.Compose([
        A.Resize(height=target_height, width=target_width),
    ])


class WholeImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.samples = []

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row['filename'], int(row['label'])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = self.img_dir / filename

        img = Image.open(img_path).convert('RGB')
        img = np.array(img).astype(np.uint8)

        if self.transform:
            img = self.transform(image=img)['image']

        img = img.astype(np.float32) / 255.0
        return mx.array(img), mx.array(label)


class WholeImageTrainer:
    def __init__(self, model, optimizer, num_classes=NUM_CLASSES):
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes

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
            "probs": all_probs,
            "targets": all_targets
        }

    def fit(self, train_loader, val_loader, num_epochs, checkpoint_dir=None,
            stage2_epoch=None, stage2_lr=None, stage2_weight_decay=None):
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_auc = 0.0
        training_start = time.time()
        epoch_times = []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            if stage2_epoch is not None and epoch == stage2_epoch:
                print(
                    f"\n>>> Stage 2: Unfreezing backbone, LR={stage2_lr}, "
                    f"weight_decay={stage2_weight_decay}"
                )
                self.model.unfreeze_all()
                wd = stage2_weight_decay or 0.0
                self.optimizer = optim.AdamW(learning_rate=stage2_lr, weight_decay=wd)

            train_metrics = self.train_epoch(train_loader, epoch)
            epoch_times.append(train_metrics['epoch_time'])
            loss, acc = train_metrics['loss'], train_metrics['accuracy']
            print(f"Training - Loss: {loss:.4f}, Acc: {acc:.4f}")
            epoch_time = train_metrics['epoch_time']
            throughput = train_metrics['throughput']
            print(f"  Epoch time: {epoch_time:.1f}s, Throughput: {throughput:.2f} img/s")

            val_metrics = self.validate(val_loader)
            v_loss = val_metrics['val_loss']
            v_acc, v_auc = val_metrics['val_accuracy'], val_metrics['val_auc']
            print(f"Validation - Loss: {v_loss:.4f}, Acc: {v_acc:.4f}, AUC: {v_auc:.4f}")

            peak_memory_gb = mx.get_peak_memory() / (1024**3)

            lr = self.optimizer.learning_rate
            if hasattr(lr, 'item'):
                lr = lr.item()

            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_metrics['loss'],
                "train_accuracy": train_metrics['accuracy'],
                "val_loss": val_metrics['val_loss'],
                "val_accuracy": val_metrics['val_accuracy'],
                "val_auc": val_metrics['val_auc'],
                "learning_rate": lr,
                "epoch_time_sec": train_metrics['epoch_time'],
                "throughput_img_per_sec": train_metrics['throughput'],
            }
            if peak_memory_gb is not None:
                log_dict["peak_memory_gb"] = peak_memory_gb
            wandb.log(log_dict)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="datasets/prep/cbis-ddsm-whole")
    parser.add_argument("--patch-weights", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage1-lr", type=float, default=1e-4)
    parser.add_argument("--stage1-weight-decay", type=float, default=0.001)
    parser.add_argument("--stage2-lr", type=float, default=1e-5)
    parser.add_argument("--stage2-weight-decay", type=float, default=0.01)
    parser.add_argument("--target-height", type=int, default=896)
    parser.add_argument("--target-width", type=int, default=1152)
    parser.add_argument("--train-csv", type=str, default="train.csv")
    parser.add_argument("--val-csv", type=str, default="val.csv")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    IMG_DIR = DATA_DIR / "img"
    TRAIN_CSV = DATA_DIR / args.train_csv
    VAL_CSV = DATA_DIR / args.val_csv

    wandb.init(
        project="cm3070-mammography",
        entity="lex",
        name=args.run_name,
        config={
            "task": "whole-image-classification",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "stage1_epochs": args.stage1_epochs,
            "stage1_lr": args.stage1_lr,
            "stage1_weight_decay": args.stage1_weight_decay,
            "stage2_lr": args.stage2_lr,
            "stage2_weight_decay": args.stage2_weight_decay,
            "target_height": args.target_height,
            "target_width": args.target_width,
            "backbone": args.backbone,
            "patch_weights": args.patch_weights,
            "optimizer": "adam",
            "dataset": args.data_dir,
        }
    )

    print("Loading datasets...")
    train_transform = get_train_transform(args.target_height, args.target_width)
    val_transform = get_val_transform(args.target_height, args.target_width)

    train_dataset = WholeImageDataset(str(TRAIN_CSV), str(IMG_DIR), train_transform)
    val_dataset = WholeImageDataset(str(VAL_CSV), str(IMG_DIR), val_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_labels = [label for _, label in train_dataset.samples]
    n_benign = sum(1 for label in train_labels if label == 0)
    n_malignant = sum(1 for label in train_labels if label == 1)
    print(f"Training class distribution: Benign={n_benign}, Malignant={n_malignant}")

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("\nCreating whole image classifier...")
    print(f"  Backbone: {args.backbone}")
    print(f"  Patch weights: {args.patch_weights}")

    model = create_whole_image_classifier(
        patch_weights_path=args.patch_weights,
        backbone_name=args.backbone,
        patch_num_classes=5,
        num_classes=2
    )

    print("\nStage 1: Training top layers only (backbone frozen)")
    model.freeze_backbone()

    optimizer = optim.AdamW(learning_rate=args.stage1_lr, weight_decay=args.stage1_weight_decay)
    trainer = WholeImageTrainer(model, optimizer, NUM_CLASSES)

    print(
        f"Stage 1: Epochs 1-{args.stage1_epochs}, "
        f"LR={args.stage1_lr}, weight_decay={args.stage1_weight_decay}"
    )
    print(
        f"Stage 2: Epochs {args.stage1_epochs+1}-{args.epochs}, "
        f"LR={args.stage2_lr}, weight_decay={args.stage2_weight_decay}"
    )

    training_stats = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        checkpoint_dir=Path("checkpoints") / args.run_name,
        stage2_epoch=args.stage1_epochs,
        stage2_lr=args.stage2_lr,
        stage2_weight_decay=args.stage2_weight_decay
    )

    print("\nTraining complete!")
    print("\n" + "=" * 50)
    print("TRAINING COMPUTATIONAL METRICS")
    print("=" * 50)
    total_hours = training_stats['total_time_sec'] / 3600
    print(f"Total training time: {training_stats['total_time_sec']:.1f}s ({total_hours:.2f} hours)")
    print(f"Average epoch time: {training_stats['avg_epoch_time']:.1f}s")
    print(f"Best validation AUC: {training_stats['best_val_auc']:.4f}")
    peak_mem = mx.get_peak_memory() / (1024**3)
    print(f"Peak memory usage: {peak_mem:.2f} GB")
    print("=" * 50)

    print("\nEvaluating on test set...")
    TEST_CSV = DATA_DIR / "test.csv"
    test_dataset = WholeImageDataset(str(TEST_CSV), str(IMG_DIR), val_transform)
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

    final_log = {
        "test_loss": test_metrics['val_loss'],
        "test_accuracy": test_metrics['val_accuracy'],
        "test_auc": test_metrics['val_auc'],
        "total_training_time_sec": training_stats['total_time_sec'],
        "total_training_time_hours": training_stats['total_time_sec'] / 3600,
        "avg_epoch_time_sec": training_stats['avg_epoch_time'],
    }
    final_log["peak_memory_gb"] = mx.get_peak_memory() / (1024**3)
    wandb.log(final_log)

    wandb.finish()


if __name__ == "__main__":
    main()
