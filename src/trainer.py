from pathlib import Path
from typing import Dict, Optional
import argparse
import csv
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import sys
# Use vendored mlx-image to path for debugging
# (I may remove this one I figure out this nan issue)
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "mlx-image" / "src"))
from mlxim.model import create_model
from mlxim.data import DataLoader
from mlxim.data._base import Dataset
from src.model_utils import freeze_backbone

from PIL import Image
import numpy as np
import cv2
import albumentations as A
import wandb


def get_train_transform(image_size: int = 512):
    """Get training augmentation pipeline."""
    return A.Compose([
        # Crops
        A.RandomSizedCrop(min_max_height=(256, 480), size=(image_size, image_size), p=0.4),

        # Flips
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Downscale - simulates lower resolution acquisitions
        A.OneOf([
            A.Downscale(scale_range=(0.75, 0.95), interpolation_pair={'downscale': cv2.INTER_AREA, 'upscale': cv2.INTER_LINEAR}, p=0.1),
            A.Downscale(scale_range=(0.75, 0.95), interpolation_pair={'downscale': cv2.INTER_AREA, 'upscale': cv2.INTER_LANCZOS4}, p=0.1),
            A.Downscale(scale_range=(0.75, 0.95), interpolation_pair={'downscale': cv2.INTER_LINEAR, 'upscale': cv2.INTER_LINEAR}, p=0.8),
        ], p=0.125),

        # Contrast - simulates exposure variations
        A.OneOf([
            A.RandomToneCurve(scale=0.3, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.4, 0.5),
            brightness_by_max=True, p=0.5)
        ], p=0.5),

        # Geometric - simulates positioning variations
        A.OneOf([
            A.Affine(
                scale=(0.85, 1.15), rotate=(-30, 30),
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.2, 0.2)},
                border_mode=cv2.BORDER_CONSTANT,
                p=0.6
            ),
            A.ElasticTransform(
                alpha=1, sigma=20, interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT, approximate=False, p=0.2
            ),
            A.GridDistortion(
                num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT, normalized=True, p=0.2
            ),
        ], p=0.5),
    ], p=0.9)


class CSVDataset(Dataset):
    """Dataset that loads images from CSV file with filename and label columns."""

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform=None,
        image_size: int = 224
    ):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.image_size = image_size
        self.samples = []

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                label = int(row['label'])
                self.samples.append((filename, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = self.img_dir / filename

        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size))
        img = np.array(img).astype(np.uint8)

        # Apply albumentations transform (expects uint8, returns uint8)
        if self.transform:
            img = self.transform(image=img)['image']

        # Normalize to float32 [0, 1]
        img = img.astype(np.float32) / 255.0

        img = mx.array(img)
        label = mx.array(label)

        return img, label


class Trainer:
    """Trainer for fine-tuning models with mlx-image."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        num_classes: int = 2
    ):
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes

    def loss_fn(self, model, inputs, targets):
        logits = model(inputs)
        loss = mx.mean(nn.losses.cross_entropy(logits, targets))
        return loss

    def train_step(self, inputs, targets):
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        loss, grads = loss_and_grad_fn(self.model, inputs, targets)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        return loss

    def compute_accuracy(self, logits, targets):
        predictions = mx.argmax(logits, axis=1)
        correct = mx.sum(predictions == targets)
        accuracy = correct / len(targets)
        return accuracy

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            loss = self.train_step(inputs, targets)

            # Accumulate metrics
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

    def validate(self, val_loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, targets in val_loader:
            logits = self.model(inputs)
            loss = mx.mean(nn.losses.cross_entropy(logits, targets))

            predictions = mx.argmax(logits, axis=1)
            correct = mx.sum(predictions == targets)

            batch_size = len(targets)
            total_loss += loss.item() * batch_size
            total_correct += correct.item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy
        }

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        checkpoint_dir: Optional[Path] = None,
        unfreeze_epoch: Optional[int] = None,
        unfreeze_lr: Optional[float] = None
    ):
        """Train the model for multiple epochs."""
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Unfreeze and lower learning rate if specified
            if unfreeze_epoch is not None and epoch == unfreeze_epoch:
                print(f"\n>>> Unfreezing backbone and setting LR to {unfreeze_lr}")
                self.model.unfreeze()
                # Reinitialize optimizer with new learning rate for all parameters
                self.optimizer = optim.Adam(learning_rate=unfreeze_lr)

            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"Training Loss: {train_metrics['loss']:.4f}")

            # Validation
            val_metrics = self.validate(val_loader)
            print(f"Validation Loss: {val_metrics['val_loss']:.4f}")
            print(f"Validation Accuracy: {val_metrics['val_accuracy']:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_metrics['loss'],
                "val_loss": val_metrics['val_loss'],
                "val_accuracy": val_metrics['val_accuracy'],
                "learning_rate": self.optimizer.learning_rate.item() if hasattr(self.optimizer.learning_rate, 'item') else self.optimizer.learning_rate
            })

            # Save best model
            if checkpoint_dir and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                checkpoint_path = checkpoint_dir / "best_model.npz"
                self.model.save_weights(str(checkpoint_path))
                print(f"Saved best model to {checkpoint_path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train a model on CBIS-DDSM dataset")
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
        "--weights",
        type=str,
        help="Path to pretrained weights (.npz) to load before training",
        default=None
    )
    args = parser.parse_args()

    DATA_DIR = Path("datasets/prep/cbis-ddsm")
    IMG_DIR = DATA_DIR / "img"
    TRAIN_CSV = DATA_DIR / "train.csv"
    VAL_CSV = DATA_DIR / "val.csv"
    TEST_CSV = DATA_DIR / "test.csv"

    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    # Unfreeze after this many epochs (0-indexed)
    UNFREEZE_EPOCH = 2
    # Lower LR for fine-tuning
    UNFREEZE_LR = 1e-5
    IMAGE_SIZE = 224
    NUM_CLASSES = 2
    MODEL_NAME = args.model_name

    # Initialize wandb
    wandb.init(
        project="cm3070-mammography",
        name=args.run_name,
        config={
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "unfreeze_epoch": UNFREEZE_EPOCH,
            "unfreeze_lr": UNFREEZE_LR,
            "image_size": IMAGE_SIZE,
            "num_classes": NUM_CLASSES,
            "model": MODEL_NAME,
            "optimizer": "adam",
            "dataset": "cbis-ddsm",
            "frozen_backbone": True,
            "augmentation": True,
            "pretrained_weights": args.weights,
            "run_name": args.run_name
        }
    )

    print("Loading datasets...")
    train_transform = get_train_transform(image_size=IMAGE_SIZE)
    train_dataset = CSVDataset(
        csv_path=str(TRAIN_CSV),
        img_dir=str(IMG_DIR),
        transform=train_transform,
        image_size=IMAGE_SIZE
    )
    val_dataset = CSVDataset(
        csv_path=str(VAL_CSV),
        img_dir=str(IMG_DIR),
        image_size=IMAGE_SIZE
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

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

    optimizer = optim.Adam(learning_rate=LEARNING_RATE)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        num_classes=NUM_CLASSES
    )

    print("\nStarting training...")
    print(f"Phase 1 (epochs 1-{UNFREEZE_EPOCH}): Training head only with LR={LEARNING_RATE}")
    print(f"Phase 2 (epochs {UNFREEZE_EPOCH+1}-{NUM_EPOCHS}): Fine-tuning entire model with LR={UNFREEZE_LR}")

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=Path("checkpoints") / args.run_name,
        unfreeze_epoch=UNFREEZE_EPOCH,
        unfreeze_lr=UNFREEZE_LR
    )

    print("\nTraining complete!")

    print("\nRunning inference on test set...")
    test_dataset = CSVDataset(
        csv_path=str(TEST_CSV),
        img_dir=str(IMG_DIR),
        image_size=IMAGE_SIZE
    )
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    best_checkpoint = Path("checkpoints") / "best_model.npz"
    if best_checkpoint.exists():
        print(f"Loading best model from {best_checkpoint}")
        model.load_weights(str(best_checkpoint))

    test_metrics = trainer.validate(test_loader)
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_metrics['val_loss']:.4f}")
    print(f"  Test Accuracy: {test_metrics['val_accuracy']:.4f}")

    wandb.log({
        "test_loss": test_metrics['val_loss'],
        "test_accuracy": test_metrics['val_accuracy']
    })

    wandb.finish()


if __name__ == "__main__":
    main()
