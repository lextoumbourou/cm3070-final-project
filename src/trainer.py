from pathlib import Path
from typing import Dict, Optional
import csv
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlxim.model import create_model
from mlxim.data import DataLoader
from mlxim.data._base import Dataset
from PIL import Image
import numpy as np
import wandb


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
        img = np.array(img).astype(np.float32) / 255.0

        if self.transform:
            img = self.transform(img)

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
        checkpoint_dir: Optional[Path] = None
    ):
        """Train the model for multiple epochs."""
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

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
                "val_accuracy": val_metrics['val_accuracy']
            })

            # Save best model
            if checkpoint_dir and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                checkpoint_path = checkpoint_dir / "best_model.npz"
                self.model.save_weights(str(checkpoint_path))
                print(f"Saved best model to {checkpoint_path}")


def main():
    """Main training script."""
    DATA_DIR = Path("datasets/prep/cbis-ddsm")
    IMG_DIR = DATA_DIR / "img"
    TRAIN_CSV = DATA_DIR / "train.csv"
    VAL_CSV = DATA_DIR / "val.csv"
    TEST_CSV = DATA_DIR / "test.csv"

    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 224
    NUM_CLASSES = 2

    # Initialize wandb
    wandb.init(
        project="cm3070-mammography",
        config={
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "image_size": IMAGE_SIZE,
            "num_classes": NUM_CLASSES,
            "model": "resnet50",
            "optimizer": "adam",
            "dataset": "cbis-ddsm"
        }
    )

    print("Loading datasets...")
    train_dataset = CSVDataset(
        csv_path=str(TRAIN_CSV),
        img_dir=str(IMG_DIR),
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

    model = create_model("efficientnet_b0", num_classes=NUM_CLASSES)

    optimizer = optim.Adam(learning_rate=LEARNING_RATE)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        num_classes=NUM_CLASSES
    )

    print("\nStarting training...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=Path("checkpoints")
    )

    print("\nTraining complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
