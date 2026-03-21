"""
Inference script for multiclass (5-class) model on ROI test data.

Aggregates 5-class predictions to binary for comparison with binary ROI models:
  - Benign: class 1 (benign mass) + class 3 (benign calc)
  - Malignant: class 2 (malignant mass) + class 4 (malignant calc)
  - Background (class 0) is included in benign for aggregation

Classes:
  0: Background
  1: Benign mass
  2: Malignant mass
  3: Benign calcification
  4: Malignant calcification
"""

import argparse
import csv
import sys
from pathlib import Path

import albumentations as A
import mlx.core as mx
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "mlx-image" / "src"))
from mlxim.model import create_model

NUM_CLASSES = 5
CLASS_NAMES = ["Background", "Benign mass", "Malignant mass", "Benign calc", "Malignant calc"]

# Mapping for binary aggregation: indices for benign and malignant classes
BENIGN_CLASSES = [0, 1, 3]
MALIGNANT_CLASSES = [2, 4]


def get_inference_transform(output_size: int = 224):
    return A.Compose([
        A.Resize(height=output_size, width=output_size),
    ])


def load_samples(csv_path: str):
    """Load samples from CSV file."""
    samples = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((row['filename'], int(row['label'])))
    return samples


def preprocess_image(img_path: Path, transform):
    """Load and preprocess image using albumentations transform."""
    img = Image.open(img_path).convert('RGB')
    img = np.array(img).astype(np.uint8)
    img = transform(image=img)['image']
    img = img.astype(np.float32) / 255.0
    return mx.array(img)


def run_inference(model, samples, img_dir: Path, image_size: int = 224, batch_size: int = 32):
    """
    Run inference and return both 5-class probabilities and aggregated binary probabilities.
    """
    model.eval()
    transform = get_inference_transform(output_size=image_size)

    all_logits = []
    all_probs_5class = []
    all_probs_binary = []
    all_labels = []
    all_predictions_5class = []

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
        batch_images = []
        batch_labels = []

        for filename, label in batch_samples:
            img_path = img_dir / filename
            img = preprocess_image(img_path, transform)
            batch_images.append(img)
            batch_labels.append(label)

        inputs = mx.stack(batch_images)
        logits = model(inputs)
        probs = mx.softmax(logits, axis=1)

        # 5-class predictions
        predictions_5class = mx.argmax(logits, axis=1)

        # Binary aggregation: P(malignant) = P(class 2) + P(class 4)
        probs_np = np.array(probs)
        malignant_probs = probs_np[:, MALIGNANT_CLASSES].sum(axis=1)

        all_logits.extend(np.array(logits).tolist())
        all_probs_5class.extend(probs_np.tolist())
        all_probs_binary.extend(malignant_probs.tolist())
        all_labels.extend(batch_labels)
        all_predictions_5class.extend(np.array(predictions_5class).tolist())

        print(f"Processed {min(i + batch_size, len(samples))}/{len(samples)}", end='\r')

    print()
    return {
        'logits': np.array(all_logits),
        'probs_5class': np.array(all_probs_5class),
        'probs_binary': np.array(all_probs_binary),
        'labels': np.array(all_labels),
        'predictions_5class': np.array(all_predictions_5class)
    }


def compute_binary_metrics(probs, labels, threshold: float = 0.5):
    """Compute binary classification metrics."""
    predictions = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    return {
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def compute_5class_metrics(predictions, labels_binary, probs_5class):
    """
    Compute 5-class distribution metrics.
    """
    # For benign samples (label=0), what classes did the model predict?
    benign_mask = labels_binary == 0
    malignant_mask = labels_binary == 1

    benign_preds = predictions[benign_mask]
    malignant_preds = predictions[malignant_mask]

    # Count predictions per class
    benign_pred_counts = {CLASS_NAMES[i]: np.sum(benign_preds == i) for i in range(NUM_CLASSES)}
    malignant_pred_counts = {
        CLASS_NAMES[i]: np.sum(malignant_preds == i) for i in range(NUM_CLASSES)
    }

    return {
        'benign_sample_predictions': benign_pred_counts,
        'malignant_sample_predictions': malignant_pred_counts,
        'total_benign': int(np.sum(benign_mask)),
        'total_malignant': int(np.sum(malignant_mask))
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run multiclass model inference on ROI test data and compute binary metrics"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to ROI data directory containing test.csv and img/"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to multiclass model weights (.npz)"
    )
    parser.add_argument(
        "--model-name", type=str, default="efficientnet_b0.mlxim",
        help="Model architecture name"
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="Image size"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold for binary aggregation"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "test.csv"
    img_dir = data_dir / "img"

    print(f"Loading model: {args.model_name} (5-class)")
    model = create_model(args.model_name, num_classes=NUM_CLASSES)

    print(f"Loading weights: {args.weights}")
    model.load_weights(args.weights)

    print(f"Loading samples from: {csv_path}")
    samples = load_samples(csv_path)
    print(f"Total samples: {len(samples)}")

    # Count label distribution
    labels = [s[1] for s in samples]
    n_benign = sum(1 for label in labels if label == 0)
    n_malignant = sum(1 for label in labels if label == 1)
    print(f"  Benign: {n_benign}, Malignant: {n_malignant}")

    print("\nRunning inference...")
    results = run_inference(
        model, samples, img_dir,
        image_size=args.image_size,
        batch_size=args.batch_size
    )

    # Binary metrics (aggregated)
    binary_metrics = compute_binary_metrics(
        results['probs_binary'],
        results['labels'],
        threshold=args.threshold
    )

    # 5-class distribution analysis
    class_metrics = compute_5class_metrics(
        results['predictions_5class'],
        results['labels'],
        results['probs_5class']
    )

    print("\n" + "=" * 60)
    print("BINARY METRICS (Aggregated from 5-class predictions)")
    print("=" * 60)
    print("P(malignant) = P(malignant_mass) + P(malignant_calc)")
    print("-" * 60)
    print(f"AUC:         {binary_metrics['auc']:.4f}")
    print(f"Sensitivity: {binary_metrics['sensitivity']:.4f} (TPR, Recall)")
    print(f"Specificity: {binary_metrics['specificity']:.4f} (TNR)")
    print(f"Accuracy:    {binary_metrics['accuracy']:.4f}")
    print("-" * 60)
    print(f"Confusion Matrix (threshold={args.threshold}):")
    print(f"  TP: {binary_metrics['tp']:4d}  FN: {binary_metrics['fn']:4d}")
    print(f"  FP: {binary_metrics['fp']:4d}  TN: {binary_metrics['tn']:4d}")

    print("\n" + "=" * 60)
    print("5-CLASS PREDICTION DISTRIBUTION")
    print("=" * 60)
    total_benign = class_metrics['total_benign']
    print(f"\nFor BENIGN samples (n={total_benign}):")
    for class_name, count in class_metrics['benign_sample_predictions'].items():
        pct = 100 * count / total_benign if total_benign > 0 else 0
        print(f"  {class_name:20s}: {count:4d} ({pct:5.1f}%)")

    total_malignant = class_metrics['total_malignant']
    print(f"\nFor MALIGNANT samples (n={total_malignant}):")
    for class_name, count in class_metrics['malignant_sample_predictions'].items():
        pct = 100 * count / total_malignant if total_malignant > 0 else 0
        print(f"  {class_name:20s}: {count:4d} ({pct:5.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()
