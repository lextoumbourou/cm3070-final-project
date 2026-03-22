import argparse
import csv
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlxim.model import create_model
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.display import print_divider, print_results_header
from src.transforms import get_patch_inference_transform, preprocess_image


def load_samples(csv_path: Path):
    samples = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((row['filename'], int(row['label'])))
    return samples


def run_inference(model, samples, img_dir: Path, image_size: int = 224, batch_size: int = 32):
    model.eval()
    transform = get_patch_inference_transform(output_size=image_size)
    all_probs = []
    all_labels = []

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
        batch_images = []
        batch_labels = []

        for filename, label in batch_samples:
            img_path = img_dir / filename
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img).astype(np.uint8)
            batch_images.append(preprocess_image(img_np, transform))
            batch_labels.append(label)

        inputs = mx.stack(batch_images)
        logits = model(inputs)
        probs = mx.softmax(logits, axis=1)[:, 1]

        all_probs.extend(probs.tolist())
        all_labels.extend(batch_labels)

        print(f"Processed {min(i + batch_size, len(samples))}/{len(samples)}", end='\r')

    print()
    return np.array(all_probs), np.array(all_labels)


def compute_metrics(probs, labels, threshold: float = 0.5):
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


def main():
    parser = argparse.ArgumentParser(description="Run inference and compute metrics")
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to data directory containing test.csv and img/"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to model weights (.safetensors)"
    )
    parser.add_argument(
        "--model-name", type=str, default="efficientnet_b0.mlxim",
        help="Model architecture name"
    )
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "test.csv"
    img_dir = data_dir / "img"

    print(f"Loading model: {args.model_name}")
    model = create_model(args.model_name, num_classes=args.num_classes)

    print(f"Loading weights: {args.weights}")
    model.load_weights(args.weights)

    print(f"Loading samples from: {csv_path}")
    samples = load_samples(csv_path)
    print(f"Total samples: {len(samples)}")

    print("Running inference...")
    probs, labels = run_inference(
        model, samples, img_dir,
        image_size=args.image_size,
        batch_size=args.batch_size
    )

    metrics = compute_metrics(probs, labels, threshold=args.threshold)

    print_results_header("RESULTS")
    print(f"AUC:         {metrics['auc']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f} (TPR, Recall)")
    print(f"Specificity: {metrics['specificity']:.4f} (TNR)")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print_divider("-")
    print(f"Confusion Matrix (threshold={args.threshold}):")
    print(f"  TP: {metrics['tp']:4d}  FN: {metrics['fn']:4d}")
    print(f"  FP: {metrics['fp']:4d}  TN: {metrics['tn']:4d}")
    print_divider()


if __name__ == "__main__":
    main()
