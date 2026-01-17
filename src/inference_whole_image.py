"""Inference script for whole image classifier."""

from pathlib import Path
import argparse
import csv
import time

import numpy as np
import mlx.core as mx
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix
import albumentations as A

from src.models.whole_image_classifier import create_whole_image_classifier


def get_inference_transform(target_height=896, target_width=1152):
    return A.Compose([
        A.Resize(height=target_height, width=target_width),
    ])


def load_samples(csv_path):
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((row['filename'], int(row['label'])))
    return samples


def preprocess_image(img_path, transform):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img).astype(np.uint8)
    img = transform(image=img)['image']
    img = img.astype(np.float32) / 255.0
    return mx.array(img)


def run_inference(model, samples, img_dir, transform, batch_size=2):
    model.eval()
    all_probs = []
    all_labels = []
    batch_times = []

    if len(samples) > 0:
        filename, _ = samples[0]
        img = preprocess_image(img_dir / filename, transform)
        _ = model(mx.expand_dims(img, 0))
        mx.eval(_)

    total_start = time.time()

    for i in range(0, len(samples), batch_size):
        batch_start = time.time()
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
        probs = mx.softmax(logits, axis=1)[:, 1]
        mx.eval(probs)

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        all_probs.extend(probs.tolist())
        all_labels.extend(batch_labels)

        print(f"Processed {min(i + batch_size, len(samples))}/{len(samples)}", end='\r')

    total_time = time.time() - total_start
    print()

    timing_stats = {
        "total_time_sec": total_time,
        "num_samples": len(samples),
        "avg_latency_per_image_ms": (total_time / len(samples)) * 1000 if samples else 0,
        "throughput_img_per_sec": len(samples) / total_time if total_time > 0 else 0,
        "avg_batch_time_ms": (sum(batch_times) / len(batch_times)) * 1000 if batch_times else 0,
    }

    return np.array(all_probs), np.array(all_labels), timing_stats


def compute_metrics(probs, labels, threshold=0.5):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--target-height", type=int, default=896)
    parser.add_argument("--target-width", type=int, default=1152)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "test.csv"
    img_dir = data_dir / "img"

    print(f"Creating model with backbone: {args.backbone}")
    model = create_whole_image_classifier(
        patch_weights_path=None,
        backbone_name=args.backbone,
        num_classes=2
    )

    print(f"Loading weights: {args.weights}")
    model.load_weights(args.weights)

    print(f"Loading samples from: {csv_path}")
    samples = load_samples(csv_path)
    print(f"Total samples: {len(samples)}")

    labels = [s[1] for s in samples]
    n_benign = sum(1 for l in labels if l == 0)
    n_malignant = sum(1 for l in labels if l == 1)
    print(f"  Benign: {n_benign}, Malignant: {n_malignant}")

    transform = get_inference_transform(args.target_height, args.target_width)

    print("\nRunning inference...")
    probs, labels, timing_stats = run_inference(
        model, samples, img_dir, transform, batch_size=args.batch_size
    )

    metrics = compute_metrics(probs, labels, threshold=args.threshold)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"AUC:         {metrics['auc']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f} (TPR, Recall)")
    print(f"Specificity: {metrics['specificity']:.4f} (TNR)")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print("-" * 50)
    print(f"Confusion Matrix (threshold={args.threshold}):")
    print(f"  TP: {metrics['tp']:4d}  FN: {metrics['fn']:4d}")
    print(f"  FP: {metrics['fp']:4d}  TN: {metrics['tn']:4d}")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("INFERENCE COMPUTATIONAL METRICS")
    print("=" * 50)
    print(f"Total inference time: {timing_stats['total_time_sec']:.2f}s")
    print(f"Number of samples: {timing_stats['num_samples']}")
    print(f"Average latency per image: {timing_stats['avg_latency_per_image_ms']:.1f}ms")
    print(f"Throughput: {timing_stats['throughput_img_per_sec']:.2f} images/sec")
    try:
        peak_mem = mx.get_peak_memory() / (1024**3)
        print(f"Peak memory usage: {peak_mem:.2f} GB")
    except:
        pass
    print("=" * 50)


if __name__ == "__main__":
    main()
