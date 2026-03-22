"""Inference script for whole image classifier."""

import argparse
import csv
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.display import print_divider, print_results_header, print_section
from src.models.whole_image_classifier import create_whole_image_classifier
from src.transforms import get_inference_transform, get_tta_transforms, preprocess_image


def load_samples(csv_path):
    samples = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((row['filename'], int(row['label'])))
    return samples


def load_and_preprocess(img_path, transform):
    """Load image from path and preprocess for inference."""
    img = Image.open(img_path).convert('RGB')
    img = np.array(img).astype(np.uint8)
    return preprocess_image(img, transform)


def run_inference(model, samples, img_dir, transform, batch_size=2, tta=False):
    model.eval()
    all_probs = []
    all_labels = []
    batch_times = []

    if len(samples) > 0:
        filename, _ = samples[0]
        img = load_and_preprocess(img_dir / filename, transform)
        _ = model(mx.expand_dims(img, 0))
        mx.eval(_)

    total_start = time.time()

    if tta:
        tta_transforms = get_tta_transforms(
            height=transform.transforms[0].height,
            width=transform.transforms[0].width
        )
        for i, (filename, label) in enumerate(samples):
            batch_start = time.time()
            img_path = img_dir / filename

            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img).astype(np.uint8)

            variants = [preprocess_image(img_np, tta_transform) for tta_transform in tta_transforms]

            inputs = mx.stack(variants)
            logits = model(inputs)
            probs = mx.softmax(logits, axis=1)[:, 1]
            mx.eval(probs)

            avg_prob = mx.mean(probs).item()

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            all_probs.append(avg_prob)
            all_labels.append(label)

            print(f"Processed {i + 1}/{len(samples)} (TTA)", end='\r')
    else:
        for i in range(0, len(samples), batch_size):
            batch_start = time.time()
            batch_samples = samples[i:i + batch_size]
            batch_images = []
            batch_labels = []

            for filename, label in batch_samples:
                img_path = img_dir / filename
                img = load_and_preprocess(img_path, transform)
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

    if len(np.unique(labels)) < 2:
        raise ValueError("AUC is undefined: labels must contain both classes.")
    auc = roc_auc_score(labels, probs)

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
    parser.add_argument(
        "--tta", action="store_true",
        help="Enable test-time augmentation (4 variants: original, h-flip, v-flip, both)"
    )
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
    n_benign = sum(1 for label in labels if label == 0)
    n_malignant = sum(1 for label in labels if label == 1)
    print(f"  Benign: {n_benign}, Malignant: {n_malignant}")

    transform = get_inference_transform(args.target_height, args.target_width)

    if args.tta:
        print("\nRunning inference with TTA (4 variants per image)...")
    else:
        print("\nRunning inference...")
    probs, labels, timing_stats = run_inference(
        model, samples, img_dir, transform, batch_size=args.batch_size, tta=args.tta
    )

    metrics = compute_metrics(probs, labels, threshold=args.threshold)

    print_results_header("RESULTS" + (" (with TTA)" if args.tta else ""))
    print(f"AUC:         {metrics['auc']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f} (TPR, Recall)")
    print(f"Specificity: {metrics['specificity']:.4f} (TNR)")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print_divider("-")
    print(f"Confusion Matrix (threshold={args.threshold}):")
    print(f"  TP: {metrics['tp']:4d}  FN: {metrics['fn']:4d}")
    print(f"  FP: {metrics['fp']:4d}  TN: {metrics['tn']:4d}")
    print_divider()

    print_section("INFERENCE COMPUTATIONAL METRICS")
    print(f"Total inference time: {timing_stats['total_time_sec']:.2f}s")
    print(f"Number of samples: {timing_stats['num_samples']}")
    print(f"Average latency per image: {timing_stats['avg_latency_per_image_ms']:.1f}ms")
    print(f"Throughput: {timing_stats['throughput_img_per_sec']:.2f} images/sec")
    peak_mem = mx.get_peak_memory() / (1024**3)
    print(f"Peak memory usage: {peak_mem:.2f} GB")
    print_divider()


if __name__ == "__main__":
    main()
