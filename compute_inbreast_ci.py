"""
Compute 95% confidence intervals for INbreast evaluation results.

Runs inference with both the pre-finetune (CBIS-DDSM) and post-finetune
(INbreast fine-tuned) models on the INbreast test set, then reports
bootstrap AUC CIs and Wilson score CIs for sensitivity/specificity.

Source: https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_confint.html
"""

import csv
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np from PIL import Image
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.whole_image_classifier import create_whole_image_classifier
from src.transforms import get_inference_transform, preprocess_image


DATA_DIR = Path(__file__).parent / "datasets/prep/inbreast-whole"
IMG_DIR = DATA_DIR / "img"

MODELS = {
    "before_finetune": Path(__file__).parent / "checkpoints/default/cbis-whole-wd-only/best_model.safetensors",
    "after_finetune":  Path(__file__).parent / "checkpoints/default/inbreast-whole-finetune/best_model.safetensors",
}


def load_samples(csv_path):
    samples = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            samples.append((row["filename"], int(row["label"])))
    return samples


def run_inference(model, samples, transform, batch_size=2):
    model.eval()
    all_probs = []

    # Warm-up
    filename, _ = samples[0]
    img = Image.open(IMG_DIR / filename).convert("RGB")
    img_np = np.array(img).astype(np.uint8)
    dummy = preprocess_image(img_np, transform)
    _ = model(mx.expand_dims(dummy, 0))
    mx.eval(_)

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        imgs = []
        for fname, _ in batch:
            img = Image.open(IMG_DIR / fname).convert("RGB")
            img_np = np.array(img).astype(np.uint8)
            imgs.append(preprocess_image(img_np, transform))
        inputs = mx.stack(imgs)
        logits = model(inputs)
        probs = mx.softmax(logits, axis=1)[:, 1]
        mx.eval(probs)
        all_probs.extend(probs.tolist())
        print(f"  {min(i + batch_size, len(samples))}/{len(samples)}", end="\r")

    print()
    return np.array(all_probs)


def bootstrap_auc_ci(y_true, y_score, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def wilson_ci(k, n):
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return lo, hi


def print_ci_report(name, probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    n_pos = tp + fn
    n_neg = tn + fp

    auc = roc_auc_score(labels, probs)
    sens = tp / n_pos if n_pos > 0 else 0.0
    spec = tn / n_neg if n_neg > 0 else 0.0
    acc  = (tp + tn) / len(labels)

    auc_lo, auc_hi = bootstrap_auc_ci(labels, probs)
    sens_lo, sens_hi = wilson_ci(tp, n_pos)
    spec_lo, spec_hi = wilson_ci(tn, n_neg)

    print(f"## {name}")
    print()
    print(f"n={len(labels)}  malignant={n_pos}  benign={n_neg}")
    print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print()
    print(f"AUC:         {auc:.3f}  95% CI [{auc_lo:.3f}–{auc_hi:.3f}]  (bootstrap n=1000)")
    print(f"Sensitivity: {sens:.3f}  95% CI [{sens_lo:.3f}–{sens_hi:.3f}]  (Wilson, k={tp}/{n_pos})")
    print(f"Specificity: {spec:.3f}  95% CI [{spec_lo:.3f}–{spec_hi:.3f}]  (Wilson, k={tn}/{n_neg})")
    print(f"Accuracy:    {acc:.3f}")
    print()
    


def main():
    samples = load_samples(DATA_DIR / "test.csv")
    labels = np.array([s[1] for s in samples])
    n_mal = labels.sum()
    n_ben = (labels == 0).sum()
    print(f"INbreast test set: {len(samples)} images ({n_mal} malignant, {n_ben} benign)")

    transform = get_inference_transform(896, 1152)
    results = {}

    for key, weights_path in MODELS.items():
        print(f"\nLoading model: {key}")
        print(f"  Weights: {weights_path}")
        model = create_whole_image_classifier(
            patch_weights_path=None,
            backbone_name="resnet50",
            num_classes=2,
        )
        model.load_weights(str(weights_path))
        print("  Running inference...")
        probs = run_inference(model, samples, transform)
        print_ci_report(key, probs, labels)


if __name__ == "__main__":
    main()
