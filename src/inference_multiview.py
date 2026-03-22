"""Multi-view fusion inference - average CC and MLO predictions per breast."""

import argparse
import csv
from pathlib import Path

import albumentations as A
import mlx.core as mx
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm

from src.display import print_divider, print_section
from src.models.whole_image_classifier import create_whole_image_classifier


def get_transform(height, width):
    return A.Compose([A.Resize(height=height, width=width)])


def get_tta_transforms(height, width):
    base = A.Resize(height=height, width=width)
    return [
        A.Compose([base]),
        A.Compose([base, A.HorizontalFlip(p=1.0)]),
        A.Compose([base, A.VerticalFlip(p=1.0)]),
        A.Compose([base, A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
    ]


def load_image(path, transform):
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.uint8)
    img = transform(image=img)['image']
    img = img.astype(np.float32) / 255.0
    return mx.array(img)


def predict(model, img):
    inputs = mx.expand_dims(img, 0)
    logits = model(inputs)
    probs = mx.softmax(logits, axis=1)
    mx.eval(probs)
    return probs[0, 1].item()


def predict_tta(model, img_path, transforms):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img).astype(np.uint8)

    probs = []
    for t in transforms:
        aug_img = t(image=img)['image']
        aug_img = aug_img.astype(np.float32) / 255.0
        aug_img = mx.array(aug_img)
        prob = predict(model, aug_img)
        probs.append(prob)

    return np.mean(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--csv", type=str, default="test.csv")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--target-height", type=int, default=896)
    parser.add_argument("--target-width", type=int, default=1152)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    img_dir = data_dir / "img"
    csv_path = data_dir / args.csv

    print("Creating model with backbone: resnet50")
    model = create_whole_image_classifier(
        patch_weights_path=None,
        backbone_name="resnet50",
        num_classes=2
    )
    print(f"Loading weights: {args.weights}")
    model.load_weights(args.weights)
    model.eval()

    # Load samples
    samples = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)

    print(f"Total images: {len(samples)}")

    # Get predictions per image
    if args.tta:
        transforms = get_tta_transforms(args.target_height, args.target_width)
        print("Running inference with TTA...")
    else:
        transform = get_transform(args.target_height, args.target_width)
        print("Running inference...")

    image_preds = {}
    for sample in tqdm(samples):
        img_path = img_dir / sample['filename']
        key = (sample['patient_id'], sample['breast_side'], sample['image_view'])

        if args.tta:
            prob = predict_tta(model, img_path, transforms)
        else:
            img = load_image(img_path, transform)
            prob = predict(model, img)

        image_preds[key] = {
            'prob': prob,
            'label': int(sample['label']),
            'patient_id': sample['patient_id'],
            'breast_side': sample['breast_side'],
            'view': sample['image_view']
        }

    # Group by breast (patient_id, breast_side)
    breast_preds = {}
    for key, pred in image_preds.items():
        patient_id, breast_side, view = key
        breast_key = (patient_id, breast_side)

        if breast_key not in breast_preds:
            breast_preds[breast_key] = {
                'probs': [],
                'label': pred['label']
            }
        breast_preds[breast_key]['probs'].append(pred['prob'])

    # Average predictions per breast
    y_true = []
    y_prob_avg = []
    y_prob_max = []

    for _breast_key, data in breast_preds.items():
        y_true.append(data['label'])
        y_prob_avg.append(np.mean(data['probs']))
        y_prob_max.append(np.max(data['probs']))

    y_true = np.array(y_true)
    y_prob_avg = np.array(y_prob_avg)
    y_prob_max = np.array(y_prob_max)
    y_pred_avg = (y_prob_avg >= 0.5).astype(int)

    # Metrics
    auc_avg = roc_auc_score(y_true, y_prob_avg)
    auc_max = roc_auc_score(y_true, y_prob_max)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_avg).ravel()
    sens_avg = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec_avg = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc_avg = (tp + tn) / len(y_true)

    print_section("IMAGE-LEVEL RESULTS")
    print(f"Images: {len(samples)}")

    # Recalculate image-level AUC
    img_probs = [image_preds[k]['prob'] for k in image_preds]
    img_labels = [image_preds[k]['label'] for k in image_preds]
    img_auc = roc_auc_score(img_labels, img_probs)
    print(f"AUC (image-level): {img_auc:.4f}")

    print_section("BREAST-LEVEL RESULTS (Multi-view Fusion)")
    print(f"Breasts: {len(breast_preds)}")
    print(f"  Benign: {sum(1 for label in y_true if label == 0)}")
    print(f"  Malignant: {sum(1 for label in y_true if label == 1)}")
    print_divider("-")
    print(f"AUC (avg fusion):  {auc_avg:.4f}")
    print(f"AUC (max fusion):  {auc_max:.4f}")
    print(f"Sensitivity (avg): {sens_avg:.4f}")
    print(f"Specificity (avg): {spec_avg:.4f}")
    print(f"Accuracy (avg):    {acc_avg:.4f}")
    print_divider("-")
    print("Confusion Matrix (avg, threshold=0.5):")
    print(f"  TP: {tp:4d}  FN: {fn:4d}")
    print(f"  FP: {fp:4d}  TN: {tn:4d}")
    print_divider()


if __name__ == "__main__":
    main()
