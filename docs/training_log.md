## Prototype

Train base model on CBIS-DDSM.

```
uv run src/trainer.py --model-name resnet50 --run-name cbis-baseline --data-dir datasets/prep/cbis-ddsm
```

Test the CBIS-DDSM.

```
uv run src/inference.py --model-name resnet50 --weights checkpoints/cbis-baseline/best_model.npz --data-dir datasets/prep/cbis-ddsm
==================================================
RESULTS
==================================================
AUC:         0.6866
Sensitivity: 0.4783 (TPR, Recall)
Specificity: 0.7547 (TNR)
Accuracy:    0.6463
--------------------------------------------------
Confusion Matrix (threshold=0.5):
  TP:  132  FN:  144
  FP:  105  TN:  323
==================================================
```

Test InBreast test set.

```
uv run src/inference.py --model-name resnet50 --weights checkpoints/cbis-baseline/best_model.npz --img-dir datasets/prep/inbreast
==================================================
RESULTS
==================================================
AUC:         0.6435
Sensitivity: 0.2000 (TPR, Recall)
Specificity: 0.9783 (TNR)
Accuracy:    0.7869
--------------------------------------------------
Confusion Matrix (threshold=0.5):
  TP:    3  FN:   12
  FP:    1  TN:   45
==================================================
```

Train InBreast train set.

```
uv run src/trainer.py --model-name resnet50 --weights checkpoints/cbis-baseline/best_model.npz --data-dir datasets/prep/inbreast --run-name inbreast-finetune
```

Retest InBreast test set.

```
uv run src/inference.py --model-name resnet50 --weights checkpoints/inbreast-finetune/best_model.npz --data-dir datasets/prep/inbreast
==================================================
RESULTS
==================================================
AUC:         0.9275
Sensitivity: 0.5333 (TPR, Recall)
Specificity: 0.9783 (TNR)
Accuracy:    0.8689
--------------------------------------------------
Confusion Matrix (threshold=0.5):
  TP:    8  FN:    7
  FP:    1  TN:   45
==================================================
```

Compare with InBreast from pure model.

```
uv run src/trainer.py --model-name resnet50 --data-dir datasets/prep/inbreast --run-name inbreast-train
```

```
uv run src/inference.py --model-name resnet50 --weights checkpoints/inbreast-train/best_model.npz --data-dir datasets/prep/inbreast
==================================================
RESULTS
==================================================
AUC:         0.9101
Sensitivity: 0.4000 (TPR, Recall)
Specificity: 1.0000 (TNR)
Accuracy:    0.8525
--------------------------------------------------
Confusion Matrix (threshold=0.5):
  TP:    6  FN:    9
  FP:    0  TN:   46
==================================================
```

## ROI

```
uv run train --model-name resnet50 --run-name cbis-roi-exp1 --data-dir datasets/prep/cbis-ddsm-roi
uv run src/inference.py --model-name resnet50 --weights checkpoints/cbis-roi-exp1/best_model.npz --data-dir datasets/prep/cbis-ddsm-roi
==================================================
RESULTS
==================================================
AUC:         0.7363
Sensitivity: 0.5254 (TPR, Recall)
Specificity: 0.7921 (TNR)
Accuracy:    0.6875
--------------------------------------------------
Confusion Matrix (threshold=0.5):
  TP:  145  FN:  131
  FP:   89  TN:  339
==================================================
```

## Patches Training

```
uv run src/trainer_multiclass.py --model-name resnet50 --run-name cbis-patch-multi --data-dir datasets/prep/cbis-ddsm-patches
Test Results:
  Test Loss: 0.7592
  Test Accuracy: 0.7197
  Per-class accuracy:
    Background: 0.8835
    Benign mass: 0.6195
    Malignant mass: 0.5265
    Benign calc: 0.5315
    Malignant calc: 0.5605
```

### Patch Training on ROI

```
uv run python src/inference_multiclass.py  --data-dir datasets/prep/cbis-ddsm-roi       --weights checkpoints/cbis-patch-multi/best_model.npz       --model-name resnet50
Loading model: resnet50 (5-class)
Downloading weights for resnet50 from HuggingFace Hub.
Loading weights: checkpoints/cbis-patch-multi/best_model.npz
Loading samples from: datasets/prep/cbis-ddsm-roi/test.csv
Total samples: 704
  Benign: 428, Malignant: 276

Running inference...
Processed 704/704

============================================================
BINARY METRICS (Aggregated from 5-class predictions)
============================================================
P(malignant) = P(malignant_mass) + P(malignant_calc)
------------------------------------------------------------
AUC:         0.7463
Sensitivity: 0.6739 (TPR, Recall)
Specificity: 0.6706 (TNR)
Accuracy:    0.6719
------------------------------------------------------------
Confusion Matrix (threshold=0.5):
  TP:  186  FN:   90
  FP:  141  TN:  287

============================================================
5-CLASS PREDICTION DISTRIBUTION
============================================================

For BENIGN samples (n=428):
  Background          :   30 (  7.0%)
  Benign mass         :  141 ( 32.9%)
  Malignant mass      :   62 ( 14.5%)
  Benign calc         :  107 ( 25.0%)
  Malignant calc      :   88 ( 20.6%)

For MALIGNANT samples (n=276):
  Background          :   22 (  8.0%)
  Benign mass         :   36 ( 13.0%)
  Malignant mass      :  102 ( 37.0%)
  Benign calc         :   27 (  9.8%)
  Malignant calc      :   89 ( 32.2%)
============================================================
```

## Whole Image Training using Patch-Backbone

```
uv run python src/trainer_whole_image.py  --run-name cbis-whole-v1 --patch-weights checkpoints/cbis-patch-multi/best_model.npz --backbone resnet50
```

Inference:

```
uv run src/inference_whole_image.py --data-dir datasets/prep/cbis-ddsm-whole --weights checkpoints/cbis-whole-v1/best_model.safetensors
Creating model with backbone: resnet50
Downloading weights for resnet50 from HuggingFace Hub.
Loading weights: checkpoints/cbis-whole-v1/best_model.safetensors
Loading samples from: datasets/prep/cbis-ddsm-whole/test.csv
Total samples: 641
  Benign: 379, Malignant: 262

Running inference...
Processed 641/641

==================================================
RESULTS
==================================================
AUC:         0.7342
Sensitivity: 0.4504 (TPR, Recall)
Specificity: 0.8443 (TNR)
Accuracy:    0.6833
--------------------------------------------------
Confusion Matrix (threshold=0.5):
  TP:  118  FN:  144
  FP:   59  TN:  320
==================================================

==================================================
INFERENCE COMPUTATIONAL METRICS
==================================================
Total inference time: 32.99s
Number of samples: 641
Average latency per image: 51.5ms
Throughput: 19.43 images/sec
Peak memory usage: 1.75 GB
==================================================
```
