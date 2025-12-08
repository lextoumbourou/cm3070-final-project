# CM3070 - Final Project

## Prototype

1. Firstly, going to download 2 datasets:
* CBIS-DDSM
* InBreast
2. Then, try to train models using a simple script.
3. Then, perform inference and report on metrics.

## Download datasets

### Download CBIS-DDSM JPEG

```
kaggle datasets download awsaf49/cbis-ddsm-breast-cancer-image-dataset --path datasets/cbis-ddsm-breast-cancer-image-dataset --unzip
```

### Download INBreast

Download from https://drive.google.com/file/d/19n-p9p9C0eCQA1ybm6wkMo-bbeccT_62/view?usp=sharing

## Notebooks

uv run jupyter lab

### 01-cbis-ddsm

EDA notebook for the CBIS-DDSM data.

## Data Processing

The preparation scripts extract ROI crops, resize images to a fixed resolution (256×256), and split the data at patient level to avoid data leakage.

### Process CBIS-DDSM

Run the CBIS-DDSM preparation script with default settings (70/10/20 train/val/test split):

```bash
uv run prepare-cbis
```

## Training

Fine-tune EfficientNet-B0 on the CBIS-DDSM dataset:

```bash
uv run train
```

## Prototype

Train base model on CBIS-DDSM.

```
uv run src/trainer.py --model-name resnet50 --run-name cbis-baseline --data-dir datasets/prep/cbis-ddsm
```

Test the CBIS-DDSM.

```
uv run src/inference.py --model-name resnet50 --weights checkpoints/cbis-baseline/best_model.npz --data-dir datasets/prep/cbis-ddsm
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

