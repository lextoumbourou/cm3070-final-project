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

EDA notebook for the CBIS-DDSM dataset.

### 02-inbreast

EDA notebook for the INBreast dataset.

### 03-preprocess

Preprocessing pipeline demonstrations.

### 04-patches

Patch extraction experiments.

### 05-vindr

EDA notebook for the VinDr Mammogram dataset.

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

## Testing

Run unit tests:

```bash
uv run python -m pytest tests/
```
