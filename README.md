# On-Device Fine-Tuning for Privacy-Preserving Mammography Classification

## CM3070 - Final Project

This is the repository for my final project for my BSc in Computer Science at University of London.

The aim of this project is demonstrate that we can build effective Breast Cancer Mammography classification models on Apple Silicon hardware, to support offline fine-tuning on hospital specific datasets.

The project comprises of two major parts:

1. **Web-based Interface** - A web-based interface to support easy and accessible fine-tuning.
2. **Model Training Scripts** - A series of scripts for training models and baselining different approachs to support our claims.

## Project Requirements

### Git LFS

This repository uses [Git LFS](https://git-lfs.github.com/) to store large files like model weights.

Install Git LFS before cloning:

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Then initialise
git lfs install
```

If you've already cloned without LFS, pull the actual files with:

```bash
git lfs pull
```

### Python Dependencies

We utilise `uv` throughout the project, as it provides one of the best tools for managing Python dependenices.

Uv can be installed following [these](https://docs.astral.sh/uv/getting-started/installation/) instructions.

Then, the project dependencies can be installed with the sync command, as follows:

```bash
uv sync
```

## Web-based Interface

To run the web-based interface, it can be run as follows:

```bash
streamlit run src/app.py
```

The web-interface is comprised of 3 sections, separated by tabs:

1. The **Project Overview** tab, where users select the model and configure training/test data folders.
2. The **Inference** tab, which provides users with the ability to classify Mammography images into benign or malignant, including batch evaluation on test datasets.
3. The **Fine-Tune** tab, which can be used to fine-tune on clinical data.

## Model Training Scripts 

The model training is broken down into a few different categories:

- Downloading datasets.
- Dataset EDA via Juypter notebooks.
- Training and inference scripts.

## Download datasets

I utilised 3 separate Mammography datasets to demonstrate base model training, and then to highlight the problem of domain shift, and show how fine-tuning can recover performance. The datasets are:

* CBIS-DDSM.
* InBreast.
* VinDr-Mammo.

### Download CBIS-DDSM

I choose to use the official CBIS-DDSM dataset.

See `01-cbis-ddsm` for details of how it was downloaded and preprocessed.

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

## Scripts

### Create Datasets for Testing UI

Create small VinDr datasets for UI testing:

```bash
uv run python scripts/create_dataset.py --preset tiny
uv run python scripts/create_dataset.py --preset small
```

Output: `datasets/prep/vindr-ui-tiny/` or `datasets/prep/vindr-ui-small/`

## Testing

Run unit tests:

```bash
uv run python -m pytest tests/
```

### Accessibility Testing

Run the accessibility audit (requires the app's model weights):

```bash
uv run pytest tests/test_accessibility.py -v --no-cov
```

This spins up the Streamlit app and checks each tab for WCAG violations using axe-core.
