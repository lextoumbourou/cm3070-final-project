# Training Log

## Summary

| Run | Dataset | Test AUC | Test Acc | Val AUC | Training Time | W&B |
|-----|---------|----------|----------|---------|---------------|-----|
| **cbis-whole-wd-only** | CBIS-DDSM | 0.735 | 0.655 | 0.835 | 7.89h | [link](https://wandb.ai/lex/cm3070-mammography/runs/i4qjands) |
| cbis-whole-final | CBIS-DDSM (train+val) | 0.737 | 0.674 | 0.714 | 10.08h | [link](https://wandb.ai/lex/cm3070-mammography/runs/hslysoda) |
| inbreast-whole-finetune | INbreast | 0.926 | 0.885 | 0.808 | 0.64h | [link](https://wandb.ai/lex/cm3070-mammography/runs/r4u682dq) |
| vindr-balanced-finetune | VinDr | 0.807 | 0.735 | 0.805 | 2.68h | [link](https://wandb.ai/lex/cm3070-mammography/runs/vp94ji8v) |

**Bold** = default model shipped with app.

---

## Base Model Training (CBIS-DDSM)

### cbis-whole-wd-only (Default Model)

**Objective:** Train whole-image classifier with weight decay regularisation.

**W&B:** https://wandb.ai/lex/cm3070-mammography/runs/i4qjands

```bash
uv run python src/trainer_whole_image.py \
    --run-name cbis-whole-wd-only \
    --data-dir datasets/prep/cbis-ddsm-whole \
    --patch-weights checkpoints/default/cbis-patch-multi/best_model.npz \
    --backbone resnet50 \
    --stage1-weight-decay 0.001 \
    --stage2-weight-decay 0.01
```

**Weights:** `checkpoints/default/cbis-whole-wd-only/best_model.safetensors`

| Metric | Value |
|--------|-------|
| Test AUC | 0.735 |
| Test AUC (TTA) | 0.745 |
| Val AUC | 0.835 |
| Test Accuracy | 0.655 |
| Sensitivity | 0.752 (TTA) |
| Specificity | 0.610 (TTA) |
| Training Time | 7.89h |
| Peak Memory | 9.50 GB |

**Notes:** Best validation AUC. Selected as default base model for fine-tuning.

---

### cbis-whole-final

**Objective:** Train on combined train+val set for maximum data utilization.

**W&B:** https://wandb.ai/lex/cm3070-mammography/runs/hslysoda

```bash
uv run python src/trainer_whole_image.py \
    --run-name cbis-whole-final \
    --data-dir datasets/prep/cbis-ddsm-whole \
    --patch-weights checkpoints/user/cbis-patch-multi/best_model.npz \
    --backbone resnet50 \
    --stage1-weight-decay 0.001 \
    --stage2-weight-decay 0.01
```

**Weights:** `checkpoints/default/cbis-whole-final/best_model.safetensors`

| Metric | Value |
|--------|-------|
| Test AUC | 0.737 |
| Val AUC | 0.714 |
| Test Accuracy | 0.674 |
| Training Time | 10.08h |

**Notes:** Slightly higher test AUC but used test-set for model selection. Not recommended as base.

---

### cbis-whole-v1

**Objective:** Initial whole-image training attempt.

**W&B:** https://wandb.ai/lex/cm3070-mammography/runs/rnc6n0zv

```bash
uv run python src/trainer_whole_image.py \
    --run-name cbis-whole-v1 \
    --patch-weights checkpoints/user/cbis-patch-multi/best_model.npz \
    --backbone resnet50
```

| Metric | Value |
|--------|-------|
| Test AUC | 0.734 |
| Test AUC (TTA) | 0.745 |
| Val AUC | 0.763 |
| Test Accuracy | 0.683 |
| Training Time | 8.65h |

---

### cbis-whole-shen-wd

**Objective:** Use Shen schedule patch weights with weight decay.

**W&B:** https://wandb.ai/lex/cm3070-mammography/runs/sy8uwh4j

| Metric | Value |
|--------|-------|
| Test AUC | 0.726 |
| Test AUC (TTA) | 0.739 |
| Val AUC | 0.790 |
| Training Time | 8.15h |

**Notes:** Shen-schedule patch weights performed worse than simpler training.

---

## Fine-tuning Experiments

### inbreast-whole-finetune

**Objective:** Fine-tune base model on INbreast dataset (Portuguese FFDM).

**W&B:** https://wandb.ai/lex/cm3070-mammography/runs/r4u682dq

```bash
uv run python src/finetune_whole_image.py \
    --run-name inbreast-whole-finetune \
    --data-dir datasets/prep/inbreast-whole \
    --weights checkpoints/default/cbis-whole-wd-only/best_model.safetensors \
    --backbone resnet50 \
    --epochs 20
```

**Weights:** `checkpoints/default/inbreast-whole-finetune/best_model.safetensors`

| Metric | Value |
|--------|-------|
| Test AUC | 0.926 |
| Val AUC | 0.808 |
| Test Accuracy | 0.885 |
| Training Time | 0.64h (38 min) |

**Notes:** Strong domain adaptation. Demonstrates fine-tuning effectiveness.

---

### vindr-balanced-finetune

**Objective:** Fine-tune on VinDr-Mammo dataset (Vietnamese hospitals).

**W&B:** https://wandb.ai/lex/cm3070-mammography/runs/vp94ji8v

```bash
uv run python src/finetune_whole_image.py \
    --run-name vindr-balanced-finetune \
    --data-dir datasets/prep/vindr-whole-balanced \
    --weights checkpoints/default/cbis-whole-wd-only/best_model.safetensors \
    --backbone resnet50 \
    --epochs 20
```

**Weights:** `checkpoints/default/vindr-balanced-finetune/best_model.safetensors`

| Metric | Value |
|--------|-------|
| Test AUC | 0.807 |
| Val AUC | 0.805 |
| Test Accuracy | 0.735 |
| Training Time | 2.68h |

---

## Patch Classifier (Backbone Pre-training)

### cbis-patch-multi

**Objective:** Train 5-class patch classifier for backbone initialization.

**W&B:** https://wandb.ai/lex/cm3070-mammography/runs/m956saga

```bash
uv run src/trainer_multiclass.py \
    --model-name resnet50 \
    --run-name cbis-patch-multi \
    --data-dir datasets/prep/cbis-ddsm-patches
```

| Metric | Value |
|--------|-------|
| Test Accuracy | 0.720 |
| Background Acc | 0.884 |
| Benign Mass Acc | 0.620 |
| Malignant Mass Acc | 0.527 |
| Benign Calc Acc | 0.532 |
| Malignant Calc Acc | 0.561 |
| Training Time | 1.78h |

**Notes:** Used as patch weights for whole-image training.

---

## Early Prototype Experiments

These experiments used the simpler ROI-based approach before implementing Shen et al.

### cbis-baseline

**W&B:** https://wandb.ai/lex/cm3070-mammography/runs/8a2752sq

| Dataset | AUC | Sensitivity | Specificity |
|---------|-----|-------------|-------------|
| CBIS-DDSM | 0.687 | 0.478 | 0.755 |
| INbreast | 0.644 | 0.200 | 0.978 |

### inbreast-finetune (ROI)

**W&B:** https://wandb.ai/lex/cm3070-mammography/runs/sw5jgbb7

| Metric | Before | After |
|--------|--------|-------|
| AUC | 0.644 | 0.928 |
| Sensitivity | 0.200 | 0.533 |
| Accuracy | 0.787 | 0.869 |

**Notes:** Early demonstration that fine-tuning recovers performance on domain-shifted data.
