"""
Various content strings for the app.

In future, could support internationalisation using something like gettext.
"""

from dataclasses import dataclass

# Model selection prefixes
VENDOR_MODEL_PREFIX = "📦 "
USER_MODEL_PREFIX = "👤 "

FINETUNE_DESCRIPTION = """
Adapt the model to your local imaging equipment by fine-tuning on your own labeled data.
"""

PROJECT_OVERVIEW_DESCRIPTION = """
Configure your model and datasets here before running inference or fine-tuning.
"""

DATASET_CONFIG_DESCRIPTION = """
Set your training and test data folders here for batch inference and fine-tuning.
"""

FOLDER_STRUCTURE_HELP = """
**Required folder structure:**
```

your_folder/
|-- benign/
|   |-- image1.png
|   |-- ...
|-- malignant/
    |-- image1.png
    |-- ...
```
"""

INFO_SET_TEST_FOLDER = "💡 Set a test folder in **Project Overview** to enable batch evaluation."
INFO_SET_TRAIN_FOLDER = "💡 Set a training folder in **Project Overview** first."

WARNING_LOW_IMAGE_COUNT = (
    "Consider adding more images for better fine-tuning results (recommended: 50+)"
)
WARNING_EXTREME_PREDICTION = (
    "**Extreme prediction detected.** "
    "Please verify this is a valid mammogram image. "
    "Non-mammography images may produce misleading results."
)


@dataclass
class ResultGuidance:

    """Used to stop guidance shown after fine-tuning based on model performance."""

    title: str
    message: str

    def format(self, **kwargs) -> str:
        return self.message.format(**kwargs)


RESULT_GUIDANCE_GOOD = ResultGuidance(
    title="**Good results.** The fine-tuned model shows strong performance on your data.",
    message="""
The model is ready for use. You can now:
- Switch to this model in the Project Overview tab
- Use it for inference on new images
""",
)

RESULT_GUIDANCE_MODERATE = ResultGuidance(
    title="**Reasonable results.** The model shows moderate performance.",
    message="""
**Suggestions to improve:**
- Add more training images (especially for the minority class)
- Try the **Thorough** preset for longer training
- Ensure consistent image quality across your dataset
""",
)

RESULT_GUIDANCE_SUBOPTIMAL = ResultGuidance(
    title="**Suboptimal results.** Performance is below typical clinical thresholds.",
    message="""
**Possible causes:**
- Too few training images (current: {total}, recommended: 50+)
- Class imbalance (benign: {benign}, malignant: {malignant})
- Inconsistent image quality or labelling errors

**Recommendations:**
- Add more labelled images, especially {cases_to_add} cases
- Try the **Thorough** preset
- Review your labels for accuracy
""",
)

RESULT_GUIDANCE_POOR = ResultGuidance(
    title="**Poor results.** The model is performing near random chance.",
    message="""
**This may indicate:**
- Insufficient training data (current: {total} images)
- Severe class imbalance
- Data quality issues or labelling errors
- The base model may not be suitable for your imaging equipment

**Recommendations:**
- Significantly increase your dataset size (aim for 100+ images)
- Balance your classes (similar numbers of benign and malignant)
- Verify all labels are correct
- Try training with the **Thorough** preset
""",
)


def get_result_guidance(auc: float, stats) -> ResultGuidance:
    if auc >= 0.80:
        return RESULT_GUIDANCE_GOOD
    elif auc >= 0.70:
        return RESULT_GUIDANCE_MODERATE
    elif auc >= 0.60:
        return RESULT_GUIDANCE_SUBOPTIMAL
    else:
        return RESULT_GUIDANCE_POOR


def format_result_guidance(auc: float, stats) -> tuple[str, str]:
    """Get formatted title and message for result guidance."""

    guidance = get_result_guidance(auc, stats)

    if auc >= 0.70:
        return guidance.title, guidance.message

    cases_to_add = "malignant" if stats.malignant < stats.benign else "benign"
    formatted_message = guidance.format(
        total=stats.total,
        benign=stats.benign,
        malignant=stats.malignant,
        cases_to_add=cases_to_add,
    )
    return guidance.title, formatted_message
