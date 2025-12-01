import os
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


def find_head_name(model):
    """Find the name of the classification head in the model."""
    if hasattr(model, 'classifier'):
        return 'classifier'
    elif hasattr(model, 'head'):
        return 'head'
    elif hasattr(model, 'fc'):
        return 'fc'
    else:
        raise ValueError("Could not find classification head in model")


def freeze_backbone(model):
    """Freeze the backbone/body parameters, only train the head."""
    # Freeze the entire model first
    model.freeze()
    # Then unfreeze only the head
    head_name = find_head_name(model)
    head = getattr(model, head_name)
    head.unfreeze()
    return model
