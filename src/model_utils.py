from mlx.utils import tree_flatten


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
    model.freeze()
    head_name = find_head_name(model)
    head = getattr(model, head_name)
    head.unfreeze()
    return model


def freeze_backbone_except_top_n(model, n_layers: int = 46):
    """
    Unfreeze the top N layers of the backbone plus the head.

    Following Shen et al. 2019, for ResNet50 the top 46 layers are unfrozen
    in stage 2 of training. This enables gradual unfreezing from top to bottom.
    """
    model.freeze()

    flat_params = tree_flatten(model.parameters())
    param_names = [name for name, _ in flat_params]
    total_layers = len(param_names)

    layers_to_unfreeze = param_names[-n_layers:] if n_layers < total_layers else param_names

    for name in layers_to_unfreeze:
        parts = name.split('.')
        obj = model
        for part in parts[:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        final_part = parts[-1]
        if final_part.isdigit():
            pass
        elif hasattr(obj, 'unfreeze'):
            obj.unfreeze()

    head_name = find_head_name(model)
    head = getattr(model, head_name)
    head.unfreeze()

    return model
