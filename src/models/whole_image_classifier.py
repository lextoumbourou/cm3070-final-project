"""
Attempt to implement the whole image part of Shen et al 2019.

Converts a patch classifier to a whole image classifier:

1. Load pretrained patch classifier (ResNet50 trained on ROI patches)
2. Remove classification head, keep convolutional backbone
3. Add VGG-style top layers for spatial aggregation (one of the approaches in paper)
4. Train on full mammograms (1152x896) with binary labels
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vendor" / "mlx-image" / "src"))
from mlxim.model import create_model
from mlxim.model.layers import AdaptiveAvgPool2d


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class TopLayers(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=(256, 128), num_classes=2, dropout=0.5):
        super().__init__()

        self.block1 = VGGBlock(in_channels, hidden_channels[0], pool=True)
        self.block2 = VGGBlock(hidden_channels[0], hidden_channels[1], pool=True)

        self.avgpool = AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels[1], num_classes)

        nn.init.he_uniform(self.block1.conv.weight)
        nn.init.he_uniform(self.block2.conv.weight)
        nn.init.glorot_uniform(self.fc.weight)

    def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class WholeImageClassifier(nn.Module):
    def __init__(self, backbone, top_layers):
        super().__init__()
        self.backbone = backbone
        self.top_layers = top_layers

    def get_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x

    def __call__(self, x):
        features = self.get_features(x)
        logits = self.top_layers(features)
        return logits

    def freeze_backbone(self):
        self.backbone.freeze()
        self.top_layers.unfreeze()

    def unfreeze_all(self):
        self.backbone.unfreeze()
        self.top_layers.unfreeze()


def create_whole_image_classifier(
    patch_weights_path=None,
    backbone_name="resnet50",
    patch_num_classes=5,
    hidden_channels=(256, 128),
    num_classes=2,
    dropout=0.5
):
    backbone = create_model(backbone_name, num_classes=patch_num_classes)

    if patch_weights_path is not None:
        print(f"Loading patch classifier weights from: {patch_weights_path}")
        backbone.load_weights(patch_weights_path)

    # For now, only support Resnet.
    # Todo: support other backbones.
    assert backbone_name == "resnet50"

    # This will be different for different backbones.
    backbone_channels = 2048

    top_layers = TopLayers(
        in_channels=backbone_channels,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        dropout=dropout
    )

    model = WholeImageClassifier(backbone, top_layers)

    return model


def test_forward_pass():
    print("Creating whole image classifier...")
    model = create_whole_image_classifier(
        patch_weights_path=None,
        backbone_name="resnet50",
        patch_num_classes=5
    )

    print("Testing forward pass with 1152x896 input...")
    batch_size = 2
    x = mx.random.normal((batch_size, 896, 1152, 3))

    features = model.get_features(x)
    print(f"Feature map shape: {features.shape}")

    logits = model(x)
    print(f"Output logits shape: {logits.shape}")

    print("Testing forward pass with 224x224 input...")
    x_patch = mx.random.normal((batch_size, 224, 224, 3))
    features_patch = model.get_features(x_patch)
    print(f"Patch feature map shape: {features_patch.shape}")

    logits_patch = model(x_patch)
    print(f"Patch output logits shape: {logits_patch.shape}")

    print("All tests passed!")


if __name__ == "__main__":
    test_forward_pass()
