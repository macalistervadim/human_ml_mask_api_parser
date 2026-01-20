import os
import tempfile

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2


import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset


# Reuse dataset settings from simple_extractor
dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


def _get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


# Global model cache (load once)
_model = None
_transform = None
_palette = None
_input_size = None


def _ensure_model(model_path: str, dataset: str = "atr"):
    global _model, _transform, _palette, _input_size
    if _model is not None:
        return
    
    num_classes = dataset_settings[dataset]['num_classes']
    _input_size = dataset_settings[dataset]['input_size']
    _palette = _get_palette(num_classes)

    # Set deterministic behavior for consistent results
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)['state_dict']
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}")

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    device = torch.device('cpu')
    model.to(device)
    model.eval()

    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])

    _model = model


def run_parsing_inference(image_bytes: bytes, model_path: str):
    _ensure_model(model_path)

    with tempfile.TemporaryDirectory(prefix="schp_") as tmp_dir:
        img_path = os.path.join(tmp_dir, "input.jpg")
        with open(img_path, "wb") as f:
            f.write(image_bytes)

        dataset = SimpleFolderDataset(
            root=tmp_dir,
            input_size=_input_size,
            transform=_transform,
        )

        image, meta = dataset[0]
        c = meta['center']
        s = meta['scale']
        w = meta['width']
        h = meta['height']

        with torch.no_grad():
            output = _model(image.unsqueeze(0))
            upsample = torch.nn.Upsample(
                size=_input_size,
                mode='bilinear',
                align_corners=True
            )
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze().permute(1, 2, 0)

            logits = transform_logits(
                upsample_output.cpu().numpy(),
                c, s, w, h,
                input_size=_input_size
            )

            parsing = np.argmax(logits, axis=2).astype(np.uint8)

        img = Image.fromarray(parsing, mode="P")
        img.putpalette(_palette)
        return img
