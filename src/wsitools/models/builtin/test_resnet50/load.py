from typing import Union
import timm
import torch
from torchvision import transforms as T


def load(*, cfg, device: str = "cuda", img_size: Union[int, None] = None, **_):
    # all behavior defined here (source of truth)
    model = timm.create_model("resnet50", pretrained=True, num_classes=0)

    if img_size is None:
        img_size = 224

    emb_dim = 2048
    # dtype = torch.bfloat16
    dtype = None
    meta = {
        "id": cfg.get("id", "test_resnet"),
        "backbone": "resnet50",
        "dynamic_img_size": False,
        "multiple_of": 224,  # we enforce Resize+CenterCrop to this size
        "img_size": img_size,
        "timm_id": "resnet50",
    }

    preprocess = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return model, preprocess, emb_dim, dtype, meta
