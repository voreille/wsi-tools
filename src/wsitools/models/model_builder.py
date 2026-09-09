import timm
import torch
import torch.nn as nn
from torchvision import transforms


def build_tx(
    pixel_mean: tuple[float, float, float],
    pixel_std: tuple[float, float, float],
    tile_size: int,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(tile_size),
            transforms.CenterCrop(tile_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=pixel_mean, std=pixel_std),
        ]
    )


def build_encoder(
    encoder_id: str,
) -> tuple[
    nn.Module,
    tuple[float, float, float],
    tuple[float, float, float],
    int,
    torch.dtype,
]:
    if encoder_id == "uni2-h":
        timm_kwargs = {
            "model_name": "hf-hub:MahmoodLab/UNI2-h",
            "pretrained": True,
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        encoder = timm.create_model(**timm_kwargs)

        embed_dim = 1536
        pixel_mean = encoder.default_cfg["mean"]
        pixel_std = encoder.default_cfg["std"]
        amp_dtype = torch.bfloat16
    elif encoder_id == "h-optimus-1":
        encoder = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        embed_dim = 1536
        pixel_mean = (0.707223, 0.578729, 0.703617)
        pixel_std = (0.211883, 0.230117, 0.177517)
        amp_dtype = torch.float16
    elif encoder_id == "h0-mini":
        encoder = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            dynamic_img_size=True,  # keep this so your hooks work on 448
        )
        embed_dim = getattr(encoder, "embed_dim", 768)
        pixel_mean = encoder.default_cfg["mean"]
        pixel_std = encoder.default_cfg["std"]
        amp_dtype = torch.float16
    else:
        raise ValueError(f"unknown encoder_id {encoder_id}")

    return (
        encoder,
        pixel_mean,
        pixel_std,
        embed_dim,
        amp_dtype,
    )
