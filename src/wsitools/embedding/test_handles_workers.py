import os

from torch.utils.data import DataLoader
from torchvision import transforms as T

from histoseg_plugin.embedding.datasets import WholeSlidePatchH5

tx = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

ds = WholeSlidePatchH5(
    "/home/val/workspaces/histoseg-plugin/output/test/patches/DHMC_0010.h5",
    "/home/val/data/DHMC_LUAD/test_corrected/DHMC_0010.tif",
    tx,
)
loader = DataLoader(ds, batch_size=32, num_workers=2, persistent_workers=True)

for i, batch in enumerate(loader):
    if i < 4:
        print(f"batch {i} with batch size: {batch['img'].shape}",)
