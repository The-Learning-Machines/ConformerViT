## Conformer-ViT

This repository extends [Vision Transformers](https://github.com/lucidrains/vit-pytorch) and [Conformers](https://github.com/lucidrains/conformer) for Image Classification and Image2Seq tasks such as OCR and Captioning.

### Usage

```python
from conformer_vit import ConformerViTForClassification
import torch

model = ConformerViTForClassification(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=144,
    depth=12,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

img = torch.randn(1, 3, 256, 256)

preds = model(img)  # (1, 1000)
```
