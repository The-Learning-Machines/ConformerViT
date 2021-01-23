## Conformer-ViT

This repository extends [Vision Transformers](https://github.com/lucidrains/vit-pytorch) and [Conformers](https://github.com/lucidrains/conformer) for Image Classification and Image2Seq tasks such as OCR and Captioning.

### Usage - Image2Seq

```python
from conformer_vit import ConformerViTForImage2Seq
import torch

model = ConformerViTForImage2Seq(
    image_size=256,
    patch_size=16,
    num_classes=150,
    dim=320,
    depth=12,
    heads=8,
    mlp_dim=1024,
    decoder_dim=640,
    output_seq_len=128,
    SOS_token=1,
    EOS_token=2,
    channels=1,
    dropout=0.1,
    emb_dropout=0.1,
    kernel_size=17,
    causal=False
)

inp = torch.randn(1, 1, 64, 256)
target_seq = torch.randint(0, 150, (1, 128))
pred = model(inp, target_seq=target_seq, teacher_forcing_ratio=0.5)
print(pred.shape)
```

### Usage - Image Classification

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

#### Acknowledgement

Code for Decoder `borrowed` from [here](https://github.com/wptoux/attention-ocr)
