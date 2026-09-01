# Stage 1 B3: RGB + Depth + Text

## Change map

Every B3-only implementation is marked with `Stage-1 B3` or `B3 change` in
comments/docstrings.

| File | B3 change |
|---|---|
| `basicsr/data/rgb_depth_text_paired_image_dataset.py` | Strict four-way LR/GT/depth/TXT pairing, robust depth normalization, synchronized spatial transforms, and UTF-8 caption loading |
| `basicsr/archs/rgb_depth_text_mambairv2_arch.py` | B1 RGB-depth fusion followed by frozen CLIP ViT-L/14 global FiLM |
| `basicsr/models/rgb_depth_text_mambairv2_model.py` | Four-modal batch handling and one-time caption encoding for aligned tiled inference |
| `options/train/mambairv2/train_S1_B3_RGBDepthText_MambaIRv2_x4.yml` | Full 500k B3 training configuration |
| `options/test/mambairv2/test_S1_B3_RGBDepthText_MambaIRv2_x4.yml` | Five-dataset B3 evaluation configuration |

No MambaIRv2 block, scan, reconstruction layer, loss, crop, augmentation,
optimizer, scheduler, batch size, or metric was changed.

## Controlled B3 design

The implementation follows the Stage-1 minimum-complexity comparison:

```text
F_R   = Conv3x3(RGB)
F_D   = Conv3x3(Depth)
F_RD  = Conv1x1(concat(F_R, F_D))
[gamma, beta] = Projection(FrozenCLIP(Text))
F_RDT = (1 + gamma) * F_RD + beta
SR     = OriginalMambaIRv2(F_RDT)
```

The text projection is zero initialized. With the same random seed, B3 starts
exactly from the B1 RGB-depth function at iteration zero. This makes
`B3 - B1` the controlled incremental contribution of text on top of depth.
CLIP is frozen and excluded from Adam as well as BasicSR checkpoints.

Trainable parameter comparison (CLIP excluded):

```text
B0 RGB:                       23,050,713
B1 RGB + depth:               23,113,179
B3 RGB + depth + text:        23,382,327
B3 text increment over B1:       269,148
```

## Required data and model files

Each HR basename must have all four non-empty/aligned inputs. Example:

```text
HR:    0001.png
LR:    0001x4.png
Depth: 0001x4.png
Text:  0001.txt
```

The paths in both YAML files are copied from the previously supplied B1/B2
reference configurations. Depth uses full-image P2/P98 normalization before
cropping. The caption is a full-image UTF-8 description and is intentionally
unchanged when a training patch is cropped or augmented.

The code performs no network download. The following Hugging Face-format CLIP
ViT-L/14 directory must exist on the training/test server:

```text
/home/BRAIN/xukai/files/clip-vit-large-patch14/
```

It must contain the CLIP configuration, tokenizer files, and model weights
(`model.safetensors` or `pytorch_model.bin`). Missing/misaligned RGB, depth, or
text files and empty captions fail immediately with the exact problematic path.

## Commands

Full training from scratch:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_S1_B3_RGBDepthText_MambaIRv2_x4.yml \
  --launcher none
```

Resume an interrupted run:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_S1_B3_RGBDepthText_MambaIRv2_x4.yml \
  --launcher none \
  --auto_resume
```

Optional 25% screening run in a separate experiment directory:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_S1_B3_RGBDepthText_MambaIRv2_x4.yml \
  --launcher none \
  --force_yml \
    name=S1_B3_RGBDepthText_MambaIRv2_x4_screen25 \
    train:total_iter=125000 \
    train:scheduler:milestones=[62500,100000,112500,118750]
```

Five-dataset test after the 500k checkpoint is available:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
  -opt options/test/mambairv2/test_S1_B3_RGBDepthText_MambaIRv2_x4.yml \
  --launcher none
```
