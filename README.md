# Tiny-ImageNet-200 Classification

**From a provided ResNet-18 baseline to 87.22 % top-1 validation accuracy — a five-week record of
what was tried, what the data suggested, and what the errors said.**

| | top-1 val acc |
|---|---:|
| Provided baseline (ResNet-18, SGD, MultiStep) | 52.96 |
| **Final (Swin-Tiny, progressive augmentation, 100 epochs)** | **87.22** |
| | **+34.26 %p** |

Tiny-ImageNet-200: 200 classes at 64×64.

Weekly reports with the full numbers: [`docs/`](docs/) (week 1–5, in Korean).

---

## Starting Point

This project began from a course-provided skeleton — a working ResNet-18 training loop with no
augmentation, no logging, and a fixed schedule. The table below is what changed.

| | Provided skeleton | This repository |
|---|---|---|
| Architectures | ResNet-18 only | ResNet-18/34/50/101/152, ViT-S, DeiT-S, **Swin-T** |
| Optimizer | SGD (lr 0.1, momentum 0.9, wd 5e-4) | **AdamW** (lr 2e-5, wd 0.05) |
| LR schedule | MultiStep, ×0.1 at epochs 20/30/35 | **Cosine annealing** |
| Regularisation | — | dropout 0.2 / attn 0.1 / drop-path 0.1 |
| Augmentation | — | Albumentations + **CutMix / MixUp**, strength-scheduled |
| Logging | stdout | **wandb**, epoch-aligned metrics |
| Reproducibility | — | seed fixed, `cudnn.deterministic=True` |
| Epochs | 40 | 100 |

CutMix and MixUp are implemented directly in `train.py` (including `rand_bbox`) rather than pulled
from a library, so the mixing probability and the Beta parameter could be driven by the
augmentation schedule described below.

---

## Experiment Log

Five weeks, measured as top-1 validation accuracy. Every row was run; the path is not a
reconstruction.

### Week 1 — read the data first, then tune

| Change | top-1 val |
|---|---:|
| Baseline (ResNet-18, SGD, MultiStep, 40 ep) | **52.96** |
| + Cosine Annealing LR | 53.22 |
| + Label Smoothing | 53.82 |
| **+ Albumentations** | **60.42** |
| + Dropout | 58.68 |
| ResNet-152 + SGD | 58.98 |
| ResNet-101 + AdamW | 54.12 |

Augmentations were chosen from inspection of the dataset rather than from a default list:

- **No vertical flip.** Many classes are objects subject to gravity; an upside-down image is not a
  plausible sample.
- **No hue/saturation shift.** Classes such as toad and frog share a shape and differ mainly in
  colour, so colour jitter was judged likely to remove the distinguishing signal.

Both exclusions were later reconsidered — dropping them outright was recorded at the time as
probably an overcorrection.

The split was also found to be skewed: **train : val : test = 91 : 4.5 : 4.5**.

### Week 2 — scale the baseline, then leave CNNs

| Change | top-1 val |
|---|---:|
| ResNet-18, 60 epochs | 59.28 |
| ResNet-18, 60 ep + batch 512 + lr 0.4 | 61.12 |
| CoAtNet-0 (RandAugment + MixUp/CutMix + AdamW + LS 0.1) | 76.18 |
| CoAtNet-0 + batch 256 + lr 0.002 | **77.38** |

The conv+attention hybrid moved the results from 61 to 77.

### Week 3 — compare at equal compute, then fix the learning rate

Three lightweight ViT-family models were compared **at matched GFLOPs (~4.5 G)**, under identical
settings (cosine LR, warmup, 224×224, seed 40, AdamW, 40 epochs, lr 1e-3):

| Model | GFLOPs | top-1 val |
|---|---|---:|
| ViT-Small | ~4–5 G | 65.80 |
| DeiT-Small | ~4.6 G | 65.80 |
| **Swin-Tiny** | ~4.5 G | **68.46** |

Swin-Tiny has the highest validation accuracy at comparable cost, so the rest of the project
uses it.

| Change | top-1 val |
|---|---:|
| Swin-T | 68.46 |
| + dropout (0.2 / 0.1 / 0.1) | 71.28 |
| + Albumentations | 70.44 |
| + MixUp/CutMix hybrid | 75.88 |
| **+ lr 1e-3 → 1e-5, warmup removed** | **85.93** |

**The single largest gain in the project is the learning rate: +10 %p.** Training was unstable
early on, so the initial learning rate was lowered from 1e-3 to 1e-5. With a starting rate that
low, warmup was judged unlikely to help and was removed.

### Week 4 — schedule the augmentation strength

Results were now underfitting, which suggests augmenting weakly at the start and strongly later.
**PS-SapAug (2024)** proposes exactly that, using a step-wise schedule. This project applies the
same idea with a **cosine ramp** instead, on the expectation that a smooth 0→1 increase trains
more evenly than discrete steps:

```python
progress     = epoch / args.epochs
aug_strength = 1 - math.cos((math.pi / 2) * progress)   # 0.0 → 1.0
```

`aug_strength` scales the transform probabilities, the CutMix/MixUp application probability, and
the Beta distribution parameter together.

Batch size also moved 64 → 128, so the learning rate was scaled 1e-5 → 2e-5, following the linear
scaling rule from *Bag of Tricks*.

| Change | top-1 val |
|---|---:|
| Swin-T + lr 1e-5 | 85.93 |
| **+ Aug Strength (cosine)** | 86.78 |
| **+ 100 epochs** | **87.22** |
| − GaussNoise | 85.14 |
| Swin-Large (batch 32, lr 3e-6, **11 epochs only**) | 92.05 |

GaussNoise was removed on the expectation that noise hurts at low resolution; the result was
worse. Swin-Large reached 92.05, but only 11 epochs were run.

### Week 5 — a bug, then error analysis

The strength schedule turned out **not to reach CutMix/MixUp**: the transform probabilities were
scaled but the mixing coefficients were not. After wiring `strength` through them as well, at a
fixed 40 epochs for comparability:

| Change | top-1 val |
|---|---:|
| Swin-T + strength | 86.78 |
| **+ strength applied to CutMix/MixUp** | **87.08** |

---

## Error Analysis

Rather than tuning augmentation blindly, the model was asked which images it fails on.

At the best epoch (37) and the last epoch (40), the five hardest classes were extracted and, for
each failing image, the predicted class distribution printed as percentages. Two patterns emerged:

| Observation | Interpretation | Change |
|---|---|---|
| `frying pan` occluded by food; `pole` small and at the frame edge | the object is not where the model expects it | **RandomResizedCrop** |
| `syringe` hidden behind a hand | partial occlusion | **CoarseDropout** |

| Change | top-1 val |
|---|---:|
| Swin-T + strength (CutMix/MixUp) | 87.08 |
| + RandomResizedCrop | 86.50 |
| + CoarseDropout | 86.96 |
| + both | 87.00 |

Neither transform improved on 87.08 at 40 epochs.

---

## Key Hyperparameters

| Argument | Default | |
|---|---|---|
| `--arch` | `swin_tiny` | resnet18/34/50/101/152, vit_small, deit_small, swin_tiny |
| `--lr_base` | `2e-5` | linearly scaled with batch size |
| `--batch_size` | `128` | |
| `--epochs` | `100` | |
| `--drop_rate` | `0.2` | |
| `--attn_drop_rate` | `0.1` | |
| `--drop_path_rate` | `0.1` | |

Optimizer AdamW (`weight_decay=0.05`), cosine annealing over `--epochs`, loss cross-entropy,
seed 42 with `cudnn.deterministic=True`.

---

## Environment

```bash
cd docker
bash build_docker.sh
sh run_docker.sh
docker attach <DOCKER_CONTAINER_NAME>
```

Docker ≥ 24.0.6, CUDA ≥ 11.6. Dataset:
[Tiny-ImageNet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip), extracted to
`./data/tiny-imagenet-200`.

The batch manager (`batch_manager.py`) is part of the course-provided skeleton and is not
included here.

### Configuration used for the reported runs

```bash
python main.py --arch swin_tiny  --batch_size 128 --lr_base 2e-5 --epochs 100
python main.py --arch resnet18   --batch_size 128 --lr_base 1e-3 --epochs 100
python main.py --arch vit_small  --batch_size 64  --lr_base 2e-5 --epochs 100
```

Checkpoints are written per epoch to `checkpoints/YYYY-MM-DD_HH:MM/`, the best model to
`best.pth.tar`, and test predictions to `best_test_preds.csv`. wandb logs train/val loss and
top-1/top-5 accuracy against a shared `epoch` axis, plus learning rate and augmentation strength.

---

## Limitations

- **Swin-Large's 92.05 is not comparable.** It is an 11-epoch reading against 100-epoch runs, and
  it was not trained to completion.
- **Vertical flip and hue/saturation augmentation were excluded on inspection alone**, never
  tested. The reports themselves note this was probably an overcorrection.
- **RandomResizedCrop and CoarseDropout did not beat the configuration they were meant to improve**
  (87.08 → 87.00). The error analysis pointed at the right failure mode; the fix did not follow.

---

## Repository Structure

```
.
├── main.py                    # model selection, optimizer, schedule, aug-strength loop
├── train.py                   # training loop with CutMix / MixUp and rand_bbox
├── val.py                     # validation and test-time prediction
├── transforms.py              # strength-parameterised Albumentations pipeline
├── utils.py                   # accuracy, AverageMeter
├── arch/resnet.py             # ResNet variants
├── docker/                    # Dockerfile, build and run scripts
├── docs/                      # weekly reports, week 1–5 (Korean)
├── size_distribution_histograms.png   # image-size distribution
└── split_image_counts.png             # train/val/test counts
```

---

## Reports

| Week | Date | Contents |
|---|---|---|
| [1](docs/week1_report.pdf) | 2025-08-14 | Dataset inspection, baseline analysis, first improvements |
| [2](docs/week2_report.pdf) | 2025-08-20 | Baseline scaling, CoAtNet |
| [3](docs/week3_report.pdf) | 2025-08-27 | ViT-family comparison at matched GFLOPs, learning-rate fix |
| [4](docs/week4_report.pdf) | 2025-09-04 | Progressive augmentation, Swin-Large |
| [5](docs/week5_report.pdf) | 2025-09-18 | Strength bug fix, error analysis |

---

## References

- [Tiny-ImageNet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
- [timm](https://github.com/huggingface/pytorch-image-models) — pretrained ViT / DeiT / Swin
- [Albumentations](https://albumentations.ai/)
- Yun et al., *CutMix*, [arXiv:1905.04899](https://arxiv.org/abs/1905.04899)
- Zhang et al., *mixup*, [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
- Liu et al., *Swin Transformer*, [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- He et al., *Bag of Tricks for Image Classification with CNNs*,
  [arXiv:1812.01187](https://arxiv.org/abs/1812.01187) — linear LR scaling
- PS-SapAug (2024) — progressive augmentation strength, adapted here with a cosine ramp
