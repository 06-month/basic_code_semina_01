# Tiny-ImageNet-200 Classification

**본 프로젝트는 AiRLab에서 수행한 첫번째 코딩 세미나의 결과물로써, Tiny-Imagenet-200 기반 Classification을 수행했다.**

제공된 ResNet기반 baseline으로부터 87.22의 Top-1 valication accuracy를 달성한 기록이다.

| | top-1 val acc |
|---|---:|
| 제공된 baseline (ResNet-18, SGD, MultiStep) | 52.96 |
| **최종 (Swin-Tiny, progressive augmentation, 100 epoch)** | **87.22** |
| | **+34.26 %p** |

Tiny-ImageNet-200: 64×64 해상도, 200개 class.

전체 수치가 담긴 주차별 보고서는 5주에 걸쳐 작성했다: [`docs/`](docs/) (1–5주차, 한국어).

---

## 출발점

이 프로젝트는 augmentation도 로깅도 없고 스케줄이 고정된 ResNet-18 기반 baseline이었다.
아래 표는 적용한 기법들의 요약이다.

| | 제공된 skeleton | 이 저장소 |
|---|---|---|
| Architecture | ResNet-18만 | ResNet-18/34/50/101/152, ViT-S, DeiT-S, **Swin-T** |
| Optimizer | SGD (lr 0.1, momentum 0.9, wd 5e-4) | **AdamW** (lr 2e-5, wd 0.05) |
| LR 스케줄 | MultiStep, epoch 20/30/35에서 ×0.1 | **Cosine annealing** |
| 정규화 | — | dropout 0.2 / attn 0.1 / drop-path 0.1 |
| Augmentation | — | Albumentations + **CutMix / MixUp**, 강도 스케줄링 |
| 로깅 | stdout | **wandb**, epoch 축 정렬 지표 |
| 재현성 | — | seed 고정, `cudnn.deterministic=True` |
| Epoch | 40 | 100 |

CutMix와 MixUp은 라이브러리를 가져다 쓰지 않고 `train.py`에 직접 구현했다(`rand_bbox` 포함). 아래 설명할
augmentation 스케줄이 mixing 확률과 Beta 파라미터를 함께 제어할 수 있어야 했기 때문이다.

---

## 실험 기록

5주간의 기록이며 top-1 validation 정확도로 측정했다. 모든 행이 실제 실행 결과이고, 사후에 재구성한
경로가 아니다.

### 1주차 — 데이터 분석 후 augmentation 적용

| 변경 | top-1 val |
|---|---:|
| Baseline (ResNet-18, SGD, MultiStep, 40 ep) | **52.96** |
| + Cosine Annealing LR | 53.22 |
| + Label Smoothing | 53.82 |
| **+ Albumentations** | **60.42** |
| + Dropout | 58.68 |
| ResNet-152 + SGD | 58.98 |
| ResNet-101 + AdamW | 54.12 |

augmentation은 기본 목록에서 고른 것이 아니라 데이터셋 분석 후 결정하였다.

- **vertical flip 제외.** 중력의 영향을 받는 사물 class가 많아서, 위아래가 뒤집힌 이미지는 있을 법한
  샘플이 아니기에 적용하지 않았다.
- **hue/saturation 변경 제외.** 두꺼비와 개구리처럼 형태를 공유하고 주로 색으로 구분되는 class가 있어서,
  색상 jitter가 구분 신호를 지워버릴 것으로 판단했다.

두 제외 결정은 나중에 재검토했고, 당시 기록에도 과도한 교정이었을 가능성이 높다고 남겨두었다.

데이터 분할이 치우쳐 있다는 것도 이때 확인했다. **train : val : test = 91 : 4.5 : 4.5**.

### 2주차 — 모델 탐색

| 변경 | top-1 val |
|---|---:|
| ResNet-18, 60 epoch | 59.28 |
| ResNet-18, 60 ep + batch 512 + lr 0.4 | 61.12 |
| CoAtNet-0 (RandAugment + MixUp/CutMix + AdamW + LS 0.1) | 76.18 |
| CoAtNet-0 + batch 256 + lr 0.002 | **77.38** |

### 3주차 — 트랜스포머 모델 탐색

ViT 계열 경량 모델 세 개를 **GFLOPs를 맞춘 조건(~4.5 G)**에서 동일한 설정으로 비교했다
(cosine LR, warmup, 224×224, seed 40, AdamW, 40 epoch, lr 1e-3).

| 모델 | GFLOPs | top-1 val |
|---|---|---:|
| ViT-Small | ~4–5 G | 65.80 |
| DeiT-Small | ~4.6 G | 65.80 |
| **Swin-Tiny** | ~4.5 G | **68.46** |

비슷한 연산 비용에서 Swin-Tiny의 validation 정확도가 가장 높아서, 이후 실험은 모두 Swin-Tiny로 진행했다.

| 변경 | top-1 val |
|---|---:|
| Swin-T | 68.46 |
| + dropout (0.2 / 0.1 / 0.1) | 71.28 |
| + Albumentations | 70.44 |
| + MixUp/CutMix 하이브리드 | 75.88 |
| **+ lr 1e-3 → 1e-5, warmup 제거** | **85.93** |

**이 프로젝트에서 가장 큰 단일 향상은 learning rate였다: +10 %p.** 초반 학습이 불안정해서 초기
learning rate를 1e-3에서 1e-5로 낮췄다. 시작 learning rate가 그렇게 낮으면 warmup이 도움이 되지 않을
것으로 판단해 함께 제거했다.

### 4주차 — augmentation 강도를 스케줄링

이 시점의 결과는 underfitting 양상이었고, 이는 초반에 약하게 augmentation하고 후반에 강하게 하는 방향을
시사한다. **PS-SapAug (2024)**가 정확히 그 방식을 제안하는데, 단계별 스케줄을 쓴다. 이 프로젝트는 같은
아이디어를 **cosine ramp**로 적용했다. 0에서 1로 매끄럽게 증가하는 편이 이산적인 단계보다 학습이 고르게
진행될 것으로 기대했기 때문이다.

```python
progress     = epoch / args.epochs
aug_strength = 1 - math.cos((math.pi / 2) * progress)   # 0.0 → 1.0
```

`aug_strength`는 transform 확률, CutMix/MixUp 적용 확률, Beta 분포 파라미터를 함께 조정한다.

batch size도 64에서 128로 옮겼기 때문에, *Bag of Tricks*의 선형 스케일링 규칙에 따라 learning rate를
1e-5에서 2e-5로 조정했다.

| 변경 | top-1 val |
|---|---:|
| Swin-T + lr 1e-5 | 85.93 |
| **+ Aug Strength (cosine)** | 86.78 |
| **+ 100 epoch** | **87.22** |
| − GaussNoise | 85.14 |
| Swin-Large (batch 32, lr 3e-6, **11 epoch만**) | 92.05 |

GaussNoise는 저해상도에서 노이즈가 해로울 것이라 보고 제거했는데, 결과는 오히려 나빠졌다. Swin-Large는
92.05에 도달했지만 11 epoch만 실행했다.

### 5주차 — 버그 및 오류 분석

강도 스케줄이 **CutMix/MixUp까지 전달되지 않고 있었다.** transform 확률은 조정되는데 mixing 계수는
그렇지 않았다. `strength`를 그쪽에도 연결한 뒤, 비교 가능하도록 40 epoch로 고정해 측정했다.

| 변경 | top-1 val |
|---|---:|
| Swin-T + strength | 86.78 |
| **+ CutMix/MixUp에 strength 적용** | **87.08** |

---

## 오류 분석

augmentation을 감으로 조정하는 대신, 모델에게 어떤 이미지에서 실패하는지를 물었다.

최고 성능 epoch(37)과 마지막 epoch(40)에서 가장 어려운 class 다섯 개를 뽑고, 실패한 이미지마다 예측
class 분포를 백분율로 출력했다. 두 가지 패턴이 나타났다.

| 관찰 | 해석 | 조치 |
|---|---|---|
| `frying pan`이 음식에 가려짐, `pole`이 작고 프레임 가장자리에 있음 | 사물이 모델이 기대하는 위치에 없다 | **RandomResizedCrop** |
| `syringe`가 손에 가려짐 | 부분 가림 | **CoarseDropout** |

| 변경 | top-1 val |
|---|---:|
| Swin-T + strength (CutMix/MixUp) | 87.08 |
| + RandomResizedCrop | 86.50 |
| + CoarseDropout | 86.96 |
| + 둘 다 | 87.00 |

40 epoch 기준으로 두 변환 모두 87.08을 넘지 못했다.

---

## 주요 하이퍼파라미터

| 인자 | 기본값 | |
|---|---|---|
| `--arch` | `swin_tiny` | resnet18/34/50/101/152, vit_small, deit_small, swin_tiny |
| `--lr_base` | `2e-5` | batch size에 따라 선형 조정 |
| `--batch_size` | `128` | |
| `--epochs` | `100` | |
| `--drop_rate` | `0.2` | |
| `--attn_drop_rate` | `0.1` | |
| `--drop_path_rate` | `0.1` | |

Optimizer는 AdamW(`weight_decay=0.05`), `--epochs`에 걸친 cosine annealing, loss는 cross-entropy,
seed 42에 `cudnn.deterministic=True`.

---

## 실행 환경

```bash
cd docker
bash build_docker.sh
sh run_docker.sh
docker attach <DOCKER_CONTAINER_NAME>
```

Docker ≥ 24.0.6, CUDA ≥ 11.6. 데이터셋:
[Tiny-ImageNet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip)을 `./data/tiny-imagenet-200`에
푼다.

batch manager(`batch_manager.py`)는 수업에서 제공한 skeleton의 일부이며 이 저장소에는 포함하지 않았다.

### 보고된 실행에 사용한 설정

```bash
python main.py --arch swin_tiny  --batch_size 128 --lr_base 2e-5 --epochs 100
python main.py --arch resnet18   --batch_size 128 --lr_base 1e-3 --epochs 100
python main.py --arch vit_small  --batch_size 64  --lr_base 2e-5 --epochs 100
```

checkpoint는 epoch마다 `checkpoints/YYYY-MM-DD_HH:MM/`에 기록되고, 최고 성능 모델은 `best.pth.tar`,
테스트 예측은 `best_test_preds.csv`에 저장된다. wandb에는 train/val loss와 top-1/top-5 정확도를 공통
`epoch` 축에 기록하며, learning rate와 augmentation 강도도 함께 남긴다.

---

## 한계

- **vertical flip과 hue/saturation augmentation의 미적용은 실제 실험을 진행하지 않은 채 제외하였다.**
- **RandomResizedCrop과 CoarseDropout은 개선하려던 설정을 넘지 못했다** (87.08 → 87.00).

---

## 디렉토리 구조

```
.
├── main.py                    # 모델 선택, optimizer, 스케줄, aug-strength 루프
├── train.py                   # CutMix / MixUp과 rand_bbox를 포함한 학습 루프
├── val.py                     # validation 및 테스트 예측
├── transforms.py              # strength로 파라미터화된 Albumentations 파이프라인
├── utils.py                   # accuracy, AverageMeter
├── arch/resnet.py             # ResNet 변형들
├── docker/                    # Dockerfile, build 및 run 스크립트
├── docs/                      # 주차별 보고서, 1–5주차 (한국어)
├── size_distribution_histograms.png   # 이미지 크기 분포
└── split_image_counts.png             # train/val/test 개수
```

---

## 보고서

| 주차 | 날짜 | 내용 |
|---|---|---|
| [1](docs/week1_report.pdf) | 2025-08-14 | 데이터셋 관찰, baseline 분석, 초기 개선 |
| [2](docs/week2_report.pdf) | 2025-08-20 | baseline 확장, CoAtNet |
| [3](docs/week3_report.pdf) | 2025-08-27 | GFLOPs를 맞춘 ViT 계열 비교, learning rate 수정 |
| [4](docs/week4_report.pdf) | 2025-09-04 | Progressive augmentation, Swin-Large |
| [5](docs/week5_report.pdf) | 2025-09-18 | strength 버그 수정, 오류 분석 |

---

## 참고 문헌

- [Tiny-ImageNet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
- [timm](https://github.com/huggingface/pytorch-image-models) — 사전학습된 ViT / DeiT / Swin
- [Albumentations](https://albumentations.ai/)
- Yun et al., *CutMix*, [arXiv:1905.04899](https://arxiv.org/abs/1905.04899)
- Zhang et al., *mixup*, [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
- Liu et al., *Swin Transformer*, [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- He et al., *Bag of Tricks for Image Classification with CNNs*,
  [arXiv:1812.01187](https://arxiv.org/abs/1812.01187) — 선형 LR 스케일링
- PS-SapAug (2024) — progressive augmentation 강도. 여기서는 cosine ramp로 변형해 적용했다.
