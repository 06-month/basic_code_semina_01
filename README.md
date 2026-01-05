# Tiny-ImageNet 200 분류 프로젝트

Tiny-ImageNet 200 데이터셋을 이용한 이미지 분류 프로젝트입니다. 다양한 모델 아키텍처(ResNet, ViT, Swin Transformer)와 최신 학습 기법들을 적용하여 높은 정확도를 달성합니다.

## 📋 주요 특징

- **다양한 모델 아키텍처 지원**
  - ResNet 계열: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
  - Transformer 계열: ViT-Small, DeiT-Small, Swin-Tiny

- **고급 데이터 증강 기법**
  - Progressive Augmentation: 학습이 진행됨에 따라 증강 강도를 점진적으로 증가
  - CutMix & MixUp: 배치 레벨 증강 기법
  - Albumentations 라이브러리 활용

- **실험 관리 및 모니터링**
  - Weights & Biases (wandb) 통합
  - 학습/검증 메트릭 실시간 추적
  - 체크포인트 자동 저장

## 🚀 사용 방법

### 환경 요구사항

- Docker >= 24.0.6
- CUDA >= 11.6
- GPU 메모리: 최소 8GB 권장

### 1. Docker 환경 구축

```bash
cd docker
bash build_docker.sh
sh run_docker.sh
docker attach <DOCKER_CONTAINER_NAME>
```

### 2. 데이터셋 다운로드

```bash
cd data
sh download_and_unzip.sh
```

데이터셋은 `./data/tiny-imagenet-200` 디렉토리에 저장됩니다.

### 3. 학습 실행 예제

본 프로젝트는 다양한 모델 아키텍처와 설정으로 실험을 수행했습니다.

#### 기본 실행 (Swin-Tiny)
```bash
python main.py --arch swin_tiny --batch_size 128 --epochs 100
```

#### ResNet18으로 실행
```bash
python main.py --arch resnet18 --batch_size 128 --lr_base 1e-3 --epochs 100
```

#### ViT-Small로 실행
```bash
python main.py --arch vit_small --batch_size 64 --lr_base 2e-5 --epochs 100
```

> 💡 **참고**: 실험은 이미 완료되었으며, 위 명령어들은 재현을 위한 예제입니다.

## 🛠️ 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--arch` | `swin_tiny` | 모델 아키텍처 선택 |
| `--lr_base` | `2e-5` | 초기 학습률 |
| `--batch_size` | `128` | 배치 크기 |
| `--epochs` | `100` | 전체 학습 에포크 수 |
| `--drop_rate` | `0.2` | Dropout 비율 |
| `--attn_drop_rate` | `0.1` | Attention Dropout 비율 |
| `--drop_path_rate` | `0.1` | Drop Path 비율 |

## 📊 프로젝트 구조

```
.
├── main.py                    # 메인 학습 스크립트
├── train.py                   # 학습 루프 (CutMix/MixUp 포함)
├── val.py                     # 검증 루프
├── transforms.py              # 데이터 증강 변환
├── utils.py                   # 유틸리티 함수
├── batch_manager.py           # 데이터 로더 (별도 구현 필요)
├── arch/
│   └── resnet.py             # ResNet 아키텍처
├── docker/
│   ├── Dockerfile            # Docker 이미지 정의
│   ├── build_docker.sh       # Docker 빌드 스크립트
│   └── run_docker.sh         # Docker 실행 스크립트
├── size_distribution_histograms.png  # 데이터셋 분석
└── split_image_counts.png            # 데이터 분할 통계
```

## 🔬 주요 기술

### 1. Progressive Augmentation
학습이 진행됨에 따라 데이터 증강 강도가 점진적으로 증가합니다:

```python
progress = epoch / args.epochs
aug_strength = 1 - math.cos((math.pi / 2) * progress)  # 0.0 → 1.0
```

### 2. CutMix & MixUp
- **CutMix**: 이미지의 일부 영역을 다른 이미지로 대체
- **MixUp**: 두 이미지를 선형 보간하여 혼합
- 학습 강도에 따라 적용 확률 조절

### 3. Cosine Annealing LR Scheduler
학습률을 코사인 함수 형태로 감소시켜 안정적인 학습을 유도합니다.

### 4. 앙상블 친화적 설계
- 여러 모델 아키텍처를 쉽게 실험 가능
- 체크포인트 자동 저장으로 모델 앙상블 구성 용이

## 📈 실험 결과

본 프로젝트는 Tiny-ImageNet 데이터셋에서 다양한 모델과 학습 기법을 적용한 실험을 완료했습니다.

### 주요 성과
- 다양한 아키텍처 실험 완료 (ResNet, ViT, Swin Transformer)
- Progressive Augmentation 및 CutMix/MixUp 효과 검증
- Wandb를 통한 체계적인 실험 관리 및 메트릭 추적

## 🧪 실험 관리 (Weights & Biases)

프로젝트는 wandb를 통해 다음 메트릭을 자동으로 추적합니다:

- **학습 메트릭**: Loss, Top-1 Accuracy, Top-5 Accuracy
- **검증 메트릭**: Loss, Top-1 Accuracy, Top-5 Accuracy
- **학습률 & 증강 강도**: Epoch별 변화 추적

### wandb 설정
```bash
wandb login
# 이후 main.py 실행 시 자동으로 로깅됨
```

## 💾 체크포인트 및 결과물

### 자동 저장 기능
- 매 에포크마다 최신 체크포인트 자동 저장
- 최고 성능 모델은 `best.pth.tar`로 별도 저장
- 저장 위치: `checkpoints/YYYY-MM-DD_HH:MM/`
- 테스트 예측 결과는 `best_test_preds.csv` 형식으로 저장

### 저장되는 정보
- 모델 가중치 (state_dict)
- Optimizer 상태
- Epoch 번호
- Top-1 및 Top-5 정확도

## 📝 구현된 기능

### Task 0: 실험 로깅 ✅
- [x] wandb 통합 완료
- [x] 학습/검증 메트릭 추적 (Loss, Top-1/Top-5 Accuracy)
- [x] Epoch별 학습률 및 증강 강도 로깅

### Task 1: 하이퍼파라미터 튜닝 ✅
- [x] 7가지 아키텍처 지원 (ResNet 5종, ViT/DeiT, Swin)
- [x] Cosine Annealing LR Scheduler 적용
- [x] AdamW Optimizer + Weight Decay

### Task 2: 데이터 증강 ✅
- [x] Progressive Augmentation (Cosine Scheduling)
- [x] CutMix & MixUp 구현
- [x] Albumentations 기반 다양한 증강 기법

### Task 3: 데이터 분석 ✅
- [x] 데이터셋 크기 분포 시각화 (`size_distribution_histograms.png`)
- [x] Train/Val/Test 분할 통계 (`split_image_counts.png`)

### Task 4: 실험 관리 ✅
- [x] 체크포인트 자동 저장 시스템
- [x] Best model 추적 및 저장
- [x] 테스트 예측 결과 CSV 출력

## 🔧 문제 해결

### CUDA Out of Memory
```bash
# 배치 크기 줄이기
python main.py --batch_size 64

# 또는 더 작은 모델 사용
python main.py --arch resnet18
```

### 학습 속도 개선
```bash
# num_workers 조정
# train.py와 val.py의 DataLoader에서 num_workers=10 → 4로 변경
```

## 📚 참고 자료

- [Tiny-ImageNet Dataset](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
- [timm Documentation](https://github.com/rwightman/pytorch-image-models)
- [Albumentations](https://albumentations.ai/)
- [CutMix Paper](https://arxiv.org/abs/1905.04899)
- [MixUp Paper](https://arxiv.org/abs/1710.09412)

## 📄 라이센스

이 프로젝트는 교육 목적으로 작성되었습니다.

## 🙋‍♂️ 기여

버그 리포트나 개선 제안은 Issue를 통해 제출해주세요.
