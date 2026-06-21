# UAAT Windows Experiment Pack

이 폴더는 이미지 전용 UAAT 실험을 빠르게 재현하기 위한 최소 실행 코드입니다.

핵심 아이디어는 다음입니다.

```text
base model이 score s를 만든다.
uncertainty module이 u를 만든다.
context feature c를 함께 본다.
PolicyNet이 샘플별 threshold tau를 만든다.
score > tau일 때만 자동 판단한다.
```

## 0. 폴더 구조

```text
uaat_windows_experiment_pack/
  requirements.txt
  README_KR.md
  src/uaat/
  experiments/
  scripts/
```

## 1. 윈도우 환경 만들기

PowerShell 또는 Anaconda Prompt를 열고 이 폴더로 이동합니다.

```bat
cd C:\UAAT\uaat_windows_experiment_pack
```

가상환경을 만듭니다.

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

GPU가 있으면 PyTorch 공식 사이트에서 Windows + Pip + CUDA 조합 명령어를 복사해 설치합니다. 예시는 아래와 같습니다. CUDA 버전은 본인 PC에 맞게 공식 사이트에서 다시 확인하세요.

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU만 쓰면 아래처럼 설치합니다.

```bat
pip install torch torchvision torchaudio
```

나머지 패키지를 설치합니다.

```bat
pip install -r requirements.txt
```

GPU 확인:

```bat
scripts\00_check_gpu.bat
```

`cuda available: True`가 나오면 GPU 사용 가능입니다. False가 나와도 CPU로 실행은 되지만 느립니다.

## 2. E1: CIFAR-10-C controlled corruption

### 2-1. CIFAR-10 base predictor 학습

```bat
scripts\01_e1_train_cifar.bat
```

결과:

```text
runs\e1_cifar_base\cifar10_resnet18.pt
runs\e1_cifar_base\train_log.json
```

### 2-2. CIFAR-10-C 다운로드와 압축 해제

CIFAR-10-C.tar를 다운로드해서 다음 구조가 되도록 풉니다.

```text
data\CIFAR-10-C\
  labels.npy
  gaussian_noise.npy
  motion_blur.npy
  jpeg_compression.npy
  brightness.npy
  contrast.npy
  ...
```

### 2-3. score, uncertainty, context CSV 만들기

```bat
scripts\02_e1_extract_cifar10c.bat
```

결과:

```text
runs\e1_cifar10c\features.csv
```

이 CSV에는 다음 열이 들어갑니다.

```text
score: 모델이 자기 예측을 믿는 정도
uncertainty: entropy와 TTA instability 기반 불확실성
error: 자동 수용하면 틀리는지 여부
split: calib 또는 test
ctx_severity, ctx_corr_*, ctx_brightness, ctx_contrast, ctx_blur_risk, ctx_saturation
```

### 2-4. 고정 임계값 vs UAAT 비교

```bat
scripts\03_policy_e1.bat
```

결과:

```text
runs\e1_cifar10c\policy\metrics.csv
runs\e1_cifar10c\policy\test_decisions.csv
runs\e1_cifar10c\policy\policy_fixed.json
runs\e1_cifar10c\policy\policy_uncertainty_grid.json
runs\e1_cifar10c\policy\policy_uaat_monotone.json
```

논문 표에 넣을 핵심 파일은 `metrics.csv`입니다.

## 3. E2: MVTec AD / MVTec LOCO AD industrial anomaly

### 3-1. 데이터셋 압축 해제

MVTec AD는 다음 구조가 되어야 합니다.

```text
data\mvtec_ad\
  bottle\
    train\good\*.png
    test\good\*.png
    test\broken_large\*.png
  cable\
  capsule\
  ...
```

MVTec LOCO AD도 같은 방식으로 둡니다.

```text
data\mvtec_loco_ad\
  breakfast_box\
  juice_bottle\
  pushpins\
  screw_bag\
  splicing_connectors\
```

### 3-2. MVTec AD feature CSV 만들기

```bat
scripts\04_e2_mvtec_extract.bat
```

처음 실행하면 torchvision이 ImageNet pretrained ResNet18 가중치를 다운로드할 수 있습니다.

### 3-3. MVTec AD 정책 평가

```bat
scripts\05_policy_e2.bat
```

LOCO를 돌리려면 `scripts\04_e2_mvtec_extract.bat`에서 root와 out_csv를 바꿉니다.

```bat
python experiments\e2_extract_mvtec_knn.py --root data\mvtec_loco_ad --out_csv runs\e2_mvtec_loco\features.csv --categories all --batch_size 64 --workers 0 --image_size 224
python experiments\train_eval_policy.py --csv runs\e2_mvtec_loco\features.csv --out_dir runs\e2_mvtec_loco\policy --target_coverage 0.80 --c_wrong 10 --c_defer 1 --group_col category
```

## 4. E3: iWildCam-WILDS real image domain shift

이 실험은 가장 무겁습니다. 데이터가 크고 다운로드도 오래 걸립니다.

```bat
scripts\06_e3_iwildcam_train_extract.bat
scripts\07_policy_e3.bat
```

빠른 테스트만 하려면 다음처럼 제한해서 실행합니다.

```bat
python experiments\e3_train_extract_iwildcam.py --data_dir data --out_dir runs\e3_iwildcam_quick --out_csv runs\e3_iwildcam_quick\features.csv --epochs 1 --batch_size 32 --workers 0 --image_size 224 --max_train_batches 100 --max_extract 2000
python experiments\train_eval_policy.py --csv runs\e3_iwildcam_quick\features.csv --out_dir runs\e3_iwildcam_quick\policy --target_coverage 0.80 --c_wrong 5 --c_defer 1
```

## 5. 결과 해석

`metrics.csv`를 엽니다.

중요한 열은 다음입니다.

```text
policy: fixed, uncertainty_grid, uaat_monotone
coverage: 자동 판단 비율
risk: c_wrong * wrong_auto_rate + c_defer * defer_rate
wrong_auto_rate: 전체 샘플 중 틀렸는데 자동 수용된 비율
auto_error_rate: 자동 수용된 샘플 안에서 틀린 비율
defer_rate: 사람 검토로 넘긴 비율
```

좋은 결과는 다음 모양입니다.

```text
uaat_monotone risk < fixed risk
coverage는 거의 같음
wrong_auto_rate가 줄어듦
uncertainty가 높을수록 tau가 올라감
```

## 6. 논문에 넣을 실험 순서

1. CIFAR-10-C paper5 quick run으로 코드가 정상인지 확인합니다.
2. CIFAR-10-C all corruptions full run을 실행합니다.
3. MVTec AD 전체 category를 실행합니다.
4. MVTec LOCO AD 전체 category를 실행합니다.
5. 시간이 있으면 iWildCam-WILDS를 실행합니다.
6. 모든 `metrics.csv`를 하나의 표로 합칩니다.
7. severity별, category별 group metrics를 그림으로 만듭니다.

## 7. 가장 흔한 오류

### CUDA가 안 잡힘

```bat
python -c "import torch; print(torch.cuda.is_available())"
```

False면 CPU로 돌아갑니다. GPU를 쓰려면 NVIDIA 드라이버와 PyTorch CUDA wheel을 다시 맞춥니다.

### 윈도우에서 DataLoader가 멈춤

`--workers 0`으로 실행하세요. 이 팩의 bat 파일은 안전하게 0으로 되어 있습니다.

### CIFAR-10-C labels.npy 오류

폴더 구조가 `data\CIFAR-10-C\labels.npy`가 되도록 확인하세요. 압축을 풀 때 폴더가 한 번 더 들어가면 경로를 바꿔야 합니다.

### MVTec 폴더 오류

카테고리 폴더 아래에 `train\good`와 `test\...`가 있어야 합니다.
