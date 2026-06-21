# UAAT E1/E2/E3 seed 3회 반복 스크립트

이 스크립트 묶음은 E1/E2/E3 실험을 seed 1, 2, 3으로 반복 실행하고, 결과를 한 파일에 몰아넣지 않고 실험별/seed별 폴더에 나누어 저장합니다.

## 1. 설치 위치

압축을 풀고 아래 파일들을 `C:\UAAT\uaat_windows_experiment_pack` 폴더 안에 복사하세요.

- `seed3_utils.py`
- `01_run_e1_seed3.py`
- `02_run_e2_mvtec_ad_seed3.py`
- `03_run_e2_mvtec_loco_seed3.py`
- `04_run_e3_iwildcam_seed3.py`
- `05_collect_seed3_summary.py`

## 2. 실행 전 기본 명령어

```bat
cd /d C:\UAAT\uaat_windows_experiment_pack
.venv\Scripts\activate
set PYTHONPATH=%CD%\src
```

## 3. 실행 순서

```bat
python 01_run_e1_seed3.py
python 02_run_e2_mvtec_ad_seed3.py
python 03_run_e2_mvtec_loco_seed3.py
python 04_run_e3_iwildcam_seed3.py
python 05_collect_seed3_summary.py
```

## 4. 결과 저장 구조

```text
runs_seed3
├─ e1_cifar10c_all
│  ├─ seed_01
│  │  ├─ features.csv
│  │  ├─ coverage_comparison.csv
│  │  ├─ coverage_grid_all.csv
│  │  ├─ coverage_matched_nearest.csv
│  │  ├─ coverage_matched_paper_targets.csv
│  │  └─ cost_sensitivity_comparison.csv
│  ├─ seed_02
│  └─ seed_03
├─ e2_mvtec_ad
├─ e2_mvtec_loco
├─ e3_iwildcam
└─ _summary
```

## 5. 중요한 점

- E3 iWildCam은 실제 학습이 들어가므로 seed 반복 의미가 큽니다.
- E2 MVTec 계열은 KNN feature extraction 기반이라 코드 내부에 random step이 거의 없다면 seed별 결과가 같거나 거의 같을 수 있습니다. 그래도 논문에는 “3 seeds were evaluated; deterministic pipelines showed negligible variation”처럼 쓸 수 있습니다.
- 스크립트가 `--seed` 옵션을 지원하면 자동으로 붙이고, 지원하지 않으면 환경변수만 설정합니다.
- 기존 파일이 이미 있으면 다시 돌리지 않고 skip합니다. 다시 돌리고 싶으면 해당 `runs_seed3\...\seed_XX` 폴더를 삭제한 뒤 실행하세요.
