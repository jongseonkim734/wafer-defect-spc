# 데이터셋 다운로드 가이드

원본 데이터는 git에 올리지 않는다(.gitignore 처리). 아래 방법으로 각자 내려받아 이 폴더에 둔다.

## 1. WM-811K LSWMD.pkl — 웨이퍼맵 불량 패턴 (메인. Large Scale Wafer Map Dataset)
- 실제 반도체 팹의 웨이퍼맵 811,457장. 46,393개의 Lot에서 수집
- 9종 불량 패턴 라벨(Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, none).
- 출처: Kaggle "WM-811K wafer map" 로 검색.
- 받는 법(둘 중 하나):
  - **웹**: Kaggle 웹사이트에서 직접 다운로드 → `data/wm811k/` 에 압축 해제.
  - **API**: `pip install kaggle` 후 토큰 설정하고
    ```bash
    kaggle datasets download -d <dataset-slug> -p data/wm811k --unzip
    ```
- 보통 `.pkl` 형태(pandas로 바로 로드). 첫 로드는 `notebooks/01_eda.py` 참고.
- Kaggle 웹사이트에서 직접 다운로드한 경우, `LSWMD.pkl`라는 이름으로 저장되게 된다.
- 구조
  - waferMap: 웨이퍼마다의 2D array. 0 → 웨이퍼 밖. 1 → 웨이퍼 내 정상 die. 2 → 웨이퍼 내 불량 die.
  - failureType: 9종 불량 패턴 라벨
  - trianTestLabel: 원 데이터의 train/test 구분...?
  - lotName, dieSize: 각종 메타데이터
- 특징
  - 811,457장 중, 라벨링이 된 웨이퍼맵은 172,950장 뿐
  - 웨이퍼맵의 크기가 통일되어있지 않아, 웨이퍼마다 die 수가 다름
  - failureType을 기반으로 불량의 원인을 추정할 수 있음

## 2. SECOM — 반도체 공정 센서 데이터 (SPC/이상탐지용)
- 반도체 공정 센서 및 계측값 로그
- 출처: UCI Machine Learning Repository "SECOM".
- 받는 법: UCI 페이지에서 SECOM 다운로드 → 그 중 `secom.data`, `secom_labels.data`를 `data/secom/` 에 둔다.
- 구조
  - secom.data: 1567행(Lot) * 590열(피처. 센서 측정값).
  - secom_labels.data: 1567행(Lot) * 2열(라벨(-1: pass, 1: fail) + 타임스탬프)
- 특징
  - 1567행 중, 불량은 104행 뿐. → 정확도보다는 F1/Recall(재현율)로 평가할 필요가 있음.
  - 센서 특성 상, NaN이 다수이므로, 결측치 대치 혹은 제거의 전처리 필요
  - 590개 피처 중, 유의미한 피처를 잘 뽑아내야 함
  - 센서 및 계측값 시계열이라서, 한계를 벗어난 Lot을 잡아낼 수 있기 때문. WM-811K와 성격이 다름.

## 폴더 배치 예시
```
data/
├── README.md
├── wm811k/
│   └── LSWMD.pkl
└── secom/
    ├── secom.data
    └── secom_labels.data
```
