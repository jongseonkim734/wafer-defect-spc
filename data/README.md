# 데이터셋 다운로드 가이드

원본 데이터는 git에 올리지 않는다(.gitignore 처리). 아래 방법으로 각자 내려받아 이 폴더에 둔다.

## 1. WM-811K — 웨이퍼맵 불량 패턴 (메인)
- 실제 반도체 팹의 웨이퍼맵 811,457장. 9종 불량 패턴 라벨(Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, none).
- 출처: Kaggle "WM-811K wafer map" 로 검색.
- 받는 법(둘 중 하나):
  - **웹**: Kaggle 웹사이트에서 직접 다운로드 → `data/wm811k/` 에 압축 해제.
  - **API**: `pip install kaggle` 후 토큰 설정하고
    ```bash
    kaggle datasets download -d <dataset-slug> -p data/wm811k --unzip
    ```
- 보통 `.pkl` 형태(pandas로 바로 로드). 첫 로드는 `notebooks/01_eda.py` 참고.
- Kaggle 웹사이트에서 직접 다운로드한 경우, `LSWMD.pkl`라는 이름으로 저장되게 된다.

## 2. SECOM — 반도체 공정 센서 데이터 (SPC/이상탐지용)
- 반도체 공정 센서 590개 피처 + 양/불량(pass/fail) 라벨 1567행. 불균형 데이터.
- 출처: UCI Machine Learning Repository "SECOM".
- 받는 법: UCI 페이지에서 SECOM 다운로드 → 그 중 `secom.data`, `secom_labels.data`를 `data/secom/` 에 둔다.

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
