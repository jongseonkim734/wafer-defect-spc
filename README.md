# wafer-defect-spc

반도체 웨이퍼 불량 데이터를 **통계적 공정관리(SPC)** 와 **머신러닝** 두 축으로 분석하는 프로젝트.
삼성전자 DS 평가및분석 직무에서 요구하는 "불량 분석 · 유효 인자 감지 · 통계적 품질관리 · 수율 개선"을
실제 데이터로 구현·증명하는 것을 목표로 한다.

> **분석 문서 구조는 6시그마의 DMAIC 프레임을 따른다.** (자격증 대신 방법론을 프로젝트로 체화)

---

## 핵심 결과 (Results)

**① 유효 인자 선별로 불량 검출력 대폭 개선 (SECOM, 로지스틱 회귀)**

| 모델 | 피처 수 | fail F1 | fail recall | 비고 |
|---|---|---|---|---|
| baseline | 590 | 0.14 | 0.19 | 불량 4/21 검출, 590피처 vs 1,253샘플 → 과적합 |
| SelectKBest k=30 | 30 | 0.28 | 0.71 | ANOVA F-검정 |
| **SelectKBest k=10** | **10** | **0.33** | **0.76** | **불량 16/21 검출 (k-sweep best)** |

- **recall 0.19 → 0.76**: 품질에서 가장 중요한 "불량을 놓치지 않기"가 크게 개선.
- 트레이드오프: precision은 낮음(false alarm 다수). 단 반도체 품질은 불량 놓침(FN)이 헛경보(FP)보다 치명적이라 recall 우선 방향이 타당.

![k-sweep: 피처 수(k) vs fail F1/recall](docs/figures/secom_feature_k_sweep.png)

**② SPC 관리도(X-bar / R) 구현 및 이탈점 검출**

![X-bar chart & R chart](docs/figures/xbar_and_r_chart.png)

> 캐비아: SECOM은 단일 공정의 균일 시계열이 아니라 subgroup을 임의(연속 5개)로 잘랐고, 관리한계선을 같은 데이터로 산출(Phase 1). 이 관리도는 **SPC 방법론 시연**이지 해당 센서 공정의 품질 판정이 아니다. (자세한 근거는 `docs/devlog.md`)

---

## 실행 (How to run)

```bash
# conda (주 환경 — Python 버전까지 재현)
conda env create -f environment.yml
conda activate wafer_env

# 또는 pip
pip install -r requirements.txt

# 단위 테스트 (SPC 로직 검증)
pytest -q
```

> 데이터 원본(WM-811K, SECOM)은 용량이 커 git에 포함하지 않는다. 다운로드 방법은 `data/README.md` 참고.

---

## 분석 단계 (JD 요구사항 직접 매핑)

> 6시그마 자격이 아니라, 삼성 평가및분석 JD에 실제로 등장하는
> "통계적 품질관리 · 불량분석 · 유효인자 감지 · 수율 개선"에 각 단계를 매핑했다.

### 1. 문제 정의
- [x] 무엇을 풀 것인가: 공정 센서 이상 탐지(SECOM) + 웨이퍼맵 불량 패턴 분류(WM-811K)
- [x] 성공 기준: 불량 recall 개선(놓침 최소화) + 관리도로 이상 Lot 검출
- [x] 왜 중요한가: 수율·품질에 직결 (불량 조기 검출 = 손실 감소)

### 2. 데이터 이해 · 전처리   ← JD: 데이터 분석 역량
- [x] 데이터 적재·구조 파악 (`notebooks/01_eda.py`) — 레거시 pkl 로드·라벨 구조 분석
- [x] 결측(median imputation)·클래스 불균형(class_weight) 처리

### 3. 통계적 분석   ← JD: "통계적 품질관리 / 통계적 공정관리 (SPC)"
- [x] 관리도(X-bar/R), 공정능력 Cp/Cpk (`src/spc.py`, 단위 테스트 `tests/`)
- [x] 590개 피처 중 유효 인자 선별(ANOVA F-검정, k-sweep)   ← JD: "유효 인자 감지"

### 4. 불량 분류 · 예측   ← JD: "A/I 기법 활용 불량 예측"
- [x] baseline 분류(로지스틱 회귀) → 유효인자 선별로 개선
- [ ] CNN 웨이퍼맵 불량분류 (유비전랩 TensorFlow 통합)
- [ ] 이상탐지(Isolation Forest / Autoencoder)

### 5. 결과 · 관리 방안   ← JD: "수율 개선 / 통계적 품질 관리"
- [ ] 결과 시각화·대시보드, 재현 가능하게 정리
- [x] 회고·한계 (`docs/devlog.md`) — 실패·의사결정 기록

---

## 폴더 구조
```
wafer-defect-spc/
├── README.md            # 프로젝트 개요 · 결과 · 실행법
├── environment.yml      # conda 환경 (재현용, 주 환경)
├── requirements.txt     # pip 패키지 (폴백)
├── .gitignore
├── data/                # 데이터셋 (원본은 git에 올리지 않음)
│   └── README.md        # 다운로드 방법
├── notebooks/           # 탐색·실험 (# %% 셀 단위 실행 가능)
│   ├── 01_eda.py        # WM-811K EDA (분포·웨이퍼맵·불균형)
│   ├── 02_spc.py        # SECOM에 관리도 적용·시각화
│   └── 03_secom_baseline.py  # baseline 분류 + 유효인자 선별
├── src/                 # 재사용 로직
│   └── spc.py           # 관리도·Cpk·이탈점 검출 (핵심 로직 직접 구현)
├── tests/               # 단위 테스트
│   ├── conftest.py      # src/ import 경로 설정
│   └── test_spc.py      # spc.py 검증 (손계산 근거 주석 포함)
└── docs/
    ├── devlog.md        # 개발일지 — "왜 이렇게 했는지" 기록 (신뢰도의 증거)
    └── figures/         # README·devlog용 차트 PNG
```

## 신뢰도 원칙 (중요)
- **매일 조금씩 커밋** — 한 번에 몰아 올리지 않는다. 잔디가 곧 "직접 했다"는 증거.
- **devlog.md에 실패·의사결정 기록** — 사람이 한 흔적을 남긴다.
- **모든 코드는 내가 설명할 수 있어야 한다** — 면접 최종 방어선. (핵심 로직·테스트 단언값을 직접 타이핑)
