# wafer-defect-spc

반도체 웨이퍼 불량 데이터를 **통계적 공정관리(SPC)** 와 **머신러닝** 두 축으로 분석하는 프로젝트.
국내 메모리 반도체의 **전공정 수율·품질**(SK하이닉스 양산기술 · 삼성전자 공정기술)에서 요구하는
"공정 이해 · 불량 분석 · 유효 인자 감지 · 통계적 공정관리 · 수율 개선"을 실제 데이터로 구현·증명하는 것을 목표로 한다.

> **분석 문서 구조는 6시그마의 DMAIC 프레임을 따른다.** (자격증 대신 방법론을 프로젝트로 체화)

---

## 핵심 결과 (Results)

> **프로젝트 구조 — 하나의 목표, 두 층위.** "불량을 놓치지 않아 수율을 지킨다"는 목표를 전공정의 두 지점에서 공략한다:
> **① 공정 중 센서 데이터**(SECOM)로 불량을 예측하고 유효 인자를 선별(원인 층), **② 공정 후 웨이퍼 불량맵**(WM-811K)을 CNN으로 분류·해석(결과 층).
> 두 층 모두 **클래스 불균형에서 소수 불량의 recall을 우선**하는 같은 품질 철학을 따르며, 그 위에 **SPC**가 통계적 공정관리의 뼈대를 이룬다.

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

**③ WM-811K 웨이퍼 불량 8종 CNN 분류**

25,519장의 웨이퍼 불량 맵을 64×64로 리사이즈해 CNN 모델로 8종 분류. 불균형(Edge-Ring 9,680장, Near-full 149장)을 보정하기 위해 class_weight="balanced"를 활용.

> 불균형 데이터라 accuracy는 다수 클래스 착시가 있어, **macro F1을 주지표**로 두고 per-class로 정직하게 평가한다.

| 지표 | 값 |
|---|---|
| **macro F1** (주지표) | **0.842** |
| 소수 클래스(Near-full, 30장) | recall 0.97 |
| 최약 클래스(Scratch) | recall 0.57 · F1 0.568 |

- **불균형 보정 검증**: 표본 30장 뿐인 Near-full까지 recall 0.97로 검출 → class_weight가 소수 클래스를 잘 보정했음을 per-class로 확인.
- **오류의 물리적 해석**: Scratch의 약 33%가 Loc로 오분류. Scratch와 Loc이 공간적으로 유사한 패턴이라 모델의 혼동이 도메인적으로 납득 가능.

![WM-811K CNN confusion matrix (행 정규화)](docs/figures/wafer_cnn_confusion.png)

- **Grad-CAM으로 오류 근거 검증**: Scratch→Loc 오분류 사례의 히트맵이 노이즈가 아니라 실제 스크래치 궤적에 집중 → 모델은 결함 위치를 옳게 짚되, 선형 궤적을 국소 덩어리(Loc)로 해석. 오류의 원인이 오작동이 아니라 **형태적 경계 모호성**임을 시각적으로 확인.

![WM-811K CNN Grad-CAM (Scratch→Loc 오분류)](docs/figures/wafer_cnn_gradcam.png)

**불량 패턴 → 전공정 공정 원인 (불량 분류를 공정 개선으로 잇는 다리)**

웨이퍼 불량맵은 EDS(전공정의 마지막 관문)에서 나온 **전공정 수율 결과물**이다. 따라서 불량 패턴 분류는 그 자체가 목적이 아니라, **어느 전공정 단계를 개선해야 하는지 진단하는 출발점**이다. 각 패턴은 대표적으로 다음 공정을 가리킨다(실제 근본원인은 FA로 확정).

| 패턴 | 대표 공정 원인 | 관련 전공정(양산기술) |
|---|---|---|
| Center | 증착/CMP 중심부 불균일, 스핀코팅 | ThinFilm · CMP · Photo |
| Donut | 반경방향(radial) 불균일 | CMP · ThinFilm |
| Edge-Ring | 가장자리 식각/증착 불균일, edge exclusion | Etch · ThinFilm |
| Edge-Loc | 가장자리 국소 이상(척 접촉·핸들링) | Etch · 장비 Set-up |
| Loc | 국소 오염·파티클 | Cleaning · 파티클 관리 |
| Near-full | 대형 공정 이탈(레시피·장비 이상) | 공정 안정화 · 장비 Set-up |
| Random | 무작위 파티클 오염(환경·자재) | Cleaning · 환경 |
| Scratch | 기계적 핸들링 손상, CMP 스크래치 | CMP · 핸들링(장비 Set-up) |

→ 8개 패턴이 모두 양산기술이 운영하는 전공정(Photo/Etch/ThinFilm/Cleaning&CMP + 장비 Set-up)으로 추적된다. **불량 분석 → 공정 원인 진단 → 조건 최적화**로 이어지는 수율 개선 고리의 앞단.

**④ Scratch 개선 실험 2종 — 공정한 A/B, 그리고 정직한 결론**

Grad-CAM 진단을 바탕으로 두 가지 처방을 **동일 조건 A/B**로 검증했으나, 둘 다 baseline을 넘지 못했다.

| 실험 | 가설 | 결과 |
|---|---|---|
| 회전/반전 증강 | 방향 불변성을 학습해 Scratch 다양성 확보 | Scratch→Loc은 줄었으나 Loc→Scratch가 폭증, 전 클래스 F1 하락(순손실) |
| 해상도↑ (64→96) | 가는 선 형태를 더 보존 | macro 0.842→0.807, Scratch F1 0.568→0.457 (64 완승) |

- **결론**: Scratch↔Loc 혼동은 전처리로 지울 수 있는 인공물이 아니라, **이 데이터·구조 규모에서의 진짜 형태적 모호성**이다. baseline(64 + class_weight)이 견고함을 두 실험이 역으로 입증. 진짜 해결은 더 깊은/attention 구조 또는 Scratch 라벨 확충 방향으로 남긴다. (상세: `docs/devlog.md` 08/22)
- 부수 수확: 큰 모델에서 val_loss·val_accuracy가 어긋나 EarlyStopping이 undertrain된 모델을 복원하는 현상을 발견하고, monitor를 분류 품질(val_accuracy)에 맞게 교정.

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

> 6시그마 자격이 아니라, 전공정 공정기술·양산기술 JD(SK하이닉스·삼성전자)에 실제로 등장하는
> "공정 운영·최적화 · 통계적 공정관리 · 불량분석 · 유효인자 감지 · 수율 개선"에 각 단계를 매핑했다.

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
