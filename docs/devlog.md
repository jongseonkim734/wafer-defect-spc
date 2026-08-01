# 개발일지 (devlog)

> 신뢰도의 증거. 매 작업 세션마다 짧게라도 남긴다. "왜 이렇게 했는지 / 무엇이 안 됐는지"를
> 적어두면 (1) 면접에서 술술 설명되고 (2) AI가 대신 만든 게 아니라는 증거가 된다.
> 형식 자유. 3줄이라도 좋다.

---

## 2026-08-01 (킥오프)
- 한 일: 프로젝트 스캐폴딩(README·spc.py·01_eda.py·devlog), WM-811K(LSWMD.pkl 2GB)·SECOM 데이터 확보, `01_eda.py`로 데이터 로드 성공 → `df.shape = (811457, 6)`.
- 다음 할 일: 셀 2 — `label_str`로 `failureType`/`trianTestLabel` 중첩 배열 펴서 분포 확인, 웨이퍼맵 1장 시각화.

### 트러블슈팅 (LSWMD.pkl 로드)
1. **`No module named 'pandas.indexes'`** — 이 pkl은 옛 pandas(0.19대)로 저장됨. 당시 모듈명 `pandas.indexes`가 지금은 `pandas.core.indexes`로 개명됨.
   → 해결: `sys.modules`에 옛 이름을 현재 모듈로 별칭 연결(원본 수정 없이 읽는 쪽에 통역기 삽입).
   ```python
   sys.modules["pandas.indexes"] = pandas.core.indexes
   sys.modules["pandas.indexes.base"] = pandas.core.indexes.base
   ```
2. **`UnicodeDecodeError: 'ascii' codec can't decode byte 0x9a`** — 이 pkl은 Python 2로 저장됨. Py3 기본 ASCII 해석이 특수바이트에서 깨짐.
   → 해결: `pd.read_pickle` 대신 `pickle.load`를 직접 쓰며 `encoding="latin1"`로 로드.
   ```python
   with open(pkl, "rb") as f:
       df = pickle.load(f, encoding="latin1")
   ```
- 배운 점: 공개 레거시 데이터셋은 저장 당시의 라이브러리/파이썬 버전 차이로 바로 안 열릴 수 있고, 파일을 바꾸는 대신 **읽는 쪽 환경을 맞춰주는** 방식이 안전하다.

### 데이터 관찰 (라벨 구조 = 원본의 한계)
- `failureType`·`trianTestLabel` 둘 다 `label_str` 적용 시 `"unlabeled"`가 대량 등장. 버그 아니라 **원본 특성**:
  전체 811,457장 중 사람이 라벨링한 건 **172,950장뿐**, 나머지 **638,507장(79%)**은 빈 배열(`array([])`) → `unlabeled`.
- 라벨링이 "불량패턴 지정 + train/test 배정"을 한 세트로 진행돼서, 두 컬럼의 `unlabeled` 개수가 거의 동일.
- **함의**: 지도학습(분류)은 라벨된 172,950장만 사용 가능(`unlabeled` 필터링 필요). 남는 638k는 향후 비지도/준지도(이상탐지) 소재.
- **근거(문헌 확인)**: 이 라벨 부족은 WM-811K만의 quirk가 아니라 **반도체 제조 데이터의 일반적 문제**. 라벨링에 전문가 수작업이 필요해 실제 팹에서도 극히 일부만 라벨됨 + 클래스 불균형이 심해, 미라벨 데이터를 함께 쓰는 **준지도학습(SSL)** 연구가 활발.
  - WM-811K에서 172,950 라벨 / 638,507 미라벨 명시: [Wafer Map Defect Patterns Semi-Supervised Classification (arXiv:2311.12840)](https://arxiv.org/pdf/2311.12840)
  - 리뷰 논문(팹 결함 검출의 데이터·알고리즘 정리): [Review of wafer defect detection in semiconductor manufacturing (J. Intelligent Manufacturing, Springer)](https://link.springer.com/article/10.1007/s10845-026-02845-z)
  - 업계 매체 설명: [Wafer Bin Map Defect Classification Using Semi-Supervised Learning (SemiEngineering)](https://semiengineering.com/wafer-bin-map-defect-classification-using-semi-supervised-learning/)

<!-- 아래로 계속 추가 -->
