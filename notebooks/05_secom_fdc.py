# %% [markdown]
# ### 05 · SECOM FDC 축소판 - 공정 이상 시점의 원인 인자 후보 자동 제시
# 실무 FDC(Fault Detection & Classification)는 센서 감지로 이탈을 탐지하고 어떤 인자가 원인인지 좁혀준다.
# 02_spc.py가 센서 감지로 이탈을 탐지함에 해당하므로, 이 노트북에서는 원인 인자 축소 방법론에 집중한다.
# 설계: Pass Lot으로 센서별 기준선을 세우고, 이상 Lot이 기준선에서 얼마나 벗어났는지(z-score)로 원인 인자 후보를 순위화한다.
# 한계: SECOM 데이터의 센서는 익명화되어있어 물리적 의미를 알 수 없다. 즉, 원인 공정을 특정하는 것은 불가능하며, 원인 인자를 축소하는 방법론을 시연하는 데에 만족한다.

# %%
# 1. SECOM 데이터 로드 + 라벨 변환 + 결측 처리
import numpy as np
import pandas as pd

# SECOM 데이터 로드
X_raw = pd.read_csv("../data/secom/secom.data", sep=r"\s+", header=None)
labels = pd.read_csv("../data/secom/secom_labels.data", sep=r"\s+", header=None)
timestamp = pd.to_datetime(labels[1], dayfirst=True, errors="coerce")

# 라벨 변환 (-1이 Pass고 1이 Fail인 것을 0이 Pass고 1이 Fail이 되도록)
# 참고: labels[0]은 라벨, labels[1]은 시간
y = (labels[0] == 1).astype(int)

# 결측 처리 (센서별 중앙값으로 대체: 03_secom_baseline.py와 동일한 전략)
X = X_raw.fillna(X_raw.median())

# 표준편차가 0인 센서는 모든 Lot에서 센서값이 동일해 이탈 판단에 쓰이지 못하므로 감시 대상에서 제외
usable_cols = X.columns[X.std() > 0]
X = X[usable_cols]

# 출력부
print("Lot 수:", len(X))
print("사용 가능한 센서 수:", X.shape[1], "/", X_raw.shape[1])
print("Fail Lot:", int(y.sum()), f"({y.mean()*100:.1f}%)")
print("기간:", timestamp.min(), "~", timestamp.max())

# %%
# 2. Pass Lot 확보 + 감시 대상 센서 선별
from sklearn.feature_selection import SelectKBest, f_classif

# Pass Lot 확보
pass_lot = (y == 0)
baseline_mean = X[pass_lot].mean()
baseline_std = X[pass_lot].std(ddof=1)

# Pass Lot 내부에서만 표준편차가 0인 센서도 감시 대상에서 제외
std_not_zero = baseline_std > 0
X = X.loc[:, std_not_zero]
baseline_mean, baseline_std = baseline_mean[std_not_zero], baseline_std[std_not_zero]

# 감시 대상 센서 선별 (불량과 통계적으로 연관된 상위 K개 센서만 감시 대상 + 03_secom_baseline.py에서는 최고의 K가 10인 것으로 판별)
# selector.get_support(): SelectorKBest로 뽑힌 상위 센서들 자리에는 True, 여타 자리에는 False.
# monitored: 감시 대상 센서 목록
K = 10
selector = SelectKBest(score_func=f_classif, k=K).fit(X, y)
monitored = X.columns[selector.get_support()]

# 출력부
# F값: 해당 센서 값이 Pass 그룹과 Fail 그룹에서 얼마나 다른 지를 나타내는 값
print("기준선 산출에 사용한 Pass Lot 수:", int(pass_lot.sum()))
print("z-score 계산이 가능한 센서 수:", X.shape[1])
print(f"\n감시 대상 센서 {K}개 (ANOVA F-검정 상위):")
for sensor in monitored:
    f_score = selector.scores_[X.columns.get_loc(sensor)]
    print(f" 센서 {sensor:>3} | F={f_score:6.1f} | 정상 평균={baseline_mean[sensor]:10.3f} | 정상 표준편차={baseline_std[sensor]:8.3f}")

# %%
# 3. z-score 기반 이상 Lot 판정 (FDC의 Detection 단계)
# 각 Lot의 센서값이 정상 기준선에서 몇 표준편차 벗어났는지 계산
# 센서값마다 단위와 크기가 제각각이라 z-score로 환산하여 비교
# 이상 Lot: 우리가 Fail로 판단한 Lot.
z = (X[monitored] - baseline_mean[monitored]) / baseline_std[monitored]

# 관리도의 3sigma 관례 적용
# 감시 센서 중 하나라도 3sigma를 벗어난 Lot은 이상 Lot으로 판정
SIGMA = 3
max_abs_z = z.abs().max(axis=1)
detected = max_abs_z > SIGMA

# 라벨과 대조하여 detected 판정이 얼마나 실제 불량을 잡아내는 지 확인
# fail_detected: True Positive
# pass_detected: False Positive
n_fail, n_pass = int((y == 1).sum()), int((y == 0).sum())
fail_detected = int((detected & (y == 1)).sum())
pass_detected = int((detected & (y == 0)).sum())

# 출력부
print(f"{SIGMA}sigma 초과로 탐지된 Lot: {int(detected.sum())}개 / 전체 {len(X)}개")
print(f"실제 Fail 중 탐지: {fail_detected:4d} / {n_fail:4d} (탐지율 {fail_detected/n_fail*100:5.1f}%)")
print(f"실제 Pass 중 탐지: {pass_detected:4d} / {n_pass:4d} (헛경보 {pass_detected/n_pass*100:5.1f}%)")

# %%
# 4. 원인 인자 후보 자동 제시 (FDC의 Classification 단계)
# 이상 Lot으로 판정된 것들에 대해, 감시 센서 중 정상 기준선에서 크게 벗어난 순으로 순위를 매김
# 이는 해당 센서와 연관된 인자(SECOM은 이 연결이 익명화 되어있어 제공할 수 없음)을 먼저 점검하는 현장 조치로 이어질 수 있음

# Lot 별로 zscore의 이탈이 가장 높은 top_n개 센서를 진단(이탈도, 측정값과 Pass Lot의 평균과 비교) 
def diagnose(lot, top_n=5):
    zscore = z.loc[lot]
    ranked = zscore.abs().sort_values(ascending=False)

    # 출력부
    label = "fail" if y[lot] == 1 else "pass"
    print(f"[Lot {lot}] {timestamp[lot]} | 실제 라벨: {label} | 최대 이탈 {zscore.abs().max():.2f} sigma")
    print(f"{'순위':<5}{'센서':<7}{'이탈(sigma)':<14}{'방향':<6}{'측정값':<14}{'정상 평균':<14}")
    for rank, sensor in enumerate(ranked.index[:top_n], start=1):
        direction = "높음" if zscore[sensor] > 0 else "낮음"
        print(f"{rank:<5}{sensor:<7}{zscore[sensor]:<14.2f}{direction:<6}"
              f"{X.loc[lot, sensor]:<14.3f}{baseline_mean[sensor]:<14.3f}")
    print()

# True Positive인 Lot들 만이 진단 대상
targets = X.index[detected & (y == 1)][:3]

# 출력부
print(f"진단 대상 (True Positive): {list(targets)}\n")
for lot in targets:
    diagnose(lot)

# %%
# 5. 진단 결과 시각화
# 감시 센서 10개의 이탈 정도를 Lot 별로 나란히 그려, 어떤 인자가 함께 움직였는 지 시각화
import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(targets), 1, figsize=(9, 3 * len(targets)), sharex=True)
axes = np.atleast_1d(axes)

for ax, lot in zip(axes, targets):
    zscore = z.loc[lot]
    colors = ["tab:red" if abs(v) > SIGMA else "tab:blue" for v in zscore]
    ax.bar([str(s) for s in zscore.index], zscore.values, color=colors)
    ax.axhline(SIGMA, color="red", ls="--", lw=1)
    ax.axhline(-SIGMA, color="red", ls="--", lw=1)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("Deviation (sigma)")
    ax.set_title(f"Lot {lot} (label: {'fail' if y[lot] == 1 else 'pass'})")

axes[-1].set_xlabel("monitored sensor")
plt.tight_layout()
plt.savefig("../docs/figures/secom_fdc_diagnosis.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
