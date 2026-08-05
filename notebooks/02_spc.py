# %% [markdown]
# ### 02 · SPC (Statistical Process Control)
# ### notebooks/02_spc.py

# %%
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent / "src"))
from spc import process_capability, xbar_r_limits, out_of_control_points

# %%
# 1. Test the process_capability
import numpy as np
data = np.random.normal(100, 2, 1000)
print(process_capability(data, lsl=94, usl=106))

# %%
# 2. Test the xbar_r_limits
import numpy as np

# 정상 공정 흉내
data = np.random.normal(100, 2, size=(20, 5)) # 평균 100, 표준편차 2, 크기 5인 subgroup 20개.
limits = xbar_r_limits(data)

print("X-bar 중심:", round(limits["x_center"], 2))
print("X-bar UCL/LCL:", round(limits["x_ucl"], 2), round(limits["x_lcl"], 2))
print("R 중심:", round(limits["r_center"], 2))

# %%
# 3. Test the out_of_control_points
import numpy as np
vals = np.array([9, 10, 11, 25, 10, -5, 10])
print(out_of_control_points(vals, ucl=20, lcl=0))

# %%
# 4. Load SECOM data and check how many NaNs
import pandas as pd
import numpy as np

secom = pd.read_csv("../data/secom/secom.data", sep=r"\s+", header=None)
print(secom.shape)
print("총 NaN:", int(secom.isna().sum().sum())) # isna -> NaN 여부를 Boolean으로 판단. sum()2개 -> 열 단위 합산 후 전체 합산

# %%
# 5. Select useful columns and split them into subgroup
# useful column -> Low NaN ratio + Variance exists
# If Variance doesn't exist, that feature is constant for every Lot -> meaningless to analyze
na_ratio = secom.isna().mean()
variance = secom.var()
good_cols = secom.columns[(na_ratio < 0.05) & (variance > 0)]
column_num = 421               # Modify this number to pick different column (different feature) between 0 ~ num(good_cols)
col = good_cols[column_num]    # [column_num]th column in good_cols
print("고른 센서 열(=좋은 첫 번째 센서 열):", col)

series = secom[col].dropna().to_numpy()    # NaN 제거 후 1D 시계열로 변환

n = 5   # size of subgroup
usable = len(series) - (len(series) % n)    # n으로 나눠떨어지게 잘랐을 때 사용할 수 있는 데이터 수
subgroups = series[:usable].reshape(-1, n)  # usable로 데이터를 자른 뒤, n열의 2차원 배열로 변환
print("subgroups shape:", subgroups.shape)

# %%
# 6. Implement Control Chart and Visualize
import matplotlib.pyplot as plt

lim = xbar_r_limits(subgroups)
xbar, R = lim["xbar"], lim["R"]
idx = np.arange(len(xbar))  # subgroup의 idx를 만듦. 이후 plot의 x축에 사용됨.
ooc_x = out_of_control_points(xbar, lim["x_ucl"], lim["x_lcl"])
ooc_r = out_of_control_points(R, lim["r_ucl"], lim["r_lcl"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

# --- X-bar chart ---
ax1.plot(idx, xbar, marker="o", ms=3, lw=0.7, color="steelblue")
ax1.axhline(lim["x_center"], color="green", lw=1, label="center")
ax1.axhline(lim["x_ucl"], color="red", ls="--", lw=1, label="UCL/LCL")
ax1.axhline(lim["x_lcl"], color="red", ls="--", lw=1)
ax1.scatter(ooc_x, xbar[ooc_x], color="red", zorder=5, label="out of control")
ax1.set_title(f"X-bar chart (sensor {col})"); ax1.set_ylabel("subgroup mean"); ax1.legend(fontsize=8)

# --- R chart ---
ax2.plot(idx, R, marker="o", ms=3, lw=0.7, color="darkorange")
ax2.axhline(lim["r_center"], color="green", lw=1)
ax2.axhline(lim["r_ucl"], color="red", ls="--", lw=1)
ax2.axhline(lim["r_lcl"], color="red", ls="--", lw=1)
ax2.scatter(ooc_r, R[ooc_r], color="red", zorder=5)
ax2.set_title("R chart"); ax2.set_ylabel("subgroup range"); ax2.set_xlabel("subgroup #")

plt.tight_layout(); plt.show()

print(f"X-bar 이탈: {len(ooc_x)}개 / R 이탈: {len(ooc_r)}개 / 총 {len(xbar)} subgroup")

# %%
