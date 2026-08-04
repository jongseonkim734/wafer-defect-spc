# %% [markdown]
# ### 02 · SPC
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
