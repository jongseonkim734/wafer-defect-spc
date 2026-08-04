# SPC (Statistical Process Control) 핵심 로직

import numpy as np

# Calculates Cp and Cpk.
# data: comes from 'data/secom/SECOM.data'.
# lsl: lower spec limit. Defined by customer.
# usl: upper spec limit. Defined by customer.
# Cp: Capability of Process. Decreases as standard deviation increases.
# Cpk: Capability of Process, Katayori. Cp + Katayori(bias level) considered.
# Normally Cpk is used and Cpk over 1.33 is considered as sufficient.
def process_capability(data, lsl, usl):
    data = np.asarray(data, dtype=float) # Fix type of 'data' as array
    mean = data.mean()
    sigma = data.std(ddof=1) # standard deviation that divided in (n-1)

    cp = (usl - lsl) / (6 * sigma)
    cpu = (usl - mean) / (3 * sigma)
    cpl = (mean - lsl) / (3 * sigma)
    cpk = min(cpu, cpl)

    return {"cp": cp, "cpk": cpk, "mean": mean, "sigma": sigma}


# Pre-calculated values of control constants.
# n: size of subgroup.
# A2: X-bar chart 한계선에 사용되는 상수. For x_ucl, x_lcl.
# D3: R chart 하한에 사용되는 상수. For r_lcl.
# D4: R chart 상한에 사용되는 상수. For r_ucl.
# 상수의 구조는 n: (A2, D3, D4)
CONTROL_CONSTANTS = {
    2: (1.880, 0.0, 3.267),
    3: (1.023, 0.0, 2.574),
    4: (0.729, 0.0, 2.282),
    5: (0.577, 0.0, 2.114),
    6: (0.483, 0.0, 2.004),
    7: (0.419, 0.076, 1.924),
}

# Calculate X-bar chart and R chart and related values.
# subgroup: 최대한 동일한 조건에 뽑은 측정값들의 묶음. (e.g., 동일 Lot의 Wafer n장)
# Control Chart: 측정값을 시간 순으로 찍고, 가운데 중심선과 위아래 관리한계선(UCL, LCL)을 그어 공정의 안정성을 보는 차트.
#              : X-bar chart and R chart 모두 Control Chart.
# X-bar chart: subgroup들의 평균(X-bar)을 시간순으로 보여준 차트
# R chart: subgroup들의 범위(Range = max-min)를 시간순으로 보여준 차트
# X-bar chart의 상한(x_ucl)과 하한(x_lcl)을 계산할 때 Rbar가 사용되므로 실무적으로는 R chart를 먼저 확인한 뒤 X-bar chart를 확인
def xbar_r_limits(subgroups):
    subgroups = np.asarray(subgroups, dtype=float)

    n = subgroups.shape[1]

    if n not in CONTROL_CONSTANTS:
        raise ValueError(f"subgroup 크기 {n}는 상수표(2~7)에 없음")
    A2, D3, D4 = CONTROL_CONSTANTS[n]

    xbar = subgroups.mean(axis=1) # 각 subgroup의 평균을 한 변수(xbar)에 저장
    r = subgroups.max(axis=1) - subgroups.min(axis=1) # 각 subgroup의 범위를 한 변수(r)에 저장

    xbarbar = xbar.mean() 
    Rbar = r.mean()

    return {
        "xbar": xbar,
        "R": r,
        "x_center": xbarbar,
        "x_ucl": xbarbar + A2 * Rbar,
        "x_lcl": xbarbar - A2 * Rbar,
        "r_center": Rbar,
        "r_ucl": D4 * Rbar,
        "r_lcl": D3 * Rbar,
    }


# To check OoC (the points that's outside of ucl or lcl) points.
# values: X-bar chart values or R chart values
def out_of_control_points(values, ucl, lcl):
    values = np.asarray(values, dtype=float)
    return np.where((values > ucl) | (values < lcl))[0]

