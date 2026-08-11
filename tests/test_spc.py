# Unit tests for src/spc.py
# Run from repo root:  pytest -q
#
# Every expected value below is computed BY HAND in the comments so the
# assertions are verifiable (and defensible in an interview), not magic numbers.

import numpy as np
import pytest

import spc


# --- process_capability -------------------------------------------------
def test_process_capability_known_values():
    # data = [9, 10, 11]
    #   mean  = (9 + 10 + 11) / 3 = 10.0
    #   var   = ((9-10)^2 + (10-10)^2 + (11-10)^2) / (3-1) = (1+0+1)/2 = 1.0
    #   sigma = sqrt(1.0) = 1.0
    data = [9, 10, 11]
    lsl, usl = 7, 13

    result = spc.process_capability(data, lsl, usl)

    assert result["mean"] == pytest.approx(10.0)
    assert result["sigma"] == pytest.approx(1.0)
    # cp = (usl - lsl) / (6 * sigma) = (13 - 7) / (6 * 1) = 6/6 = 1.0
    assert result["cp"] == pytest.approx(1.0)
    # mean(10) sits exactly at the spec midpoint (7..13 -> 10), so no bias:
    #   cpu = (13-10)/(3*1) = 1.0 ,  cpl = (10-7)/(3*1) = 1.0 ,  cpk = min = 1.0
    assert result["cpk"] == pytest.approx(1.0)


# --- CONTROL_CONSTANTS --------------------------------------------------
def test_control_constants_n5():
    # From the table for n = 5: (A2, D3, D4) = (0.577, 0.0, 2.114)
    A2, D3, D4 = spc.CONTROL_CONSTANTS[5]
    assert A2 == pytest.approx(0.577)
    assert D3 == pytest.approx(0.0)
    assert D4 == pytest.approx(2.114)


# --- xbar_r_limits ------------------------------------------------------
def test_xbar_r_limits_shapes_and_center():
    # 3 subgroups of size 5.
    #   subgroup means: 50/5=10.0 , 60/5=12.0 , 50/5=10.0
    #   grand mean (x_center) = (10 + 12 + 10) / 3 = 32/3 = 10.6667
    subgroups = np.array([
        [10, 11, 9, 10, 10],
        [12, 11, 13, 12, 12],
        [9,  10, 10, 11, 10],
    ])
    out = spc.xbar_r_limits(subgroups)

    # xbar / R should have one value per subgroup (3).
    assert out["xbar"].shape == (3,)
    assert out["R"].shape == (3,)
    assert out["x_center"] == pytest.approx(32 / 3)
    # Sanity: UCL must be above center, LCL below.
    assert out["x_ucl"] > out["x_center"] > out["x_lcl"]


def test_xbar_r_limits_rejects_unsupported_n():
    # subgroup size 8 is not in CONTROL_CONSTANTS (2..7).
    subgroups = np.zeros((2, 8))
    with pytest.raises(ValueError):
        spc.xbar_r_limits(subgroups)


# --- out_of_control_points ----------------------------------------------
def test_out_of_control_points():
    values = [5, 12, 8, 1, 9]
    ucl, lcl = 10, 3
    idx = spc.out_of_control_points(values, ucl, lcl)
    # 12 (index 1) is above UCL, 1 (index 3) is below LCL.
    assert list(idx) == [1, 3]
