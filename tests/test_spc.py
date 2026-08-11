# Unit tests for src/spc.py
# Run from repo root:  pytest -q
#
# HOW TO FILL THIS IN (do the math BY HAND — that is the point):
# For each TODO, compute the expected value yourself with a calculator,
# then assert against it. Use pytest.approx for floats.

import numpy as np
import pytest

import spc


# --- process_capability -------------------------------------------------
def test_process_capability_known_values():
    # A tiny, hand-checkable sample.
    data = [9, 10, 11]            # mean = 10, sigma(ddof=1) = 1.0
    lsl, usl = 7, 13

    result = spc.process_capability(data, lsl, usl)

    assert result["mean"] == pytest.approx(10.0)
    assert result["sigma"] == pytest.approx(1.0)
    # TODO: cp = (usl - lsl) / (6 * sigma) = ?
    assert result["cp"] == pytest.approx(...)   # <-- fill in
    # TODO: mean is centered, so cpk == cp here. Confirm and fill in.
    assert result["cpk"] == pytest.approx(...)  # <-- fill in


# --- CONTROL_CONSTANTS --------------------------------------------------
def test_control_constants_n5():
    A2, D3, D4 = spc.CONTROL_CONSTANTS[5]
    assert A2 == pytest.approx(0.577)
    # TODO: fill D3, D4 for n=5 from the table
    assert D3 == pytest.approx(...)  # <-- fill in
    assert D4 == pytest.approx(...)  # <-- fill in


# --- xbar_r_limits ------------------------------------------------------
def test_xbar_r_limits_shapes_and_center():
    # 3 subgroups of size 5.
    subgroups = np.array([
        [10, 11, 9, 10, 10],
        [12, 11, 13, 12, 12],
        [9,  10, 10, 11, 10],
    ])
    out = spc.xbar_r_limits(subgroups)

    # xbar / R should have one value per subgroup (3).
    assert out["xbar"].shape == (3,)
    assert out["R"].shape == (3,)
    # TODO: compute xbarbar (grand mean) by hand and assert x_center.
    assert out["x_center"] == pytest.approx(...)  # <-- fill in
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
