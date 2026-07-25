"""
SPC (Statistical Process Control) 핵심 로직.

★ 규칙: 함수 몸통은 직접 채운다. 아래는 시그니처와 무엇을 계산해야 하는지 힌트만.
   막히면 대화에서 '핵심 로직 코드'를 받아 이해한 뒤 직접 타이핑할 것.
"""

import numpy as np


def process_capability(data, lsl, usl):
    """
    공정능력지수 Cp, Cpk 계산.

    Cp  = (USL - LSL) / (6 * sigma)
    Cpk = min( (USL - mean) / (3*sigma),  (mean - LSL) / (3*sigma) )

    Parameters
    ----------
    data : array-like  측정값
    lsl, usl : float   규격 하한/상한 (Lower/Upper Spec Limit)

    Returns
    -------
    dict  {"cp": ..., "cpk": ..., "mean": ..., "sigma": ...}
    """
    # TODO: mean, sigma 구하고 위 공식대로 cp, cpk 계산해서 반환
    raise NotImplementedError


def xbar_r_limits(subgroups):
    """
    X-bar / R 관리도의 관리한계선(control limits) 계산.

    - 각 subgroup의 평균(x-bar)과 범위(R)를 구한다
    - 전체 평균 x-double-bar, 평균 범위 R-bar
    - 관리도 상수 A2, D3, D4 (subgroup 크기 n에 따라 표에서 결정)
      UCL_x = x̿ + A2 * R̄ ,  LCL_x = x̿ - A2 * R̄
      UCL_r = D4 * R̄     ,  LCL_r = D3 * R̄

    Parameters
    ----------
    subgroups : 2D array-like  (샘플 수 × subgroup 크기)

    Returns
    -------
    dict  중심선/관리한계 값들
    """
    # TODO: 관리도 상수표(A2, D3, D4)는 subgroup 크기 n으로 찾아 상수로 둔다
    raise NotImplementedError


def out_of_control_points(values, ucl, lcl):
    """관리한계를 벗어난 점들의 인덱스 반환 (가장 단순한 이상 규칙)."""
    # TODO: values 중 ucl 초과 또는 lcl 미만인 위치를 찾아 반환
    raise NotImplementedError
