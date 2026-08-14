# %% [markdown]
# ## 04 · WM-811K 웨이퍼맵 CNN 불량분류

# %% [markdown]
# ### 1. 라벨된 데이터 로드
# WM-811K = 811,457 wafers. Legacy pickle(old pandas / Python 2).

# %%
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas.core.indexes

# Bridge old pandas module names -> current ones (pkl saved on pandas 0.19)
sys.modules["pandas.indexes"] = pandas.core.indexes
sys.modules["pandas.indexes.base"] = pandas.core.indexes.base

# This pkl was written by Python 2 -> decode bytes with latin1
pkl = Path("../data/wm811k/LSWMD.pkl")
with open(pkl, "rb") as f:
    df = pickle.load(f, encoding="latin1")

print(df.shape)     # expected to get (811457, 6)
print(df.columns.tolist())

# %% [markdown]
# ### 2. 8종 불량 패턴만 필터 + 분포 확인
# failureType is a nested array -> unwrap to a plain string.

# %%
def label_str(x):
    return x[0][0] if len(x) > 0 else "unlabeled"

df["fail_str"] = df["failureType"].apply(label_str)

# keep only the 8 real defect patterns (drop 'unlabeled' and 'none')
labeled = df[(df["fail_str"] != "unlabeled") & (df["fail_str"] != "none")].copy()

print("labeled wafers:", len(labeled))
print(labeled["fail_str"].value_counts())

# %%
