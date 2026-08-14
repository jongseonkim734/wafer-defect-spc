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

# %% [markdown]
# ### 3. 웨이퍼맵 64*64로 리사이즈
# 웨이퍼맵은 다양한 사이즈로 존재하나 CNN은 고정된 규격을 요구하므로 리사이즈를 진행한다.
# 각 웨이퍼맵의 셀들은 0, 1, 2의 값을 가진다. (0은 빈공간, 1은 정상, 2는 불량) -> 고로 보간은 진행하지 않는다.

# %%
from skimage.transform import resize

IMG = 64 # target grid size

def resize_map(wafer_map):
    m = np.asarray(wafer_map, dtype=float)
    return resize(
        m,
        (IMG, IMG),
        order=0,                # use nearest-neighbor value (not interpolation)
        preserve_range=True,    # Preserve 0~2 range (don't autoscale it into 0~1)
        anti_aliasing=False,    # Turn off aliasing
    )

X = np.stack([resize_map(m) for m in labeled["waferMap"]])
X = X / 2.0                 # 0, 1, 2 -> 0.0, 0.5, 1.0
X = X[..., np.newaxis]      # simply add new axis with full of '1' which is considered as number of channel at CNN. CNN needs the number of channel explicitly.

print("X shape:", X.shape)  # expect (25519, 64, 64, 1)

# %% [markdown]
# ### 4. 라벨 인코딩
# 'Center' -> 3 (int) -> [0, 0, 0, 1, 0, 0, 0, 0] (one-hot that softmax + categorical_crossentropy requires)
# one-hot: just a vector
# softmax throws 8 possibilities and categorial_crossentropy compares them with one-hot.

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_int = le.fit_transform(labeled["fail_str"])   # 8 failure classes -> 0, 1, 2, 3, 4, 5, 6, 7                   
n_classes = len(le.classes_)                    # le.classes_ -> label list (Center, Donut, ..)
y = np.eye(n_classes, dtype='float32')[y_int]   # 0, 1, 2, 3, 4, 5, 6, 7 -> one-hot

print("classes:", list(le.classes_))
print("y_shape:", y.shape)                      # expect (25519, 8)
