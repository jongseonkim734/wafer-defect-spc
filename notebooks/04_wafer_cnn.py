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

# %% [markdown]
# ### 5. Train / Val / Test Stratified Split
# Rare class (e.g., Near-full only got 149 cases) should be kept its class ratio in every split
# -> that's why we use Stratified Split

# %%
from sklearn.model_selection import train_test_split

# 1) Split 20% as Test
# train_test_split -> Split half every inserted arrays.
# -> We put three arrays (X, y, y_int) and it is splitted into half
# -> Get 6 variables.
X_trainval, X_test, y_trainval, y_test, yi_trainval, yi_test = train_test_split(
    X, y, y_int, test_size=0.2, stratify=y_int, random_state=42
)

# 2) Split Train and Val (Val is 20% of trainval)
X_train, X_val, y_train, y_val, yi_train, yi_val = train_test_split(
    X_trainval, y_trainval, yi_trainval, test_size=0.2, stratify=yi_trainval, random_state=42
)

print("train:", X_train.shape[0], "\nval:", X_val.shape[0], "\ntest:", X_test.shape[0])

# %%
# ### 6. CNN 모델 정의
# [Conv2D -> Conv2D -> MaxPooling2D] *2 -> Flatten -> Dense -> Dropout -> Dense(softmax)

# %%
from tensorflow.keras import layers, models

def build_cnn(input_shape, n_classes):
    model = models.Sequential([
        # Preset: Set the input data type of model
        layers.Input(shape=input_shape),
        
        # Block 1: 3x3 size 32 filters, actication function is relu, output size is same as input size.
        # Repeat it twice. + 2x2 Pooling (reducing the size)
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Block 2: 3x3 size 32 filters, actication function is relu, output size is same as input size.
        # Repeat it twice. + 2x2 Pooling (reducing the size)
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # 2D vector -> 1D vector
        layers.Flatten(),
        # Connect 1D vector with 128 nodes
        # Use relu to add non-linearity
        layers.Dense(128, activation="relu"),
        # Reduce 50% of node (128 -> 64)
        layers.Dropout(0.5),
        # Connect 1D vector (with 64 nodes) with n_classes(=8) nodes
        # Use softmax to make sum as 1
        layers.Dense(n_classes, activation="softmax"),
    ])

    # adam: Adaptive Moment Estimation. Weight optimizing algorithm that self-fix the Learning Rate. 
    # categorial_crossentropy: compare expectation possibility with one-hot values
    # use accuracy as a metrics
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

model = build_cnn((IMG, IMG, 1), n_classes)
model.summary()

# %% [markdown]
# ### 7. 학습 (class_weight로 불균형 보정)

# %%
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

# "balanced": Higher weight for rarer classes
# -> prevents model learning on only for major classes
weights = compute_class_weight(
    "balanced", classes=np.unique(yi_train), y=yi_train,
)
class_weight = dict(enumerate(weights))   # Dictionary-ify the weights: {0: 2.5, 1:0.3, ...}
print(class_weight)

# Stop when val_loss stops improving + give 4 epoch delay on stop + keep the best epoch's weights betweenn all widgets
early = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

# Where actual training happens
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[early]
)
# %%
