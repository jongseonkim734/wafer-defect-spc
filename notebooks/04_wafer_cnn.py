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
from keras import layers, models

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
# 테스트 데이터도 불균형한 데이터이므로, CNN 학습 시에도 불균형을 보정해서 진행한다.

# %%
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping

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
# %% [markdown]
# ### 8. 평가 - per-class F1 + confusion matrix
# 7번에서는 학습을 불균형 보정을 통해 진행했다면, 8번에서는 평가 시 불균형 착시를 방지하기 위해
# 다수 class만 맞춰도 점수가 높앙지는 accuracy 대신
# 소수 클래스를 실제로 잡았는 지 더 정확히 파악하기 위해 per-class로 확인한다.

# %%
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# 사용되지 않은 테스트 데이터(X_test)를 기반으로 CNN 모델이 y값 예측(y_pred)을 진행한다.
# CNN 모델 정의에서, 마지막에 Softmax를 활용했는데,
# 그 값 중 가장 큰 값(axis=1)을 y_pred로 규명한다.
y_pred = model.predict(X_test).argmax(axis=1)

# 클래스 별 F1 report를 출력한다.
# 숫자 대신 실제 라벨명(le.classes_)을 활용한다.
print(classification_report(
    yi_test,
    y_pred,
    target_names=le.classes_,
    digits=3,
))

# Confusion matrix(혼동행렬)를 그려내고 저장한다.
# 숫자 대신 실제 라벨명(le.classes_)을 활용한다.
# 행(True Label) 기준 정규화를 진행한다. 각 행의 합이 100%가 되도록.
ConfusionMatrixDisplay.from_predictions(
    yi_test, y_pred,
    display_labels=le.classes_,
    xticks_rotation=45, cmap="Blues",
    normalize="true", values_format=".0%",
)
plt.tight_layout()
plt.savefig("../docs/figures/wafer_cnn_confusion.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 9. Grad-CAM 준비 - CNN 모델이 "어디를 보고 판단하는 지" 시각화하기.
# Grad-CAM: Gradient-weighted Class Activation Mapping. 딥러닝 모델이 이미지의 어느 부분을 보고 판단을 내렸는 지 시각적으로 보여주는 AI 기법.

# %%
import tensorflow as tf

# 네 번째(마지막) Conv2D 층 이름 탐색 (수동으로 이름을 부여하지 않았으나 Sequential이라서 이름이 자동 부여)
last_conv_layer = [l for l in model.layers if isinstance(l, layers.Conv2D)][-1].name
print("마지막 Conv2D 층:", last_conv_layer)

# grad_model: 입력값(inp)을 넣으면 마지막 Conv2D 레이어의 출력 데이터와 최종 예측을 함께 반환하는 객체.
inp = tf.keras.Input(shape=(IMG, IMG, 1))   # 64*64*1채널 웨이퍼 맵이 들어갈 입력 텐서 정의
x = inp                                     # 임시 변수 x 정의
conv_output = None                          # 빈 conv_output 정의
for layer in model.layers:                  # 학습된 layer에 x 재통과
    x = layer(x)                            
    if layer.name == last_conv_layer:       # 마지막 Conv2D 레이어의 출력 데이터를 가로채 conv_output에 저장
        conv_output = x
grad_model = models.Model(inp, [conv_output, x])


def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)             # grad_model에 실제 img_array를 넣은 결과를 conv_out, preds에 저장
        if pred_index is None:                              # pred_index가 없으면 예측 인덱스 사용
            pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]                  # 예측 인덱스의 확률값을 class_score에 저장

    # 예측 점수에 대한 Conv2D 특징맵의 gradient
    grads = tape.gradient(class_score, conv_out)            # class_score를 도출하는 데 사용된 마지막 출력 데이터(conv_out)의 픽셀 별 영향력(기울기)을 편미분하여 도출
    # print("debug(grads): ", grads)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    # 픽셀 별 영향력의 공간적 평균을 내, 채널(8개) 마다 하나의 값으로 변환
    # print("debug(pooled_grads): ", pooled_grads.shape)
    conv_out = conv_out[0]                                  # (1, H, W, C) -> (H, W, C)
    # print("debug(conv_out): ", conv_out)
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]      # 픽셀 별 영향력과 채널 단일 값(pooled_grads)를 행렬곱 -> 가능성이 높은 채널(불량 케이스)의 픽셀 정보가 증폭. -> (H, W, 1)
    # print("debug(heatmap): ", heatmap)
    heatmap = tf.squeeze(heatmap)                           # (H, W, 1) -> (H, W)
    # print("debug(heatmap): ", heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8) # 음수값 제거 및 0~1 정규화
    # print("debug(heatmap): ", heatmap)
    # print("debug(heatmap.numpy()): ", heatmap.numpy())
    return heatmap.numpy()

# %% [markdown]
# ### 10. 원본 위에 히트맵을 겹쳐 시각화

# %%
# 전체 test 예측 한 번만 계산 (인덱스 선택 및 시각화에 재사용)
y_pred_all = model.predict(X_test, verbose=0).argmax(axis=1)

# 오분류 예시 인덱스 찾기: 실제 Scratch인데 Loc로 예측된 것
# 해당 케이스 중, 앞 5개만 추출해서 보여준다.
sc, lo = le.transform(["Scratch"])[0], le.transform(["Loc"])[0]
confused = np.where((yi_test == sc) & (y_pred_all == lo))[0]
print("Scratch -> Loc 오분류 예시 인덱스:", confused[:5])

def show_gradcam(i, ax_map, ax_cam):
    img = X_test[i:i+1]                                         # (1, 64, 64, 1)
    heatmap = make_gradcam_heatmap(img, grad_model)

    # 왼쪽: 원본 웨이퍼맵
    ax_map.imshow(img[0, :, :, 0], cmap="gray")
    ax_map.set_title(f"true: {le.classes_[yi_test[i]]}")
    ax_map.axis("off")

    # 오른쪽: 원본 웨이퍼맵 + 히트맵 (32 -> 64 자동 확대)
    # jet: blue -> red. 빨간색일수록 AI가 판단에 더 참고한(활성화한) 부분이라는 뜻.
    ax_cam.imshow(heatmap, cmap="jet", alpha=0.5, extent=(0, IMG, IMG, 0), interpolation="bilinear")
    ax_cam.set_title(f"pred: {le.classes_[y_pred_all[i]]}")
    ax_cam.axis("off")

# 보고 싶은 test 인덱스 (원하는 인덱스로 변경 및 추가 가능)
idxs = [confused[0], confused[3], confused[4]]
fig, axes = plt.subplots(len(idxs), 2, figsize=(6, 3 * len(idxs)))
for row, i in enumerate(np.atleast_1d(idxs)):
    show_gradcam(i, axes[row, 0], axes[row, 1])

plt.tight_layout()
plt.savefig("../docs/figures/wafer_cnn_gradcam.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
