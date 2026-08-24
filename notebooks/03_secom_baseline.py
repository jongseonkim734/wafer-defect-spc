# %% [markdown]
# ### 03 · SECOM Baseline
# 공정 조건을 기반으로 웨이퍼의 불량 여부를 예측하는 ML 모델의 Baseline을 제작

# %%
# 1. SECOM 데이터 로드 + Fail의 수량과 비율을 확인
import pandas as pd
import numpy as np

# SECOM 데이터 로드
X = pd.read_csv("../data/secom/secom.data", sep=r"\s+", header=None)
labels = pd.read_csv("../data/secom/secom_labels.data", sep=r"\s+", header=None)

# 라벨 변환 (-1이 Pass고 1이 Fail인 것을 0이 Pass고 1이 Fail이 되도록)
# 참고: labels[0]은 라벨, labels[1]은 시간
y = (labels[0] == 1).astype(int)

# 출력부
print("X:", X.shape, "| y:", y.shape)
print(labels.head)
print("Fail:", int(y.sum()), f"({y.mean()*100:.1f}%)")

# %%
# 2. Train 데이터와 Test 데이터로 분할 (Stratified Split)
from sklearn.model_selection import train_test_split

# Train 데이터 80% + Test 데이터 20%
# Stratified Split을 활용하고 random_state를 사용해 시드를 고정해 반복성을 구현
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 출력부
print("train:", X_train.shape, "| test:", X_test.shape)
print("train fail ratio:", round(y_train.mean(), 3), "| test fail ratio:", round(y_test.mean(), 3))

# %%
# 3. 데이터 전처리
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 결측치를 median 값으로 채우는 메소드
# 평균 0, 표준편차 1로 정규화하는 메소드
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

# Train 데이터와 Test 데이터에 imputer와 scaler를 적용
# fit_transform: 데이터의 평균, 중앙값, 표준편차를 계산하고 적용
# transform: 이미 계산된 평균, 중앙값, 표준편차를 단순 적용
# Train 데이터의 평균, 중앙값, 표준편차를 계산해서 이를 Train 데이터와 Test 데이터에 적용
X_train_p = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_p = scaler.transform(imputer.transform(X_test))

# 출력부
print("Pre-processing completed:", X_train_p.shape, X_test_p.shape)

# %%
# 4. 모델 정의, 학습, 예측
# Baseline은 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

# 모델 정의
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

# 모델 학습
model.fit(X_train_p, y_train)

# 모델 예측
y_pred = model.predict(X_test_p)

# %%
# 5. 모델 평가 (confusion_matrix and F1/recall)
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix shows True Negative, False Positive, False Negative, True Positive.
# Format will be [[TN FP] [FN TP]]
# To do that, we need to prediction version of y and actual y value of test set.
print("confusion_matrix\n", confusion_matrix(y_test, y_pred), "\n")

# Classification Report shows precision, recall, f1-score and support.
# precision: TP / (FP + TP). Ratio of actual failures between those are picked as failure by model.
# recall: TP / (FN + TP). Ratio of failure found by model between all failures.
# f1-score:  조화평균 of precision and recall.
# support: number of y_test for each (pass and fail).
print(classification_report(y_test, y_pred, target_names=["pass(0)", "fail(1)"], digits=3))

# %%
# 6. Effective Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif

k = 10
# selector -> select specific amount of features (that's reason why it need y to fit unlikely imputer and scaler)
# To be detailed, selector is supervised training and other two is unsupervised training.
selector = SelectKBest(score_func=f_classif, k=k)   # Use ANOVA F-test and select top k features
X_train_sel = selector.fit_transform(X_train_p, y_train)    # Train with picked features
X_test_sel = selector.transform(X_test_p)   # Just transform as it's test set

selected_idx = selector.get_support(indices=True)
print("selected shape:", X_train_sel.shape)
print("selected feature indices", selected_idx)

# %%
# 7. Re-train with selected top k features and evaluation
# LogisticRegression, confusion_matrix, classification_report are already imported above.
model_sel = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model_sel.fit(X_train_sel, y_train)
y_pred_sel = model_sel.predict(X_test_sel)

print("confusion_matrix\n", confusion_matrix(y_test, y_pred_sel), "\n")
print(classification_report(y_test, y_pred_sel, target_names=["pass(0)", "fail(1)"], digits=3))

# %%
# 8. Draw a k-swip and a curve to find out optimistic k value
# 그 전에, 우리가 10이 가장 좋았다고 하는 것을 어딘가에 dp하면 좋겠다. 이거 진짜 github 꾸미는 방법 어디서 알아내서, 보고서로 적든 뭘 하든 해야겠는데? 안 되면 지금 devlog라고 더 열심히 적어보자...
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score

k_nomis = [3, 5, 10, 15, 20, 30, 50, 80, 120]
f1s, recs = [], []
for kk in k_nomis:
    sel = SelectKBest(f_classif, k=kk)
    X_tr_sel = sel.fit_transform(X_train_p, y_train)
    X_te_sel = sel.transform(X_test_p)
    m_sel = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    m_sel.fit(X_tr_sel, y_train)
    y_pd_sel = m_sel.predict(X_te_sel)

    f1s.append(f1_score(y_test, y_pd_sel))
    recs.append(recall_score(y_test, y_pd_sel))
    print(f"k={kk:3d} | fail F1={f1s[-1]:.3f} | fail recall={recs[-1]:.3f}")

plt.figure(figsize=(8, 4))
plt.plot(k_nomis, f1s, marker="o", label="fail F1")
plt.plot(k_nomis, recs, marker="s", label="fail recall")
plt.axhline(0.14, color="grey", ls="--", lw=1, label="Baseline F1 (all 590 features)")
plt.xlabel("k (number of selected features)"); plt.ylabel("score")
plt.title("SECOM: feature count vs fail-class performance")
plt.legend(); plt.tight_layout()

# save SECOM k-score chart as png
plt.savefig("../docs/figures/secom_feature_k_sweep.png", dpi=150, bbox_inches="tight")

plt.show()

# Pick optimistic k value as the one with hightest f1 score
optimistic_k = k_nomis[int(np.argmax(f1s))]
print("best k by F1:", optimistic_k)

# %%
