# %% [markdown]
# ### 03 SECOM Baseline
# #### To make a baseline model of ML for do the defect 예측 분류

# %%
# 1. Read secom data and check quantity and ratio of failure(defect).
import pandas as pd
import numpy as np

X = pd.read_csv("../data/secom/secom.data", sep=r"\s+", header=None)
labels = pd.read_csv("../data/secom/secom_labels.data", sep=r"\s+", header=None)

# Original database -> -1: pass, 1: fail
# Converted into -> 1: fail, 0: pass
# lables[0] is label, labels[1] is date and time. So we only use labels[0].
# btw, y became 1D value by this line.
y = (labels[0] == 1).astype(int)

print("X:", X.shape, "| y:", y.shape)
# print(X.head)
print(labels.head)
print("Fail:", int(y.sum()), f"({y.mean()*100:.1f}%)")

# %%
# 2. Split the train set and test set based on Stratified split.
from sklearn.model_selection import train_test_split

# Train set 80%. Test set 20%.
# Enable stratify split.
# Fix seed to provide repeatablity of code.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("train:", X_train.shape, "| test:", X_test.shape)
print("train fail ratio:", round(y_train.mean(), 3), "| test fail ratio:", round(y_test.mean(), 3))

# %%
# 3. Data pre-processing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# imputer -> fill NaN with median value.
imputer = SimpleImputer(strategy="median")
# scaler -> standard-ify all features (sensors) with mean=0, var=1.
scaler = StandardScaler()

# fit_transform -> calculate values (median, mean, std) of original data + transform the original data.
# transform -> use pre-calculated values to only transform the original data.
# In our case, we calculate based on train set and implement it into train set AND test set.
X_train_p = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_p = scaler.transform(imputer.transform(X_test))

print("Pre-processing completed:", X_train_p.shape, X_test_p.shape)

# %%
# 4. Model Learning
# The baseline is logistic regression
from sklearn.linear_model import LogisticRegression

# model defining. 1000 iteration + up-weight to rare fail classes as there are so less (~6%) failures.
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

# model fitting
model.fit(X_train_p, y_train)

y_pred = model.predict(X_test_p)

# %%
# 5. Model Evaluation (confusion_matrix and F1/recall)
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
plt.legend(); plt.tight_layout(); plt.show()

# Pick optimistic k value as the one with hightest f1 score
optimistic_k = k_nomis[int(np.argmax(f1s))]
print("best k by F1:", optimistic_k)

# %%
