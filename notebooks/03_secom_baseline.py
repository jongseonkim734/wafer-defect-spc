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

# NaN is filled with median value.
imputer = SimpleImputer(strategy="median")
# All features (sensors) are standard-ify with avg 0, var 1.
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
