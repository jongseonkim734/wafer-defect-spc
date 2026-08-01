# %% [markdown]
# # 01 · EDA — 웨이퍼맵 첫 탐색
# # EDA: Exploratory Data Analysis
# 목표(킥오프): 분석 모델을 만들기 전에, 데이터가 로드되고, 웨이퍼맵 1장을 그려보면서 생김새, 분포, 결측, 이상, 불균형 등을 짧게 파악.
# ★ 아래 TODO는 직접 채운다. 막히면 대화에서 로직을 받아 이해 후 타이핑.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. 데이터 로드
# WM-811K는 보통 pandas pickle(.pkl). 경로는 data/README.md 참고.
import os
print(os.getcwd())
from pathlib import Path
pkl = Path("../data/wm811k/LSWMD.pkl")
print("찾는 경로:", pkl.resolve())
print("존재 여부:", pkl.exists())

# %%
# 데이터 로드 코드
import sys
import pickle
import pandas.core.indexes

# 구 pandas 모듈 이름과 현재 이름 간 다리 놓기
sys.modules["pandas.indexes"] = pandas.core.indexes
sys.modules["pandas.indexes.base"] = pandas.core.indexes.base

# Python 2 pickle이라 latin1으로 디코딩해서 로드
with open(pkl, "rb") as f:
    df = pickle.load(f, encoding="latin1")

# 로드 결과 확인
print(df.shape)
print(df.columns.tolist())
df.head()

# %% [markdown]
# ## 2. 불량 패턴 분포
# failureType 컬럼의 클래스별 개수 → 불균형 확인.

# %%
# trianTestLabel과 failureType의 중첩 array를 벗기는 함수
def label_str(x):
    return x[0][0] if len(x) > 0 else "unlabeled"

df["split_str"] = df["trianTestLabel"].apply(label_str)
df["fail_str"] = df["failureType"].apply(label_str)

# 중첩 array가 벗겨진 split_str에 대해서 표로 내부 항목들의 개수 집계
print("WM-811K Test or Training data distribution")
print(df["split_str"].value_counts())

# 중첩 array가 벗겨진 fail_str에 대해서 unlabeled와 none을 제외하고
# 막대그래프로 불량 패턴 8종의 개수 시각화
counts = df["fail_str"].value_counts()
defects = counts.drop(["unlabeled", "none"], errors="ignore")
ax = defects.plot(kind="bar", figsize=(8, 4), color="indianred")
ax.set_ylabel("count")
ax.set_title("WM-811K 8 failure type distribution")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. 웨이퍼맵 1장 그려보기

# unlabeled도 아니고 none도 아닌 값 중 첫 번째 항목의 df 내 위치를 파악한 뒤
# 해당 위치의 waferMap 열과 fail_str을 활용해 그림을 그림
# waferMap 셀 하나는 2D 배열(0=빈공간, 1=정상, 2=불량 식). imshow로 표시.
labeled = df[(df["fail_str"] != "unlabeled") & (df["fail_str"] != "none")]
sample = labeled.iloc[0]
plt.imshow(sample["waferMap"]); plt.title(sample["fail_str"]); plt.colorbar(); plt.show()

# %%
# TODO:
# sample = df["waferMap"].iloc[0]
# plt.imshow(sample); plt.title(df["failureType"].iloc[0]); plt.show()

# %% [markdown]
# ## 오늘의 회고 → docs/devlog.md 에 3줄 기록
# - 무엇을 확인했나 / 예상과 달랐던 점 / 다음 할 일
