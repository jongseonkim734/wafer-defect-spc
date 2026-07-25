# %% [markdown]
# # 01 · EDA — 웨이퍼맵 첫 탐색
# 목표(킥오프): 데이터가 로드되고, 웨이퍼맵 1장을 그려본다.
# ★ 아래 TODO는 직접 채운다. 막히면 대화에서 로직을 받아 이해 후 타이핑.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. 데이터 로드
# WM-811K는 보통 pandas pickle(.pkl). 경로는 data/README.md 참고.

# %%
# TODO: df = pd.read_pickle("../data/wm811k/<파일명>.pkl")
# TODO: df.shape, df.columns, df.head() 로 구조 확인
df = None  # <- 채우기

# %% [markdown]
# ## 2. 불량 패턴 분포
# failureType 컬럼의 클래스별 개수 → 불균형 확인.

# %%
# TODO: df["failureType"].value_counts() 시각화 (bar chart)

# %% [markdown]
# ## 3. 웨이퍼맵 1장 그려보기
# waferMap 셀 하나는 2D 배열(0=빈공간, 1=정상, 2=불량 식). imshow로 표시.

# %%
# TODO:
# sample = df["waferMap"].iloc[0]
# plt.imshow(sample); plt.title(df["failureType"].iloc[0]); plt.show()

# %% [markdown]
# ## 오늘의 회고 → docs/devlog.md 에 3줄 기록
# - 무엇을 확인했나 / 예상과 달랐던 점 / 다음 할 일
