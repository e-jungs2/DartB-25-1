# 캐글로리
## Porto Seguro 필사노트
---
### 1. 라이브러리 불러오기
```python
import collections as cl
import functools as ft
import gc
import os.path
import pprint
import time
import warnings as w

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import torch as t
import torch.nn.functional as tnf

from sklearn.model_selection import train_test_split
from IPython.display import display as ipython_display
```
- 데이터 전처리, 시각화, 모델링을 위한 전반적인 라이브러리 셋업.
- PyTorch까지 포함되어 있지만, 이 노트에서는 주로 사용되지 않음.

---

### 2. EDA 함수
```python
def println(*args, **kwargs):
    print(*args, **kwargs)
    print("")

def pprintln(title, *args, **kwargs):
    print(f"{title}:")
    kwargs.setdefault("indent", 2)
    pprint.pp(*args, **kwargs)
    print("")
```
- 출력 결과를 보기 좋게 정리하기 위한 간단한 보조 함수.

```python
plot_layout = (1, 1)
next_subplot = 0

def plt_layout(rows, cols, figsize):
    global plot_layout, next_subplot
    plot_layout = (rows, cols)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    fig.tight_layout()
    next_subplot = 0
    return fig

def plt_next(title):
    global plot_layout, next_subplot
    next_subplot += 1
    ax = plt.subplot(plot_layout[0], plot_layout[1], next_subplot)
    plt.title(title)
    return ax

def plt_hist(title, *args, **kwargs):
    plt_next(title)
    return plt.hist(*args, **kwargs)

def plt_scatter(t1, t2, d1, d2, *args, **kwargs):
    ax = plt_next(f"{t1} vs. {t2}")
    ax.set_xlabel(t1)
    ax.set_ylabel(t2)
    return plt.scatter(d1, d2, *args, **kwargs)

def plt_show():
    plt.show()
    next_subplot = 0

```
- 시각화 레이아웃 설정 함수.

```python
def gen_histograms_and_tables(data, columns, target_col, target_cols, with_all=True):
    target_vals = sorted(data[target_col].unique())
    need_complement = len(target_vals) > 2
    cols = len(target_vals)
    rows = 2 if need_complement else 1
    plot_height = rows * 3
    colors = [
        (0.9, 0.0, 0.0),
        (0.9, 0.7, 0.0),
        (0.0, 0.8, 0.1),
        (0.9, 0.2, 0.9),
    ]
    colors = {
        tv: color
        for tv, color in zip(target_vals, colors)
    }
    background_color = "lightgrey"
    all_color = "darkblue"
    data_grouped = {
        tv: data[data[target_col] == tv] for tv in target_vals

      }

    for col in columns:
        if col in target_cols:
            continue

        print(f"#### {col}")

        unique_vals = sorted(data[col].unique())
        hist_kwargs = {
            "bins": min(20, len(unique_vals)),
            "range": (data[col].min(), data[col].max()),
        }
        crosstab = {}
        header = [f"{col} ↓ / Target →"]
        with_crosstab = col.endswith("_cat") or col.endswith("_bin")

        if with_crosstab:
            crosstab = {v: [] for v in unique_vals}
            hist_kwargs["bins"] = len(unique_vals)

        plt_layout(rows, cols, (12, plot_height)).set_facecolor(background_color)

        for tv in target_vals:
            if with_all:
                plt_next(f"{tv} vs. all").set_facecolor(background_color)
                plt.hist(data[col], label="all", color=all_color, **hist_kwargs)
            else:
                plt_next(tv).set_facecolor(background_color)
            plt.hist(
                data_grouped[tv][col],
                label=tv,
                color=colors.get(tv, "black"),
                **hist_kwargs,
            )
            plt.legend()

            if with_crosstab:
                header.append(tv)
                add_to_crosstab(crosstab, data_grouped[tv][col], unique_vals)

        if need_complement:
            for tv in target_vals:
                plt_next(f"not {tv} vs. all").set_facecolor(background_color)
                plt.hist(data[col], label="all", color=all_color, **hist_kwargs)
                plt.hist(
                    data_grouped_inv[tv][col],
                    label=tv,
                    color=colors.get(tv, "black"),
                    **hist_kwargs,
                )
                plt.legend()

        plt_show()

        if with_crosstab:
            header.append("Total")
            add_to_crosstab(crosstab, data[col], unique_vals)
            print_crosstab(header, crosstab, col)
        else:
            print_stats(data, col, target_vals, data_grouped)

        print("")

def add_to_crosstab(crosstab, values, unique_vals):
    dist = dict(cl.Counter(values))
    perc_scale = 100.0 / len(values)

    for k in unique_vals:
        count = dist.get(k, 0)
        perc = count * perc_scale
        crosstab[k].append((count, perc))


def print_crosstab(header, crosstab, col):
    width_0 = min(45, max(45, len(header[0])))
    width = 16
    totals = [0] * (len(header) - 1)

    print(
        f"| {truncate_feature_name(header[0], width_0):{width_0}} | "
        + " | ".join(f"{c:>{width}}" for c in header[1:])
        + " |"
    )
    print(
        f"| {'-' * width_0} | "
        + " | ".join("-" * width for c in header[1:])
        + " |"
    )

    for key, row in crosstab.items():
        print(
            f"| {truncate_feature_name(str(key), width_0):{width_0}} | "
            +  " | ".join(f"{prc:>{width-1}.3f}%" for cnt, prc in row[:-1])
            + f" | {row[-1][0]:>{width}}"
            +  " |"
        )

        for i, (cnt, prc) in enumerate(row):
            totals[i] += cnt

    print(
        f"| {'-':{width_0}} | "
        + " | ".join(f"{'-':>{width}}" for c in header[1:])
        + " |"
    )
    print(
        f"| {'Total':{width_0}} | "
        + " | ".join(f"{t:>{width}}" for t in totals)
        + " |"
    )


def truncate_feature_name(text, max_len):
    if len(text) <= max_len:
        return text

    keep = (max_len - 3) // 2

    return text[:keep] + "..." + text[-keep:]

def print_stats(data, col, target_vals, data_grouped):
    width = 11
    width_0 = 32
    table = [
        [truncate_feature_name(col, width_0), "Min", "Min (!= -1)", "Max", "Mean", "Var", "Skewness", "Kurtosis"],
        ["-" * width_0] + ["-" * width] * 7,
    ]

    for t, d in [(tv, data_grouped[tv][col]) for tv in target_vals] + [(None, None), ("All", data[col])]:
        if d is None:
            table.append(["-"] * 8)
            continue

        table.append(
            [t, d.min(), d[d != -1].min(), d.max(), d.mean(), d.var() ** 0.5, d.skew(), d.kurt()]
        )

    for row in table:
        print(
            "| "
            + " | ".join(
                format_stat_cell(i, c, width_0 if i == 0 else width)
                for i, c in enumerate(row)
            )
            + " |"
        )

def format_stat_cell(i, c, width):
    if isinstance(c, str):
        return f"{c:{width}}" if i == 0 else f"{c:>{width}}"

    return f"{c:{width}.3f}" if i == 0 else f"{c:>{width}.3f}"
```
- 타겟별로 히스토그램, 교차표, 기술통계 자동 출력.
- 연속형 변수는 평균, 분산, 왜도, 첨도 포함한 통계 출력.

---

### 3. 데이터 로딩
```python
# 경로 설정
data_path = "/kaggle/input/porto-seguro-safe-driver-prediction"
if not os.path.isdir(data_path):
    data_path = "/content/drive/MyDrive/다트비/캐글로리/"

# 데이터 읽기
train = pd.read_csv(os.path.join(data_path, "train.csv"))
test = pd.read_csv(os.path.join(data_path, "test.csv"))
```
---

### 4. 데이터 불러오기 및 샘플 분할

```python
rnd_subset = train[np.random.rand(len(train)) <= 0.05]

rnd_X = rnd_subset.drop(columns=["id", target_col])
rnd_y = rnd_subset[target_col]
rnd_X_train, rnd_X_test, rnd_y_train, rnd_y_test = train_test_split(
    rnd_X,
    rnd_y,
    test_size=0.2,
    random_state=4242,
    shuffle=True,
    stratify=rnd_y
)
```
- `train.csv`, `test.csv` 파일을 지정된 경로에서 읽어옴.
- `np.random.rand()`를 이용해 학습 데이터의 5%를 랜덤 샘플링함.
- `id` 컬럼은 분석에 필요 없으므로 제거하고, 입력(X)과 타겟(y) 분리.
- `train_test_split()`을 통해 학습/검증 데이터를 8:2 비율로 분할.
- `stratify`를 통해 클래스 불균형을 고려하여 분포를 유지함.

---
### 5. EDA 유틸 함수 정리

```python
def print_distr(title, data, col):
    print(f"#### {title}: {col}")

    for k, v in sorted(dict(cl.Counter(data[col])).items(), key=lambda e: e[0]):
        print(f"{k:20}: {v:9}   ({v * 100 / len(data):6.3f}%)")

    print("")


def print_nans(train, test):
    print("#### NaNs")
    print(f"{'Feature':>20} {'Train':>22}   {'Test':>22}")

    for col in test.columns:
        train_nans = np.sum(train[col] == -1)
        test_nans = np.sum(test[col] == -1)

        if train_nans or test_nans:
            train_nans_p = train_nans * 100 / len(train)
            test_nans_p = test_nans * 100 / len(test)
            print(f"{col:>20} {train_nans:9} ({train_nans_p:9.5f}%)   {test_nans:9} ({test_nans_p:9.5f}%)")

    print("")

print_distr("train", train, target_col)
print_distr("rnd_subset", rnd_subset, target_col)
print_nans(train, test)
gen_histograms_and_tables(
    train,
    [col for col in train.columns if col != "id"],
    target_col,
    target_cols,
    with_all=False
)
```

- `print_distr()`는 특정 컬럼에 대해 값별로 몇 개가 있는지, 그리고 그 비율(%)을 출력해줌. EDA에서 범주형 변수 분포 볼 때 유용함.
- `print_nans()`는 -1로 결측치 표시된 경우를 찾아서 train/test 각각 몇 개 있는지 출력함. 퍼센트까지 계산해줌.
- `gen_histograms_and_tables()`는 히스토그램과 테이블을 그려주는 함수. 시각화 + 통계적 요약용.

---

### 6. 결측치 처리 함수

```python
def clear_nans(data_frame):
    for col in ("ps_reg_03", "ps_car_14", "ps_car_12"):
        data_frame.loc[data_frame[col] == -1, col] += 1

clear_nans(train)
clear_nans(test)
clear_nans(rnd_subset)
```

- -1로 되어 있는 결측치를 +1 처리해서 정수 범위를 유지함. 단순하지만 빠르고 일관된 처리 방식.

---

### 7. 범주형 + 수치형 변환 클래스 (StdCatTransformer)

```python
class StdCatTransformer(skl.base.BaseEstimator, skl.base.TransformerMixin):
    ...
```

- Sklearn pipeline에 넣을 수 있는 전처리용 커스텀 Transformer 클래스
- `standardize=True`일 경우 수치형 컬럼 표준화
- 범주형은 기본은 원-핫, `bitstring=True`이면 비트 단위 인코딩 처리

---

### 8. 모델 실험/튜닝용 GridSearch + 평가 지표 모음

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, RandomForestClassifier,
    StackingClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, make_scorer,
    precision_score, recall_score
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, train_test_split
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
```

```python
GS_JOBS = 6
CLS_JOBS = 3
RUN_EXPERIMENTS = False
```

- `GS_JOBS`는 GridSearch에 사용할 병렬 작업 수
- `RUN_EXPERIMENTS`를 False로 두면 실험 안 돌아감 (개발 단계 디버깅용.. 버그 잡기.)

---

#### 지니계수 관련 함수

```python
def gini_score(y_true, y_pred): ...
def normalized_gini_score(y_true, y_pred): ...
```

- `normalized_gini_score`는 gini를 0~1 스케일로 정규화함
- 지니 계수를 사용하는 이유 : 정답과 예측 확률 간의 정렬 품질을 평가하는 지니 계수를 사용함.
---

#### 실험 실행 함수들

```python
def run_grid_search(classifier, param_grid, tune, resample=None): ...
def run_grid_search_with_scoring(...): ...
```

- 주어진 모델(classifier)에 대해 여러 스코어(F1, BA, NG)로 GridSearch를 수행
- `resample` 인자에 따라 SMOTE나 RandomOverSampling 사용 가능
- 내부적으로 `StdCatTransformer` 자동 포함시켜 전처리 + 분류 한번에 처리

```python
def print_scores(model, title, X_train, X_test, y_train, y_test): ...
def print_scores_pred(...): ...
```

- 모델이 예측한 결과에 대해 F1, 정밀도, 재현율, balanced accuracy, normalized gini 등 여러 지표 출력.
- `print_scores_pred`는 학습/테스트셋의 예측 비율과 평가 점수를 나란히 출력해줌.

### 9. 추론 및 시각화 (Prediction & Visualization)

```python
model.train(False)

with t.no_grad():
    pred_prob_train = t.nn.functional.sigmoid(model(X_train.to(device))).cpu().numpy()
    pred_prob_test = t.nn.functional.sigmoid(model(X_test.to(device))).cpu().numpy()

# 훈련셋 기준 threshold 구하기 (단순 로그용 - Gini score엔 영향 없음)
pos_ratio = (np.sum(y_train.cpu().numpy() == 1) / len(y_train))
threshold = np.percentile(pred_prob_train, 100.0 - pos_ratio * 100.0)
print(f"{threshold=:.5f}")

# 확률 기반 이진 분류 결과 도출
pred_train = (pred_prob_train > threshold) * 1
pred_test = (pred_prob_test > threshold) * 1

# 주요 평가 지표 출력
print_scores_pred(
    "",
    pred_train.flatten(),
    pred_test.flatten(),
    pred_prob_train.flatten(),
    pred_prob_test.flatten(),
    y_train.cpu().numpy().flatten(),
    y_test.cpu().numpy().flatten(),
)

# 예측 확률 히스토그램 시각화
ax = plt.subplots(1, 2, figsize=(9, 3))[1]

ax[0].hist(
    [
        pred_prob_train[pred_prob_train <= threshold],
        pred_prob_train[pred_prob_train > threshold]
    ],
    bins=200,
    stacked=True
)
ax[0].set_title("Predictions (train)")

ax[1].hist(
    [
        pred_prob_test[pred_prob_test <= threshold],
        pred_prob_test[pred_prob_test > threshold]
    ],
    bins=200,
    stacked=True
)
ax[1].set_title("Predictions (test)")

plt.show()

# 에폭별 손실 시각화
plt.figure(figsize=(5, 3))
plt.plot(losses["train"][5:], label="train")
plt.plot(losses["test"][5:], label="test")
plt.legend()
plt.title("Training losses (after the 5th epoch)")
plt.show()
```

---
- `model.train(False)`는 모델을 평가 모드로 바꿔 드롭아웃/배치정규화 등을 비활성화함.
- `t.no_grad()` 안에서는 gradient 계산을 하지 않으므로, 메모리 효율 + 추론 속도 향상됨.
- `sigmoid`를 통해 이진 분류 확률값(0~1) 출력 → `threshold` 기준으로 0/1 예측.
- `threshold`는 학습 데이터의 positive 비율을 기준으로 정함. (예: positive가 3.6%면 상위 3.6% 확률을 1로 분류)
- `print_scores_pred()`로 정밀도, 재현율, F1 등 다양한 지표 확인 가능.
- 시각화 2개:
  - 예측 확률 분포 히스토그램 (train/test)
  - 에폭별 손실 그래프 (5번째 이후부터)

→ 모델이 어떻게 예측했고, 과적합은 없는지 시각적으로 확인할 수 있음.

### 느낀점, 깨달은 점
코드가 너무 가독성 떨어져서 읽기 힘들고, 흐름을 파악하지 못하겠음.. ㅠㅠ

## 스터디 후
- kaggle api 가져와서 연동하면 자동으로 kaggle 점수도 나옴.
