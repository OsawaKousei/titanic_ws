import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from filer import filter
from mixup import mixup
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import all_estimators
from xgboost import callback

########################################################################
## sklearnの全てのモデルを試すプログラム-ここでの得点はあてにならない
########################################################################

all_estimators(type_filter="classifier")


warnings.filterwarnings("ignore")


# 乱数を固定する関数
def reset_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    # sklearnのシードを固定
    np.random.seed(seed)


# 乱数を固定
reset_seed(622)

# データの読み込み
path = "./adversarial_validation/fixed_data/"
TRAIN = pd.read_csv(path + "X.csv")
PRED = pd.read_csv(path + "Y.csv")

# TRAINからPassengerIdを切り捨て
TRAIN = TRAIN.drop("PassengerId", axis=1)
# TRAINからPerishedを取得
Y = TRAIN["Perished"]
# TRAINからPerishedを削除
X = TRAIN.drop("Perished", axis=1)

# PREDからPassengerIdを取得
PassengerId = PRED["PassengerId"]
# PREDからPassengerIdを削除
X_pred = PRED.drop("PassengerId", axis=1)

# 訓練データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# データの形状を確認
print("X_train_scaled.shape: ", x_train.shape)
print("X_test_scaled.shape: ", x_test.shape)
print("y_train.shape: ", y_train.shape)
print("y_test.shape: ", y_test.shape)
print("X_pred.shape: ", X_pred.shape)

# ハイパーパラメータの設定
THRESHOLD = 0.4

# 訓練データを水増し
# X_train, y_train = mixup(X_train, y_train, MIX_ALPHA, MIX_NUM)

kf = KFold(n_splits=3, shuffle=True, random_state=1)

# モデル名とスコアを保持するリスト
model_scores = []

for name, Estimator in all_estimators(type_filter="classifier"):

    try:
        model = Estimator()
        if "score" not in dir(model):
            continue
        scores = cross_validate(
            model, x_train, y_train, cv=kf, scoring=["accuracy"]
        )
        model_scores.append(
            {"name": name, "mean": scores["test_accuracy"].mean()}
        )
    except:
        pass

# スコアが高い順にソート
model_scores = sorted(model_scores, key=lambda x: -x["mean"])
# スコアを表示
for model_score in model_scores:
    print(model_score)
