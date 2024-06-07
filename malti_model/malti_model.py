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
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from xgboost import callback

warnings.filterwarnings("ignore")


# 乱数を固定する関数
def reset_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    # sklearnのシードを固定
    np.random.seed(seed)


# 乱数を固定
reset_seed(622)

# データの読み込み
path = "./malti_model/fixed_data/"
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
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# データの形状を確認
print("X_train_scaled.shape: ", X_train.shape)
print("X_test_scaled.shape: ", X_test.shape)
print("y_train.shape: ", y_train.shape)
print("y_test.shape: ", y_test.shape)
print("X_pred.shape: ", X_pred.shape)

# ハイパーパラメータの設定
THRESHOLD = 0.4
LEARNING_RATE = 0.1
MAX_DEPTH = 6

# 訓練データを水増し
# X_train, y_train = mixup(X_train, y_train, MIX_ALPHA, MIX_NUM)


# 採用する特徴量を25個から20個に絞り込む
# select = SelectKBest(k=25)

param_grid = {
    "criterion": ["gini", "entropy"],
    "n_estimators": [25, 100, 500, 1000, 2000],
    "min_samples_split": [0.5, 2, 4, 10],
    "min_samples_leaf": [1, 2, 4, 10],
    "bootstrap": [True, False],
}

# grid = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=1), param_grid=param_grid
# )
# grid = grid.fit(X_train, y_train)

# print(grid.best_score_)
# print(grid.best_params_)

# # parameterをtxtファイルに保存
# filter.save_dict(grid.best_params_, "./malti_model/params/params.txt")

clf = RandomForestClassifier(
    random_state=10,
    warm_start=True,  # 既にフィットしたモデルに学習を追加
    n_estimators=26,
    max_depth=6,
    max_features="sqrt",
)
# pipeline = make_pipeline(select, clf)
pipeline = make_pipeline(clf)
pipeline.fit(X_train, y_train)

# pipeline = make_pipeline(select, grid.best_estimator_)
# pipeline.fit(X_train, y_train)


# フィット結果の表示
cv_result = cross_validate(pipeline, X_train, y_train, cv=10)
print("mean_score = ", np.mean(cv_result["test_score"]))
print("mean_std = ", np.std(cv_result["test_score"]))

# --------　採用した特徴量 ---------------
# 採用の可否状況
# mask = select.get_support()

# # 項目のリスト
# list_col = list(X_train.columns[1:])

# # 項目別の採用可否の一覧表
# for i, j in enumerate(list_col):
#     print("No" + str(i + 1), j, "=", mask[i])

# # シェイプの確認
# X_selected = select.transform(X)
# print("X.shape={}, X_selected.shape={}".format(X.shape, X_selected.shape))

# testデータに対しての予測
test_pred = pipeline.predict(X_test)
# test_predをDataFrameに変換
test_pred = pd.DataFrame(test_pred)
# 閾値を超えたら1, そうでなければ0にする
test_pred = (test_pred > THRESHOLD).astype(int)
# y_testをDataFrameに変換
y_test = pd.DataFrame(y_test)
# 正解率を計算
accuracy = 0.0
for i in range(len(y_test)):
    if y_test.iloc[i, 0] == test_pred.iloc[i, 0]:
        accuracy += 1
accuracy /= len(y_test)
# 正解率を表示
print("accuracy: ", accuracy)

# 予測
pred = pipeline.predict(X_pred)
# predをDataFrameに変換
pred = pd.DataFrame(pred)
# 閾値を超えたら1, そうでなければ0にする
pred = (pred > THRESHOLD).astype(int)
# predにPerishedという列名をつける
pred.columns = ["Perished"]
# PassengerIdと結合
submission = pd.concat([PassengerId, pred], axis=1)
# submissionを保存
submission.to_csv("./malti_model/predictions/submission.csv", index=False)
