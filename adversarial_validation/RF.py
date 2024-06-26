import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from filer import filter
from mixup import mixup
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from xgboost import callback

########################################################################
## ランダムフォレストによる予測
########################################################################

warnings.filterwarnings("ignore")


# 乱数を固定する関数
def reset_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    # sklearnのシードを固定
    np.random.seed(seed)


# 乱数を固定
reset_seed(53)

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

##
## adversarial_validationにより、predとtrainで分布の異なる特徴量を削除
##

# Xから"Familysize"と"Title_3.0"と"test"を削除
X = X.drop("Family_size", axis=1)
X = X.drop("Title_3.0", axis=1)
X = X.drop("test", axis=1)
# predから"Familysize"と"Title_3.0"と"test"を削除
PRED = PRED.drop("Family_size", axis=1)
PRED = PRED.drop("Title_3.0", axis=1)
PRED = PRED.drop("test", axis=1)

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
THRESHOLD = 0.5

# 訓練データを水増し
# X_train, y_train = mixup(X_train, y_train, MIX_ALPHA, MIX_NUM)


# 採用する特徴量の数を設定　25は全て
select = SelectKBest(k=25)

param_grid = {
    "criterion": ["gini", "entropy"],
    "n_estimators": [25, 100, 500, 1000, 2000],
    "min_samples_split": [0.5, 2, 4, 10],
    "min_samples_leaf": [1, 2, 4, 10],
    "bootstrap": [True, False],
}

##
## グリッドサーチで最適なパラメータを探そうとしたが、時間がかかる上に、既定値の方が精度が高かったため、コメントアウト
## cf:https://qiita.com/sudominoru/items/1c21cf4afaf67fda3fee
##

# grid = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=1), param_grid=param_grid
# )
# grid = grid.fit(X_train, y_train)

# print(grid.best_score_)
# print(grid.best_params_)

# # parameterをtxtファイルに保存
# filter.save_dict(grid.best_params_, "./adversarial_validation/params/params.txt")

# clf = RandomForestClassifier(
#     random_state=10,
#     warm_start=True,  # 既にフィットしたモデルに学習を追加
#     n_estimators=26,
#     max_depth=6,
#     max_features="sqrt",
# )
# # pipeline = make_pipeline(select, clf)
# pipeline = make_pipeline(clf)
# pipeline.fit(X_train, y_train)

# # pipeline = make_pipeline(select, grid.best_estimator_)
# # pipeline.fit(X_train, y_train)


# # フィット結果の表示
# cv_result = cross_validate(pipeline, X_train, y_train, cv=10)
# print("mean_score = ", np.mean(cv_result["test_score"]))
# print("mean_std = ", np.std(cv_result["test_score"]))

clf = RandomForestClassifier(
    random_state=10,
    warm_start=True,  # 既にフィットしたモデルに学習を追加
    n_estimators=26,
    max_depth=10,
    max_features="sqrt",
)


pipeline = make_pipeline(select, clf)
pipeline.fit(X_train, y_train)


# testデータに対しての予測
test_pred = pipeline.predict(X_test)

# test_predをcsvファイルに保存
test_pred = pd.DataFrame(test_pred)
test_pred.to_csv(
    "./adversarial_validation/predictions/test_pred.csv", index=False
)
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

# x_predをcsvファイルに保存
X_pred = pd.DataFrame(X_pred)
X_pred.to_csv("./adversarial_validation/predictions/X_pred.csv", index=False)

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
submission.to_csv(
    "./adversarial_validation/predictions/submission.csv", index=False
)
