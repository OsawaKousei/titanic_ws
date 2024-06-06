import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from filer import filter
from mixup import mixup
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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
path = "./copy_model/fixed_data/"
TRAIN = pd.read_csv(path + "X.csv")
PRED = pd.read_csv(path + "Y.csv")
prefix = pd.read_csv(path + "prefix.csv")

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


# xgb.DMatrixによってオリジナルAPIで使用できる形にします
dtrain = xgb.DMatrix(
    X_train, label=y_train, feature_names=X_train.columns.to_list()
)
dtest = xgb.DMatrix(
    X_test, label=y_test, feature_names=X_test.columns.to_list()
)

dpred = xgb.DMatrix(X_pred, feature_names=X_pred.columns.to_list())

# 先にxgb_paramsとしてパラメータを設定しておきます
xgb_params = {  # 目的関数
    "objective": "reg:squarederror",
    # 学習に用いる評価指標
    "eval_metric": "rmse",
    # boosterに何を用いるか
    "booster": "gbtree",
    # learning_rateと同義
    "eta": 0.1,
    # 木の最大深さ
    "max_depth": 6,
    # random_stateと同義
    "seed": 2525,
}

# 学習過程を取得するための変数を用意
evals_result = {}
reg = xgb.train(  # 上で設定した学習パラメータを使用
    params=xgb_params,
    dtrain=dtrain,
    # 学習のラウンド数
    num_boost_round=50000,
    # early stoppinguのラウンド数
    early_stopping_rounds=15,
    # 検証用データ
    evals=[(dtrain, "train"), (dtest, "eval")],
    # 上で用意した変数を設定
    evals_result=evals_result,
)

# testデータに対して予測
pred = reg.predict(dtest)
# 予測値が閾値を超えたら1, そうでなければ0にする
pred = (pred > THRESHOLD).astype(int)
# 予測値をDataFrameに変換
pred = pd.DataFrame(pred)  # Convert pred to a DataFrame
# predにPerishedという列名をつける
pred.columns = ["Perished"]
# y_testをDataFrameに変換
y_test = pd.DataFrame(y_test)
# y_testにPerishedという列名をつける
y_test.columns = ["Perished"]

# predとy_testを保存
pred.to_csv("pred.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# 正解率を計算
accuracy = 0.0
for i in range(len(y_test)):
    if y_test.iloc[i, 0] == pred.iloc[i, 0]:
        accuracy += 1
accuracy /= len(y_test)
# 小数第3位まで表示
accuracy = round(accuracy, 3)
# 正解率を表示
print("pre accuracy: ", accuracy)

# # predの各行ごとに
# for i in range(len(prefix)):
#     # prefixのperishedが-1かどうかを判定
#     if not prefix.loc[i, "Perished"] == -1:
#         # prefixのPerishedが-1でない場合、predのi行目をprefixのPerishedに代入
#         pred.loc[i, "Perished"] = prefix.loc[i, "Perished"]

# 正解率を計算
accuracy = 0.0
for i in range(len(y_test)):
    if y_test.iloc[i, 0] == pred.iloc[i, 0]:
        accuracy += 1
accuracy /= len(y_test)
# 正解率を表示
print("accuracy: ", accuracy)

# trainデータに対してのloss推移をplot
plt.plot(evals_result["train"]["rmse"], label="train rmse")
# testデータに対してのloss推移をplot
plt.plot(evals_result["eval"]["rmse"], label="eval rmse")
plt.grid()
plt.legend()
plt.xlabel("rounds")
plt.ylabel("rmse")
plt.show()

# 予測
pred = reg.predict(dpred)
# 予測値が閾値を超えたら1, そうでなければ0にする
pred = (pred > THRESHOLD).astype(int)
# 予測値をDataFrameに変換
pred = pd.DataFrame(pred)  # Convert pred to a DataFrame
# predにPerishedという列名をつける
pred.columns = ["Perished"]
# PassengerIdと結合
submission = pd.concat([PassengerId, pred], axis=1)
# submissionを保存
submission.to_csv("submission.csv", index=False)
