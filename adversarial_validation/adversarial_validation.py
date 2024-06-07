import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split

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

# TRAINからPassengerIdとPerishedを切り捨て
TRAIN = TRAIN.drop(["PassengerId", "Perished"], axis=1)
# PREDからPassengerIdを切り捨て
PRED = PRED.drop("PassengerId", axis=1)

# TRAINとPREDを結合
TRAIN = pd.concat([TRAIN, PRED], axis=0)

# TRAINをシャッフル
TRAIN = TRAIN.sample(frac=1).reset_index(drop=True)

# TRAINからtestを取得
Y = TRAIN["test"]
# TRAINからtestを削除
X = TRAIN.drop("test", axis=1)


THRESHOLD = 0.5

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=78
)

# xgb.DMatrixによってオリジナルAPIで使用できる形にします
dtrain = xgb.DMatrix(
    X_train, label=y_train, feature_names=X_train.columns.to_list()
)
dtest = xgb.DMatrix(
    X_test, label=y_test, feature_names=X_test.columns.to_list()
)

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
pred.columns = ["test"]
# y_testをDataFrameに変換
y_test = pd.DataFrame(y_test)
# y_testにPerishedという列名をつける
y_test.columns = ["test"]

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

# modelのfeature importanceをplot
xgb.plot_importance(reg)
plt.show()
# gainの場合
xgb.plot_importance(reg, importance_type="gain")
plt.show()

# # trainデータに対してのloss推移をplot
# plt.plot(evals_result["train"]["rmse"], label="train rmse")
# # testデータに対してのloss推移をplot
# plt.plot(evals_result["eval"]["rmse"], label="eval rmse")
# plt.grid()
# plt.legend()
# plt.xlabel("rounds")
# plt.ylabel("rmse")
# plt.show()
