import warnings

import numpy as np
import pandas as pd
from fancyimpute import IterativeImputer
from sklearn.discriminant_analysis import StandardScaler

warnings.filterwarnings("ignore")


########################################################################
## データの前処理を行うプログラム
########################################################################

# データの読み込み
path = "~/titanic_ws/84.9/data/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

# trainとtestに'test'という列を追加
train["test"] = 0
test["test"] = 1

# trainとtestを連結してdataとする
data = pd.concat([train, test], sort=False)

# 'Cabin' の先頭文字を取得
data["Cabin"] = data["Cabin"].str[0]
# 欠損値を 'M' で補完
data["Cabin"].fillna("M", inplace=True)
# 'Cabin' を数値に変換
data["Cabin"] = data["Cabin"].map(
    {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "T": 7,
        "M": 8,
    }
)

# 'Embarked' の欠損値を最頻値で補完
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# 'Ticket'の数字でない部分を最瀕値で補完
data["Ticket"] = data["Ticket"].str.extract("(\d+)", expand=False)
data["Ticket"].fillna(data["Ticket"].mode()[0], inplace=True)

# 'ticket' を標準化
sc = StandardScaler()
data["Ticket"] = sc.fit_transform(data[["Ticket"]])


# 'Sex' と 'Embarked' のカテゴリカル特徴量を数値に変換
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})


# 'Name' カラムから敬称を抽出し、新しい特徴量 'Title' として追加
data["Title"] = data["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)  # type: ignore

# 敬称を数値に変換（頻度が低い敬称を 'Rare' としてまとめる）
rare_titles = [
    "Don",
    "Rev",
    "Dr",
    "Mme",
    "Ms",
    "Major",
    "Lady",
    "Sir",
    "Mlle",
    "Col",
    "Capt",
    "Countess",
    "Jonkheer",
]
data["Title"] = data["Title"].replace(rare_titles, "Rare")
data["Title"] = data["Title"].map(
    {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
)
# 欠損値を '0' に変換
data["Title"].fillna(0, inplace=True)

# 'Fare' の欠損値を中央値で補完
data["Fare"].fillna(data["Fare"].median(), inplace=True)

# 'Fare'を標準化
sc = StandardScaler()
data["Fare_std"] = sc.fit_transform(data[["Fare"]])

# 'Family_size' と 'Family_survival' を追加
# 名前の名字を取得して'Last_name'に入れる
data["Last_name"] = data["Name"].apply(lambda x: x.split(",")[0])

data["Family_survival"] = 0.5  # デフォルトの値
# Last_nameとFareでグルーピング
for grp, grp_df in data.groupby(["Last_name", "Fare"]):

    if len(grp_df) != 1:
        # (名字が同じ)かつ(Fareが同じ)人が2人以上いる場合
        for index, row in grp_df.iterrows():
            smax = grp_df.drop(index)["Perished"].max()
            smin = grp_df.drop(index)["Perished"].min()
            passID = row["PassengerId"]

            if smax == 1.0:
                data.loc[data["PassengerId"] == passID, "Family_survival"] = 1
            elif smin == 0.0:
                data.loc[data["PassengerId"] == passID, "Family_survival"] = 0
            # グループ内の自身以外のメンバーについて
            # 1人でも生存している → 1
            # 生存者がいない(NaNも含む) → 0
            # 全員NaN → 0.5

# チケット番号でグルーピング
for grp, grp_df in data.groupby("Ticket"):
    if len(grp_df) != 1:
        # チケット番号が同じ人が2人以上いる場合
        # グループ内で1人でも生存者がいれば'Family_survival'を1にする
        for ind, row in grp_df.iterrows():
            if (row["Family_survival"] == 0) | (row["Family_survival"] == 0.5):
                smax = grp_df.drop(ind)["Perished"].max()
                smin = grp_df.drop(ind)["Perished"].min()
                passID = row["PassengerId"]
                if smax == 1.0:
                    data.loc[
                        data["PassengerId"] == passID, "Family_survival"
                    ] = 1
                elif smin == 0.0:
                    data.loc[
                        data["PassengerId"] == passID, "Family_survival"
                    ] = 0

# Family_sizeの作成
data["Family_size"] = data["SibSp"] + data["Parch"] + 1
# 1, 2~4, 5~の3つに分ける
data["Family_size_bin"] = 0
data.loc[
    (data["Family_size"] >= 2) & (data["Family_size"] <= 4), "Family_size_bin"
] = 1
data.loc[
    (data["Family_size"] >= 5) & (data["Family_size"] <= 7), "Family_size_bin"
] = 2
data.loc[(data["Family_size"] >= 8), "Family_size_bin"] = 3


# 特徴量を選択
features_ = [
    "Pclass",
    "Embarked",
    "SibSp",
    "Parch",
    "Title",
    "Family_size",
]

# MICE補完の設定
imputer = IterativeImputer(
    random_state=42, max_iter=20, min_value=0, max_value=80
)

# 欠損値補完の実行
data["Age"] = imputer.fit_transform(
    np.hstack((data[features_], np.expand_dims(data["Age"], axis=1)))
)[:, -1]

# 補完後の年齢の統計量を確認
print("補完後の年齢の統計量 (data):")
print(data["Age"].describe())

# 負の年齢があるかを確認
negative_ages_data = data[data["Age"] < 0]
if not negative_ages_data.empty:
    print(f"負の年齢が {len(negative_ages_data)} 件あります。")
    # 負の値を持つ行のインデックス
    neg_idx_data = negative_ages_data.index
    # 年齢を再補完（負の値を0歳に設定し、再補完）
    data.loc[neg_idx_data, "Age"] = np.nan
    data["Age"] = imputer.fit_transform(
        np.hstack((data[features_], np.expand_dims(data["Age"], axis=1)))
    )[:, -1]
    print("再補完後の年齢の統計量 (data):")
    print(data["Age"].describe())

# ダミー変数化
data = pd.get_dummies(
    data=data,
    columns=["Title", "Pclass", "Family_survival", "Cabin", "Embarked"],
)

# 訓練データと本番データに再分割
train = data.iloc[: len(train)]
test = data.iloc[len(train) :].reset_index(drop=True)

# 特徴量を選択
features = [
    col
    for col in train.columns
    if col.startswith(("Pclass_", "Title_", "Family_survival_", "Cabin_"))
] + ["Sex", "SibSp", "Parch", "Family_size", "test"]

# 特徴量を取得
X = train[features + ["Age"]]
Y = test[features + ["Age"]]

# 標準化
# scaler = StandardScaler()
# X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
# Y = pd.DataFrame(scaler.transform(Y), columns=Y.columns)  # type: ignore

# 特徴量をfloat型に変換
X = X.astype("float")
Y = Y.astype("float")

# xにPerishedとPassengerIdを追加
X["Perished"] = train["Perished"]
X["PassengerId"] = train["PassengerId"]
# yにPassengerIdを追加
Y["PassengerId"] = test["PassengerId"]


# csvファイルに保存
X.to_csv("./adversarial_validation/fixed_data/X.csv", index=False)
Y.to_csv("./adversarial_validation/fixed_data/Y.csv", index=False)
