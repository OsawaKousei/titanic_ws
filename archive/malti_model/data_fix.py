import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from fancyimpute import IterativeImputer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# データの読み込み
path = "~/titanic_ws/malti_model/data/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

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

# 欠損値を Embarked='S', Pclass=3 の平均値で補完
fare = data.loc[
    (data["Embarked"] == "S") & (data["Pclass"] == 3), "Fare"
].median()
data["Fare"] = data["Fare"].fillna(fare)

# 'Fare'を標準化
sc = StandardScaler()
data["Fare_std"] = sc.fit_transform(data[["Fare"]])

# Faer_stdの欠損値を平均値で補完
data["Fare_std"].fillna(data["Fare_std"].mean(), inplace=True)

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

Ticket_Count = dict(data["Ticket"].value_counts())
data["TicketGroup"] = data["Ticket"].map(Ticket_Count)
data.loc[
    (data["TicketGroup"] >= 2) & (data["TicketGroup"] <= 4), "Ticket_label"
] = 2
data.loc[
    (data["TicketGroup"] >= 5) & (data["TicketGroup"] <= 8)
    | (data["TicketGroup"] == 1),
    "Ticket_label",
] = 1
data.loc[(data["TicketGroup"] >= 11), "Ticket_label"] = 0

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


# Age を Pclass, Sex, Parch, SibSp からランダムフォレストで推定
# 推定に使用する項目を指定
age_df = data[["Age", "Pclass", "Sex", "Parch", "SibSp"]]

# ラベル特徴量をワンホットエンコーディング
age_df = pd.get_dummies(age_df)

# 学習データとテストデータに分離し、numpyに変換
known_age = age_df[age_df.Age.notnull()].values
unknown_age = age_df[age_df.Age.isnull()].values

# 学習データをX, yに分離
X = known_age[:, 1:]
y = known_age[:, 0]

# ランダムフォレストで推定モデルを構築
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)

# 推定モデルを使って、テストデータのAgeを予測し、補完
predictedAges = rfr.predict(unknown_age[:, 1::])
data.loc[(data.Age.isnull()), "Age"] = predictedAges

# dead list, survie list

# NameからSurname(苗字)を抽出
data["Surname"] = data["Name"].map(lambda name: name.split(",")[0].strip())

# 同じSurname(苗字)の出現頻度をカウント(出現回数が2以上なら家族)
data["FamilyGroup"] = data["Surname"].map(data["Surname"].value_counts())


# 家族で16才以下または女性の生存率
Female_Child_Group = data.loc[
    (data["FamilyGroup"] >= 2)
    & ((data["Age"] <= 16) | (data["Sex"] == "female"))
]
Female_Child_Group = Female_Child_Group.groupby("Surname")["Perished"].mean()
print("FC")
print(Female_Child_Group)
# 家族で16才超えかつ男性の生存率
Male_Adult_Group = data.loc[
    (data["FamilyGroup"] >= 2) & (data["Age"] > 16) & (data["Sex"] == "male")
]
Male_Adult_List = Male_Adult_Group.groupby("Surname")["Perished"].mean()
print("MA")
print(Male_Adult_List)

# デッドリストとサバイブリストの作成
Dead_list = set(
    Female_Child_Group[Female_Child_Group.apply(lambda x: x == 1)].index
)
print(Dead_list)
Survived_list = set(
    Male_Adult_List[Male_Adult_List.apply(lambda x: x == 0)].index
)
print(Survived_list)


# デッドリストとサバイブリストをSex, Age, Title に反映させる
# data.loc[
#     (data["Perished"].isnull())
#     & (data["Surname"].apply(lambda x: x in Dead_list)),
#     ["Sex", "Age", "Title"],
# ] = [0, 28.0, 0]
# data.loc[
#     (data["Perished"].isnull())
#     & (data["Surname"].apply(lambda x: x in Survived_list)),
#     ["Sex", "Age", "Title"],
# ] = [1, 5.0, 1]

# ダミー変数化
data = pd.get_dummies(
    data=data,
    columns=[
        "Title",
        "Pclass",
        "Family_survival",
        "Cabin",
        "Embarked",
        "Ticket_label",
        "Family_size_bin",
    ],
)

# 訓練データと本番データに再分割
train = data.iloc[: len(train)]
test = data.iloc[len(train) :].reset_index(drop=True)

# 特徴量を選択
features = [
    col
    for col in train.columns
    if col.startswith(
        (
            "Pclass_",
            "Title_",
            "Family_survival_",
            "Cabin_",
            # "Embarked_",
            # "Ticket_label_",
            # "Family_size_bin_",
        )
    )
] + [
    "Sex",
    # "Fare_std",
    "Age",
    "Parch",
    "SibSp",
    "Family_size",
]

# 特徴量を取得
X = train[features]
Y = test[features]

# # 標準化
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
Y = pd.DataFrame(scaler.transform(Y), columns=Y.columns)  # type: ignore

# 特徴量をfloat型に変換
X = X.astype("float")
Y = Y.astype("float")

# xにPerishedとPassengerIdを追加
X["Perished"] = train["Perished"]
X["PassengerId"] = train["PassengerId"]
# yにPassengerIdを追加
Y["PassengerId"] = test["PassengerId"]


# csvファイルに保存
X.to_csv("./malti_model/fixed_data/X.csv", index=False)
Y.to_csv("./malti_model/fixed_data/Y.csv", index=False)
