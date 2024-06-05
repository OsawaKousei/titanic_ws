import warnings

import matplotlib.pyplot as plt  # import the library to draw the graph
import numpy as np
import pandas as pd
from fancyimpute import IterativeImputer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# データの読み込み
path = "~/titanic_ws/data/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

# Cabinカラムを削除
train.drop(columns=["Cabin"], inplace=True)
test.drop(columns=["Cabin"], inplace=True)

# 'Embarked' の欠損値を最頻値で補完
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
test["Embarked"].fillna(test["Embarked"].mode()[0], inplace=True)

# 'Sex' と 'Embarked' のカテゴリカル特徴量を数値に変換
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
train["Embarked"] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})
test["Embarked"] = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# 'Name' カラムから敬称を抽出し、新しい特徴量 'Title' として追加
train["Title"] = train["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)  # type: ignore
test["Title"] = test["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)  # type: ignore

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
train["Title"] = train["Title"].replace(rare_titles, "Rare")
test["Title"] = test["Title"].replace(rare_titles, "Rare")
train["Title"] = train["Title"].map(
    {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
)
test["Title"] = test["Title"].map(
    {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
)

# 欠損値を '0' に変換
train["Title"].fillna(0, inplace=True)
test["Title"].fillna(0, inplace=True)

# 特徴量を選択
features = ["Pclass", "Sex", "Embarked", "SibSp", "Parch", "Fare", "Title"]

# 特徴量をスケーリング
scaler = StandardScaler()
train_scaled_features = scaler.fit_transform(train[features])
test_scaled_features = scaler.transform(test[features])

# MICE補完の設定
imputer = IterativeImputer(
    random_state=42, max_iter=20, min_value=0, max_value=80
)

# 欠損値補完の実行
train["Age"] = imputer.fit_transform(
    np.hstack((train_scaled_features, np.expand_dims(train["Age"], axis=1)))
)[:, -1]
test["Age"] = imputer.transform(np.hstack((test_scaled_features, np.expand_dims(test["Age"], axis=1))))[:, -1]  # type: ignore

# 補完後の年齢の統計量を確認
print("補完後の年齢の統計量 (train):")
print(train["Age"].describe())
print("補完後の年齢の統計量 (test):")
print(test["Age"].describe())

# 負の年齢があるかを確認
negative_ages_train = train[train["Age"] < 0]
negative_ages_test = test[test["Age"] < 0]
if not negative_ages_train.empty or not negative_ages_test.empty:
    print(
        f"負の年齢が {len(negative_ages_train) + len(negative_ages_test)} 件あります。"
    )
    # 負の値を持つ行のインデックス
    neg_idx_train = negative_ages_train.index
    neg_idx_test = negative_ages_test.index
    # 年齢を再補完（負の値を0歳に設定し、再補完）
    train.loc[neg_idx_train, "Age"] = np.nan
    test.loc[neg_idx_test, "Age"] = np.nan
    train["Age"] = imputer.fit_transform(
        np.hstack(
            (train_scaled_features, np.expand_dims(train["Age"], axis=1))
        )
    )[:, -1]
    test["Age"] = imputer.transform(np.hstack((test_scaled_features, np.expand_dims(test["Age"], axis=1))))[:, -1]  # type: ignore
    print("再補完後の年齢の統計量 (train):")
    print(train["Age"].describe())
    print("再補完後の年齢の統計量 (test):")
    print(test["Age"].describe())

# 目的変数と説明変数に分割
X = train[features + ["Age"]]
y = train["Perished"]

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特徴量のスケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train_scaled.shape: ", X_train_scaled.shape)
print("X_test_scaled.shape: ", X_test_scaled.shape)
print("y_train.shape: ", y_train.shape)
print("y_test.shape: ", y_test.shape)
