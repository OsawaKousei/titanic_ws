import os
import random
import warnings

import matplotlib.pyplot as plt  # import the library to draw the graph
import numpy as np
import pandas as pd
import tensorflow as tf

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from fancyimpute import IterativeImputer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# 乱数を固定する関数
def reset_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(seed)  # random関数のシードを固定
    np.random.seed(seed)  # numpyのシードを固定
    tf.random.set_seed(seed)  # tensorflowのシードを固定


# 乱数を固定
reset_seed(28)


# データの読み込み
path = "~/titanic_ws/data/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

# trainとtestを連結してdataとする
data = pd.concat([train, test], sort=False)
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

# trainとtestに再分割
train = data.iloc[: len(train)]
test = data.iloc[len(train) :].reset_index(drop=True)


# 特徴量を選択
features = [
    "Pclass",
    "Sex",
    "Embarked",
    "SibSp",
    "Parch",
    "Fare",
    "Title",
    "Ticket",
    "Family_survival",
    "Family_size",
    "Cabin",
]

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

# csvファイルに保存
X.to_csv("X.csv", index=False)
y.to_csv("y.csv", index=False)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特徴量のスケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# データの形状を確認
print("X_train_scaled.shape: ", X_train_scaled.shape)
print("X_test_scaled.shape: ", X_test_scaled.shape)
print("y_train.shape: ", y_train.shape)
print("y_test.shape: ", y_test.shape)

# ハイパーパラメータの設定
BATCH_SIZE = 100
WEIGHT_DECAY = 0.1
LEARNING_RATE = 0.001
EPOCH = 300
DROPOUT = 0.5
THRESHOLD = 0.5

transform = torchvision.transforms.ToTensor()
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.int64)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.int64)
# inputをx_train, targetをy_trainとしてdatasetを作成
train_dataset = TensorDataset(X_train_scaled, y_train)
test_dataset = TensorDataset(X_test_scaled, y_test)
# datasetをDataLoaderに渡す
trainloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
)
testloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
)


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(12, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 200)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 2)

        # add batch normalization
        self.bn1 = nn.BatchNorm1d(50)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(200)
        self.bn4 = nn.BatchNorm1d(100)
        self.bn5 = nn.BatchNorm1d(50)
        # Add dropout layer
        self.dropout = nn.Dropout(DROPOUT)

        # add softmax
        self.softmax = nn.Softmax()

        # weight initialization
        nn.init.kaiming_normal_(
            self.fc1.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.fc2.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.fc3.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.fc4.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.fc5.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.fc6.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = self.fc6(x)
        return x


device = torch.device("cuda:0")
net = Net()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

train_acc_value = []  # trainingのaccuracyを保持するlist
test_acc_value = []  # testのaccuracyを保持するlist

# 学習
for epoch in range(EPOCH):
    print("epoch", epoch + 1)  # epoch数の出力
    # train dataを使って学習する
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(
            device
        )  # データをGPUに送る
        optimizer.zero_grad()  # 勾配の初期化
        outputs = net(inputs)  # inputをモデルに入れる
        loss = criterion(outputs, labels)  # lossの計算
        loss.backward()  # 勾配の計算
        optimizer.step()  # 重みの更新

    sum_correct = 0  # 正解数の合計
    accuracy = 0.0  # 正答率
    sum_total = 0  # dataの数の合計

    # train dataを使ってテストをする(パラメータ更新がないようになっている)
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # outputをsoftmaxに通す
        outputs = F.softmax(outputs, dim=1)
        # 閾値を超えたら1, そうでなければ0にする
        outputs = (outputs[:, 1] > THRESHOLD).long()
        # 正解数を計算
        sum_correct += (outputs == labels).sum().item()

    # 正答率を計算
    accuracy = float(sum_correct / len(train_dataset))
    # 正答率を出力
    print("train accuracy: ", accuracy)
    # traindataのaccuracyをグラフ描画のためにlistに保持
    train_acc_value.append(float(accuracy))

    sum_correct = 0  # 正解数の合計
    accuracy = 0  # 正答率
    sum_total = 0  # dataの数の合計
    acuuracy_maen = 0  # 正答率の平均

    # test dataを使ってテストをする
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = F.softmax(outputs, dim=1)
        outputs = (outputs[:, 1] > THRESHOLD).long()
        sum_correct += (outputs == labels).sum().item()

    accuracy = float(sum_correct / len(test_dataset))
    print("test accuracy: ", accuracy)
    test_acc_value.append(float(accuracy))

# 最後10エポックの平均を出力
accuracy_maen = np.mean(test_acc_value[-10:])
# 少数第3位まで出力
accuracy_maen = round(accuracy_maen, 3)
print("accuracy_maen: ", accuracy_maen)

plt.figure(figsize=(6, 6))  # グラフ描画用

# 以下グラフ描画
plt.plot(range(EPOCH), train_acc_value)
plt.plot(range(EPOCH), test_acc_value, c="#00ff00")
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.xlabel("EPOCH")
plt.ylabel("ACCURACY")
plt.legend(["train acc", "test acc"])
plt.title("accuracy")
# 最後10エポックの平均をグラフに追加
plt.axhline(y=float(accuracy_maen), color="r", linestyle="--")
# 最後10エポックの平均をテキストでグラフ中央に追加
plt.text(
    EPOCH / 2,
    0.5,
    f"accuracy_maen: {accuracy_maen}",
    ha="center",
    va="center",
    color="r",
)
plt.savefig("accuracy_image.png")
plt.show()
