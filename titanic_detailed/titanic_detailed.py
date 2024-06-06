import os
import warnings

import matplotlib.pyplot as plt  # import the library to draw the graph
import numpy as np
import pandas as pd

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from early_stopping import EarlyStopping
from model import Net
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# 乱数を固定する関数
def reset_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    # pytorchのシードを固定
    torch.manual_seed(seed)
    # sklearnのシードを固定
    np.random.seed(seed)


# 乱数を固定
reset_seed(622)

# データの読み込み
path = "./titanic_detailed/fixed_data/"
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
BATCH_SIZE = 100
WEIGHT_DECAY = 0.5
LEARNING_RATE = 0.0001
EPOCH = 500
DROPOUT = 0.1
THRESHOLD = 0.5
PATIENCE = 1000  # 早期終了のパラメータ

# データをpytorchのtensorに変換
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.int64).squeeze()
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.int64).squeeze()
X_pred = torch.tensor(X_pred.values, dtype=torch.float32)

# inputをx_train, targetをy_trainとしてdatasetを作成
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
pred_dataset = TensorDataset(X_pred, torch.zeros(X_pred.shape[0]))
# datasetをDataLoaderに渡す
trainloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
)
testloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
)
predloader = DataLoader(
    pred_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
)


device = torch.device("cuda:0")
net = Net(DROPOUT)
net = net.to(device)
# 早期終了の設定
earlystopping = EarlyStopping(PATIENCE, verbose=False)
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
    sum_loss = 0  # lossの合計

    # test dataを使ってテストをする
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
        outputs = F.softmax(outputs, dim=1)
        outputs = (outputs[:, 1] > THRESHOLD).long()
        sum_correct += (outputs == labels).sum().item()

    accuracy = float(sum_correct / len(test_dataset))
    print("test accuracy: ", accuracy)
    test_acc_value.append(float(accuracy))

    earlystopping(sum_loss, net)
    if earlystopping.early_stop:
        print("Early stopping")
        # ハイパーパラメータをtxtファイルに保存
        with open("./titanic_detailed/params/hyperparameter.txt", "w") as f:
            f.write(
                f"batch_size: {BATCH_SIZE}\n"
                f"weight_decay: {WEIGHT_DECAY}\n"
                f"learning_rate: {LEARNING_RATE}\n"
                f"epoch: {epoch}\n"
                f"dropout: {DROPOUT}\n"
                f"threshold: {THRESHOLD}\n"
            )
        break

# 最後10エポックの平均を出力
accuracy_maen = np.mean(test_acc_value[-10:])
# 少数第3位まで出力
accuracy_maen = round(accuracy_maen, 3)
print("accuracy_maen: ", accuracy_maen)

# ハイパーパラメータをtxtファイルに保存
with open("./titanic_detailed/params/hyperparameter.txt", "w") as f:
    f.write(
        f"batch_size: {BATCH_SIZE}\n"
        f"weight_decay: {WEIGHT_DECAY}\n"
        f"learning_rate: {LEARNING_RATE}\n"
        f"epoch: {epoch}\n"
        f"dropout: {DROPOUT}\n"
        f"threshold: {THRESHOLD}\n"
    )

plt.figure(figsize=(6, 6))  # グラフ描画用

# 以下グラフ描画
plt.plot(range(len(train_acc_value)), train_acc_value)
plt.plot(range(len(test_acc_value)), test_acc_value)
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
    f"accuracy: {accuracy_maen}",
    ha="center",
    va="center",
    color="r",
)
plt.savefig("./titanic_detailed/graphs/accuracy_image.png")
plt.show()

# モデルで予測
y_pred = []
for inputs, _ in predloader:
    inputs = inputs.to(device)
    outputs = net(inputs)
    outputs = F.softmax(outputs, dim=1)
    outputs = (outputs[:, 1] > THRESHOLD).long()
    y_pred.extend(outputs.tolist())

# 予測結果をPassengerIdと結合
y_pred = pd.DataFrame(y_pred, columns=["Perished"])
y_pred = pd.concat([PassengerId, y_pred], axis=1)
# 予測結果をcsvファイルに保存
y_pred.to_csv("./titanic_detailed/predictions/submission.csv", index=True)  # type: ignore
