import numpy as np
import pandas as pd


def mixup(train_x, train_y, alpha=0.1):  # type: ignore
    # train_x,train_y:入力データとラベル
    # alpha:ベータ分布のパラメータ

    # 乱数の固定
    np.random.seed(622)

    # mixup
    length = len(train_x)  # Changed variable name from 'l' to 'length'
    train_x = train_x.values
    train_y = train_y.values
    # mix_x,mix_yをpandas.DataFrame型で初期化
    mix_x = []
    mix_y = []
    for i in range(length):  # Changed variable name from 'l' to 'length'
        j = np.random.randint(
            length
        )  # Changed variable name from 'l' to 'length'
        mix_x.append(alpha * train_x[i] + (1 - alpha) * train_x[j])
        mix_y.append(alpha * train_y[i] + (1 - alpha) * train_y[j])

    # mixと元のデータを結合
    mix_x = np.vstack([train_x, mix_x])
    mix_y = np.hstack([train_y, mix_y])

    # Convert mix_x and mix_y to pandas DataFrame
    mix_x = pd.DataFrame(mix_x)
    mix_y = pd.DataFrame(mix_y)

    # csvファイルに保存
    mix_x.to_csv("./data_augmentation/augmented_data/mix_x.csv", index=False)
    mix_y.to_csv("./data_augmentation/augmented_data/mix_y.csv", index=False)

    # 形状を確認
    print("mix_x.shape: ", mix_x.shape)
    print("mix_y.shape: ", mix_y.shape)

    return pd.DataFrame(mix_x), pd.DataFrame(mix_y)


# # csvファイルの読み込み
# data = pd.read_csv("./data_augmentation/fixed_data/X.csv")

# # train_x,train_yに分割
# train_y = data["Perished"]
# # PerishedとPassengerIdを削除
# train_x = data.drop(["Perished", "PassengerId"], axis=1)

# # mixup
# mix_x, mix_y = mixup(train_x, train_y)  # type: ignore
