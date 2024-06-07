import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def filter(data: pd.DataFrame) -> pd.DataFrame:

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
    Female_Child_Group = Female_Child_Group.groupby("Surname")[
        "Perished"
    ].mean()
    # 家族で16才超えかつ男性の生存率
    Male_Adult_Group = data.loc[
        (data["FamilyGroup"] >= 2)
        & (data["Age"] > 16)
        & (data["Sex"] == "male")
    ]
    Male_Adult_List = Male_Adult_Group.groupby("Surname")["Perished"].mean()

    # デッドリストとサバイブリストの作成
    Dead_list = set(
        Female_Child_Group[Female_Child_Group.apply(lambda x: x == 0)].index
    )
    Survived_list = set(
        Male_Adult_List[Male_Adult_List.apply(lambda x: x == 1)].index
    )

    # デッドリストとサバイブリストの該当者の'perished'を0, 1に置換
    data.loc[
        (data["Perished"].isnull())
        & (data["Surname"].apply(lambda x: x in Dead_list)),
        ["Perished"],
    ] = 0
    data.loc[
        (data["Perished"].isnull())
        & (data["Surname"].apply(lambda x: x in Survived_list)),
        ["Perished"],
    ] = 1

    return data
