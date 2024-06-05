# ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings

warnings.filterwarnings('ignore')

# データの読み込み
path = '~/titanic_ws/data/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# Cabinカラムを削除
train.drop(columns=['Cabin'], inplace=True)
test.drop(columns=['Cabin'], inplace=True)

# 'Embarked' の欠損値を最頻値で補完
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)

# 'Sex' と 'Embarked' のカテゴリカル特徴量を数値に変換
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 'Name' カラムから敬称を抽出し、新しい特徴量 'Title' として追加
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 敬称を数値に変換（頻度が低い敬称を 'Rare' としてまとめる）
rare_titles = ['Don', 'Rev', 'Dr', 'Mme', 'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer']
train['Title'] = train['Title'].replace(rare_titles, 'Rare')
test['Title'] = test['Title'].replace(rare_titles, 'Rare')
train['Title'] = train['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
test['Title'] = test['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})

# 欠損値を '0' に変換
train['Title'].fillna(0, inplace=True)
test['Title'].fillna(0, inplace=True)

# 特徴量を選択
features = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch', 'Fare', 'Title']

# 特徴量をスケーリング
scaler = StandardScaler()
train_scaled_features = scaler.fit_transform(train[features])
test_scaled_features = scaler.transform(test[features])

# MICE補完の設定
imputer = IterativeImputer(random_state=42, max_iter=20, min_value=0, max_value=80)

# 欠損値補完の実行
train['Age'] = imputer.fit_transform(np.hstack((train_scaled_features, np.expand_dims(train['Age'], axis=1))))[:, -1]
test['Age'] = imputer.transform(np.hstack((test_scaled_features, np.expand_dims(test['Age'], axis=1))))[:, -1]

# 補完後の年齢の統計量を確認
print("補完後の年齢の統計量 (train):")
print(train['Age'].describe())
print("補完後の年齢の統計量 (test):")
print(test['Age'].describe())

# 負の年齢があるかを確認
negative_ages_train = train[train['Age'] < 0]
negative_ages_test = test[test['Age'] < 0]
if not negative_ages_train.empty or not negative_ages_test.empty:
    print(f"負の年齢が {len(negative_ages_train) + len(negative_ages_test)} 件あります。")
    # 負の値を持つ行のインデックス
    neg_idx_train = negative_ages_train.index
    neg_idx_test = negative_ages_test.index
    # 年齢を再補完（負の値を0歳に設定し、再補完）
    train.loc[neg_idx_train, 'Age'] = np.nan
    test.loc[neg_idx_test, 'Age'] = np.nan
    train['Age'] = imputer.fit_transform(np.hstack((train_scaled_features, np.expand_dims(train['Age'], axis=1))))[:, -1]
    test['Age'] = imputer.transform(np.hstack((test_scaled_features, np.expand_dims(test['Age'], axis=1))))[:, -1]
    print("再補完後の年齢の統計量 (train):")
    print(train['Age'].describe())
    print("再補完後の年齢の統計量 (test):")
    print(test['Age'].describe())

# 目的変数と説明変数に分割
X = train[features + ['Age']]
y = train['Perished']

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徴量のスケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# モデルの定義
models = {
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(max_iter=1000),
    "GradientBoosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

# ハイパーパラメータ調整用のパラメータグリッド
param_grids = {
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30]
    },
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "solver": ["liblinear"]
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [(50, 50), (100,)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001]
    },
    "GradientBoosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },
    "LightGBM": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "num_leaves": [31, 50, 100]
    },
    "CatBoost": {
        "iterations": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "depth": [3, 5, 7]
    }
}

# グリッドサーチによるハイパーパラメータ調整
best_estimators = {}
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')  # LightGBMの警告を無視

for model_name in models:
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=StratifiedKFold(n_splits=5), scoring="accuracy")
    grid_search.fit(X_train_scaled, y_train)
    best_estimators[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best score for {model_name}: {grid_search.best_score_}")

# アンサンブルモデルの定義
voting_clf = VotingClassifier(estimators=[(name, best_estimators[name]) for name in best_estimators], voting='soft')
voting_clf.fit(X_train_scaled, y_train)

# アンサンブルモデルのクロスバリデーション評価
scores = cross_val_score(voting_clf, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
print(f"Cross-validated accuracy: {scores.mean()} +/- {scores.std()}")

# モデルの評価
results = {
    "Model": [],
    "Accuracy": [],
    "F1 Score": [],
    "AUC-ROC": []
}

for model_name, model in best_estimators.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    print(f"Accuracy of {model_name}: {accuracy}")
    results["Model"].append(model_name)
    results["Accuracy"].append(accuracy)
    results["F1 Score"].append(f1)
    results["AUC-ROC"].append(auc_roc)

# アンサンブルモデルの評価
y_pred_ensemble = voting_clf.predict(X_test_scaled)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble)
auc_roc_ensemble = roc_auc_score(y_test, y_pred_ensemble)
print(f"Accuracy of Ensemble: {accuracy_ensemble}")
results["Model"].append("Ensemble")
results["Accuracy"].append(accuracy_ensemble)
results["F1 Score"].append(f1_ensemble)
results["AUC-ROC"].append(auc_roc_ensemble)

# 結果の可視化
results_df = pd.DataFrame(results)

plt.figure(figsize=(12, 8))
sns.barplot(x="Accuracy", y="Model", data=results_df, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.xlabel("Accuracy")
plt.ylabel("Model")
plt.show()

# 精度の向上を数値的に示す
improvement = accuracy_ensemble - results_df["Accuracy"].min()
print(f"Ensembleによる精度の向上: {improvement}")

# 混同行列と分類レポートの表示
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ensemble))
print("Classification Report:")
print(classification_report(y_test, y_pred_ensemble))
