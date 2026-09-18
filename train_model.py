#1 ライブラリの設定
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, classification_report,accuracy_score

from pathlib import Path

import joblib
#2 ファイル場所の設定
project_dir=Path(__file__).resolve().parent

 #学習材料のデータの場所の設定
data_path=(
    project_dir
    /"notebooks"/"Dry_Bean_Dataset.csv"
)

 #モデルを移す先のフォルダを設定
model_dir=project_dir/"models"

 #モデルを移すファイル名の設定
model_path=model_dir/"bean_model.joblib"

model_dir.mkdir(exist_ok=True)

#3 データの読み込み

df=pd.read_csv(data_path)

X=df.drop(columns=["Class"])

y=df["Class"]

#4 学習データとテストデータに分割

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

#5 pipelineの定義

model=Pipeline([
    (
        "imputer",SimpleImputer(strategy="median")
    ),
    (
        "classifier",RandomForestClassifier(random_state=42)
    )
])

#6 モデルの学習

model.fit(X_train,y_train)

#7 テストデータで評価

predict=model.predict_proba(X_test)
y_preb=model.predict(X_test)

test_auc=roc_auc_score(y_test,predict,multi_class="ovr",average="macro",labels=model.classes_)

print("AUCスコア",test_auc)
cv=StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_auc=cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="roc_auc_ovo"
)

accuracy=accuracy_score(y_test,y_preb)
print("5分割交差3",cv_auc)

print(classification_report(y_test,y_preb))

#8 モデルと関連情報を保存

artifact={
    "model":model,
    "feature_names":X.columns.tolist(),
    "default_values":X_train.median().to_dict(),
    "accuracy":accuracy
}

joblib.dump(artifact,model_path)

print()
print(f"モデルを保存しました:{model_path}")
