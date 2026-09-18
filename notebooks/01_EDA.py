import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv(r"C:\Users\ychir\Downloads\seed sort ai\notebooks\Dry_Bean_Dataset.csv")
pd.set_option('display.max_columns',None)
print(df.head())
print(df.info())
print(len(df))
print(df.describe())

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve

y=df["Class"]

drop_columns=["Class"]
X=df.drop(columns=drop_columns)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

dummy_tree=DummyClassifier(strategy="uniform")
normal_tree=DecisionTreeClassifier(max_depth=4,min_samples_leaf=20,random_state=42)
forest_tree=RandomForestClassifier(random_state=42)

dummy_tree.fit(X_train,y_train)
normal_tree.fit(X_train,y_train)
forest_tree.fit(X_train,y_train)

dummy_predict=dummy_tree.predict_proba(X_test)
normal_predict=normal_tree.predict_proba(X_test)
forest_predict=forest_tree.predict_proba(X_test)

test_auc_1=roc_auc_score(y_test,dummy_predict,multi_class="ovr",average="macro",labels=dummy_tree.classes_)
test_auc_2=roc_auc_score(y_test,normal_predict,multi_class="ovr",average="macro",labels=normal_tree.classes_)
test_auc_3=roc_auc_score(y_test,forest_predict,multi_class="ovr",average="macro",labels=forest_tree.classes_)
print(normal_tree.classes_)
print("dummy",test_auc_1)
print("normal",test_auc_2)
print("forest",test_auc_3)
cv=StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
cv_auc_1=cross_val_score(
    dummy_tree,
    X,
    y,
    cv=cv,
    scoring="roc_auc_ovo"
)
cv_auc_2=cross_val_score(
    normal_tree,
    X,
    y,
    cv=cv,
    scoring="roc_auc_ovo"
)
cv_auc_3=cross_val_score(
    forest_tree,
    X,
    y,
    cv=cv,
    scoring="roc_auc_ovo"
)

print("5分割交差1",cv_auc_1)
print("5分割交差2",cv_auc_2)
print("5分割交差3",cv_auc_3)

#特徴量の重要度
forest_importance=permutation_importance(
    forest_tree,
    X_test,
    y_test,
    scoring="roc_auc_ovo",
    n_repeats=20,
    random_state=42
)

importance_table=pd.DataFrame({
    "特徴量":X.columns,
    "重要度":forest_importance.importances_mean,
    "標準偏差":forest_importance.importances_std
})
importance_table = (
    importance_table
    .sort_values("重要度", ascending=False)
)
print(importance_table)

#誤分類分析
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

y_pred=forest_tree.predict(X_test)

print(classification_report(y_test,y_pred))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues",
    xticks_rotation=45
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

result_df=X_test.copy()
result_df["Actual"]=y_test
result_df["Predicted"]=y_pred

#正解したかどうかの判定
result_df["Correct"]=(
    result_df["Actual"]==result_df["Predicted"]
)

error_df=result_df[result_df["Correct"]==False]

print("テストデータ数", len(result_df))
print("誤分類数",len(error_df))
print("誤分類率",len(error_df)/len(result_df))

print(error_df.head())

#どのクラス同士が間違えたか
error_pairs=(
    error_df
    .groupby(["Actual","Predicted"])
    .size()
    .reset_index(name="Count")
    .sort_values("Count",ascending=False)
)

print(error_pairs)

class_result = (
    result_df
    .groupby("Actual")["Correct"]
    .agg(
        Total="count",
        Correct_Count="sum"
    )
)

class_result["Error_Count"] = (
    class_result["Total"] - class_result["Correct_Count"]
)

class_result["Accuracy"] = (
    class_result["Correct_Count"] / class_result["Total"]
)

class_result["Error_Rate"] = (
    class_result["Error_Count"] / class_result["Total"]
)

print(
    class_result.sort_values(
        "Error_Rate",
        ascending=False
    )
)