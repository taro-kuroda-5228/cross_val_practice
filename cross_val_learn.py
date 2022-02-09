# シンプルなロジスティック回帰分析
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)

logreg = LogisticRegression().fit(X_train, y_train)

score = logreg.score(X_test, y_test)
print("Test set score: {}".format(score))

# 交差検証
from sklearn.model_selection import cross_val_score

logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target)

print("Cross-Validation scores: {}".format(scores))

import numpy as np

print("Average score: {}".format(np.mean(scores)))

# 層化k分割交差検証（Stratified kk-fold cross-validation）
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# 単純なk分割交差検証
kfold = KFold(n_splits=3)
print(
    "Cross-validation scores: \n{}".format(
        cross_val_score(logreg, iris.data, iris.target, cv=kfold)
    )
)

# 層化k分割交差検証
stratifiedkfold = StratifiedKFold(n_splits=3)
print(
    "Cross-validation scores: \n{}".format(
        cross_val_score(logreg, iris.data, iris.target, cv=stratifiedkfold)
    )
)


# 一般的に，回帰には単純なk分割交差検証，クラス分類には層化k分割交差検証が用いられる。
