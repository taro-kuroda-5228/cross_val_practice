# 層化k分割交差検証（Stratified k-fold cross-validation）
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


class KFoldVal:
    """K-fold cross-validation, simple or stratified. Generaly, regression use simple one and clasify use stratified one.
    Args:
        data: sklearn.utils.Bunch, dataset of scikit-learn like iris-data-set
    """

    def __init__(self, data):
        self.data = data

    def simple_kfold(self):
        self.logreg = LogisticRegression()
        self.kfold = KFold(n_splits=3)
        self.simple_cross_val_score = cross_val_score(
            self.logreg, self.data.data, self.data.target, cv=self.kfold
        )
        return f"Cross-Validation scores:{self.simple_cross_val_score}"

    def stratified_kfold(self):
        self.logreg = LogisticRegression()
        self.stratifiedkfold = StratifiedKFold(n_splits=3)
        self.stratified_cross_val_score = cross_val_score(
            self.logreg, self.data.data, self.data.target, cv=self.stratifiedkfold
        )
        return f"Cross-Validation scores:{self.stratified_cross_val_score}"
