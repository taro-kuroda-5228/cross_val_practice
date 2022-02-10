# 交差検証
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import numpy as np


class CrossVal:
    """Cross validation analysis
    Args:
        data: sklearn.utils.Bunch, dataset of scikit-learn like iris-data-set
    """

    def __init__(self, data):
        self.data = data

    def cross_val(self):
        self.logreg = LogisticRegression()
        self.scores = cross_val_score(self.logreg, self.data.data, self.data.target)
        self.avg_score = np.mean(self.scores)
        return f"Cross-Validation scores:{self.scores}, Average score:{self.avg_score}"
