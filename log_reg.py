# シンプルなロジスティック回帰分析
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class LogReg:
    """Simple logistic regression analysis.
    Args:
        data: sklearn.utils.Bunch, dataset of scikit-learn like iris-data-set
    """

    def __init__(self, data):
        self.data = data

    def log_reg(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data.data, self.data.target, random_state=0
        )
        self.logreg = LogisticRegression().fit(self.X_train, self.y_train)
        self.score = self.logreg.score(self.X_test, self.y_test)
        return self.score
