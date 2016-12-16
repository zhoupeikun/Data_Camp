from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn import svm
import feature_extractor_clf

class Classifier(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=200, C=5000, gamma=10):
        self.n_components = n_components
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        self.clf = Pipeline([
            ('pca', PCA(n_components=self.n_components)),
            ('clf', svm.SVC(C=self.C, kernel='rbf', gamma=self.gamma, probability=True))
        ])
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

