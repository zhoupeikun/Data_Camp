from sklearn import linear_model

from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import logistic
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xg


class Classifier(BaseEstimator):
    def __init__(self):
        self.n_components = 20
        self.n_estimators = 300


        self.clf = Pipeline([
            ('pca', PCA(n_components=self.n_components)),
            # ('pca', RandomizedPCA(n_components=self.n_components, iterated_power=3, whiten=False)),
            # ('clf', RandomForestClassifier(n_estimators=self.n_estimators,
            #                                random_state=42))
            # ("clf1", GradientBoostingClassifier(n_estimators=self.n_estimators, learning_rate=1.0 ))
            # # ('clf', SVC(kernel='rbf',random_state=42, probability=True, C=10000000)),
            # ('clf2', xg.XGBClassifier(nthread=1, n_estimators=500))
            ('clf1', RandomForestClassifier(n_estimators=self.n_estimators,
                                           random_state=42)),
            # ('clf2', xg.XGBClassifier(nthread=1, n_estimators=500)),
            ('clf', SVC(kernel='rbf', random_state=42, probability=True, C=10000000))

        ])

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
