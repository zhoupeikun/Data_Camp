from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
# from xgboost import XGBClassifier
from sklearn.svm import SVC


class Classifier(BaseEstimator):
    def __init__(self):
        self.n_components = 100
        self.n_estimators = 300

        # clfXgboost = XGBClassifier(learning_rate=0.1,
        #                            n_estimators=1000,
        #                            max_depth=5,
        #                            min_child_weight=1,
        #                            gamma=0,
        #                            subsample=0.8,
        #                            colsample_bytree=0.8,
        #                            objective='binary:logistic',
        #                            nthread=4,
        #                            scale_pos_weight=1,
        #                            seed=27)
        self.clf = Pipeline([
            # ('pca', PCA(n_components=self.n_components)),
            ('pca', RandomizedPCA(n_components=self.n_components, iterated_power=3, whiten=False)),
            # ('clf', RandomForestClassifier(n_estimators=self.n_estimators,
            #                                random_state=42)
            ('clf', SVC(kernel='rbf',random_state=42, probability=True, C=10000000)),
            # ('clf', clfXgboost)
        ])

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
