# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
# from sklearn.base import BaseEstimator
# import numpy as np
# import xgboost as xg
#
#
# class Regressor(BaseEstimator):
#     def __init__(self):
#         self.n_components = 100
#         self.n_estimators = 40
#         self.learning_rate = 0.2
#         self.list_molecule = ['A', 'B', 'Q', 'R']
#         self.dict_reg = {}
#         for mol in self.list_molecule:
#             self.dict_reg[mol] = Pipeline([
#                 ('pca', PCA(n_components=self.n_components)),
#                 ('reg', GradientBoostingRegressor(
#                     n_estimators=self.n_estimators,
#                     learning_rate=self.learning_rate,
#                     random_state=42)),
#                 ('reg2',xg.XGBRegressor(nthread=1,n_estimators=500))
#             ])
#         self.gbr = GradientBoostingRegressor(n_estimators=self.n_estimators,
#                                              learning_rate=self.learning_rate,
#                                              random_state=42)
#
#     def fit(self, X, y):
#         y_pred = np.zeros(X.shape[0])
#         for i, mol in enumerate(self.list_molecule):
#             ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]
#             XX_mol = X[ind_mol]
#             y_mol = y[ind_mol]
#             self.dict_reg[mol].fit(XX_mol, np.log(y_mol))
#
#         # X = np.concatenate([X, y_pred[:, None]], axis=1)
#         # self.gbr.fit(X, np.log(y.astype(float)))
#
#     def predict(self, X):
#         y_pred = np.zeros(X.shape[0])
#         for i, mol in enumerate(self.list_molecule):
#             ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]
#             XX_mol = X[ind_mol]
#             y_pred[ind_mol] = np.exp(self.dict_reg[mol].predict(XX_mol))
#         # X = np.concatenate([X, y_pred[:, None]], axis=1)
#         # y_pred = np.exp(self.gbr.predict(X))
#         return y_pred


import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import SVC


class Regressor(BaseEstimator):

    def __init__(self, split=5000):
        # self.n_components = 100
        # self.n_estimators = 40
        # self.learning_rate = 0.2
        # self.list_molecule = ['A', 'B', 'Q', 'R']
        # self.dict_reg = {}
        # for mol in self.list_molecule:
        #     self.dict_reg[mol] = Pipeline([
        #         ('pca', PCA(n_components=self.n_components)),
        #         ('reg', GradientBoostingRegressor(
        #             n_estimators=self.n_estimators,
        #             learning_rate=self.learning_rate,
        #             random_state=42)),
        #         ('reg2', xg.XGBRegressor(nthread=1, n_estimators=500))
        #     ])
        # self.gbr = GradientBoostingRegressor(n_estimators=self.n_estimators,
        #                                      learning_rate=self.learning_rate,
        #                                      random_state=42)
        self.split = split
        self.svc_split = SVC(C=10000000)
        self.svc_small = SVC(C=10000000)
        self.svc_big = SVC(C=10000000)

    def fit(self, X, y):
        is_small = np.array(y < self.split).astype(np.int)

        small = np.array(y < self.split)
        self.svc_split.fit(X, is_small)
        self.svc_small.fit(X[small], y[small])
        self.svc_big.fit(X[~small], y[~small])

    def predict(self, X):
        # y_pred = np.zeros(X.shape[0])
        # for i, mol in enumerate(self.list_molecule):
        #     ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]
        #     XX_mol = X[ind_mol]
        #     y_pred[ind_mol] = np.exp(self.dict_reg[mol].predict(XX_mol))
        # # X = np.concatenate([X, y_pred[:, None]], axis=1)
        # # y_pred = np.exp(self.gbr.predict(X))
        # return y_pred
        res = np.zeros(X.shape[0])
        is_small = self.svc_split.predict(X).astype(np.bool)

        small_pred = self.svc_small.predict(X[is_small])
        big_pred = self.svc_big.predict(X[~is_small])
        res[is_small] = small_pred
        res[~is_small] = big_pred
        return res
