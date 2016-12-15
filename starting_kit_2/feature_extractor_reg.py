# import numpy as np
# # import pandas as pd



import numpy as np
# import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor, LinearRegression

labels = np.array(['A', 'B', 'Q', 'R'])

r = RANSACRegressor(LinearRegression())

def get_l1_slope(array):
    n_features = 200
    return r.fit(np.arange(n_features).reshape((n_features, 1)),
                 array[:n_features]).estimator_.coef_[0]


def get_l2_slope(x):
    n = x.shape[0]
    return np.dot(range(n), x) / n - (n - 1) / 2. * x.mean()


class FeatureExtractorReg(object):
    def __init__(self):
        self.pca_n_components = 20

    def fit(self, X_df, y_df):
        X = np.array([np.array(dd) for dd in X_df['spectra']])
        self.pca = PCA(n_components=self.pca_n_components).fit(X)

    def transform(self, X_df):
        XX = np.array([np.array(dd) for dd in X_df['spectra']])
        return self.pca.transform(XX)
        return XX
