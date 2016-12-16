import numpy as np
import pandas as pd
from scipy import signal as signal
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractorClf(BaseEstimator, TransformerMixin):
    def __init__(self, window_length=25, polyorder=4):
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, X_df, y=None):
        return self

    def transform(self, X_df):
        n_samples = X_df.shape[0]
        XX = np.array([np.array(dd) for dd in X_df['spectra']])
        vial = pd.get_dummies(X_df['vial'], drop_first=True)
        XX = signal.savgol_filter(XX, axis=1, window_length=self.window_length, polyorder=self.polyorder, mode='nearest')
        XX = signal.detrend(XX, axis=1)
        #XX = np.concatenate([XX, vial], axis=1)
        return XX
