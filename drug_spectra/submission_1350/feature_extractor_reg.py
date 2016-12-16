import numpy as np
from sklearn.decomposition import PCA
labels = np.array(['A', 'B', 'Q', 'R'])

class FeatureExtractorReg(object):
    def __init__(self):
        self.pca_n_components = 30
#         pass

#     def fit(self, X_df, y):
#         pass
    
#     def transform(self, X_df):                                                   
#         XX = np.array([np.array(dd) for dd in X_df['spectra']])                  
#         XX -= np.median(XX, axis=1)[:, None]                                     
#         XX /= np.sqrt(np.sum(XX ** 2, axis=1))[:, None]                          
#         XX = np.concatenate([XX, X_df[labels].values], axis=1)                   
#         return XX   

    def fit(self, X_df, y_df):
        self.X = np.array([np.array(dd) for dd in X_df['spectra']])
        self.pca = PCA(n_components=self.pca_n_components).fit(self.X)
    
    def transform(self, X_df):
        XX = np.array([np.array(dd) for dd in X_df['spectra']])
        return self.pca.transform(XX)
        return XX