import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn import preprocessing
import numpy as np

class Classifier(BaseEstimator):
    def __init__(self):
        self.params={'colsample_bytree': 0.55, 'silent': 1, 'eval_metric': 'merror', 'nthread': 1, 'min_child_weight': 3.0, 'n_estimators': 460.0, 'subsample': 0.55, 'eta': 0.25, 'objective': 'multi:softprob', 'num_class': 4, 'max_depth': 3, 'gamma': 0.7000000000000001}
        self.trainRound = 100

    def fit(self, X, y):
        self.lbl_enc = preprocessing.LabelEncoder()
        self.label_to_num = self.lbl_enc.fit_transform(y)
        train_xgb = xgb.DMatrix(X, self.label_to_num)
        self.clf= xgb.train(self.params, train_xgb, self.trainRound)

    def predict(self, X):
        num_to_label= self.lbl_enc.inverse_transform(self.label_to_num)
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num_to_label[i] for i in y])
    

    def predict_proba(self, X):
        test_xgb  = xgb.DMatrix(X)
        return self.clf.predict(test_xgb)
