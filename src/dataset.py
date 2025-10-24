import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, file_path, n_features, n_targets):
        ds = pd.read_csv(file_path)
        self.X = ds.iloc[:, : n_features].to_numpy()
        self.Y = ds.iloc[:, -n_targets: ].to_numpy()
        self.l = [0,0]
        self.h = [0,0]
        self.min = [0,0]
        self.max = [0,0]
    
    def split(self, ratio, transposed=False):
        num = int(np.floor(ratio*self.X.shape[0]))
        x_train = self.X[ : num , :]
        x_test = self.X[num: , :]
        y_train = self.Y[ : num , :]
        y_test = self.Y[num: , :]
        if transposed:
            return x_train.T, x_test.T, y_train.T, y_test.T
        else:
            return x_train, x_test, y_train, y_test
    def normalizeX(self, l, h):
        self.l[0] = l
        self.h[0] = h
        self.min[0] = self.X.min()
        self.max[0] = self.X.max()
        self.X = l + (h-l) * (self.X - self.X.min())/(self.X.max()-self.X.min())
    
    def normalizeY(self, l, h):
        self.l[1] = l
        self.h[1] = h
        self.min[1] = self.Y.min()
        self.max[1] = self.Y.max()
        self.Y = l + (h-l) * (self.Y - self.Y.min())/(self.Y.max()-self.Y.min())
    
    def denormalize(self, arr, x_or_y):
        if x_or_y == 'x':
            l = self.l[0]
            h = self.h[0]
            g = self.max[0]
            s = self.min[0]
        else:
            l = self.l[1]
            h = self.h[1]
            g = self.max[1]
            s = self.min[1]        
        return s + (arr-l)/(h-l) * (g-s)