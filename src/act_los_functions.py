import numpy as np
class Functions_f_df:
    def __init__(self, f, df):
        self.f = f
        self.df = df

#Activation Functions

#Relu
def relu_f(x):
    return np.maximum(0, x)
def relu_df(x):
    return np.where(x > 0, 1, 0)

#Leaky Relu
def lrelu_f(x):
    return np.where(x > 0, x, x*0.01)
def lrelu_df(x):
    return np.where(x > 0, 1, 0.01)

#Linear
def lin_f(x):
    return x
def lin_df(x):
    return np.ones_like(x)


Relu = Functions_f_df(relu_f, relu_df)
Linear = Functions_f_df(lin_f, lin_df)

#Loss Functions

#MSE
def mse_f(x_true, x_pred):
    return np.mean(np.power(x_pred-x_true, 2))
def mse_df(x_true, x_pred):
    return 2 * (x_pred-x_true)
MSE_Loss = Functions_f_df(mse_f, mse_df)