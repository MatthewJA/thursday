import keras 
import numpy as np
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.svm import *

def model_sklearn(train_x, train_y, method, params):
    train_x = train_x.reshape((train_x.shape[0], -1))
    
    model = method(**params)
    model = model.fit(train_x, train_y)
    
    return model  

