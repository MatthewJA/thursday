"""Creates various Sklearn models"""

import keras 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def RFT(train_x, train_y, params):
    """Instantiates random forest tree."""
    
    train_x = train_x.reshape((train_x.shape[0], -1))
    
    model = RandomForestClassifier(**params)
    model = model.fit(train_x, train_y)
    
    return model  

def logistic_reg(train_x, train_y, params):
    """Instantiates logistic regression."""
    
    train_x = train_x.reshape((train_x.shape[0], -1))
    
    model = LogisticRegression(**params)
    model = model.fit(train_x, train_y)
    
    return model

def svm(train_x, train_y, params):
    """Instantiates linear support vector classifier."""
    
    train_x = train_x.reshape((train_x.shape[0], -1))
    
    model = LinearSVC(**params)
    model = model.fit(train_x, train_y)
    
    return model