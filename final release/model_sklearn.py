"""Creates various Sklearn models"""
import inspect
import pickle

import keras 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

class sklearn_model:
    
    def __init__(self, Model, name, path=None, seed=None):
        self.Model = Model
        self.name = name
        self.path = path
        self.seed = seed
        
    def load(self):    
        
        if self.path is None:
            path = self.name + ".pkl"
            
        else:
            path = self.path + self.name + ".pkl"
            
        with open(path, 'rb') as f:
            model = pickle.load(f)
            
        return model
        
    def train(self, train_x, train_y, params):
        """Instantiates and trains an sklearn model""" 
        train_x = np.reshape(train_x, (np.shape(train_x)[0], -1))
        
        model = self.Model(**params, random_state=self.seed)
        model = model.fit(train_x, train_y)
        
        if self.path is None:
            path = self.name + ".pkl"
            
        else:
            path = self.path + self.name + ".pkl"
        
        with open(path, 'wb') as f:
            pickle.dump(model, f)
            
        print("model saved as " + path)
        
        return self

    