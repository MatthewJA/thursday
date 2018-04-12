"""Creates various Sklearn models"""
import pickle

import keras 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

class SklearnModel:
    """Instantize sklearn classifer object.

    Run train method to train the classifer. Use load method to return
    the trained model.

    Attributes:
    Model: Sklearn classifier.
    name: name of the file at which the model is saved. 
        No file extension.
    data_path: Path of data folder. Defults to current directory
    seed: Random seed. All our tests used 0.
    """   
    def __init__(self, Model, name, data_path=None, seed=None):
        self.Model = Model
        self.name = name
        self.data_path = data_path
        self.seed = seed

    def load(self):    
        """Loads trained Sklearn Model from file"""
        if self.data_path is None:
            path = self.name + ".pkl"

        else:
            path = self.data_path + self.name + ".pkl"

        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model

    def train(self, train_x, train_y, params):
        """Trains sklearn model.

        Using base set of images, a random combination is augmentations within
        a defined range is applied to generate a new batch of data.

        # Arguments
            images: Array of images with shape (samples, dim, dim, 1).
            labels: Array of labels with shape (samples ,).
            params: Dictionary of parameters specific to Model.

        # Returns
            Reference to the object.
        """ 
        train_x = np.reshape(train_x, (np.shape(train_x)[0], -1))

        model = self.Model(**params, random_state=self.seed)
        model = model.fit(train_x, train_y)

        if self.data_path is None:
            path = self.name + ".pkl"

        else:
            path = self.data_path + self.name + ".pkl"

        with open(path, 'wb') as f:
            pickle.dump(model, f)

        print("model saved as " + path)

        return self

    