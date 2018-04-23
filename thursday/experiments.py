import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB

from format_data import augment_data, get_data, get_fr_data, resize
from models import SklearnModel, HOGNet

print("Experiment 1: Classifing FR-I vs FR-II")
print("Training on two thirds of class-wise balenced FR Data")
print("Testing on a third of class-wise balenced FR Data")

# Setting Seed
seed = 0

# Setting Data Generator
datagen = augment_data(rotation_range=180, zoom_range=0.2, shift_range=0.0, flip=True)

# Getting data
train_i, test_i = get_fr_data("data/data.h5", split_ratio=(1/3), seed=seed)

with h5py.File("data/data.h5", 'r') as data:
    train_x = np.asarray(data['images'])[train_i]
    train_y = np.asarray(data['labels'])[train_i]
    test_x = np.asarray(data['images'])[test_i]
    test_y = np.asarray(data['labels'])[test_i]
    
    # Formatting images
    train_x = resize(train_x, dim=256)
    test_x = resize(test_x, dim=256)
    train_x = np.expand_dims(train_x, axis=3)
    test_x = np.expand_dims(test_x, axis=3)

    
# Instantiating Classifiers
random_forest = SklearnModel(RandomForestClassifier, datagen=datagen, nb_augment=10, seed=seed)
naive_bayes = SklearnModel(GaussianNB, datagen=datagen, nb_augment=5, seed=seed)
hognet = HOGNet(datagen=datagen, batch_size=64, steps_per_epoch=50, 
                max_epoch=100, patience=5, gap=2, seed=seed)

classifiers = [random_forest, naive_bayes, hognet]

for classifier in classifiers:
    classifier = classifier.fit(train_x, train_y)
    predictions = classifier.predict(test_x)

    print (predictions)

    correct = test_y == predictions
    print('Accuracy: {:.02%}'.format(correct.mean()))
