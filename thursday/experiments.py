import h5py
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB

from format_data import add_random
from format_data import append_random
from format_data import augment
from format_data import generate_labels_fri
from format_data import generate_labels_frii
from format_data import get_aniyan
from format_data import get_fr
from format_data import get_random
from format_data import initial_resizing
from format_data import resize_array
from models import HOGNet
from models import SklearnModel

print("Experiment 1: Classifing FR-I vs FR-II")
print("Training on two thirds of class-wise balenced FR Data")
print("Testing on a third of class-wise balanced FR Data")

data_path = 'data'
save_path = 'saved_models'

# Setting Paths
raw_fr_data_data = Path.cwd() / data_path / 'data_raw.h5'
fr_data_path = Path.cwd() / data_path / 'data.h5'
random_path = Path.cwd() / data_path / 'random_1000.h5'
everything_path = Path.cwd() / data_path / 'everything.h5'

if not Path(raw_fr_data_data).is_file():
    raise print("Error: " + raw_fr_data_data + " is not a valid path and/or file")

if not Path(random_path).is_file():
    raise print("Error: " + random_path + " is not a valid path and/or file")

if not Path(fr_data_path).is_file():
    initial_resizing(raw_fr_data_data, fr_data_path)

if not Path(everything_path).is_file():
    add_random(fr_data_path, random_path, everything_path)

# Setting Seed
seed = 0

# Setting Data Generator
datagen = augment(rotation_range=180, zoom_range=0.2, shift_range=0.0, flip=True)

# Getting data
train_i, test_i = get_fr(everything_path, split_ratio=(1/3), seed=seed)

with h5py.File(everything_path, 'r') as data:
    train_x = np.asarray(data['images'])[train_i]
    test_x = np.asarray(data['images'])[test_i]
    labels = np.asarray(data['labels'])
    
    # Formatting images
    train_x = resize_array(train_x, dim=256)
    test_x = resize_array(test_x, dim=256)
    train_x = np.expand_dims(train_x, axis=3)
    test_x = np.expand_dims(test_x, axis=3)

    # Formating
    train_y = np.where(labels[train_i] == 1, False, True)
    test_y = np.where(labels[test_i] == 1, False, True)
    
# Instantiating Classifiers
random_forest = SklearnModel(RandomForestClassifier, datagen=datagen, 
                nb_augment=10, seed=seed)

naive_bayes = SklearnModel(GaussianNB, datagen=datagen, nb_augment=5, seed=seed)
hognet = HOGNet(datagen=datagen, batch_size=64, steps_per_epoch=50, 
                max_epoch=100, patience=5, gap=2, seed=seed)

classifiers = [random_forest, naive_bayes, hognet]

for classifier in classifiers:
    print ("Running " + classifier.name)

    classifier = classifier.fit(train_x, train_y)
    predictions = classifier.predict(test_x)

    print ("Class predictions for testing data")
    print (predictions)

    correct = test_y == predictions
    print('Accuracy: {:.02%}'.format(correct.mean()))

print("Experiment 2: Classifing FR-I vs Random Sources")

# Getting data
train_i, test_i = get_random(everything_path, split_ratio=(1/3), seed=seed)

with h5py.File(everything_path, 'r') as data:
    train_x = np.asarray(data['images'])[train_i]
    test_x = np.asarray(data['images'])[test_i]
    labels = np.asarray(data['labels'])
    
    # Formatting images
    train_x = resize_array(train_x, dim=256)
    test_x = resize_array(test_x, dim=256)
    train_x = np.expand_dims(train_x, axis=3)
    test_x = np.expand_dims(test_x, axis=3)

    # Formating
    train_y = np.where(labels[train_i] == 1, True, False)
    test_y = np.where(labels[test_i] == 1, True, False)


for classifier in classifiers:
    print ("Running " + classifier.name)

    classifier = classifier.fit(train_x, train_y)
    predictions = classifier.predict(test_x)

    print ("Class predictions for testing data")
    print (predictions)

    correct = test_y == predictions
    print('Accuracy: {:.02%}'.format(correct.mean()))

