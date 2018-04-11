"""Functions for formatting, preprocessing, and augmentation."""

import sys
import os
import pickle

from astropy.coordinates import SkyCoord
import astropy.io.ascii  as asc
import astropy.units as u
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from sklearn.utils import shuffle


def rescale_imgs(images, dim):
    """Rescales array of images to specified dimensions."""
    size = images.shape[0]
    imgs = np.zeros((size, dim, dim))

    for i in range(size):
        imgs[i, :, :] = resize(images[i, :, :], (dim, dim))

    return np.expand_dims(imgs, axis=3)
        
def get_data(first_path, second_path, dim=256, seed=0, conf_matrix=True, flip_train_test=False):
    """Formats data into training and testing sets.
    
    Takes a base dataset and a second dataset in specific formats. 
    The base dataset should contain all samples that are in the 
    second dataset. Coordinate information is first extracted and then
    converted into the universal SkyCoord format. These are then cross
    matched. Sources within a set radius of each other are defined 
    assumed to be the same object. All image in the second dataset that
    are also in the first are extracted. The remaining images are in the
    base dataset are formated into a balanced training set and the 
    extracted images are formatted into a balanced testing set.
    
    
    # Arguments
        first_path: File path of the base dataset. Must be a h5py file 
            of the exact format as the one used for this script 
            (see https://github.com/josh-marsh/thursday)
        second_path: File path of the second dataset. Must be a .tsv file 
            of the exact format as the one used for this script 
            (see https://github.com/josh-marsh/thursday)
        dim: Length and width that the images are rescaled two as an 
            integer.
        seed: Random seed. All our tests used 0.
        conf_matrix: If true, train_y and test_y are returned as confusion
            matrices. If False, an array of bools. Defaults to True
        flip_train_test: If true, the training and testing data are reversed.
            That is, images in the second dataset are used for training and
            vice versa. Defaults to False
        
        
    # Returns
        Training and testing data in the form:
        (train_x, train_y, test_x, test_y). There are an equal number of 
        FRIs and FRIIs in each set, with the rest discarded. The shapes 
        of train_x and test_x are (nb_samples, dim, dim, 1).

    """
    # Loading data
    with h5py.File(first_path, 'r') as data:
        images = data['images'].value
        labels = data['labels'].value.astype(bool)
        
        fri_data = data['fri_data']
        frii_data = data['frii_data']
        
        # Converting coordinates into a Astropy Skycoord readable format
        fri_ra = fri_data['ra'].value.astype(str).reshape((-1, 1))
        fri_de = fri_data['dec'].value.astype(str).reshape((-1, 1))

        frii_ra = frii_data['_RA'].value.astype(str).reshape((-1, 1))
        frii_de = frii_data['_DE'].value.astype(str).reshape((-1, 1))
        
        fri_c = np.hstack((fri_ra, fri_de))
        frii_c = np.hstack((frii_ra, frii_de))

    # Opening Catalogues
    asu = asc.read(second_path).to_pandas()
    
    # Splitting classes
    fri_images = images[~labels]
    frii_images = images[labels]
     
    # Coverting coordinates of wildly varying formats to SkyCoords
    asu_ra = asu.iloc[2:]['RAJ2000']
    asu_dec = asu.iloc[2:]['DEJ2000']
    
    coord_asu = SkyCoord(ra=asu_ra, dec=asu_dec, unit=('hourangle', 'deg'))
    coord_fri = SkyCoord(ra=fri_c[:, 0], dec=fri_c[:, 1], unit=('hourangle', 'deg'))
    coord_frii = SkyCoord(ra=frii_c[:, 0], dec=frii_c[:, 1], unit=('deg', 'deg'))
   
    # Finding nearest FRI's and FRIIs to each object in ASU
    idx_i, sep_i, _i = coord_fri.match_to_catalog_sky(coord_asu)
    idx_ii, sep_ii, _ii = coord_frii.match_to_catalog_sky(coord_asu)

    # For sources within our set threshold, we assume that they are the same object.
    fri_in_asu = sep_i <= 5 * u.arcsec
    frii_in_asu = sep_ii <= 5 * u.arcsec

    # Extacting FR images that are in asu
    asu_fri = fri_images[fri_in_asu]
    asu_frii = frii_images[frii_in_asu]

    leftover_fri = fri_images[~fri_in_asu]
    leftover_frii = frii_images[~frii_in_asu]
 
    # Generating Labels
    asu_fri_labels = np.zeros(asu_fri.shape[0], dtype=bool)
    asu_frii_labels = np.ones(asu_frii.shape[0], dtype=bool)

    leftover_fri_labels = np.zeros(leftover_fri.shape[0], dtype=bool)
    leftover_frii_labels = np.ones(leftover_frii.shape[0], dtype=bool)

    # Combining FRI's and FRII's and shuffling
    asu_fr = np.vstack((asu_fri, asu_frii))
    asu_fr_labels = np.concatenate((asu_fri_labels, asu_frii_labels), axis=0) 

    leftover_fr = np.vstack((leftover_fri, leftover_frii))
    leftover_fr_labels = np.concatenate((leftover_fri_labels, leftover_frii_labels), axis=0)             
    # Shuffling
    asu_fr_, asu_fr_labels_ = shuffle(asu_fr, asu_fr_labels, random_state=seed)
    leftover_fr_, leftover_fr_labels_ = shuffle(leftover_fr, leftover_fr_labels, random_state=seed) 

    # Setting training and testing sets
    if not flip_train_test:
        train_x = asu_fr_
        train_y = asu_fr_labels_

        test_x = leftover_fr_
        test_y = leftover_fr_labels_

    else:
        train_x = leftover_fr_
        train_y = leftover_fr_labels_

        test_x = asu_fr_
        test_y = asu_fr_labels_

    #Rescaling Images
    train_x = rescale_imgs(train_x, dim)
    test_x = rescale_imgs(test_x, dim)

    # Convert class vectors to binary class matrices (for Keras Dense)
    if conf_matrix:
        train_y = np_utils.to_categorical(train_y, 2)
        test_y = np_utils.to_categorical(test_y, 2)
    
    return train_x, train_y, test_x, test_y    

def add_noise(image):
        """ Adds tiny random Gaussian Noise to each image. 
        
         Applies the function to each image to ensure there are no
         vanishing and exploding gradients. Fixes an Error that 
         occurred when sections of an image had pixel intensity values 
         of zero due to formatting errors, resulting in intensity 
         gradients of zero that broke the arctan2 function in our custom
         keras layer"""
        
        image += 10e-10 * np.random.randn(image.shape[0], image.shape[1], 1)
        return image

    
def data_gen(rotation_range=180, zoom_range=0.2, shift_range=0.0, flip=True):
    """ Initializes data generator."""
    
    # Defining Data Generator
    datagen = ImageDataGenerator(
                rotation_range=rotation_range,
                zoom_range=zoom_range,
                width_shift_range=shift_range,
                height_shift_range=shift_range,
                horizontal_flip=flip,
                vertical_flip=flip,
                fill_mode ="nearest",
                preprocessing_function=add_noise)
    
    return datagen


def data_pregenerate(images, labels, datagen, batch_size, nb_epoch, seed):
    """Pre-generates data for Sklearn models.
    
    Using base set of images, a random combination is augmentations within
    a defined range is applied to generate a new batch of data.
    
    
    # Arguments
        images: Array of images with shape (samples, dim, dim, 1)
        labels: Array of labels
        datagen: The data generator outputed by the data_gen function.
        batch_size: Number of sample per batch.
        nb_epoch: The number of epochs to generate for.
        seed: Random seed. All our tests used 0.
        
    # Returns
        Array of augmented images and their corresponding labels.

    """
    samples = images.shape[0]
    
    # the .flow() command below generates batches of randomly transformed images
    gen = datagen.flow(images, labels, batch_size=batch_size, seed=seed)

    # Generate empty data arrays
    pro_images = np.zeros((images.shape[0] * nb_epoch, images.shape[1],
                           images.shape[2], 1))
    pro_labels = np.zeros((labels.shape[0] * nb_epoch))
    
    for epoch in range(1, nb_epoch+1):
        batch = 1
        b = batch_size
        b_start = samples * (epoch-1)

        for X_batch, Y_batch in gen:
            if batch < (samples / b):
                cut_start = b_start + b*(batch-1)
                cut_stop = b_start + batch*b
                
                pro_images[cut_start:cut_stop, :, :, :] = X_batch
                pro_labels[cut_start:cut_stop] = Y_batch
                
            elif batch == int(samples / b):
                break

            else:
                cut_start = b_start + b*(batch-1)
                cut_stop = b_start + b*(batch-1) + X_batch.shape[0] % b
                
                pro_images[cut_start:cut_stop, :, :, :] = X_batch
                pro_labels[cut_start:cut_stop] = Y_batch
                break

            batch += 1
        
    return pro_images, pro_labels


def data_pregen(train_x, train_y, test_x, test_y, 
                datagen, batch_size, nb_epoch, seed):
    """Applies data_pregenerate to both the training and testing test"""
    train_x, train_y = data_pregenerate(images=train_x, labels=train_y,
                                        datagen=datagen, batch_size=batch_size,
                                        nb_epoch=nb_epoch, seed=seed)
    
    test_x, test_y  = data_pregenerate(images=test_x, labels=test_y, 
                                       datagen=datagen, batch_size=batch_size,
                                       nb_epoch=nb_epoch, seed=seed)
    
    return train_x, train_y, test_x, test_y
