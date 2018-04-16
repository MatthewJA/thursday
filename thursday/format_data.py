"""Functions for formatting, preprocessing, and augmentation."""

import math
import os
import pickle
import sys

from astropy.coordinates import SkyCoord
import astropy.io.ascii  as asc
import astropy.units as u
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize as skimage_resize
from sklearn.utils import shuffle


def resize(images, dim=None):
    """Rescales array of images to specified dimensions."""
    size = np.shape(images)[0]
    imgs = np.zeros((size, dim, dim))

    for i in range(size):
        imgs[i, :, :] = skimage_resize(images[i, :, :], (dim, dim))

    return imgs


def get_data(fr_data_path, aniyan_path):
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
        fr_data_path: File path of the base dataset. Must be a h5py file 
            of the exact format as the one used for this script 
            (see https://github.com/josh-marsh/thursday)
        aniyan_path: File path of the second dataset. Must be a .tsv file 
            of the exact format as the one used for this script 
            (see https://github.com/josh-marsh/thursday)
        seed: Random seed. All our tests used 0.
       
    # Returns
        Training and testing data in the form:
        (train_x, train_y, test_x, test_y). There are an equal number of 
        FRIs and FRIIs in each set, with the rest discarded. The shapes 
        of train_x and test_x are (samples, dim, dim, 1).
    """
    # Loading data
    with h5py.File(fr_data_path, 'r') as data:
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
    aniyan = asc.read(aniyan_path).to_pandas()
    
    # Splitting classes
    fri_ind  = ~labels
    frii_ind = labels
     
    # Coverting coordinates of wildly varying formats to SkyCoords
    aniyan_ra = aniyan.iloc[2:]['RAJ2000']
    aniyan_dec = aniyan.iloc[2:]['DEJ2000']
    
    coord_aniyan = SkyCoord(ra=aniyan_ra, dec=aniyan_dec, unit=('hourangle', 'deg'))
    coord_fri = SkyCoord(ra=fri_c[:, 0], dec=fri_c[:, 1], unit=('hourangle', 'deg'))
    coord_frii = SkyCoord(ra=frii_c[:, 0], dec=frii_c[:, 1], unit=('deg', 'deg'))
   
    # Finding nearest FRI's and FRIIs to each object in aniyan
    idx_i, sep_i, _i = coord_fri.match_to_catalog_sky(coord_aniyan)
    idx_ii, sep_ii, _ii = coord_frii.match_to_catalog_sky(coord_aniyan)

    # For sources within our set threshold, we assume that they are the same object.
    fri_in_aniyan = sep_i <= 5 * u.arcsec
    frii_in_aniyan = sep_ii <= 5 * u.arcsec

    fri_leftover = ~fri_in_aniyan
    frii_leftover = ~frii_in_aniyan

    aniyan_fr = np.concatenate((fri_in_aniyan, frii_in_aniyan), axis=0)
    leftover_fr = np.concatenate((fri_leftover, frii_leftover), axis=0)

    # Training and testing sets
    train_i = aniyan_fr
    test_i = leftover_fr

    return train_i, test_i


def add_noise(image):
        """ Adds tiny random Gaussian Noise to each image. 
        
         Applies the function to each image to ensure there are no
         vanishing and exploding gradients. Fixes an Error that 
         occurred when sections of an image had pixel intensity values 
         of zero due to formatting errors, resulting in intensity 
         gradients of zero that broke the arctan2 function in our custom
         keras layer.
         """
        image += 10e-10 * np.random.randn(image.shape[0], image.shape[1], 1)
        return image

    
def augment_data(rotation_range=180, zoom_range=0.2, shift_range=0.0, flip=True):
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
