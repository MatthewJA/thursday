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
    size = images.shape[0]
    imgs = np.zeros((size, dim, dim))

    for i in range(size):
        imgs[i, :, :] = skimage_resize(images[i, :, :], (dim, dim))

    return imgs


def get_data(fr_data_path, aniyan_path, seed=0):
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
        seed: Seed value to consistently initialize the random number 
                generator.  
       
    # Returns
        Locations training and testing data with shape 
        [samples,]. There are an equal number of FRIs and FRIIs in each 
        set, with the rest discarded.
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

    # Fectching indices
    fri_in_aniyan = np.where(fri_in_aniyan)[0]
    frii_in_aniyan = np.where(frii_in_aniyan)[0] + np.where(fri_ind)[0].shape[0]

    fri_leftover = np.where(fri_leftover)[0]
    frii_leftover = np.where(frii_leftover)[0] + np.where(fri_ind)[0].shape[0]

    # Finding the number of samples in the smallest fr class
    # So there are an equal amount of each class in each set
    aniyan_cut = np.min(np.array([fri_in_aniyan.shape[0], frii_in_aniyan.shape[0]]))
    leftover_cut = np.min(np.array([fri_leftover.shape[0], frii_leftover.shape[0]]))

    # Shuffling before discarding
    fri_in_aniyan = shuffle(fri_in_aniyan, random_state=seed)
    frii_in_aniyan = shuffle(frii_in_aniyan, random_state=seed)
    fri_leftover = shuffle(fri_leftover, random_state=seed)
    frii_leftover = shuffle(frii_leftover, random_state=seed)

    # Discarding samples
    fri_in_aniyan = fri_in_aniyan[:aniyan_cut]
    frii_in_aniyan = frii_in_aniyan[:aniyan_cut]
    fri_leftover = fri_leftover[:leftover_cut]
    frii_leftover = frii_leftover[:leftover_cut]

    aniyan = np.concatenate((fri_in_aniyan, frii_in_aniyan), axis=0)
    leftover = np.concatenate((fri_leftover, frii_leftover), axis=0)

    # Shuffling FRI's and FRII's
    aniyan = np.sort(shuffle(aniyan, random_state=seed), axis=0)
    leftover = np.sort(shuffle(leftover, random_state=seed), axis=0)

    # Training and testing sets
    train_i = leftover
    test_i = aniyan

    # Shuffling FRI's and FRII's
    train_i = shuffle(train_i, random_state=seed)
    test_i = shuffle(test_i, random_state=seed)

    return train_i, test_i

def get_fr_data(fr_data_path, split_ratio=0.5, seed=0):
    """Splits FR data into a training and testing set.
    
    # Arguments
        fr_data_path: File path of the base dataset. Must be a h5py file 
            of the exact format as the one used for this script 
            (see https://github.com/josh-marsh/thursday)
        split_ratio: Factor of the data used for testing. The value 
            is rounded to the nearest indice. Default is 0.5.
        seed: Seed value to consistently initialize the random number 
            generator.
       
    # Returns
        Locations training and testing data  with shape [samples,]. 
        There are an equal number of FRIs and FRIIs in each set, with the
        rest discarded.
    """
    # Loading data
    with h5py.File(fr_data_path, 'r') as data:
        labels = data['labels'].value.astype(bool)
    
    # Splitting classes
    fri_ind  = np.where(~labels)[0]
    frii_ind = np.where(labels)[0]

    # Finding the number of samples in the smallest fr class
    # So there are an equal amount of each class in each set
    cut = np.min(np.array([fri_ind.shape[0], frii_ind.shape[0]]))
    cut = cut - cut % 2    # So there can be an even train/test split

    # Shuffling before discarding
    fri_ind = shuffle(fri_ind, random_state=seed)
    frii_ind = shuffle(frii_ind, random_state=seed)

    # Discarding samples
    fri_ind = np.sort(fri_ind[:cut], axis=0)
    frii_ind = np.sort(frii_ind[:cut], axis=0)

    # Slitting into training and testing sets
    split_i = int(np.round(cut * split_ratio))

    train_fri = fri_ind[split_i:]
    train_frii = frii_ind[split_i:]
    test_fri = fri_ind[:split_i]
    test_frii = frii_ind[:split_i]

    train_i = np.concatenate((train_fri, train_frii), axis=0)
    test_i = np.concatenate((test_fri, test_frii), axis=0)

    # Shuffling FRI's and FRII's
    train_i = shuffle(train_i, random_state=seed)
    test_i = shuffle(test_i, random_state=seed)

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
    """ Initializes data generator.

    Arguments:
        rotation_range: Int. Degree range for random rotations.
        shift_range: Float (fraction of total width). Range for random 
            horizontal and vertical shifts.
        zoom_range: Float or [lower, upper]. Range for random zoom. If a 
            float, [lower, upper] = [1-zoom_range, 1+zoom_range]
        horizontal_flip: Boolean. Randomly flip inputs horizontally.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}. Default 
            is 'nearest'. Points outside the boundaries of the input are filled
            according to the given mode
        preprocessing_function: function that will be implied on each input. 
            The function will run after the image is resized and augmented. The 
            function should take one argument: one image (Numpy tensor with 
            rank 3), and should output a Numpy tensor with the same shape.
    """
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
