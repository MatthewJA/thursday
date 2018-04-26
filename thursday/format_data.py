"""Functions for formatting, preprocessing, and augmentation."""

import math
import os
import pickle
import sys

from astropy.coordinates import SkyCoord
import astropy.io.ascii  as asc
import astropy.units as u
import h5py
from keras.preprocessing.image import apply_transform
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize as skimage_resize
from sklearn.utils import shuffle


def resize_array(images, dim=None):
   """Rescales array of images to specified dimensions."""
   size = images.shape[0]
   imgs = np.zeros((size, dim, dim))

   for i in range(size):
      imgs[i, :, :] = skimage_resize(images[i, :, :], (dim, dim))

   return imgs


def get_aniyan(fr_data_path, aniyan_path, seed=0):
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
   
   # FRI ind
   fri_ind = np.where(labels==1)
   
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
   frii_in_aniyan = np.where(frii_in_aniyan)[0] + fri_ind.shape[0]

   fri_leftover = np.where(fri_leftover)[0]
   frii_leftover = np.where(frii_leftover)[0] + fri_ind.shape[0]

   # Concatenating
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


def get_fr(fr_data_path, split_ratio=0.5, seed=0):
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
   """
   # Loading data
   with h5py.File(fr_data_path, 'r') as data:
      labels = np.asarray(data['labels'])
   
   # Splitting classes
   fri_i  = np.where(labels==1)[0]
   frii_i = np.where(labels==2)[0]

   # Slitting into training and testing sets
   cut = int(np.round(split_ratio * fri_i.shape[0]))
   train_fri = fri_i[cut:]
   test_fri = fri_i[:cut]

   cut = int(np.round(split_ratio * frii_i.shape[0]))
   train_frii = frii_i[cut:]
   test_frii = frii_i[:cut]

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


def augment(rotation_range=180, zoom_range=0.2, shift_range=0.0, flip=True):
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


def shift(x, row_ind, col_ind, row_axis=0, col_axis=1, channel_axis=2,
             fill_mode='constant', cval=0.):
   """Performs a random spatial shift of a Numpy image tensor.
   
   # Arguments
      x: Input tensor. Must be 3D.
      wrg: Width shift range, as a float fraction of the width.
      hrg: Height shift range, as a float fraction of the height.
      row_axis: Index of axis for rows in the input tensor.
      col_axis: Index of axis for columns in the input tensor.
      channel_axis: Index of axis for channels in the input tensor.
      fill_mode: Points outside the boundaries of the input
         are filled according to the given mode
         (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
      cval: Value used for points outside the boundaries
         of the input if `mode='constant'`.
         
   # Returns
      Shifted Numpy image tensor.
   """
   h, w = x.shape[row_axis], x.shape[col_axis]
   tx = row_ind - (h / 2)
   ty = col_ind - (w / 2) 
   translation_matrix = np.array([[1, 0, tx],
                           [0, 1, ty],
                           [0, 0, 1]])

   transform_matrix = translation_matrix  # no need to do offset
   x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
   
   return x


def center_on_brightest(x):
   dim = np.asarray(x.shape)
   max_i = np.zeros((dim[0], 2))
   x_shift = np.zeros(dim)

   for i in range(dim[0]):
      max_i[i, :] = np.unravel_index(np.argmax(x[i, :, :]), dims=dim[1:])
      
      centered = shift(np.expand_dims(x[i, :, :],  2), max_i[i, 0], max_i[i, 1])
      x_shift[i, :, :] = centered[:, :, 0]
      
   #radius = np.sqrt((max_i[:, 0]-dim[1]/2)**2 + (max_i[:, 1]-dim[2]/2)**2)
   #in_range = np.where(radius < dim[1]/3)
   
   x = x_shift #[in_range]
   
   return x


def initial_resizing(fr_raw_data_path, fr_data_path, dim=300):
   """Resizes raw ouput of fri-frii-download.
   
   Takes ouput h5py file of fri-frii-download.ipynb and returns a 
   h5py file with the exact same format, but with resized images.
   
   # Arguments
      fr_raw_data_path: File path of output of 
         fri-frii-download.ipynb
      fr_data_path: File path of output file
   """
   with h5py.File(fr_raw_data_path, 'r') as data:
     images = resize_array(np.asarray(data['images'].value), dim=dim)
     labels = data['labels'].value
     
   with h5py.File(fr_data_path, 'w') as f:
      f.create_dataset('images', data=images)

      with h5py.File(fr_raw_data_path, 'r') as data: 
         f.copy(data['fri_data'], 'fri_data')
         f.copy(data['frii_data'], 'frii_data')
         f.copy(data['labels'], 'labels')


def add_random(fr_data_path, random_path, output_path):
   """Dumps random sources and FR sources into a h5py file.
   
   Empty images and images with nans are removed from inputed 
   random radio images. The ouput h5py file has the exact 
   file structure of the inputed fr_data_path file. The remaining
   random images are concatenated with the fr images. FRI labels 
   are set to 0.0, FRII labels are set to 1.0, and random source 
   labels are set to 2.0.
   
   # Arguments
      fr_data_path: File path of H5py file that contains FR 
         sources.
      random_path: File path of H5py file that contains 
         random sources. 
      output_path: File path of output file.
   """
   with h5py.File(random_path, 'r') as data:
      random = np.asarray(data['images'].value)
    
      means = np.mean(np.mean(random, axis=-1), axis=-1)
      empty = means == 0.0
      error = np.isnan(means)
      discard = empty | error

      random_i = np.where(~discard)
      random = random[random_i]

      random = center_on_brightest(random)
    
   with h5py.File(fr_data_path, 'r') as data:
      images = np.asarray(data["images"].value) 
      images = center_on_brightest(images)
      
      labels = np.where(np.asarray(data['labels']), 2, 1)

   images = np.concatenate((images, random), axis=0)
   labels = np.concatenate((labels, np.full((random.shape[0],), 
                    fill_value=0)), axis=0)

   with h5py.File(output_path, 'w') as f:
      f.create_dataset('images', data=images)
      f.create_dataset('labels', data=labels)

      with h5py.File(fr_data_path, 'r') as data: 
         f.copy(data, 'fri_data')
         f.copy(data, 'frii_data')


def append_random(everything_path, random_path):
   """Appends new random sources to data h5py File.
   
   Adds new random sources to a h5py file that is or has
   the same format as the output file of add_random_data.
   
   # Arguments
      everything_path: File path of H5py file that is or 
         has the same format as the output of add_random_data
      random_path: File path of H5py file that contains 
   """
   with h5py.File(random_path, 'r') as data:
      random = np.asarray(data['images'].value)
    
      means = np.mean(np.mean(random, axis=-1), axis=-1)
      empty = means == 0.0
      error = np.isnan(means)
      discard = empty | error

      random_i = np.where(~discard)
      random = random[random_i]

      random = center_on_brightest(random)

   with h5py.File(everything_path, 'a') as data:
      data["images"].resize((data["images"].shape[0] + random.shape[0]), axis=0)
      data["images"][-random.shape[0]:] = random
   
      data["labels"].resize((data["labels"].shape[0] + random.shape[0]), axis=0)
      data["labels"][-random.shape[0]:] = np.full((random.shape[0],), 
                                 fill_value=0)


def get_random(everything_path, split_ratio=0.5, seed=0):
   """Get train and test indices for FR vs random classification.
   
   # Arguments
      everything_path: File path of the h5py file that contains FR 
         and random radio sources. Must be or has the same format 
         as the output file of the add_random_data function.
      split_ratio: Factor of the data used for testing. The value 
         is rounded to the nearest indice. Default is 0.5.
      seed: Seed value to consistently initialize the random number 
         generator.
      
   # Returns
      Locations training and testing data  with shape [samples,].
   """
   # Loading data
   with h5py.File(everything_path, 'r') as data:
      labels = data['labels'].value
   
   # Splitting classes
   fri_i  = np.where(labels==1)[0]
   frii_i = np.where(labels==2)[0]
   rand_i = np.where(labels==0)[0]

   # Shuffling
   fri_i = shuffle(fri_i, random_state=seed)
   frii_i = shuffle(frii_i, random_state=seed)
   rand_i = shuffle(rand_i, random_state=seed)

   # Splitting into training and testing sets
   cut = int(np.round(split_ratio * fri_i.shape[0]))
   train_fri = fri_i[cut:]
   test_fri = fri_i[:cut]

   cut = int(np.round(split_ratio * frii_i.shape[0]))
   train_frii = frii_i[cut:]
   test_frii = frii_i[:cut]

   cut = int(np.round(split_ratio * rand_i.shape[0]))
   train_rand = rand_i[cut:]
   test_rand = rand_i[:cut]

   train_i = np.concatenate((train_fri, train_frii, train_rand), axis=0)
   test_i = np.concatenate((test_fri, test_frii, test_rand), axis=0)

   return train_i, test_i


def generate_labels_fri(train_i, test_i, labels):
   """Gets train and test labels for FRI vs Random classification.
   
   # Arguments
      train_i: Training indices. Output of get_random_data function.
         Shape of [samples,].
      test_i: Testing indices. Output of get_random_data function.
         Shape of [samples,].
      labels: label array. 'labels' dataset in data h5py file as
         numpy array with shape [samples,]
      
   # Returns
      Training and testing labels, with FRI sources as True
      and FRII and Random sources as False.
   """
   train = labels[train_i]
   test = labels[test_i]

   train_y = train == 1
   test_y = test == 1

   return train_y, test_y


def generate_labels_frii(train_i, test_i, labels):
   """Gets train and test labels for FRII vs Random classification.
   
   # Arguments
      train_i: Training indices. Output of get_random_data function.
         Shape of [samples,].
      test_i: Testing indices. Output of get_random_data function.
         Shape of [samples,].
      labels: label array. 'labels' dataset in data h5py file as
         numpy array with shape [samples,]
      
   # Returns
      Training and testing labels, with FRII sources as True
      and FRI and Random sources as False.
   """
   train = labels[train_i]
   test = labels[test_i]

   train_y = train == 2
   test_y = test == 2

   return train_y, test_y

