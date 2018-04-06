
# coding: utf-8

# In[45]:


import numpy as np
import pickle
import matplotlib.pyplot as plt
import h5py
from skimage.transform import resize
from astropy.io import ascii
#import ask_first
from sklearn.utils import shuffle
from keras.utils import np_utils
from astropy.coordinates import SkyCoord
import astropy.units as u

import sys, os


# In[46]:

def get_data(first_path, second_path, dim = 256, seed = 0, conf_matrix = True, flip_train_test = False):
    # Loading data
    data = h5py.File(first_path, 'r')

    images = np.asarray(data['images'])
    labels = np.asarray(data['labels'])

    # Opening Catalogues
    asu = ascii.read(second_path).to_pandas()

    # Opening Entire First Catalogue with ask_first script (Only for reference)
    #first = ask_first.read_catalogue("catalog_14dec17.bin")


    # Splitting classes
    fri_images = images[np.where(labels == False)]
    frii_images = images[np.where(labels == True)]

    # Extracting data 
    fri_data = data['fri_data']
    frii_data = data['frii_data']

    # Converting coordinates into a Astropy Skycoord readable format
    frii_c = np.hstack((np.asarray(list(frii_data['_RA'])).reshape(-1, 1), np.asarray(list(frii_data['_DE'])).reshape(-1, 1)))
    fri_c = np.hstack((np.asarray(list(fri_data['ra'])).astype(str).reshape((-1, 1)), np.asarray(list(fri_data['dec'])).astype(str).reshape((-1, 1))))
    
    # Coverting coordinates of wildly varying formats to SkyCoords
    coord_asu = SkyCoord(ra=asu.iloc[2:]['RAJ2000'], dec=asu.iloc[2:]['DEJ2000'], unit=('hourangle', 'deg'))
    coord_fri = SkyCoord(ra=fri_c[:, 0], dec=fri_c[:, 1], unit=('hourangle', 'deg'))
    coord_frii = SkyCoord(ra=frii_c[:, 0], dec=frii_c[:, 1], unit=('deg', 'deg'))
    #coord_first = SkyCoord(ra=first.iloc[ind_first]['RA'], dec=first.iloc[ind_first]['Dec'], unit=('hourangle', 'deg'))
    
    #first_plot = SkyCoord(ra=first['RA'], dec=first['Dec'], unit=("hourangle", "deg"))

    #plt.scatter(first_plot.ra.deg, first_plot.dec.deg, label="first")
    #plt.scatter(coord_asu.ra.deg, coord_asu.dec.deg, label="asu", alpha=0.5)
    #plt.scatter(coord_fri.ra.deg, coord_fri.dec.deg, label="fri")
    #plt.scatter(coord_frii.ra.deg, coord_frii.dec.deg, label="frii", alpha=0.5)
    #plt.legend()
    #plt.show()

    # Finding nearest FRI's and FRIIs to each object in ASU
    idx_i, sep_i, _i = coord_fri.match_to_catalog_sky(coord_asu)
    idx_ii, sep_ii, _ii = coord_frii.match_to_catalog_sky(coord_asu)

    # For sources within our set threshold, we assume that they are the same object.
    fri_in_asu = (sep_i <= 5*u.arcsec)
    frii_in_asu = (sep_ii <= 5*u.arcsec)

    # Extacting FR images that are in asu
    asu_fri = fri_images[fri_in_asu]
    asu_frii = frii_images[frii_in_asu]

    leftover_fri = fri_images[np.invert(fri_in_asu)]
    leftover_frii = frii_images[np.invert(frii_in_asu)]

    #Generating Labels
    asu_fri_lab = np.zeros(asu_fri.shape[0], dtype=bool)
    asu_frii_lab = np.ones(asu_frii.shape[0], dtype=bool)

    leftover_fri_lab = np.zeros(leftover_fri.shape[0], dtype=bool)
    leftover_frii_lab = np.zeros(leftover_frii.shape[0], dtype=bool)

    # Combining FRI's and FRII's and shuffling
    asu_fr = np.vstack((asu_fri, asu_frii))
    asu_fr_lab = np.concatenate((asu_fri_lab, asu_frii_lab), axis=0) 

    leftover_fr = np.vstack((leftover_fri, leftover_frii))
    leftover_fr_lab = np.concatenate((leftover_fri_lab, leftover_frii_lab), axis=0)             

    #Shuffling
    asu_fr_, asu_fr_lab_ = shuffle(asu_fr, asu_fr_lab, random_state = seed)  
    leftover_fr, leftover_fr_lab = shuffle(leftover_fr, leftover_fr_lab, random_state = seed) 

    # Setting training and testing sets
    if flip_train_test == False:
        train_x = asu_fr_
        train_y = asu_fr_lab_

        test_x = leftover_fr
        test_y = leftover_fr_lab


    elif flip_train_test == True:
        train_x = leftover_fr
        train_y = leftover_fr_lab

        test_x = asu_fr_
        test_y = asu_fr_lab_

    
    # Defining rescaling function
    def rescale_imgs(images,dim=dim):
        size = images.shape[0]
        imgs = np.zeros((size, dim, dim))

        for i in range(size):
            imgs[i, :, :] = resize(images[i, :, :], (dim, dim))

        return np.expand_dims(imgs, axis=3)

    
    

    #Rescaling Images
    sys.stdout = open(os.devnull, "w") # don't show printed output
    
    train_x = rescale_imgs(train_x, dim)
    test_x = rescale_imgs(test_x, dim)

    sys.stdout = sys.__stdout__ # show printed output

    # Convert class vectors to binary class matrices (for Keras Dense)
    if conf_matrix == True:
        train_y = np_utils.to_categorical(train_y, 2)
        test_y = np_utils.to_categorical(test_y, 2)

        
    return train_x, train_y, test_x, test_y    


import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# Adding tiny Gaussian Noise to ensure there are no vanishing and exploding gradients
def add_noise(image):
        image += (10e-10)*np.random.randn(image.shape[0], image.shape[1], 1)
        return image

    
def data_gen(rotation_range=180, zoom_range=0.2, shift_range = 0.0, flip=True):
    
    # Defining Data Generator
    datagen = ImageDataGenerator(
                rotation_range = rotation_range,
                zoom_range = zoom_range,
                width_shift_range = shift_range,
                height_shift_range = shift_range,
                horizontal_flip = flip,
                vertical_flip = flip,
                fill_mode = "nearest",
                preprocessing_function=add_noise)
    
    return datagen


def data_pregenerate(images, labels, datagen, batch_size, data_mult, seed):
    samples = images.shape[0]

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    gen = datagen.flow(images, labels, batch_size=batch_size, seed=seed)

    pro_images = np.zeros((images.shape[0]*data_mult, images.shape[1], images.shape[2], 1))
    pro_labels = np.zeros((labels.shape[0]*data_mult, 2))
    
    for e in range(1, data_mult+1):
        batch = 1
        b = batch_size
        b_start = samples*(e-1)

        for X_batch, Y_batch in gen:
            if batch < (samples/b):
                pro_images[b_start+b*(batch-1):b_start+batch*b, :, :, :] = X_batch
                pro_labels[b_start+b*(batch-1):b_start+batch*b, :] = Y_batch
                
            elif batch == (samples/b):
                break

            else: 
                pro_images[b_start+b*(batch-1):b_start+b*(batch-1) + X_batch.shape[0]%b, :, :, :] = X_batch
                pro_labels[b_start+b*(batch-1):b_start+b*(batch-1) + X_batch.shape[0]%b, :] = Y_batch
                break

            batch += 1
        
    return pro_images, pro_labels


def data_pregen(train_x, train_y, test_x, test_y, datagen, batch_size, nb_epoch, seed):

    train_x, train_y = data_pregenerate(images=train_x, labels=train_y, datagen=datagen, batch_size=batch_size, data_mult = nb_epoch, seed=seed)
    test_x, test_y  = data_pregenerate(images=test_x, labels=test_y, datagen=datagen, batch_size=batch_size, data_mult = nb_epoch, seed=seed)
    
    return train_x, train_y, test_x, test_y