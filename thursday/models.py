"""Creates various Sklearn models"""

import math
import pickle

import keras 
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import History
from keras.callbacks import ReduceLROnPlateau
from keras.engine.topology import Layer
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import multiply
from keras.models import load_model
from keras.models import Model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
import tensorflow as tf

K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last')

class SklearnModel:
    """Wrapper for sklearn classifiers.

    This creates a wrapper that can be instantiated to any 
    sklearn classifer that takes input data with shape 
    (samples, features) and label data with shape (samples,). 
    Using it's train method can generate a trained model, and
    load the same trained model using the load method at any 
    point on the script. 
    """   
    def __init__(self, Model, datagen=None, 
                              nb_augment=None, 
                              seed=0, **kwargs):
        """Attributes:
            Model: Sklearn classifier.
            datagen: The output of the data_gen function to apply 
                random augmentations to our data in real time 
                (a keras.preprocessing.image.ImageDataGenerator 
                object)
            nb_augment: Int. Factor by which the number of samples 
                is increased by random augmentations.
            seed: Seed value to consistently initialize the random 
                number generator.
            **kwargs: Keyword arguments passed to Sklearn. 
        """
        self.Model = Model
        self.datagen = datagen
        self.nb_augment = nb_augment
        self.seed = seed
        self.kwargs = kwargs

        self.model = None
        self.path = None

        self.name = Model.__name__

    def fit(self, train_x, train_y):
        """Trains sklearn model.

        # Arguments:
            train_x: Array of images with shape [samples, dim, dim, 
                1].
            train_y: Array of labels with shape [samples ,].

        # Returns:
            self
        """ 
        # Shuffling
        train_x, train_y = shuffle(train_x, train_y, 
                                   random_state=self.seed)

        # Augmenting data 
        if self.datagen is not None:
            train_x, train_y = self.augment(
                                    train_x, 
                                    train_y, 
                                    batch_size=32, 
                                    nb_augment=self.nb_augment)

        # Shuffling
        train_x, train_y = shuffle(train_x, train_y, 
                                   random_state=self.seed)

        # Flattening images
        train_x = train_x.reshape((np.shape(train_x)[0], -1))
        try:
            model = self.Model(random_state=self.seed, 
                                class_weight='balanced',
                                **self.kwargs)
        except TypeError:
            try:
                model = self.Model(class_weight='balanced', 
                                   **self.kwargs)
            except TypeError:
                model = self.Model(**self.kwargs)

        model = model.fit(train_x, train_y)

        self.model = model

        return self

    def predict_proba(self, test_x):
        """ Probability estimates for samples.

        # Arguments:
            test_x: Array of images with shape [samples, dim, 
            dim, 1].

        # Returns:
            predictions: Probability estimates for test_x.
        """
        # Flattening images
        test_x = test_x.reshape((test_x.shape[0], -1))

        predictions = self.model.predict_proba(test_x)

        return predictions

    def predict(self, test_x):
        """Predicting class labels for samples in test_x.
        
        # Arguments:
            test_x: Array of images with shape [samples, 
                dim, dim, 1].

        # Returns:
            predictions: Class predictions for test_x.

        """
        pred = self.predict_proba(test_x)

        if pred.shape[1] == 2:
            pred = pred.max(axis=1)

        pred = np.around(pred).astype('bool')

        return pred      

    def score(self, test_x, test_y):

        """Computes the balanced accuracy of a predictor.
        # Arguments:
            test_x: Array of images with shape [samples, dim, 
            dim, 1].
            test_y: (n_examples,) (masked) array of predicted 
                labels.
        # Returns:
            Balanced accuracy.
        """
        pred = self.predict(test_x)

        
        cm = sklearn.metrics.confusion_matrix(test_y, pred)
        cm = cm.astype(float)

        tp = cm[1, 1]
        n, p = cm.sum(axis=1)
        tn = cm[0, 0]
        if not n or not p:
            return None
    
        ba = (tp / p + tn / n) / 2
        
        return ba  

    def save(self, path=None):
        """Saves model as pickle file.
        # Arguments:
            path: File path to save the model. Must be a 
                .pk file.

        # Returns:
            self
        """

        if path is None:
            path = self.name + '.pk'

        self.path = path
        
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

        return self
    
    def load(self, path=None):    
        """Loads trained Sklearn Model from disk.

        # Arguments:
            path: File path load saved the model from..

        # Returns:
            self
        """
        if path is None:
            if self.path is not None:
                path =  self.path
            else:
                print ("no model can be found")

        with open(path, 'rb') as f:
            model = pickle.load(f)

        self.model = model

        return self

    def augment(self, images, labels, datagen=None, 
                                      batch_size=32, 
                                      nb_augment=None):
        """Augments data for Sklearn models.
        
        Using base set of images, a random combination is 
        augmentations within a defined range is applied to 
        generate a new batch of data.
        
        # Arguments
            images: Array of images with shape 
                (samples, dim, dim, 1)
            labels: Array of labels
            datagen: The data generator outputed by the 
                data_gen function.
            batch_size: Number of sample per batch.
            nb_augment: factor that the data is increased by 
                via augmentation
            seed: Seed value to consistently initialize the 
                random number generator.
            
        # Returns
            Array of augmented images and their corresponding 
            labels.
        """
        if nb_augment is None:
            nb_augment = self.nb_augment

        if datagen is None:
            datagen = self.datagen

        # Number of images
        samples = np.shape(images)[0]
        
        # .flow() generates batches of randomly augmented images
        gen = datagen.flow(images, labels, batch_size=batch_size, 
                                           shuffle=True, 
                                           seed=self.seed)

        # Generate empty data arrays
        x = np.zeros((images.shape[0] * nb_augment, 
                               images.shape[1],
                               images.shape[2], 
                               1))
        y = np.zeros((labels.shape[0] * nb_augment))
        
        for step in range(1, nb_augment+1):
            batch = 1
            b = batch_size
            b_start = samples * (step-1)

            for X_batch, Y_batch in gen:
                if batch < (samples / b):
                    cut_start = b_start + b * (batch-1)
                    cut_stop = b_start + batch * b
                    
                    x[cut_start:cut_stop, :, :, :] = X_batch
                    y[cut_start:cut_stop] = Y_batch
                    
                elif batch == int(samples / b):
                    break

                else:
                    odd = X_batch.shape[0]
                    cut_start = b_start + b * (batch-1)
                    cut_stop = b_start + b * (batch-1) + odd % b
                    
                    x[cut_start:cut_stop, :, :, :] = X_batch
                    y[cut_start:cut_stop] = Y_batch
                    break

                batch += 1
        
        return x, y

    
class HOGNet:
    """Wrapper for our hognet keras model.

    This creates a class that acts as a wrapper around 
    our custom keras model, aka hognet. The train method
    trains the model on our data generator to our specifed 
    training paramerts. Using the load method, we can load
    the fully trained model from the disk at any point in 
    the script. This method can be useful when using 
    notebooks, as training time can be significant.
    """     
    def __init__(self, datagen=None, batch_size=32, 
                                     steps_per_epoch=50, 
                                     max_epoch=100, 
                                     patience=5, 
                                     gap=2, 
                                     seed=None):
        """Attributes:
            datagen: The output of the data_gen function 
                to apply random augmentations to our data 
                in real time (a keras.preprocessing.image.
                ImageDataGenerator object)
            batch_size: number of images per batch (i.e 
                number of images generated per batch)
            steps_per_epoch: number of batchs per step (i.e
                number of batches generated by datagen per 
                step)
            max_epoch: maximum number of epochs the model for. 
                The model should stop training automatically 
                when the loss stops decreasing.
            patience: number of epochs with no improvement 
                after which training will be stopped.
            gap: Number of layers that have their weights 
                unfrozen per training cycle.
            seed: Seed value to consistently initialize the 
                random number generator.
        """
        self.datagen = datagen
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.max_epoch = max_epoch 
        self.patience = patience
        self.gap = gap
        self.seed = seed

        self.model = None
        self.history = None
        self.class_weight = None

        self.prewitt_x = None
        self.prewitt_y = None
        self.cent = None

        self.name = "HOGNet"

        # Setting random number generator seeds.
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # Model HyperParameters
        # Following values are base on past research and much testing
        bins = 8       # number of bins in histogram
        cell_dim = 8   # height and width of the cells  
        block_dim = 2  # if changed, must add more block layers. Don't attempt. 
        bs = bin_stride_length =  1 

        # Number of cells along each dim 
        cell_nb = 256 // cell_dim
        assert not 256 % cell_dim
        
        # Defining Values
        w = 2*np.pi/bins     # width of each bin
        # centers of each bin
        centers = np.arange(-np.pi, np.pi, w) + 0.5 * w  
        
        # Weights for x and y Conv2D's to calculate image gradients
        prewitt_x = np.array([[-1, 0, 1], 
                              [-1, 0, 1], 
                              [-1, 0, 1]])
        prewitt_y = np.array([[-1,-1,-1], 
                              [ 0, 0, 0], 
                              [ 1, 1, 1]])
        
        # Reshaping Prewitt opperators to required shape
        prewitt_x = prewitt_x.reshape((1, 3, 3, 1, 1))
        prewitt_y = prewitt_y.reshape((1, 3, 3, 1, 1))
        self.prewitt_x = prewitt_x.astype('float64')
        self.prewitt_y = prewitt_y.astype('float64') 
        # Adding tiny gaussian noise
        self.prewitt_x += 0.01 * np.random.randn(1, 3, 3, 1, 1)
        self.prewitt_y += 0.01 * np.random.randn(1, 3, 3, 1, 1)

        # Generating weights for histogram construction
        self.cent = np.vstack((np.sin(centers), np.cos(centers)))
        self.cent = self.cent.reshape((1, 1, 1, 2, bins))

        # Generating Filters for the block Operations
        def create_block_filters(block_dim):
            filters = np.zeros((block_dim ** 2, 
                                block_dim, 
                                block_dim))
            count = 0 
            for i in range(block_dim):
                for j in range(block_dim):
                    filters[count, i, j] = 1
                    count += 1
            return filters   

        # block_dim must be 2
        # Increasing this will require adding:
        # tf.nn.depthwise_conv2d functions. 
        # There is a depthwise_conv2d for each element in a single filter 
        # there are block_dim^2 elements in a filter
        block_filters = create_block_filters(block_dim)
    
        # Reshaping to satisfy required shape for weight array
        # copying each filter along the last axis 
        # Must have shape:
        #[filter_height, filter_width, in_channels, channel_multiplier] 
        # (see Tensorflow docs for tf.nn.depthwise_conv2d)
        b_shp = block_filters.shape
        block_filters = block_filters.reshape((b_shp[0], 
                                               b_shp[1], 
                                               b_shp[2], 
                                               1, 1))
        block_filters = block_filters.repeat(bins, axis=3)
        
        # Converting filters to tensors
        filt1 = tf.convert_to_tensor(block_filters[0, :, :, :, :])
        filt2 = tf.convert_to_tensor(block_filters[1, :, :, :, :])
        filt3 = tf.convert_to_tensor(block_filters[2, :, :, :, :])
        filt4 = tf.convert_to_tensor(block_filters[3, :, :, :, :])
        filt1 = tf.cast(filt1, dtype=tf.float32)
        filt2 = tf.cast(filt2, dtype=tf.float32)
        filt3 = tf.cast(filt3, dtype=tf.float32)
        filt4 = tf.cast(filt4, dtype=tf.float32)

        def calculate_magnitudes(conv_stacked):
            mags = tf.norm((conv_stacked), axis=3)
            return mags

        def calculate_angles(conv_stacked):
            angles = tf.atan2(conv_stacked[:, :, :, 1], 
                              conv_stacked[:, :, :, 0])
            return angles

        def calculate_sin_cos(angles):
            sin = K.sin(angles)
            cos = K.cos(angles)
            sin_cos = K.stack([sin, cos], axis =-1)
            return sin_cos

        def block1(cells):
            c_blocks = tf.nn.depthwise_conv2d(cells, filt1, 
                                    strides=(1, bs, bs, 1),
                                    padding="SAME")
            return c_blocks

        def block2(cells):
            c_blocks = tf.nn.depthwise_conv2d(cells, filt2, 
                                    strides=(1, bs, bs, 1),
                                    padding="SAME")
            return c_blocks 

        def block3(cells):
            c_blocks = tf.nn.depthwise_conv2d(cells, filt3, 
                                    strides=(1, bs, bs, 1),
                                    padding="SAME")
            return c_blocks

        def block4(cells):
            c_blocks = tf.nn.depthwise_conv2d(cells, filt4, 
                                    strides=(1, bs, bs, 1),
                                    padding="SAME")
            return c_blocks

        def block_norm_function(block_layer):
            c = 0.00000001
            divisor = tf.expand_dims(tf.sqrt(tf.norm(
                                     block_layer, 
                                     axis=-1) + c), 
                                     axis=-1)
            block_norm = tf.div(block_layer, divisor)
            return block_norm

        def hog_norm_function(block_layer):
            c = 0.00000001
            divisor = tf.expand_dims(tf.sqrt(tf.norm(
                                    block_layer, 
                                    axis=-1) + c), 
                                    axis=-1)     
            hog_norms = tf.div(block_layer, divisor)
            
            divisor = tf.expand_dims(tf.sqrt(tf.norm(
                                    hog_norms, 
                                    axis=-1) + c),
                                    axis=-1)
            hog_norm = tf.div(hog_norms, divisor)
            return hog_norm

        # Building Model 
        inputs = Input(shape=(256, 256, 1))

        # Convolutions
        x_conv = Conv2D(1, (3,3), strides=(1, 1), 
                                  padding="same", 
                                  data_format="channels_last", 
                                  trainable=True, 
                                  use_bias=False)(inputs)

        y_conv = Conv2D(1, (3,3), strides=(1, 1), 
                                  padding="same", 
                                  data_format="channels_last", 
                                  trainable=True, 
                                  use_bias=False)(inputs)

        # Stacking: Can't have multiple inputs into a single layer 
        # i.e mags and angles
        conv_stacked = Concatenate(axis=-1)(
                                  [x_conv, y_conv])

        # Calculating the gradient magnitudes and angles
        mags = Lambda(calculate_magnitudes, 
                                output_shape=(256, 256))(
                                conv_stacked)
        mags = Lambda(lambda x: K.stack((mags, mags), 
                        axis=-1))(mags)  # To enable sin_cos_vec 
        angles = Lambda(calculate_angles, 
                        output_shape=(256, 256))(conv_stacked)

        # Calculating the components of angles in the x and y direction
        # Then multiplying by magnitudes, giving angle vectors
        sin_cos = Lambda(calculate_sin_cos, 
                                output_shape=(256, 256, 2), 
                                name="sin_cos")(angles)
        sin_cos_vec = multiply([sin_cos, mags], 
                                name="sin_cos_mag")
        # Applying each filter to each angle vector.
        # Each filter: a single bin unit vector
        # Result is array with shape (img_height, img_width, bins) 
        # Each bin contains an angle vector that contributes to each bin
        # Relu activation function to remove negative projections.
        votes = Conv2D(8, kernel_size=(1, 1), 
                          strides=(1, 1), 
                          activation="relu", 
                          trainable=True, 
                          bias=False, 
                          name="votes")(sin_cos_vec) 

        # A round about way of splitting the image (i.e vote array) 
        # into a bunch of non-overlapping cells of size 
        # (cell_dim, cell_dim)
        # Concatenating values at each bin level.
        # Gives shape (cell_nb, cell_nb, bins)
        # Result: array of cells with histograms along last axis
        cells = AveragePooling2D(pool_size=cell_dim, 
                                 strides=cell_dim)(votes)
        cells = Lambda(lambda x: x * (cell_dim ** 2))(cells)

        # Bin Operations
        # Assuming that bin shape = (2, 2)
        # Ad hoc way of  grouping the cells...
        # ...into overlapping blocks of 2 * 2 cells each. 
        # Two h or w consecutive blocks overlap by two cells,
        # i.e block strides. 
        # As a consequence, each internal cell is covered by four blocks 
        # if bin_dim=2
        block1_layer = Lambda(block1, trainable=True)(cells)
        block2_layer = Lambda(block2, trainable=True)(cells)
        block3_layer = Lambda(block3, trainable=True)(cells)
        block4_layer = Lambda(block4, trainable=True)(cells)

        block_layer = Concatenate(axis=-1)([block1_layer, 
                                            block2_layer, 
                                            block3_layer, 
                                            block4_layer])

        # normalize each block feature by its Euclidean norm
        block_norm = Lambda(block_norm_function)(block_layer)
        hog_norm = Lambda(hog_norm_function)(block_norm)

        # Block 1
        x = Conv2D(4, (3, 3), padding='same')(hog_norm)
        x = Activation('relu')(x)
        x = Conv2D(4, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(4, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        # Block 2
        x = Conv2D(8, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(8, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(8, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        # Dense Block
        x = Flatten()(x) 
        x = Dense(50, activation='relu')(x) 
        x = Dropout(0.2)(x) 
        x = Dense(50, activation='relu')(x) 
        x = Dropout(0.2)(x) 
        x = Dense(50, activation='relu')(x) 
        x = Dropout(0.2)(x) 

        output = Dense(1, activation = "sigmoid")(x) 

        # Building Model
        model = Model(inputs=inputs, outputs=output)

        self.model = model

    def fit(self, train_x, train_y, val_x=None, val_y=None):
        """Fits HOGNet parameters to training data.
    
        Using the set hyperparameters, our custom initial 
        weights are generated, our custom keras layers are 
        defined in tensorflow, and our model is constructed
        in keras. The resulting model is trained either 
        with or without validation data. The data is 
        generated in real time using the prefined datagen 
        function. The learning rate is reduced when the loss 
        plateaus. When the model stops improving for a set
        number of epochs (patience), training stops, and a  
        model is returned. 
        
        # Arguments
            train_x: Images to train on with shape [samples, 
                dim, dim, 1]
            train_y: Image labels to train on as a confusion 
                matrix with shape [samples, 2] 
            val_x: Images to validate with. By default, no 
                validation data is used 
            val_y: Labels to validate with. By default, no 
                validation data is used. 

        # Returns
            An updated state where self.model is a trained 
            keras model. 
        """
        # Setting random number generator seeds.
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # Setting weights
        self.model.layers[1].set_weights(self.prewitt_x)
        self.model.layers[2].set_weights(self.prewitt_y)
        self.model.layers[9].set_weights(self.cent)
        
        # Checking for validation data
        if val_x is None and val_y is None:
            validation = False
        else:
            validation = True
        
        if self.datagen is None:
            print ("no data generator has been inputed")
            raise 

        # Shuffling
        train_x, train_y = shuffle(train_x, 
                                   train_y, 
                                   random_state=self.seed)
           
        # Setting Class weights
        self.class_weight = compute_class_weight('balanced',
                                               np.unique(train_y),
                                               train_y)
        # Constructing class for callbacks
        class LossHistory(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.loss = []
                self.acc = []

                if validation is True:
                    self.val_loss = []
                    self.val_acc = []

            def on_batch_end(self, batch, logs={}):
                self.loss.append(logs.get('loss'))
                self.acc.append(logs.get('acc'))

                if validation is True:
                    self.val_loss.append(logs.get('val_loss'))
                    self.val_acc.append(logs.get('val_acc'))        
        

        # Freazing Layers 20 to 27 (the Conv2D blocks)
        start = 20
        stop = 28

        i = 0
        for layer in self.model.layers:
            if i < start:
                continue
            elif i >= stop:
                continue
            else:
                layer.trainable = False
            i = i+1
        
        # Compiling Model
        self.model.compile(optimizer='adam', 
                           loss='binary_crossentropy', 
                           metrics=["accuracy"])

        for layer_block in range(start, stop-self.gap, self.gap):
            if validation is True:
                callback = [EarlyStopping(
                                monitor='val_loss', 
                                patience=self.patience),
                            ReduceLROnPlateau(
                                monitor='val_loss', 
                                patience=5, 
                                verbose=1),
                            History()]
                
                self.model.fit_generator(
                        self.datagen.flow(train_x, train_y, 
                            batch_size=self.batch_size, 
                            shuffle=True), 
                        steps_per_epoch=self.steps_per_epoch, 
                        epochs=self.max_epoch,
                        validation_data=self.datagen.flow(val_x, val_y, 
                            batch_size=self.batch_size, 
                            shuffle=True),
                        validation_steps=math.ceil(self.steps_per_epoch / 5), 
                        callbacks=callback,
                        class_weight=self.class_weight)
    
            else:
                callback = [EarlyStopping(
                                monitor='loss', 
                                patience=self.patience),
                            ReduceLROnPlateau(
                                monitor='loss', 
                                patience=5, 
                                verbose=1),
                            History()]
                self.model.fit_generator(
                        self.datagen.flow(train_x, train_y, 
                            batch_size=self.batch_size, 
                            shuffle=True), 
                        steps_per_epoch=self.steps_per_epoch, 
                        epochs=self.max_epoch, 
                        callbacks=callback,
                        class_weight=self.class_weight)
    
            if self.history is None:
                self.history = callback[2].history
    
            else:
                for metric in callback[2].history:
                    sub_metric = callback[2].history[metric]
                    self.history[metric].append(sub_metric)
    
                
            i = 0
            for layer in self.model.layers:
                if i < layer_block:
                    continue

                elif i >= (layer_block + self.gap):
                    continue

                else:
                    layer.trainable = True
                i += 1
                
        return self

    def predict_proba(self, test_x):
        """ Probability estimates for samples.

        # Arguments:
            test_x: Array of images with shape [samples, dim, 
                dim, 1].

        # Returns:
            predictions: Probability estimates for test_x as a 
                confusion matrix. Shape of [samples, 2].
        """
        predictions =  self.model.predict(
                                    test_x, 
                                    batch_size=
                                    self.batch_size)

        return predictions

    def predict(self, test_x):
        """Predicting class labels for samples in test_x.
        
        # Arguments:
            test_x: Array of images with shape [samples, dim, 
                dim, 1].

        # Returns:
            predictions: Probability estimates for test_x as 
                a confusion matrix. Shape of [smaples, 2].
        """
        predictions =  self.predict_proba(test_x)
        predictions = np.around(predictions).astype('bool')

        return predictions

    def score(self, test_x, test_y):

        """Computes the balanced accuracy of a predictor.
        # Arguments:
            test_x: Array of images with shape [samples, dim, 
            dim, 1].
            test_y: (n_examples,) (masked) array of predicted 
                labels.
        # Returns:
            Balanced accuracy.
        """
        predictions = self.predict(test_x)
        
        cm = sklearn.metrics.confusion_matrix(test_y, predictions)
        cm = cm.astype(float)

        tp = cm[1, 1]
        n, p = cm.sum(axis=1)
        tn = cm[0, 0]
        if not n or not p:
            return None
    
        ba = (tp / p + tn / n) / 2
        
        return ba
        
    def save(self, path=None):
        """Serialize model weights to HDF5.

        # Arguments:
            path: File path to save the model weights. Must be 
                a .h5 file.

        # Returns:
            self
        """
        if path is None:
            path = self.name + '.h5'

        self.model.save_weights(path)

        self.path = path

        return self
     
    def load(self, path=None):
        """Loads weights into keras Model from disk.

        Loads trained Sklearn Model from disk.

        # Arguments:
            path: File path to load saved weights from from.

        # Returns:
            self
        """
        if path is None:
            path = self.path
        
        # load weights into new model
        self.model.load_weights(path)
        
        return self

