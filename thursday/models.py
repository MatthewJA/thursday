"""Creates various Sklearn models"""

import math
import pickle

import keras 
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import History
from keras.callbacks import ReduceLROnPlateau
from keras.engine.topology import Layer
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
from sklearn.svm import LinearSVC
import tensorflow as tf

K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last')


class SklearnModel:
    """Wrapper for sklearn classifiers.

    This creates a wrapper that can be instantiated to any sklearn
    classifer that takes input data with shape (samples, features)
    and label data with shape (samples,). Using it's train method 
    can generate a trained model, and load the same trained model 
    using the load method at any point on the script. 

    Attributes:
    Model: Sklearn classifier.
    name: name of the file at which the model is saved. 
        No file extension.
    path: Path of data folder. Dfaults to current directory
    seed: Random seed. All our tests used 0.
    kwargs: All other parameters specific to Model type. 
    """   
    def __init__(self, Model, datagen=None, nb_augment=None, seed=0, **kwargs):
        self.Model = Model
        self.datagen = datagen
        self.nb_augment = nb_augment
        self.seed = seed
        self.kwargs = kwargs

        self.model = None
        self.path = None

    def fit(self, train_x, train_y):
        """Trains sklearn model.

        # Arguments
            images: Array of images with shape (samples, dim, dim, 1).
            labels: Array of labels with shape (samples ,).

        # Returns
            Updated state where self.model is a fully trained sklearn classifer
        """ 
        # Augmenting data 
        if self.datagen is not None:
            train_x, train_y = self.augment(train_x, train_y, batch_size=train_x.shape[0], nb_augment=self.nb_augment)

        # Flattening images
        train_x = np.reshape(train_x, (np.shape(train_x)[0], -1))

        model = self.Model(random_state=self.seed, **self.kwargs)
        model = model.fit(train_x, train_y)

        self.model = model

        return self


    def predict_proba(self, test_x):
        """ Probability estimates for samples"""
        # Flattening images
        test_x = np.reshape(test_x, (np.shape(test_x)[0], -1))

        predictions = self.model.predict(test_x)

        return predictions

    def predict(self, test_x):
        """Predicting class labels for samples in test_x."""
        predictions = self.predict_proba(test_x)
        predictions = np.around(predictions)

        return predictions

    def save(self, path=None):
        """Saves model as pickle file"""
        self.path = path
        
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

        return self
    
    def load(self, path=None):    
        """Loads trained Sklearn Model from disk"""
        if path is None:
            if self.path is not None:
                path =  self.path
            else:
                print ("no model can be found")

        with open(path, 'rb') as f:
            model = pickle.load(f)

        self.model = model

        return self

    def augment(self, images, labels, datagen=None, batch_size=32, nb_augment=None):
        """Augments data for Sklearn models.
        
        Using base set of images, a random combination is augmentations within
        a defined range is applied to generate a new batch of data.
        
        # Arguments
            images: Array of images with shape (samples, dim, dim, 1)
            labels: Array of labels
            datagen: The data generator outputed by the data_gen function.
            batch_size: Number of sample per batch.
            nb_augment: factor that the data is increased by via augmentation
            seed: Random seed. All our tests used 0.
            
        # Returns
            Array of augmented images and their corresponding labels.
        """
        if nb_augment is None:
            nb_augment = self.nb_augment

        if datagen is None:
            datagen = self.datagen

        # Number of images
        samples = np.shape(images)[0]
        
        # the .flow() command below generates batches of randomly transformed images
        gen = datagen.flow(images, labels, batch_size=batch_size, seed=self.seed)

        # Generate empty data arrays
        pro_images = np.zeros((images.shape[0] * nb_augment, images.shape[1],
                               images.shape[2], 1))
        pro_labels = np.zeros((labels.shape[0] * nb_augment))
        
        for epoch in range(1, nb_augment+1):
            batch = 1
            b = batch_size
            b_start = samples * (epoch-1)

            for X_batch, Y_batch in gen:
                if batch < (samples / b):
                    cut_start = b_start + b * (batch-1)
                    cut_stop = b_start + batch * b
                    
                    pro_images[cut_start:cut_stop, :, :, :] = X_batch
                    pro_labels[cut_start:cut_stop] = Y_batch
                    
                elif batch == samples // b:
                    break

                else:
                    cut_start = b_start + b * (batch-1)
                    cut_stop = b_start + b * (batch-1) + X_batch.shape[0] % b
                    
                    pro_images[cut_start:cut_stop, :, :, :] = X_batch
                    pro_labels[cut_start:cut_stop] = Y_batch
                    break

                batch += 1
        
        return pro_images, pro_labels

    
class HOGNet:
    """Wrapper for our hognet keras model.

    This creates a class that acts as a wrapper around our custom 
    keras model, aka hognet. The train method trains the model on our 
    data generator to our specifed training paramerts.
    Using the load method, we can load the fully trained model from the
    disk at any point in the script. This method can be useful when using 
    notebooks, as training time can be significant.


    We introduce a tensor-based implementation of the HOG feature 
    extractor, built in Keras and Tensorflow. It is, to our knowledge,
    the first of its kind. To restrict all operations to the tensor
    domain, we approximate the previously employed histogram bin
    construction methodology, known as voting via bi-linear interpolation,
    as the scalar projection of angle vectors onto bin unit vector. Since
    this is defined as a simple dot product between the two aforementioned
    vectors, we can implement this as a Convolution2D operation, with the
    angle vectors as the input and the bin unit vectors as the filters, 
    with each filter representing a single unit vector that operates on
    every angle vector in the image. A Relu activation function operates
    on the output to curb the influence of negative projections. The 
    reshaping required for the bin operations broken into separate 
    depthwise convolution2D operations (via Tensorflow) followed via a 
    Concatenation.

    As both Tensorflow allows the automatic computation of gradients, 
    unlike the SkImage HOG, our tensor-based HOG enables backpropagation
    through HOG. This opens up an array of previously unexplored 
    possibilities, such as allowing our initial weights to train, boosting
    accuracy by roughly 2% with a logistic regression. 
    
    The primary benefit of this classier is that it allows us to utilize
    the power of state of the art convolutional neural networks like 
    VGG19 on an extremely small set of data (the convolutional blocks 
    of this implementation are a vastly simpler version of our highest 
    performing model, which is heavily inspired by VGG19). Our 
    tensor-based HOG feature extractor simultaneously reduces 
    dimensionality of the vertical and horizontal axes (by 8 each in
    our tests) and while drastically increasing the dimensionality on
    the third axis (by a factor of 32 in our tests, with 8 bins and a
    block area of 2^2) by way of unique feature maps. Each of these 
    feature maps, representing intensity gradients in varying angle 
    ranges in a different normalization region for each block, can be
    thought of as the output of a different filter from a Cov2D layer
    followed by a max-pooling layer (or more accurately, an 
    average-pooling layer). The reduced dimensionality of the first two 
    axes allows us to use VGG19. Repeated Max-pooling operations reduce
    the output of the final convolution block, reducing the number of weights 
    in the Dense classification block to a point where training on less 
    than 70 samples per class is feasible. 
    
    All classifiers we tested (which granted was not very many)
    trained up 5-100 times faster (if at all) on the output array of the 
    tensor-based HOG than raw images. Our guess is that our tensor-based
    HOG drastically reduces problem complexity, and thus the time it 
    takes for the optimizer to converge on the weights. It also slighly 
    increased accuracy.

    Attributes:
    name: name of the file at which the model is saved. 
        No file extension.
    datagen: The output of the data_gen function to apply random 
        augmentations to our data in real time 
        (a keras.preprocessing.image.ImageDataGenerator object)
    batch_size: number of images per batch (i.e number of images 
                generated per batch)
    steps_per_epoch: number of batchs per epoch (i.e number of 
        batches generated by datagen per epoch)
    max_epoch: maximum number of epochs the model for. The model 
        should stop training automatically when the loss stops 
        decreasing.
    patience: number of epochs with no improvement after which 
        training will be stopped.
    seed: Random seed. All our tests used 0.
    """     
    def __init__(self, datagen=None, batch_size=32, steps_per_epoch=50, max_epoch=100, patience=5, seed=None):
        self.datagen = datagen
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.max_epoch = max_epoch 
        self.patience = patience
        self.seed = seed

        self.model = None
        self.history = None

        self.prewitt_x = None
        self.prewitt_y = None
        self.cent = None

        # Setting random number generator seeds for numpy and tensorflow
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # Model HyperParameters
        # The following values were chosen based on both past research and thorough testing
        bins = 8         # number of bins in histogram
        cell_dim = 8     # height and width of the cells  
        block_dim = 2    # if changed, must add more block layers. Not recommended.
        bs = bin_stride_length =  1 

        # Number of cells along each dim 
        cell_nb = 256 // cell_dim
        assert not 256 % cell_dim
        
        # Defining Values
        w = 2*np.pi/bins     # width of each bin
        centers = np.arange(-np.pi, np.pi, w) + 0.5 * w   # centers of each bin
        
        # Weights for the x and y convolutions to calculate image gradients
        self.prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).reshape((1, 3, 3, 1, 1)) + 0.01 * np.random.randn(1, 3, 3, 1, 1)
        self.prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).reshape((1, 3, 3, 1, 1)) + 0.01 * np.random.randn(1, 3, 3, 1, 1)

        self.cent = np.vstack((np.sin(centers), np.cos(centers))).reshape((1, 1, 1, 2, 8))

        # Generating Filters for the block Operations
        def create_block_filters(block_dim):
            filters = np.zeros((block_dim ** 2, block_dim, block_dim))

            count = 0 
            for i in range(block_dim):
                for j in range(block_dim):
                    filters[count, i, j] = 1
                    count += 1
            return filters   

        # block_dim must be 2
        # Increasing this will require adding more tf.nn.depthwise_conv2d functions. 
        # There is a depthwise_conv2dfor each element in a single filter 
        # there are block_dim^2 elements in a filter
        block_filters = create_block_filters(block_dim)
    
        # Reshaping to satisfy required shape for weight array
        # copying each filter along the last axis 
        # Must have shape [filter_height, filter_width, in_channels, channel_multiplier] 
        # (see Tensorflow docs for tf.nn.depthwise_conv2d)
        b_shp = block_filters.shape
        block_filters = block_filters.reshape((b_shp[0], b_shp[1], b_shp[2], 1, 1))
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
            angles = tf.atan2(conv_stacked[:, :, :, 1], conv_stacked[:, :, :, 0])
            return angles

        def calculate_sin_cos(angles):
            sin = K.sin(angles)
            cos = K.cos(angles)
            sin_cos = K.stack([sin, cos], axis =-1)
            return sin_cos

        def block1(cells):
            c_blocks = tf.nn.depthwise_conv2d(cells, filt1, strides=(1, bs, bs, 1),
                                            padding="SAME")
            return c_blocks

        def block2(cells):
            c_blocks = tf.nn.depthwise_conv2d(cells, filt2, strides=(1, bs, bs, 1),
                                            padding="SAME")
            return c_blocks 

        def block3(cells):
            c_blocks = tf.nn.depthwise_conv2d(cells, filt3, strides=(1, bs, bs, 1),
                                            padding="SAME")
            return c_blocks
        def block4(cells):
            c_blocks = tf.nn.depthwise_conv2d(cells, filt4, strides=(1, bs, bs, 1),
                                            padding="SAME")
            return c_blocks

        def block_norm(bins_layer):
            c = 0.00000001
            denominator = tf.expand_dims(tf.sqrt(tf.norm(block_layer, axis=-1) + c), -1)
            block_norm = tf.div(block_layer, denominator)
            return block_norm

        def hog_norm(bins_layer):
            c = 0.00000001
            denominator = tf.expand_dims(tf.sqrt(tf.norm(bins_layer, axis=-1) + c), -1)     
            hog_norms = tf.div(bins_layer, denominator)
            
            denominator = tf.expand_dims(tf.sqrt(tf.norm(hog_norms, axis=-1) + c), -1)
            hog_norms = tf.div(hog_norms, denominator)
            return hog_norms

        # Building Model 
        inputs = Input(shape=(256, 256, 1), name="input")

        # Convolutions
        x_conv = Conv2D(1, (3,3), strides=(1, 1), padding="same", data_format="channels_last", 
                        trainable=True, use_bias=False, name="conv_x")(inputs)
        y_conv = Conv2D(1, (3,3), strides=(1, 1), padding="same", data_format="channels_last", 
                        trainable=True, use_bias=False, name="conv_y")(inputs)

        # Stacking you cannot multiple layer (i.e mags and angles)
        conv_stacked = Concatenate(axis=-1, name="Conv_Stacked")([x_conv, y_conv])

        # Calculating the gradient magnitudes and angles
        mags = Lambda(calculate_magnitudes, output_shape=(256, 256), name="mags1")(conv_stacked)
        mags = Lambda(lambda x: K.stack((mags, mags), axis=-1), name="mags2")(mags)   # To enable sin_cos_vec 
        angles = Lambda(calculate_angles, output_shape=(256, 256), name="angles")(conv_stacked)

        # Calculating the components of angles in the x and y direction
        # Then multiplying by magnitudes, giving angle vectors
        sin_cos = Lambda(calculate_sin_cos, output_shape=(256, 256, 2), name="sin_cos")(angles)
        sin_cos_vec = multiply([sin_cos, mags], name="sin_cos_mag")

        # Applying each filter (representing a single bin unit vector) to every angle vector in the image.
        # Result is an array with shape (img_height, img_width, bins) 
        # where each bin contains an angle vectors that contribution to each bin
        # Relu activation function to remove negative projections (represented by negative scalar dot products).
        votes = Conv2D(8, kernel_size=(1, 1), strides=(1, 1), activation="relu", trainable=True, 
                       bias=False, name="votes")(sin_cos_vec) 

        # A round about way of splitting the image (i.e vote array) 
        # into a bunch of non-overlapping cells of size (cell_dim, cell_dim)
        # then concateting values at each bin level, giving shape of (cell_nb, cell_nb, bins)
        # Result is an array of cells with histograms along the final axis
        cells = AveragePooling2D(pool_size=cell_dim, strides=cell_dim, name="cells")(votes)
        cells = Lambda(lambda x: x * (cell_dim ** 2), name="cells2")(cells)

        # Bin Operations
        # Assuming that bin shape = (2, 2)
        # A round about way of grouping the cells into overlapping blocks of 2 * 2 cells each. 
        # Two horizontally or vertically consecutive blocks overlap by two cells, that is, the block strides. 
        # As a consequence, each internal cell is covered by four blocks (if bin_dim=2).
        block1_layer = Lambda(block1, trainable=True, name="block1")(cells)
        block2_layer = Lambda(block2, trainable=True, name="block2")(cells)
        block3_layer = Lambda(block3, trainable=True, name="block3")(cells)
        block4_layer = Lambda(block4, trainable=True, name="block4")(cells)
        block_layer = Concatenate(axis=-1)([block1_layer, block2_layer, block3_layer, block4_layer])

        # normalize each block feature by its Euclidean norm
        block_norm_layer = Lambda(block_norm, name="norm1")(block_layer)
        hog_norm_layer = Lambda(hog_norm, name="norm2")(block_norm_layer)

        # Block 1
        x = Conv2D(4, (3, 3), activation='relu', padding='same', name='block1_conv1')(hog_norm_layer)
        x = Conv2D(4, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = Conv2D(4, (3, 3), activation='relu', padding='same', name='block1_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block2_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Dense
        x = Flatten(name='flat')(x)
        x = Dense(50, activation='relu', name='fc1')(x)
        x = Dropout(0.2)(x)
        x = Dense(50, activation='relu', name='fc2')(x)
        x = Dropout(0.2)(x)
        x = Dense(50, activation='relu', name='fc3')(x)
        x = Dropout(0.2)(x)
        logistic_reg = Dense(2, activation = "softmax", trainable=True, name="lr")(x)

        # Building Model
        model = Model(inputs=inputs, outputs=logistic_reg)

        self.model = model

    def fit(self, train_x, train_y, val_x=None, val_y=None):
        """Defines and trains hognet.
    
        Using the set hyperparameters, our custom initial weights are 
        generated, our custom keras layers are defined in tensorflow, and 
        our model is constructed in keras. The resulting model is trained 
        either with or without validation data. The data is generated in 
        real time using the prefined datagen function. The learning rate 
        is reduced when the loss plateaus. When the model stops improving 
        for a set number of epochs (patience), training stops, and a  
        model is returned. 
        
         
        
        # Arguments
            train_x: Images to train on with shape (samples, dim, dim, 1)
            train_y: Image labels to train on as a confusion matrix with 
                shape (samples, 2) 
            val_x: Images to validate with. By default, no validation data 
                is used 
            val_y: Labels to validate with. By default, no validation data 
                is used. 

        # Returns
            An updated state where self.model is a trained keras model. 
        """
        # Setting random number generator seeds for numpy and tensorflow
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
        
        # Compiling Model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', 
                      metrics=["accuracy"])

        if validation is True:
            callback = [EarlyStopping(monitor='val_loss', patience=self.patience),
                    ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1),
                    LossHistory()]
            
            self.model.fit_generator(self.datagen.flow(train_x, train_y, batch_size=self.batch_size, shuffle=True), 
                                steps_per_epoch=self.steps_per_epoch, epochs=self.max_epoch,
                                validation_data=self.datagen.flow(val_x, val_y, batch_size=self.batch_size,
                                                                 shuffle=True),
                                validation_steps=math.ceil(self.steps_per_epoch / 5), callbacks=callback)

        else:
            callback = [EarlyStopping(monitor='loss', patience=self.patience),
                    ReduceLROnPlateau(monitor='loss', patience=5, verbose=1),
                    LossHistory()]
            self.model.fit_generator(self.datagen.flow(train_x, train_y, batch_size=self.batch_size, shuffle=True), 
                            steps_per_epoch=self.steps_per_epoch, epochs=self.max_epoch, 
                            callbacks=callback)

        self.history = callback[2]

        return self

    def predict_proba(self, test_x):
        """ Probability estimates for samples"""
        predictions =  self.model.predict(test_x, batch_size=self.batch_size)

        return predictions

    def predict (self, test_x):
        """Predicting class labels for samples in test_x."""
        predictions =  self.predict_proba(test_x)
        predictions = np.around(predictions)

        return predictions


    def save (self, path=None):
        """Serialize model weights to HDF5"""
        self.model.save_weights(path)

        self.path = path

        return self

        
    def load(self, path=None):
        """Loads weights into keras Model from disk"""
        if path is None:
            path = self.path
        
        # load weights into new model
        self.model.load_weights(path)
        
        return self

