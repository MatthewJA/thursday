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

    This creates a wrapper that can be instatised to any sklearn
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
    """   
    def __init__(self, Model, name, path=None, seed=None):
        self.Model = Model
        self.name = name
        self.path = path
        self.seed = seed

    def train(self, train_x, train_y, params):
        """Trains sklearn model.

        # Arguments
            images: Array of images with shape (samples, dim, dim, 1).
            labels: Array of labels with shape (samples ,).
            params: Dictionary of parameters specific to Model.

        # Returns
            A trained model.
        """ 
        model = self.Model(**params, random_state=self.seed)
        model = model.fit(train_x, train_y)

        if self.path is None:
            path = self.name + ".pkl"

        else:
            path = self.path + self.name + ".pkl"

        with open(path, 'wb') as f:
            pickle.dump(model, f)

        print("model saved as " + path)

        return model
    
    def load(self):    
        """Loads trained Sklearn Model from file"""
        
        if self.path is None:
            path = self.name + ".pkl"

        else:
            path = self.path + self.name + ".pkl"

        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model
    

class HognetModel:
    """Wrapper for our hognet keras model.

    This creates a class that acts as a wrapper around our custom 
    keras model i.e hognet. The train method constructs and trains
    the model on our data generator to our specifed training paramerts.
    Using the load method, we can load the fully trained model from the
    disk at any point in the script. This method can be useful when using 
    notebooks, as training time can be significant.

    Attributes:
    name: name of the file at which the model is saved. 
        No file extension.
    path: Path of data folder. Defaults to current directory
    seed: Random seed. All our tests used 0.
    """     
    def __init__(self, name, path=None, seed=None):
        self.name = name
        self.path = path
        self.seed = seed

    def train (self, train_x, train_y, val_x=None, val_y=None, datagen=None, 
               batch_size=100, steps_per_epoch=50, max_epoch=100, patience=5, plot=True):
        """Defines and trains hognet.
    
        Using the set hyperparameters, our custom initial weights are 
        generated, our custom keras layers are defined in tensorflow, and 
        our model is constructed in keras. The resulting model is trained 
        either with or without validation data. The data is generated in 
        real time using the prefined datagen function. The learning rate 
        is reduced when the loss plateaus. When the model stops improving 
        for a set number of epochs (patience), training stops, and a  
        model is returned. 
        
        About the model:
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
        accuracy by roughly 2%. 
        
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
        train up 5-100 times faster (if at all) on the output array of the 
        tensor-based HOG than raw images. Our guess is that our tensor-based
        HOG drastically reduces problem complexity, and thus the time it 
        takes for the optimizer to converge on the weights. It also slighly 
        increased accuracy. 
        
        # Arguments
            train_x: Images to train on with shape (samples, dim, dim, 1)
            train_y: Image labels to train on as a confusion matrix with 
                shape (samples, 2) 
            val_x: Images to validate with. By default, no validation data 
                is used 
            val_y: Labels to validate with. By default, no validation data 
                is used. 
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
            patience:
            plot: if true, prints model summary and plots the acc and loss on 
                graphs

        # Returns
            A trained keras model. The model is also saved automatically and 
            can be called by the load method.

        """
        # Model HyperParameters
        # the following values were chosen based on both past research and thorough testing
        bins = 8                    # number of bins in histogram
        cell_dim = 8                # height and width of the cells  
        block_dim = 2               # if changed, must add more block layers. Not recommended.
        bs = bin_stride_length = 1  

        # Number of cells along each dim 
        cell_nb = 256 // cell_dim
        assert not 256 % cell_dim
        
        # Defining Values
        w = 2*np.pi/bins     # width of each bin
        centers = np.arange(-np.pi, np.pi, w) + 0.5 * w   # centers of each bin
        

        # Defining Weights for the vertical and horizontal convolutions to calculate the image gradients
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).reshape((1, 3, 3, 1, 1)) + 0.01 * np.random.randn(1, 3, 3, 1, 1)
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).reshape((1, 3, 3, 1, 1)) + 0.01 * np.random.randn(1, 3, 3, 1, 1)

        cent = np.vstack((np.sin(centers), np.cos(centers))).reshape((1, 1, 1, 2, 8))

        # Generating Filters for the Bin Operations
        def create_bin_filters(block_dim):
            filters = np.zeros((block_dim ** 2, block_dim, block_dim))

            count = 0 
            for i in range(block_dim):
                for j in range(block_dim):
                    filters[count, i, j] = 1
                    count += 1
            return filters   

        # block_dim must be 2
        # Increasing this will require the adding of some more tf.nn.depthwise_conv2d functions. 
        # There is one for each element in a single filter (i.e block_dim^2)

        b_flt = create_bin_filters(2)
        
        # copying each filter anlong the last axis to satisfy the required shape for weights array 
        # (see Tensorflow docs for tf.nn.depthwise_conv2d)
        bin_filters_n = b_flt.reshape((b_flt.shape[0], b_flt.shape[1], b_flt.shape[2], 1)).repeat(bins, 3) 

        # Reshaping to satisfy required shape for weight array
        bin_filters = bin_filters_n.reshape(bin_filters_n.shape[0], bin_filters_n.shape[1],
                                            bin_filters_n.shape[2], bin_filters_n.shape[3], 1).astype(np.float32)
        
        filt1 = tf.convert_to_tensor(bin_filters[0, :, :, :, :])
        filt2 = tf.convert_to_tensor(bin_filters[1, :, :, :, :])
        filt3 = tf.convert_to_tensor(bin_filters[2, :, :, :, :])
        filt4 = tf.convert_to_tensor(bin_filters[3, :, :, :, :])

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

        def bins1(cells):
            c_bins = tf.nn.depthwise_conv2d(cells, filt1, strides = (1, bs, bs, 1), padding="SAME")
            return c_bins

        def bins2(cells):
            c_bins = tf.nn.depthwise_conv2d(cells, filt2, strides = (1, bs, bs, 1), padding="SAME")
            return c_bins 

        def bins3(cells):
            c_bins = tf.nn.depthwise_conv2d(cells, filt3, strides = (1, bs, bs, 1), padding="SAME")
            return c_bins

        def bins4(cells):
            c_bins = tf.nn.depthwise_conv2d(cells, filt4, strides = (1, bs, bs, 1), padding="SAME")
            return c_bins

        def bin_norm(bins_layer):
            bins_norm = tf.div(bins_layer, tf.expand_dims(tf.sqrt(tf.norm(bins_layer, axis=-1)+0.00000001), -1))
            return bins_norm

        def hog_norm(bins_layer):
            hog_norms = tf.div(bins_layer, tf.expand_dims(tf.sqrt(tf.norm(bins_layer, axis=-1)+0.00000001), -1))
            hog_norms_2 = tf.div(hog_norms, tf.expand_dims(tf.sqrt(tf.norm(hog_norms, axis=-1)+0.00000001), -1))
            return hog_norms_2

        # Building Model 
        inputs = Input(shape = (256, 256, 1), name="input")

        # Convolutions
        x_conv = Conv2D(1, (3,3), strides=1, padding="same", data_format="channels_last", 
                        trainable=True, use_bias=False, name="conv_x")(inputs)
        y_conv = Conv2D(1, (3,3), strides=1, padding="same", data_format="channels_last", 
                        trainable=True, use_bias=False, name="conv_y")(inputs)

        # Stacking since it appears you cannot have more than a single input into a layer (i.e mags and angles)
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

        # A round about way of splitting the image (i.e vote array) into a bunch of non-overlapping cells of size (cell_dim, cell_dim)
        # then concateting values at each bin level, giving shape of (cell_nb, cell_nb, bins)
        # Result is an array of cells with histograms along the final axis
        cells = AveragePooling2D(pool_size=cell_dim, strides=cell_dim, name="cells")(votes)
        cells = Lambda(lambda x: x * (cell_dim ** 2), name="cells2")(cells)

        # Bin Operations
        # Assuming that bin shape = (2, 2)
        # A round about way of grouping the cells into overlapping blocks of 2 * 2 cells each. 
        # Two horizontally or vertically consecutive blocks overlap by two cells, that is, the block strides. 
        # As a consequence, each internal cell is covered by four blocks (if bin_dim=2).
        bins1_layer = Lambda(bins1, trainable=True, name="bins1")(cells)
        bins2_layer = Lambda(bins2, trainable=True, name="bins2")(cells)
        bins3_layer = Lambda(bins3, trainable=True, name="bins3")(cells)
        bins4_layer = Lambda(bins4, trainable=True, name="bins4")(cells)
        bins_layer = Concatenate(axis=-1)([bins1_layer, bins2_layer, bins3_layer, bins4_layer])

        # normalize each block feature by its Euclidean norm
        bins_norm_layer = Lambda(bin_norm, name="norm1")(bins_layer)
        hog_norm_layer = Lambda(hog_norm, name="norm2")(bins_norm_layer)

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

        # Setting weights
        model.layers[1].set_weights(prewitt_x)
        model.layers[2].set_weights(prewitt_y)
        model.layers[9].set_weights(cent)

        # Model Summary
        if plot:
            model.summary()
        
        # Checking for validation data
        if val_x is None and val_y is None:
            validation = False
        else:
            validation = True
        
        if datagen is None:
            print ("no data generator has been inputed")
            raise p
    
        # Setting Seed
        seed = self.seed
           
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
        model.compile(optimizer='adam', loss='binary_crossentropy', 
                      metrics=["accuracy"])

        if validation is True:
            callback = [EarlyStopping(monitor='val_loss', patience=patience),
                    ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1),
                    LossHistory()]
            
            model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size, shuffle=True), 
                                steps_per_epoch=steps_per_epoch, epochs=max_epoch,
                                validation_data=datagen.flow(val_x, val_y, batch_size=batch_size,
                                                                 shuffle=True),
                                validation_steps=math.ceil(steps_per_epoch / 5), callbacks = callback)

        else:
            callback = [EarlyStopping(monitor='loss', patience=patience),
                    ReduceLROnPlateau(monitor='loss', patience=5, verbose=1),
                    LossHistory()]
            model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size, shuffle=True), 
                            steps_per_epoch=steps_per_epoch, epochs = max_epoch, 
                            callbacks = callback)
        
        # Printing metrics
        if plot:
            if validation:
                plt.plot(callback[2].acc)
                plt.plot(callback[2].val_acc)
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.grid(True)
                plt.show() 

                plt.plot(callback[2].loss)
                plt.plot(callback[2].val_acc)
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.grid(True)
                plt.show() 


            else:
                plt.plot(callback[2].acc)
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train'], loc='upper left')
                plt.grid(True)
                plt.show() 

                plt.plot(callback[2].loss)
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train'], loc='upper left')
                plt.grid(True)
                plt.show()  
        
        """
        # Saving Model
        if self.path is None:
            model_path = self.name + "_model.json"
            weight_path = self.name + "_weights.h5"

        else:
            model_path = self.path + self.name + "_model.json"
            weight_path = self.path + self.name + "_weights.h5"
            
       
        # serialize model to HDF5
        model.save_weights(weight_path)

        # serialize model to JSON
        model_json = model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        """

        return model
        
    def load(self):
        """Loads trained keras Model from disk"""
        
        if self.path is None:
            model_path = self.name + "_model.json"
            weight_path = self.name + "_weights.h5"

        else:
            model_path = self.path + self.name + "_model.json"
            weight_path = self.path + self.name + "_weights.h5"
            
        # load json and create model
        json_file = open(model_path, 'r')
        model_json = json_file.read()
        json_file.close()
        
        model = model_from_json(model_json)
        
        # load weights into new model
        model.load_weights(weight_path)
        

        return model



    