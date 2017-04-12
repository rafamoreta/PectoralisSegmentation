from keras import models as models
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, Activation, Flatten
from keras import optimizers as optimizers
from keras import metrics as metrics
from keras import backend as K
from keras.models import load_model

class ArchitectureParameters(object):
    def __init__(self, num_classes=2, image_width=512, image_height=512, num_channels=1, class_code=None):
        """Constructor.

        Parameters
        ----------
        num_classes : int with number of classes that are going to be segmented.

        image_width : int image width size.

        image_height : int image height size.

        num_channels : int number of channels (depth) of the image.

        class_code : list of int with the code of the different classes.

        """

        self.num_classes = num_classes
        self.image_width = image_width
        self.image_height = image_height
        self.num_channels = num_channels  # Grayscale images
        if class_code == None:
            self.class_code = [13335, 13336, 13591, 13592, 17943, 17944]
        else:
            self.class_code = class_code

        self.init = 'glorot_uniform'
        self.W_regularizer = None
        self.b_regularizer = None

        self.training_learning_rate = 0.001

        self.loss = [metrics.categorical_crossentropy]
        self.optimizer = optimizers.Adam(lr=0.001)

class TrainingParameters(object):
    """Constructor.

    Parameters
    ----------
    None.

    """
    num_epochs = 1
    training_batch_size = 10
    training_decay_rate = 0.99  # Needed for learning rate decrease
    verbose = 1

class UNETNetwork(object):
    def __init__(self, architecture_parameters):
        """Constructor.

        Parameters
        ----------
        architecture_parameters : class with the architecture parameters of the neural network.

        """

        self.params = architecture_parameters

        self.model = None
        self.num_classes = self.params.num_classes
        self.image_width = self.params.image_width
        self.image_height = self.params.image_height
        self.num_channels = self.params.num_channels

        self.init = 'glorot_uniform'
        self.W_regularizer = self.params.W_regularizer
        self.b_regularizer = self.params.b_regularizer

        self.training_learning_rate = self.params.training_learning_rate

        self.loss = [metrics.categorical_crossentropy]
        self.optimizer = optimizers.Adam(lr=0.001)

    def build_UNET(self):
        """Creates the structure in keras of an U-Net.

        Parameters
        ----------

        Returns
        -------
        model : keras model with the model structure of a U-Net.

        """
        inputs = Input((self.image_width, self.image_height, self.num_channels))

        ## Part 1
        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(inputs)
        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        ## Part 2
        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(pool1)
        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        ## Part 3
        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(pool2)
        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        ## Part 4
        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(pool3)
        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        ## Part 5
        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(pool4)
        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(conv5)

        ## Part 6
        up_aux = UpSampling2D(size=(2, 2))(conv5)
        conv_up_aux = Convolution2D(256, 2, 2, activation='relu', border_mode='same',
                                    init=self.init,
                                    W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(up_aux)

        up6 = merge([conv_up_aux, conv4], mode='concat', concat_axis=3)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(up6)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(conv6)

        ## Part 7
        up_aux = UpSampling2D(size=(2, 2))(conv6)
        conv_up_aux = Convolution2D(256, 2, 2, activation='relu', border_mode='same',
                                    init=self.init,
                                    W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(up_aux)

        up7 = merge([conv_up_aux, conv3], mode='concat', concat_axis=3)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(up7)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(conv7)

        ## Part 8
        up_aux = UpSampling2D(size=(2, 2))(conv7)
        conv_up_aux = Convolution2D(256, 2, 2, activation='relu', border_mode='same',
                                    init=self.init,
                                    W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(up_aux)

        up8 = merge([conv_up_aux, conv2], mode='concat', concat_axis=3)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(up8)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(conv8)

        ## Part 9
        up_aux = UpSampling2D(size=(2, 2))(conv8)
        conv_up_aux = Convolution2D(256, 2, 2, activation='relu', border_mode='same',
                                    init=self.init,
                                    W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(up_aux)

        up9 = merge([conv_up_aux, conv1], mode='concat', concat_axis=3)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(up9)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same',
                              init=self.init,
                              W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(conv9)

        conv10 = Convolution2D(self.num_classes, 1, 1, activation='sigmoid',
                               init=self.init,
                               W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer)(conv9)

        self.model = models.Model(input=inputs, output=[conv10])

        return self.model

    def load_UNET(self, model_path):
        """Load the already trained model from path.

        Parameters
        ----------
        model_path : str with the path of the pre-trained model.

        Returns
        -------
        model : keras model of the trained model in input path.

        """

        self.model = load_model(model_path)

        return self.model

    def compile_UNET(self ,model, train_params):
        """Compiles in keras the model structure built before.

        Parameters
        ----------
        model : keras model with the built structure of the net model.

        Returns
        -------
        model : compiled keras model.

        """

        self.model = model

        self.model.compile(optimizer=optimizers.Adam(lr=self.training_learning_rate),
                           loss=self.loss,
                           metrics=[self.IoU])

        return self.model

    def IoU(self, y_true, y_pred):
        """Calculates the Intersection over Union (IoU) measurement between label and predicted data.

        Parameters
        ----------
        y_true : np.array label value.
        y_pred : np.array predicted value.

        Returns
        -------
        IoU : float number with IoU measure.

        """

        y_true = K.round(K.clip(y_true, 0, 1))
        y_pred = K.round(K.clip(y_pred, 0, 1))

        tp = K.sum(K.sum(K.sum((y_true * y_pred), axis=2), axis=1), axis=0)
        fn = K.sum(K.sum(K.sum(((y_true + y_pred) * y_true), axis=2), axis=1), axis=0) - tp * 2.
        fp = K.sum(K.sum(K.sum(((y_true + y_pred) * y_pred), axis=2), axis=1), axis=0) - tp * 2.

        IoU = K.mean((tp) / (tp + fn + fp))
        return IoU
