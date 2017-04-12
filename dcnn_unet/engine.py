import numpy as np
from keras import callbacks as callbacks

class LossHistory(callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.IoU = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.IoU.append(logs.get('IoU'))

class Engine():
    def __init__(self, network, output_net_path=None):
        """Constructor.

        Parameters
        ----------
        network : UNETNetwork object.

        output_net_path : str with the path where the net should be saved.

        """
        if output_net_path == None:
            self.output_path = 'output_net.hd5f'
        else:
            self.output_path = output_net_path

        self.network = network
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.model = None

    def predict(self, images, model_path=None, model=None):
        """Uses keras predict function for predicting data with a NN model.

        Parameters
        ----------
        images : numpy array with the images that are going to be introduce in the neural network.

        model_path : str with the path where the net is saved.

        model : keras model if it has been built previously.

        Returns
        -------
        pred_labels : numpy array with predicted labels
        """

        self.test_images = images

        if model_path != None:
            self.model = self.network.load_UNET(model_path)
        else:
            self.model = model

        pred_labels = self.model.predict(self.test_images, verbose=1)

        return pred_labels

    def predict_pectoralis(self, selection, images, model_path):
        """Prediction of pectoralis introducing CT images.

        Parameters
        ----------
        selection : int with 1,2 or 3 depending on the wanted segmentation.

        images : numpay array with the images to be segmented.

        model_path : keras model if it has been built previously.

        Returns
        -------
        pred_labels : numpy array with predicted labels.
        """

        pred_labels = []

        if selection == 1:
            pred_labels = self.predict(images, model_path)
        elif selection == 2:
            pred_labels = self.predict(images, model_path)

        elif selection  == 3:
            print model_path
            pred_labels = self.predict(images, model_path)

        elif selection  == 4:
            print model_path
            pred_labels_pect = self.predict(images, model_path[0])
            pred_labels_fat = self.predict(images, model_path[1])
            pred_labels = np.concatenate((pred_labels_pect,pred_labels_fat[:,:,:,1:3]), axis=3)

        return pred_labels

    def fit(self, images, labels, train_params, model=None):
        print('Starting Engine Fit...')

        self.train_images = images
        self.train_labels = labels

        # Compile
        if model == None:
            self.model = self.network.build_UNET()
            self.model = self.network.compile_UNET(self.model, train_params)
        else:
            self.model = model

        # CallBacks
        p = self.output_path + "neural_netwrok_train.hdf5"
        checkpointer = callbacks.ModelCheckpoint(filepath=p, verbose=1, save_best_only=True, monitor='loss')

        model_ReduceLROnPlateau = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto',
                                                    cooldown=0, min_lr=0.001)

        history = LossHistory()

        # Training
        training_history = self.model.fit(self.train_images, self.train_labels,
                                          batch_size=train_params.training_batch_size,
                                          nb_epoch=train_params.num_epochs,
                                          verbose=train_params.verbose, shuffle=True,
                                          callbacks=[checkpointer, history]
                                          )

        json_string = self.model.to_json()

        print('Engine Fit Finished')
        return training_history, history, self.model, json_string

    def evaluate(self, images, labels, model_path=None, model=None):
        self.test_images = images
        self.test_labels = labels

        if model_path != None:
            self.model = self.network.load_UNET(model_path)  # Poner que weights coger
        else:
            self.model = model

        pred_metrics = self.model.evaluate(self.test_images, self.test_labels, verbose=1)

        return pred_metrics
