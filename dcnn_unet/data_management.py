import SimpleITK as sitk
import numpy as np
from image_reader_writer import ImageReaderWriter

class DataManagement():
    def __init__(self, architecture_params):
        """Constructor.

        Parameters
        ----------
        architecture_parameters : class with the architecture parameters of the neural network.

        """

        self.image_height = architecture_params.image_height
        self.image_width = architecture_params.image_width
        self.num_channels = architecture_params.num_channels
        self.num_classes = architecture_params.num_classes
        self.class_code = architecture_params.class_code

    def get_images_from_nrrd(self, path, slice_num=None):
        """Get in a numpy array the slice from the input nrrd.

        Parameters
        ----------
        path : str
            Path of nrrd.
        slice_num : int
            Number of the slice form the nrrd to be segmented.

        Returns
        -------
        images : numpy array
            Array with the slice with shape (1,image_height,image_width,num_channels)
        """

        # Reading Images
        image_nrrd = sitk.ReadImage(path)
        CTSlices = sitk.GetArrayFromImage(image_nrrd)

        if slice_num != None:
            CTSlices = CTSlices[slice_num:slice_num+1,:,:]

        num_images = CTSlices.shape[0]

        # Pre-processing images
        # Normalizing
        numpy_array = CTSlices.astype(np.float32)
        numpy_array[numpy_array < -1024] = -1024
        numpy_array[numpy_array > 1500] = 1500
        numpy_array = (numpy_array + 1024) / (1500 + 1024) - 0.5

        CTSlices_new = numpy_array

        # UNET Data shape transform
        images = np.reshape(CTSlices_new, (num_images, self.image_height, self.image_width, self.num_channels))

        return images

    def get_labels_from_nrrd(self, path):
        """Transform the labels in Keras format for UNET.

        Parameters
        ----------
        path : str
            Path of nrrd.

        Returns
        -------
        labels_cat : numpy array
            Array with labels in binary form.
        """

        # Labels
        image_nrrd = sitk.ReadImage(path)
        labels = sitk.GetArrayFromImage(image_nrrd)

        ## Pre-processing images
        # Labeles
        labels_cat = np.zeros(labels.shape)

        for i, code in enumerate(self.class_code):
            condition = (labels == code)
            index = np.where(condition)
            labels_cat[index] = (i+1)

        labels_cat = (np.arange(self.num_classes) == labels_cat[:, :, :, None]) * 1.0

        return labels_cat

    def get_labels_from_nrrd_as_oneclass(self, path):
        """Transform the labels in Keras format for UNET if the segmentatoin is going to be just one class.

        Parameters
        ----------
        path : str
            Path of nrrd.

        Returns
        -------
        labels_cat : numpy array
            Array with labels in binary form.
        """

        # Labels
        image_nrrd = sitk.ReadImage(path)
        labels = sitk.GetArrayFromImage(image_nrrd)

        ## Pre-processing images
        # Labeles
        labels_cat = np.zeros(labels.shape)

        for i, code in enumerate(self.class_code):
            condition = (labels == code)
            index = np.where(condition)
            labels_cat[index] = 1

        labels_cat = (np.arange(self.num_classes) == labels_cat[:, :, :, None]) * 1.0

        return labels_cat

    def save_labels_as_original_nrrd(self, slice_labels_unique, slice_num, output_path, CT_path):
        """Save resultant labels from a numpy array to path to a nrrd.

        Parameters
        ----------
        slice_labels_unique : numpy array
            Labels to be saved.
        slice_num: int
            Slice number from the original nrrd.
        output_path: str
            Path to be saved the nrrd. File name must be included.
        CT_path: str
            Path form the original nrrd.

        """

        image_reader_writer = ImageReaderWriter()

        sitk_CT = image_reader_writer.read(CT_path)

        numpy_CT = image_reader_writer.sitkImage_to_numpy(sitk_CT)
        labels_unique_CT = np.zeros(numpy_CT.shape)

        labels_unique_CT[:,:,slice_num] = slice_labels_unique.transpose([2,1,0])[:,:,0]

        sitk_image = image_reader_writer.numpy_to_sitkImage(labels_unique_CT, sitk_image_tempate=sitk_CT)
        image_reader_writer.write(sitk_image,output_path)

    def save_labels_as_nrrd(self, labels, output_path):
        """Save labels as nrrd.

        Parameters
        ----------
        labels : numpy array
            Labels to be saved.
        output_path: str
            Path to be saved the nrrd. File name must be included.

        """

        sitk.WriteImage(sitk.GetImageFromArray(labels.astype('int16')), (output_path))

    def save_labels_as_np(self, labels, output_path):
        """Save labels as npy file.

        Parameters
        ----------
        labels : numpy array
            Labels to be saved.
        output_path: str
            Path to be saved the numpy array. File name must be included ending with '.npy'.

        """

        # Save Results
        np.save(output_path, labels)

    ## Post-processing
    def get_slice_labels_unique(self, labels, selection):
        """From binary labels to an unique image with all labels joined together.

        Parameters
        ----------
        labels : numpy array
        selection: int
            Selection of the segmentation process.

        Returns
        -------
        labels_unique : numpy array
            Single image with all the labels joined together.

        """

        labels_unique = np.zeros(labels.shape[0:-1])

        th = 0.5
        labels[(labels > th)] = 1
        labels[(labels < th)] = 0

        # Closing
        labels = self.close_labels(labels)

        # Unifying labels
        if self.num_classes == 2:
            labels[(labels == 1)] = 67 # Creo que esta mal
            labels_unique = labels
        else:
            if selection == 4:
                labels = self.delimit_fat_segmentation(labels) ## solo si se segmenta una sola slice

            for i in np.linspace(self.num_classes - 1, 1, self.num_classes - 1, dtype='int16'):
                labels_unique = self.unify_labels(labels[:, :, :, i], labels_unique, self.class_code[i-1])

        return labels_unique

    def unify_labels(self, labels, labels_unique, code):
        """Function to unify on eclass og label to labels unique variable with the label class code.

        Parameters
        ----------
        labels : numpy array
            Binary label of an specific class.
        labels_unique: numpy array
            Array to be saved where the class form the label is.
        code: int
            Code number of the label class.

        Returns
        -------
        labels_unique : numpy array
            Array with labels in binary form.
        """
        condition = (labels == 1)
        index = np.where(condition)
        labels_unique[index] = code

        return labels_unique

    def close_labels(self, labels):
        """Function which make a morphological close to the segmentation to all classes.

        Parameters
        ----------
        labels : numpy array
            Labels to perform the close.

        Returns
        -------
        labels : numpy array
            Array after performing the close.
        """


        labels = labels.astype('int16')

        for i in range(1,self.num_classes):
            for j in range(labels.shape[0]):
                aux = sitk.GetImageFromArray(labels[j, :, :, i])
                labels[j, :, :, i] = sitk.GetArrayFromImage(sitk.BinaryFillhole(aux))

        return labels

    def delimit_fat_segmentation(self, labels):
        """Function needed when segmenting fat and pectoralis at the same time, in which it deletes segmented parts
        of the fat when it goes beyond pectoralis limits.

        Parameters
        ----------
        labels : numpy array

        Returns
        -------
        labels : numpy array

        """

        condition = labels[0,:, :, 1:5] == 1
        where_ones = np.where(condition)
        where_ones = np.array(where_ones)

        y_axis_min = where_ones[1].min()
        y_axis_max = where_ones[1].max()

        y_range = range(y_axis_min, y_axis_max)
        x_axis_max = list()

        for i in y_range:
            condition = labels[0, :, i, 1:5] == 1
            where_ones = np.where(condition)
            where_ones = np.array(where_ones)
            if condition.sum() == 0:
                x_axis_max.append(x_axis_max[-2])
            else:
                x_axis_max.append(where_ones[0].max())

        y_direction = range(y_axis_min) + range(y_axis_max, self.image_width)

        index_to0_fat_y = np.ix_([0], range(self.image_width), y_direction,range(5,7))

        for i, val in enumerate(x_axis_max):
            index_to0_fat_x = np.ix_([0],range(val, self.image_height), [y_range[i]],range(5,7))
            labels[index_to0_fat_x] = 0
        labels[index_to0_fat_y] = 0

        return labels
