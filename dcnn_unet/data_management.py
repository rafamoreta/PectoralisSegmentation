import SimpleITK as sitk
import numpy as np

class DataManagement():
    def __init__(self, architecture_params):
        """
        Constructor
        Args:
            architecture_params: ArchitectureParameters object
        """

        self.image_height = architecture_params.image_height
        self.image_width = architecture_params.image_width
        self.num_channels = architecture_params.num_channels
        self.num_classes = architecture_params.num_classes
        self.class_code = architecture_params.class_code

    def get_images_from_nrrd(self, path, slice_num=None):
        ## Reading Images
        # Images
        image_nrrd = sitk.ReadImage(path)
        CTSlices = sitk.GetArrayFromImage(image_nrrd)

        if slice_num != None:
            CTSlices = CTSlices[slice_num:slice_num+1,:,:]

        num_images = CTSlices.shape[0]

        ## Pre-processing images
        # Normalizing
        numpy_array = CTSlices.astype(np.float32)
        numpy_array[numpy_array < -1024] = -1024
        numpy_array[numpy_array > 1500] = 1500
        numpy_array = (numpy_array + 1024) / (1500 + 1024) - 0.5

        CTSlices_new = numpy_array

        # UNET Data
        images = np.reshape(CTSlices_new, (num_images, self.image_height, self.image_width, self.num_channels))

        return images

    def get_labels_from_nrrd(self, path):
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

    def unify_labels(self, labels, labels_unique, code):
        condition = (labels == 1)
        index = np.where(condition)
        labels_unique[index] = code

        return labels_unique

    def save_labels_as_nrrd(self, output_path, labels):

        labels_unique = np.zeros(labels.shape[0:-1])

        th = 0.5
        labels[(labels > th)] = 1
        labels[(labels < th)] = 0

        if self.num_classes == 2:
            labels[(labels == 1)] = 67
            labels_unique = labels
        else:
            for i in np.linspace(self.num_classes - 1, 1, self.num_classes - 1, dtype='int16'):
                labels_unique = self.unify_labels(labels[:, :, :, i], labels_unique, self.class_code[i-1])

        sitk.WriteImage(sitk.GetImageFromArray(labels_unique.astype('int16')), (output_path))

        return labels_unique

    def save_labels_as_np(self, path, labels):
        # Save Results
        np.save((path + 'pred_labels.npy'), labels)
