import SimpleITK as sitk
import numpy as np
from image_reader_writer import ImageReaderWriter

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

        return images, image_nrrd

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

    def save_labels_as_nrrd(self, output_path, labels, image_nrrd, selection=None):

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


        sitk_CT = ImageReaderWriter.read(CT_path)
        sitk_image = ImageReaderWriter.numpy_to_sitkImage(labels_unique, sitk_image_template=sitk_CT)
        ImageReaderWriter.write(sitk_image,output_path)
        #sitk.WriteImage(sitk.GetImageFromArray(labels_unique.astype('int16')), (output_path))
        #sitk.WriteImage(sitk.GetImageFromArray(labels.astype('int16')), (output_path))

        return labels_unique

    def save_labels_as_np(self, path, labels):
        # Save Results
        np.save((path + 'pred_labels.npy'), labels)

    ## Post-processing
    def close_labels(self, labels):
        labels = labels.astype('int16')

        for i in range(1,self.num_classes):
            for j in range(labels.shape[0]):
                aux = sitk.GetImageFromArray(labels[j, :, :, i])
                labels[j, :, :, i] = sitk.GetArrayFromImage(sitk.BinaryFillhole(aux))

        return labels

    def delimit_fat_segmentation(self, labels):
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
