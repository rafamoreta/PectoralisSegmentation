import numpy as np
import tensorflow as tf
import SimpleITK as sitk


from data_management import DataManagement
from unet_network import ArchitectureParameters, TrainingParameters, UNETNetwork
from engine import Engine


## Main

def segment_pectoralis(data_path, slice_num, selection, model_path, output_path, verbose=0):
    print('Test')

    ## Step 1: Parameters Selection
    if verbose:
        print('Step 1: Starting Parameters Selection...')
    # Selection = 1:
    # Segmentation of background and all pectoralis as one class
    if selection == 1:
        num_classes = 2  # length(class_code)+1 (Taking into account background)
        class_code = [13335, 13336, 13591, 13592] # Code representing each label
        image_width = 512
        image_height = 512
        num_channels = 1
        if model_path == None:
            model_path = '/path'

    # Selection = 2:
    # Segmentation of background and each pectoralis individual
    elif selection == 2:
        num_classes = 5  # length(class_code)+1 (Taking into account background)
        class_code = [13335, 13336, 13591, 13592] # Code representing each label
        image_width = 512
        image_height = 512
        num_channels = 1
        if model_path == None:
            model_path = '/path'

    # Selection = 3:
    # Segmentation of background each pectoralis individual and right and left fat
    elif selection == 3:
        num_classes = 7  # length(class_code)+1 (Taking into account background)
        class_code = [13335, 13336, 13591, 13592, 17943, 17944] # Code representing each label
        image_width = 512
        image_height = 512
        num_channels = 1
        if model_path == None:
            model_path = ['/path']
            model_path.append['/path2']


    architecture_params = ArchitectureParameters(num_classes=num_classes,
                                                image_width=image_width,
                                                image_height=image_height,
                                                num_channels=num_channels,
                                                class_code=class_code)
    if verbose:
        print('Step 1 Done.')

    ## Step 2: Get Images from ".nrrd"
    if verbose:
        print('Step 2: Starting Get Images...')

    data_management = DataManagement(architecture_params)
    test_images = data_management.get_images_from_nrrd(data_path, slice_num)

    if verbose:
        print('Step 2 Done.')

    ## Step 3: Network creation
    if verbose:
        print('Step 3: Starting UNETNetwork...')

    unet_network = UNETNetwork(architecture_params)

    if verbose:
        print('Step 3 Done.')

    ## Step 4: Engine Testing
    if verbose:
        print('Step 4: Starting Engine Testing...')

    unet_engine = Engine(output_path, unet_network)
    pred_labels = unet_engine.predict_pectoralis(selection, test_images, model_path)

    if verbose:
        print('Step 4 Done.')

    ## Step 5: Save Segmentation Prediction
    if verbose:
        print('Step 5: Starting Save Segmentation Prediction...')

    pred_labels_unique = data_management.save_labels_as_nrrd(output_path, pred_labels)


    return pred_labels, pred_labels_unique

def train(data_folder, output_folder):
    print('Train')

    # Params
    num_classes = 2  # length(class_code)+1 (Taking into account background)
    class_code = [13335, 13336]
    image_width = 512
    image_height = 512
    num_channels = 1
    architecture_params = ArchitectureParameters(num_classes=num_classes,
                                                       image_width=image_width,
                                                       image_height=image_height,
                                                       num_channels=num_channels,
                                                       class_code=class_code)
    train_params = TrainingParameters()
    print('Params done')

    # Get Images
    extract_data = DataManagement(architecture_params)
    train_images = extract_data.get_images_from_nrrd(data_folder)
    train_labels = extract_data.get_labels_from_nrrd(data_folder)
    print('Get images done')

    # Netwrok
    unet_network = UNETNetwork(architecture_params)
    print('Netwrok construction done')

    # Engine Training
    unet_engine = Engine(output_folder, unet_network)
    training_history, history = unet_engine.fit(train_images, train_labels, train_params)


def get_labelmap(input_data, slice, unet_model_checkpoint_folder, unet_num_classes, unet_labelmap_output_path):
    return "Labelmap would be generated in " + unet_labelmap_output_path
##

from optparse import OptionParser
import argparse

if __name__ == "__main__":
    pass
    #parser = OptionParser(description='Pectoralis Segmentation')
    parser = argparse.ArgumentParser(description='Pectoralis Segmentation')
    parser.add_argument('-operation', dest='operation', help='TRAIN, TEST',type=str, required=True)
    parser.add_argument('-data_path', dest='data_path', help='Path of CT',type=str, required=True)
    parser.add_argument('-test_selection', dest='test_selection', help='1,2,3',type=int, required=True)
    parser.add_argument('-model_path', dest='model_path', help='Path of Model',type=str, required=False)
    parser.add_argument('-model_path2', dest='model_path2', help='Path of Model2',type=str, required=False)
    parser.add_argument('-output_path', dest='output_path', help='Output path for saving results',type=str, required=False)
    parser.add_argument('-slice_num', dest='slice_num', help='Slice of the CT',type=str, required=False)
    parser.add_argument('-verbose', dest='verbose', help='Verbose prcoess',type=bool, required=False)
    args = parser.parse_args()

    if args.operation == 'TRAIN':
        # image_folder = '/Users/acil-user/Documents/1 - Rafael Moreta/ProyectBWH/ProjectData_clean/Cont_1_clean.nrrd'
        # label_folder = 'H'
        # output_folder = '/Users/acil-user/Documents/1 - Rafael Moreta/ProyectBWH/Results/'
        # nn_path = '/Users/acil-user/Documents/1 - Rafael Moreta/ProyectBWH/unet_GPU_multiclass_nc5_1400im_16ep_lr001.hdf5'
        # train(image_folder, label_folder, output_folder)
        pass

    elif args.operation == 'TEST':
        data_path = '/Users/acil-user/Documents/1 - Rafael Moreta/ProyectBWH/ProjectData_clean/Cont_1_clean.nrrd'
        output_path = '/Users/acil-user/Documents/1 - Rafael Moreta/ProyectBWH/Results/pred_labels.nrrd'
        model_path = '/Users/acil-user/Projects/unet_models/pectoralis_segmentation_nc5/unet_multiclass_nc5.hdf5'
        # model_path = '/Users/acil-user/Projects/unet_models/pectoralis_segmentation_nc2/unet_nc2.hdf5'

        slice_num = 90
        selection = 2
        verbose = 1

        segment_pectoralis(data_path, slice_num, selection, model_path, output_path, verbose)



