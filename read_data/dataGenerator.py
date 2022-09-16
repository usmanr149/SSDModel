import numpy as np
import keras
import tensorflow as tf

from getData import read_data, label_dimensions_normalized, resize_images_and_labels
from bbox_utils import match_priors_with_gt
from image_augmentations.augmentations import returnPatches, horizontalFlipImageAndLabels, verticalFlipImageAndLabels

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs,
                label_folder_path,
                image_folder_path, 
                prior_boxes,
                prior_boxes_point_form,
                batch_size = 8, 
                n_classes = 5, 
                image_height = 300,
                image_width = 300,
                normalize = True,
                shuffle = True,
                image_extension = '.png',
                training = True):
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.label_folder_path = label_folder_path
        self.image_folder_path = image_folder_path
        self.prior_boxes = prior_boxes
        self.prior_boxes_point_form = prior_boxes_point_form
        self.image_height = image_height
        self.image_width = image_width
        self.normalize = normalize
        self.image_extension = image_extension
        self.training = training
        self.on_epoch_end()

        """
        Inputs:
            list_IDs: name of files used to look data in label_folder_path and image_folder_path
            label_folder_path: path to where labels are stored, need to be in .txt format
            image_folder_path: path to where images are stored, need to be in png
            prior_boxes: precalculated prior boxes in (c_x, c_y, w, h)
            prior_boxes_point_form: precomputed prior boxes in (x_min, y_min, x_max, y_max)
            batch_size: int
            n_classes: number of classes in the dataset, don't include background

        """

    def __len__(self):
        return int( np.floor( len(self.list_IDs) / self.batch_size ) )

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, list_IDs_temp):

        X = np.empty([self.batch_size, self.image_height, self.image_width, 3])
        y_label = None
        y_loc = None

        for i, file_name in enumerate(list_IDs_temp):
            image, labelled_gt_box_coords = read_data(  file_name, 
                                        self.image_folder_path, 
                                        self.label_folder_path,
                                        self.image_extension
                                     )
            
            if self.training:
                if np.random.rand() > 0.5:
                    image, labelled_gt_box_coords = returnPatches(image, labelled_gt_box_coords)
                if np.random.rand() > 0.5:
                    image, labelled_gt_box_coords = horizontalFlipImageAndLabels(image, labelled_gt_box_coords)
                if np.random.rand() > 0.8:
                    image, labelled_gt_box_coords = verticalFlipImageAndLabels(image, labelled_gt_box_coords)

            # take care of images with no labels
            # if no label then the whole image is a background
            if len(labelled_gt_box_coords) == 0:
                labelled_gt_box_coords = [[0, 0, 0 , image.shape[1], image.shape[0]]]

            image, labelled_gt_box_coords = resize_images_and_labels(image, labelled_gt_box_coords, self.image_height, self.image_width)
            if self.normalize:
                X[i,] = image / self.image_width
            else:
                X[i,] = image / self.image_width

            labelled_gt_box_coords_normallized = label_dimensions_normalized(labelled_gt_box_coords, self.image_height, self.image_width)

            gt_labels = [l[0] for l in labelled_gt_box_coords_normallized]
            gt_boxes_normalized = [l[1:] for l in labelled_gt_box_coords_normallized]
            
            offset, one_hot_encoded_label = match_priors_with_gt(   
                                                            self.prior_boxes, 
                                                            self.prior_boxes_point_form, 
                                                            tf.constant([gt_boxes_normalized]), 
                                                            tf.constant([gt_labels]), 
                                                            number_of_labels = self.n_classes + 1, 
                                                            threshold = 0.5)

            if y_label == None:
                y_label = one_hot_encoded_label
                y_loc = offset
            else:
                y_label = tf.concat([y_label, one_hot_encoded_label], axis = 0)
                y_loc = tf.concat([y_loc, offset], axis = 0)

        return X, [y_loc, y_label]

if __name__ == '__main__':
    # dg = DataGenerator(list_IDs, 
    #                label_path,
    #                image_path, 
    #                prior_boxes,
    #                boxes)
    pass