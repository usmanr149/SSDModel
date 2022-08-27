import numpy as np
import keras

from getData import read_data, label_dimensions_to_point_form
from bbox_utils import match_priors_with_gt

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs,
                label_folder_path,
                image_folder_path, 
                prior_boxes,
                prior_boxes_point_form,
                batch_size = 8, 
                n_classes = 11, 
                image_height = 300,
                image_width = 300,
                shuffle = True):
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
        self.on_epoch_end()

    def __len__(self):
        return int( np.floot( len(self.list_IDs) / self.batch_size ) )

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_temp = [self.list_Ids[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arrange(len(self.IDs))

        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, list_IDs_temp):

        X = np.empty([self.batch_size, self.image_height, self.image_width, 3])
        y_label = None
        y_loc = None

        for i, file_name in enumerate(list_IDs_temp):
            image, labels = read_data(  file_name, 
                                        self.image_path, 
                                        self.label_path, 
                                        self.image_height, 
                                        self.image_width)

            X[i,] = image

            labels = label_dimensions_to_point_form(labels, self.image_height, self.image_width)

            gt_labels = [l[0] for l in labels]
            gt_boxes = [l[1:] for l in labels]

            offset, one_hot_encoded_label = match_priors_with_gt(   
                                                            self.prior_boxes, 
                                                            self.prior_boxes_point_form, 
                                                            gt_boxes, 
                                                            gt_labels, 
                                                            number_of_labels = self.n_classes, 
                                                            threshold = 0.5)

            if y_label == None:
                y_label = one_hot_encoded_label
                y_loc = offset
            else:
                y_label = tf.concat([y_label, one_hot_encoded_label], axis = 1)
                y_loc = tf.concat([y_loc, offset], axis = 1)

        return X, [y_loc, y_label]




        