
import tensorflow as tf
import numpy as np

# feature_map_shapes = [38, 19, 10, 5, 3, 1]
# aspect_ratios = [
#     [1, 2, 0.5],
#     [1, 2, 3, 0.5, 0.333],
#     [1, 2, 3, 0.5, 0.333],
#     [1, 2, 3, 0.5, 0.333],
#     [1, 2, 0.5],
#     [1, 2, 0.5]
# ]

def calculate_scale_of_default_boxes(k, m, s_max = 0.9, s_min = 0.2):
    """
    m = number_of_feature_maps
    s_k = s_min + (s_max - s_min) * (k - 1)/(m - 1)
    width_k = s_k * sqrt(aspect_ratio)
    height_k = s_k / sqrt(aspect_ratio)
    """
    return s_min + (s_max - s_min) * (k - 1) / (m - 1)

def generate_default_boxes(feature_map_shapes, number_of_feature_maps, aspect_ratios):
    """
    feature map shapes for VGG: [38, 19, 10, 5, 3, 1]
    """

    assert len(feature_map_shapes) == number_of_feature_maps, 'number of feature maps needs to be {0}'.format(len(feature_map_shapes))
    assert len(feature_map_shapes) == len(aspect_ratios), 'Need aspect ratios for all feature maps'

    prior_boxes = []

    for k, f_k in enumerate(feature_map_shapes):
        s_k = calculate_scale_of_default_boxes(k, m = number_of_feature_maps)
        s_k_prime = np.sqrt(s_k * calculate_scale_of_default_boxes(k + 1, m = 6))
        for i in range(f_k):
            for j in range(f_k):
                cx = (i + 0.5) / f_k
                cy = (j + 0.5) / f_k
                prior_boxes.append([cx, cy, s_k_prime, s_k_prime])

                for ar in aspect_ratios[k]:
                    # height, width for numpy
                    prior_boxes.append([cx, cy, s_k*np.sqrt(ar), s_k/np.sqrt(ar)])

    prior_boxes = tf.convert_to_tensor(prior_boxes, dtype=tf.float32)
    return tf.clip_by_value(prior_boxes, clip_value_min = 0., clip_value_max = 1.)


# Adapted from https://gist.github.com/escuccim/d0be49ccfc6084cdc784a67339f130dd
def box_overlap_iou(boxes, gt_boxes):
    """
    Args:
        boxes: shape (1, total boxes, x_min, y_min, x_max, y_max)
        gt_boxes: shape (1, total label, x_min  y_min, x_max, y_max)

    Returns:
        Tensor with shape (batch_size, total boxes, total label)
    """
    box_x_min, box_y_min, box_x_max, box_y_max = tf.split(boxes, 4, axis = 1)
    gt_boxes_x_min, gt_boxes_y_min, gt_boxes_x_max, gt_boxes_y_max = tf.split(gt_boxes, 4, axis = 2)

    # From https://www.tensorflow.org/api_docs/python/tf/transpose
    intersection_x_min = tf.maximum(box_x_min, tf.transpose(gt_boxes_x_min, perm=[0, 2, 1]))
    intersection_y_min = tf.maximum(box_y_min, tf.transpose(gt_boxes_y_min, perm=[0, 2, 1]))

    intersection_x_max = tf.minimum(box_x_max, tf.transpose(gt_boxes_x_max, perm=[0, 2, 1]))
    intersection_y_max = tf.minimum(box_y_max, tf.transpose(gt_boxes_y_max, perm=[0, 2, 1]))

    # need to take care of boxes that don't overlap at all
    intersection_area = tf.maximum(intersection_x_max - intersection_x_min, 0) * tf.maximum(intersection_y_max - intersection_y_min, 0)

    boxes_areas = (box_x_max - box_x_min) * (box_y_max - box_y_min)
    gt_box_areas = (gt_boxes_x_max - gt_boxes_x_min) * (gt_boxes_y_max - gt_boxes_y_min)

    union = (boxes_areas + tf.transpose(gt_box_areas, perm=[0, 2, 1])) - intersection_area

    return tf.maximum(intersection_area / union, 0)


def match_priors_with_gt(prior_boxes, boxes, gt_boxes, gt_labels, number_of_labels = 3, threshold = 0.5):
    
    """
    prior boxes: (1, 8732, c_x, c_y, w, h)
    boxes are box coordinate representation of prior: (1, 8732, x_min, y_min, x_max, y_max)
    gt_boxes need to be (1, number of labels, x_min, y_min, x_max, y_max)
    gt_labels need to be like 1, [1,2,3]

    0 is background, so the gt_labels is the number of labels in the dataset + 1
    class 0 is reserved.
    """

    # number of rows for the IOU map the is the number of gt_boxes
    IOU_map = box_overlap_iou(boxes, gt_boxes)

    # convert ground boxes labels to box label format
    gt_box_label = convert_to_centre_dimensions_form(gt_boxes)

    # select the box with the highest IOU
    # highest_overlap_idx = tf.math.argmax(IOU_map, axis = 0)

    # find the column idx with the highest IOU at each row
    max_IOU_idx_per_row = tf.math.argmax(IOU_map, axis = 2)
    # find the max value per row
    max_IOU_per_row = tf.reduce_max(IOU_map, axis = 2)

    # threshold IOU
    max_IOU_above_threshold = tf.greater(max_IOU_per_row, threshold)

    # map the gt boxes to the prior boxes with the highest overlap
    gt_box_label_map = tf.gather(gt_box_label, max_IOU_idx_per_row, batch_dims = 1)
    # get the offset, offcet (delta_cx, delta_cy, delta_width, delta_height)
    gt_box_label_map_offsets = calculate_offset_from_gt(gt_box_label_map, prior_boxes)
    # remove from gt_boxes_map where overlap with prior boxes is less than 0.5
    gt_boxes_map_offset_suppressed = tf.where( tf.expand_dims(max_IOU_above_threshold, -1),  
                                        gt_box_label_map_offsets, tf.zeros_like(gt_box_label_map))
    

    gt_labels_map = tf.gather(gt_labels, max_IOU_idx_per_row, batch_dims = 1)
    # suppress the label where IOU with the gt boxes is < 0.5
    gt_labels_map_suppressed = tf.where( max_IOU_above_threshold, 
                                        gt_labels_map, tf.zeros_like(gt_labels_map))
    gt_labels_one_hot_encoded = tf.one_hot(gt_labels_map_suppressed, number_of_labels)

    return gt_boxes_map_offset_suppressed, gt_labels_one_hot_encoded

def calculate_offset_from_gt(gt_boxes_mapped_to_prior, prior_boxes):
    return gt_boxes_mapped_to_prior - tf.expand_dims(prior_boxes, axis=0)

def convert_to_box_form(boxes):
    """
    Input:
        (number_of_labels, c_x, c_y, width, height)
    Output:
        (number_of_labels, x_min, y_min, x_max, y_max)
    """

    box_coordinates = tf.concat([   boxes[:, :2] - boxes[:, 2:] / 2, 
                                    boxes[:, :2] + boxes[:, 2:] / 2 ], 
                                    axis = 1)

    return tf.clip_by_value(box_coordinates, clip_value_min = 0., clip_value_max = 1.)

def convert_to_centre_dimensions_form(boxes):
    """
    Input:
        boxes: (1, number_of_labels, x_min, y_min, x_max, y_max)
    Output:
        (1, number_of_labels, c_x, c_y, width, height)
    """

    coordinates = tf.concat([
                [
                        (boxes[:, :, 0] + boxes[:, :, 2]) / 2, 
                        (boxes[:, :, 1] + boxes[:, :, 3]) / 2,
                        boxes[:, :, 2] - boxes[:, :, 0],
                        boxes[:, :, 3] - boxes[:, :, 1]
                ]], axis = 1)
    # need the output in the same format as input, could be imporived
    coordinates = tf.transpose(coordinates, perm=[1,2,0])
    return tf.clip_by_value(coordinates, clip_value_min = 0., clip_value_max = 1.)


