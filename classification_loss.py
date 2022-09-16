import tensorflow as tf

class SSDLoss:
    def __init__(self, negative_mining_ratio = 3, alpha = 1.0):
        self.negative_mining_ratio = negative_mining_ratio
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Compute smooth L1 loss, see references.
        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
                contains the ground truth bounding box coordinates, where the last dimension
                contains `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box coordinates.
        Returns:
            The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        References:
            https://arxiv.org/abs/1504.08083
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        Compute the softmax log loss.
        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape (batch_size, #boxes, #classes)
                and contains the ground truth bounding box categories.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.
        Returns:
            The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        '''
        # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
        y_pred = tf.maximum(y_pred, 1e-15)
        # Compute the log loss
        log_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return log_loss

    def localization_loss(self, actual_deltas, pred_delta):
        """
        input:
            actual_deltas: (batch_size, number of prior boxes, [delta_cx, delta_cy, delta_w, delta_h, pos_cond])
            pred_delta = (batch_size, number of prior boxes, [delta_cx, delta_cy, delta_w, delta_h])

        outputs:
            loc_loss: Huber loss over all prior boxes with IOU > threshold (defined elsewhere)
            over ground label boxes
        """

        batch_size = tf.shape(pred_delta)[0]

        # huber loss over all preds
        # huber_loss = tf.keras.losses.Huber(
        #     reduction=tf.keras.losses.Reduction.NONE
        # )

        # localization_loss_for_all_priors = huber_loss(actual_deltas[:, :, :4], pred_delta)

        localization_loss_for_all_priors = self.smooth_L1_loss(actual_deltas[:, :, :4], pred_delta)

        localization_loss_for_all_priors = tf.cast(localization_loss_for_all_priors, tf.float32)

        # localization loss is only for default prior boxes with IOU > 0.5 over ground truth boxes
        localization_loss_for_all_priors = localization_loss_for_all_priors * actual_deltas[:, :, 4]

        total_pos_boxes = tf.reduce_sum(actual_deltas[:, :, 4:], axis=1)

        # If an image has no labels, the loc loss for that image should be 0.
        no_label_mask = tf.reduce_any( tf.not_equal(total_pos_boxes, tf.constant(0.)), axis = 1 )

        total_pos_boxes = tf.where(total_pos_boxes <= 0, 1e-6, total_pos_boxes)

        loss = self.alpha * tf.where(tf.transpose([no_label_mask]), 
                                                    localization_loss_for_all_priors / total_pos_boxes, 
                                                    tf.constant(0.))

        return tf.reduce_sum(loss, axis = 1) * tf.cast(batch_size, dtype=tf.float32)

    def confidence_loss(self, actual_labels, pred_labels):
        """
        inputs:
            actual_labels = (batch_size, number of prior boxes, total labels)
            pred_labels = (batch_size, number of prior boxes, total labels)
        
        outputs:
            conf_loss = loss per class
        """

        batch_size = tf.shape(pred_labels)[0]

        # categorical_cross_entropy = tf.losses.CategoricalCrossentropy(
        #     reduction=tf.losses.Reduction.NONE
        # )

        # confidence_loss_for_all = categorical_cross_entropy(actual_labels, pred_labels)

        confidence_loss_for_all = self.log_loss(actual_labels, pred_labels)

        confidence_loss_for_all = tf.cast(confidence_loss_for_all, tf.float32)

        pos_cond = tf.reduce_any( tf.equal(actual_labels[..., 1:], tf.constant(1.0)), axis = 2)
        pos_mask = tf.cast(pos_cond, dtype=tf.float32)
        total_pos_boxes = tf.reduce_sum(pos_mask, axis=1)

        pos_loss = pos_mask * confidence_loss_for_all

        # hard negative mining
        # set positive cases to 0
        neg_cond = tf.reduce_any( tf.equal(actual_labels[..., :1], tf.constant(1.0)), axis = 2)
        
        confidence_loss_for_all = tf.where(neg_cond, confidence_loss_for_all, 
                                        tf.constant(0., tf.float32)
                                        )
        
        # If there are no positive positive boxes in the select top k
        neg_boxes_for_empty_images = 0
        total_neg_boxes = tf.cast(total_pos_boxes * self.negative_mining_ratio, tf.int32)
        no_neg_boxes_mask = tf.not_equal(total_neg_boxes, tf.constant(0))
        total_neg_boxes = tf.where(no_neg_boxes_mask, total_neg_boxes, tf.constant(neg_boxes_for_empty_images))

        # sort by positive example
        loss_sorted_indices = tf.argsort(confidence_loss_for_all, direction="DESCENDING")
        loss_sorted_rank = tf.argsort(loss_sorted_indices)

        neg_mining_cond = tf.less(loss_sorted_rank, tf.expand_dims(total_neg_boxes, axis=1))
        neg_mining_mask = tf.cast(neg_mining_cond, dtype=tf.float32)

        neg_loss = neg_mining_mask * confidence_loss_for_all
        # total_boxes = total_pos_boxes + tf.cast(total_neg_boxes, tf.float32)
        total_pos_boxes = tf.where(total_pos_boxes <= 0, 1e-6, total_pos_boxes)
        total_loss = (pos_loss + neg_loss) / tf.expand_dims(total_pos_boxes, axis = 1)

        # If an image has no labels, the conf loss for that image should be 0.
        # no_label_mask = tf.reduce_any( tf.not_equal(tf.expand_dims(total_pos_boxes, axis = 1), tf.constant(0.)), axis = 1 )

        return tf.reduce_sum(total_loss, axis = 1) * tf.cast(batch_size, dtype=tf.float32)
        # tf.where(tf.transpose([no_label_mask]), 
        #                             total_loss, 
        #                             tf.constant(0.))

    




