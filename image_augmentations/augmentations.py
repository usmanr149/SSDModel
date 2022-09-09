import os
import glob

import cv2
import numpy as np

def calculateArea(min_x, min_y, max_x, max_y):
    return (max_x - min_x) * (max_y - min_y)    

def checkOverLap(min_x1, min_y1, max_x1, max_y1, min_x2, min_y2, max_x2, max_y2, perecent_overlap = 0.2):

    """
    Input: (min_x,min_y)--|
            |             |
            |             |
            |--max_x,max_y|
            min_x1, min_y1, max_x1, max_y1 of first rectangle
            min_x2, min_y2, max_x2, max_y2 of second rectable
    Output: bool
        return False if the first rectangles is > percent_overlap with in the second overlap
    """

    min_x = max(min_x1, min_x2)
    min_y = max(min_y1, min_y2)
    
    max_x = min(max_x1, max_x2)
    max_y = min(max_y1, max_y2)
    
    if max_x - min_x < 0 or max_y - min_y < 0:
        return False
    bbox_area = calculateArea(min_x1, min_y1, max_x1, max_y1)
    intersection_area = calculateArea(min_x, min_y, max_x, max_y)
    if intersection_area / bbox_area > perecent_overlap:
        return True
    return False

def create_patches(title, image_path, label_path, image_dest = '', label_dest = ''):

    image_path = os.path.join(image_path, title + '.png')
    label_path = os.path.join(label_path, title + '.txt')

    image = cv2.imread(image_path)

    label = open(label_path)

    f = label.readlines()

    boxes_in_bounds = []

    # from https://stackoverflow.com/questions/31968588/extract-a-patch-from-an-image-given-patch-center-and-patch-scale
    # define some values
    patch_center = np.array([np.random.randint(400, 500), np.random.randint(400, 500)])
    patch_scale = np.random.uniform()

    while patch_scale < 0.3:
        patch_scale = np.random.uniform()

    # calc patch position and extract the patch
    smaller_dim = np.min(image.shape[0:2])
    patch_size = int(patch_scale * smaller_dim)
    patch_x = max(0, int(patch_center[0] - patch_size / 2.))
    patch_y = max(0, int(patch_center[1] - patch_size / 2.))

    for box in f:
        label_name, label_num, min_x, min_y, max_x, max_y = box.split(',')
        min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
        if checkOverLap(min_x, min_y, max_x, max_y, patch_x, patch_y, patch_x+patch_size, patch_y+patch_size) == False:
            continue
        else:
            min_x, min_y, max_x, max_y = max(0, int(min_x) - patch_x), max(0, int(min_y) - patch_y), \
            min(patch_size, int(max_x) - patch_x), min(patch_size, int(max_y) - patch_y)

            boxes_in_bounds.append([label_name, label_num, min_x, min_y, max_x, max_y])

    label_dest = os.path.join(label_dest, '{0}_{1}_{2}_{3}.txt'.format(title, 
                                            patch_center[0], 
                                            patch_center[1], 
                                            str(round(patch_scale, 2)).replace('.', '')))

    print(label_dest)

    with open(label_dest, 'w') as f:
        for line in boxes_in_bounds:
            s = ','.join([str(l) for l in line])
            f.write(f"{s}\n")

    patch_image = image[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size]

    image_dest = os.path.join(image_dest, '{0}_{1}_{2}_{3}.png'.format(title, 
                                            patch_center[0], 
                                            patch_center[1], 
                                            str(round(patch_scale, 2)).replace('.', '')))

    cv2.imwrite(image_dest, patch_image)

    return

if __name__ == '__main__':
    image_path = '/home/usman/workspace/fisheye_images/rgb_images/'
    label_path = '/home/usman/workspace/fisheye_images/box_2d_annotations'

    image_dest = '/home/usman/workspace/fisheye_images/train_images'
    label_dest = '/home/usman/workspace/fisheye_images/train_labels'

    file_names = glob.glob('/home/usman/workspace/fisheye_images/train_labels/*.txt')

    # list_IDs = [f.split('/')[-1].replace('.txt', '') for f in file_names]

    list_IDs = ['07600_RV']

    images_processed = 0

    for title in list_IDs:
        for _ in range(5):
            create_patches(title, image_path, label_path)#, image_dest, label_dest) 
        images_processed+=1
        if images_processed > 1000:
            break


