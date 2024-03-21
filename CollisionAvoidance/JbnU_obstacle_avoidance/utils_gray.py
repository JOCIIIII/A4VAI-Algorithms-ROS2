import sys 
import cv2, os
import numpy as np

 

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 300, 300, 1
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    img = cv2.imread(os.path.join(data_dir, image_file.strip()),0)
   
    
    
    
    
    return  img



def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    image = np.expand_dims(image, axis=2)
    
 
    return image


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
   
    """
    Combine all preprocess functions into one
    """
   # image = crop(image)
    
    
    image = resize(image)

    #image = rgb2yuv(image)
    
    
    return image


def choose_image(data_dir, center, vx,vy,vz,vyaw):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    
    return load_image(data_dir, center), vx,vy,vz,vyaw


def random_flip(image, vx,vz,vyaw):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        vx = -vx
        vz = -vz
        vyaw = -vyaw
    return image, vx,vz,vyaw


def random_translate(image, vx,vy,vz,vyaw, range_x, range_y,range_z,range_yaw):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    trans_z = range_z * (np.random.rand() - 0.5)
    trans_yaw = range_yaw * (np.random.rand() - 0.5)
    vx += trans_x * 0.002
    vy += trans_y * 0.002
    vz += trans_z * 0.002
    vyaw += trans_yaw * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, vx,vy,vz,vyaw


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(data_dir, center, vx,vy,vz,vyaw, range_x=100, range_y=10,range_z = 10,range_yaw=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, vx,vy,vz,vyaw = choose_image(data_dir, center, vx,vy,vz,vyaw)

    #image, vx,vy,vz,vyaw = random_flip(image, vx,vy,vz,vyaw)
    #image, vx,vy,vz,vyaw  = random_translate(image, vx,vy,vz,vyaw, range_x, range_y,range_z, range_yaw)
    #image = random_shadow(image)
    #image = random_brightness(image)
    return image, vx,vy,vz,vyaw 


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    
    
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty([batch_size,4])
    
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center = image_paths[index]
            vx = steering_angles[index][0]
            vy = steering_angles[index][1]
            vz = steering_angles[index][2]
            vyaw = steering_angles[index][3]
            # argumentation
           
            image = load_image(data_dir, center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = [vx,vy,vz,vyaw]
            i += 1
            if i == batch_size:
                break
        yield images, steers


