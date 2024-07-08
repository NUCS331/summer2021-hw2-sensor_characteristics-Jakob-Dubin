import rawpy
import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def crop(imgs, xDimMin, xDimMax, yDimMin, yDimMax):
    """
    this crops the image defined by the following arguments
    
    Hint: This should be a one liner ! DO NOT USE for loop
    
    args:
        imgs(np.ndarray): numpy array of all images to be cropped
        xDimMin(int): this is the x Dimension minimum
        xDimMax(int): this is the x Dimension maximum
        yDimMin(int): this is the y Dimension minimum
        yDimMax(int): this is the y Dimension maximum
        
    returns:
        images (np.ndarray): cropped version of all the images
    """
    return imgs[xDimMin:xDimMax, yDimMin:yDimMax, :, :]
    raise NotImplementedError

def channel_filter(images):
    #raise NotImplementedError
    """
    this filters the image to a specific channel of the bayers pattern.

    BAYER PATTERN:
    Here is a wikipedia page of the Bayer Filter:
        https://en.wikipedia.org/wiki/Bayer_filter
    
    Quick runthrough, what the Bayer pattern/filter is arranging 3 colors, green; red; blue,
    in a pattern to filter light which allows for the cameras to capture the colored images.
    The pattern contains 50% green, 25% red, and 25% blue. So what this channel filter function
    does is breaks an image into the separate filters created by the camera, and we can do
    analysis on each of these filters!
    
    Hint: You have 50% green pixels. Just take every second here to keep the same image dimension
    as for the R and B channel

    EXAMPLE:
    If we look at the bayer pattern, every second pixel for every other row is red, so we can say
    the x dimension be shifted by 1 over to be able to look at every red pixel, and look at every
    other row.
    
    args:
        imgs(np.ndarray): numpy array of all images to be filtered
    
    returns:
        filteredImages(np.ndarray): filtered version of the image, whose dimensions are 
        (X,Y, RGB, timestamp of image)
    

    if imgs.ndim == 2:
        # Single grayscale image
        imgs = imgs[np.newaxis, :, :]  # Add a dummy timestamp dimension
    elif imgs.ndim == 3:
        # Multiple images or a single color image
        if imgs.shape[-1] == 3:
            raise ValueError("Expected grayscale images for Bayer pattern, but found color channels.")
        # Add a dummy timestamp dimension if necessary
        if imgs.shape[0] in [1, 3]:  # Potentially single image with channels or timestamps
            imgs = imgs[np.newaxis, :, :]

    height, width, num_timestamps = imgs.shape
    
    # Initialize arrays for each channel
    red_channel = np.zeros_like(imgs)
    green_channel = np.zeros_like(imgs)
    blue_channel = np.zeros_like(imgs)
    
    # Extract the red channel (odd rows, even columns)
    red_channel[1::2, ::2, :] = imgs[1::2, ::2, :]
    
    # Extract the blue channel (even rows, odd columns)
    blue_channel[::2, 1::2, :] = imgs[::2, 1::2, :]
    
    # Extract the green channel
    green_channel[::2, ::2, :] = imgs[::2, ::2, :]  # Green pixels in even rows and even columns
    green_channel[1::2, 1::2, :] = imgs[1::2, 1::2, :]  # Green pixels in odd rows and odd columns
    
    # Stack the channels to get the filtered image
    filtered_images = np.stack((red_channel, green_channel, blue_channel), axis=-2)
    
    return filtered_images"""

    if images.ndim != 4:
        print("Input must be a 4D array with shape (height, width, num_images, num_channels).")
        return np.array([])

    height, width, num_images, num_channels = images.shape

    # Calculate new dimensions (height and width should be halved)
    new_height = height // 2
    new_width = width // 2

    # Initialize the output array for the RGB channels
    filtered_images = np.zeros((new_height, new_width, 3, num_images, num_channels), dtype=images.dtype)

    for img_index in range(num_images):
        for chan_index in range(num_channels):
            img = images[:, :, img_index, chan_index]

            # Extracting red channel
            red_channel = img[0::2, 1::2]  # Red pixels are at even rows and odd columns
            
            # Extracting green channel
            green_channel_1 = img[0::2, 0::2]  # Green pixels are at even rows and even columns
            green_channel_2 = img[1::2, 1::2]  # Green pixels are also at odd rows and odd columns
            green_channel = (green_channel_1 + green_channel_2) / 2
            
            # Extracting blue channel
            blue_channel = img[1::2, 0::2]  # Blue pixels are at odd rows and even columns

            # Assign the channels to the output array
            filtered_images[:, :, 0, img_index, chan_index] = red_channel
            filtered_images[:, :, 1, img_index, chan_index] = green_channel
            filtered_images[:, :, 2, img_index, chan_index] = blue_channel

    return filtered_images

    raise NotImplementedError
