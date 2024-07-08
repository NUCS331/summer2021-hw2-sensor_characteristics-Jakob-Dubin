import rawpy
import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def plot_overlayed_hist(data,loc,sensitivity,size):
    x, y = loc
    h, w = size
    
    # Extract pixel intensities at the specified location and size
    pixel_data = data[y:y+h, x:x+w, :, :, :]  # Shape: (h, w, #colors, #images, #sensitivity)

    # Reshape to flatten dimensions except sensitivity
    pixel_data = pixel_data.reshape((h * w * pixel_data.shape[2], -1))  # Shape: (h*w*#colors, #images*#sensitivity)
    
    # Calculate number of sensitivities
    num_sensitivities = len(sensitivity)

    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms for each sensitivity setting
    for idx, sens in enumerate(sensitivity):
        hist_data = pixel_data[:, idx::num_sensitivities].ravel()  # Select every num_sensitivities-th column
        ax.hist(hist_data, bins=50, alpha=1, ec="k", density=True, histtype='stepfilled', label=f"Sensitivity {sens}")

 
    
    ax.set_title(f"x = {x} | y = {y}")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.legend()

    # Display the plot
    plt.show()


def get_pixel_location(img_shape,N_x,N_y):
    #raise NotImplementedError
    """
    
    Takes the shape of an image and number of to be gridded points in X and Y direction 
    to sample equally spaced points on the 2D-grid
    
    We want to exclude points at the boundaries.
    
    E.g., if our image has 100 x 100 and we want to sample 3 x 4 points we would do the following 
    
    25 50 75 for the x-coordinate
    and
    20 40 60 80 for the y-coordinate
    
    Those 2 vectors then need to converted into 2 matrices for X and Y positions (use meshgrid)
    
    the following numpy functions can come in handy to develop this function:
    
    1. np.arange
    2. np.meshgrid
    3. np.round to cast to integer values 
    4. np.astype(np.uint16) as we want integer values for the coordinates
    
    Input:
    
    Output:
    
    """
    height, width = img_shape[0], img_shape[1]

    # Calculate step sizes
    step_x = (width - 1) / (N_x + 1)
    step_y = (height - 1) / (N_y + 1)

    # Generate 1D arrays of x and y coordinates
    x_coords = np.arange(step_x, width, step_x, dtype=np.uint16)
    y_coords = np.arange(step_y, height, step_y, dtype=np.uint16)

    # Use meshgrid to create 2D matrices of x and y coordinates
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    return X, Y
    #raise NotImplementedError