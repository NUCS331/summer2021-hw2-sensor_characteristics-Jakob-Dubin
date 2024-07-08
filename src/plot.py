import rawpy
import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_with_colorbar(img,vmax=0):
    """
    args:
        vmax: The maximal value to be plotted
    """
    ax = plt.gca()
    if vmax == 0:
        im = ax.imshow(img, cmap='gray')
    else:
        im = ax.imshow(img, cmap='gray', vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    

def plot_input_histogram(imgs,sensitivity):
    """
    
    The imgs variable consists of 1 image captured per different camera sensitivity (ISO) settings. plot_input_histogram
    visualize the histograms for each image in a subplot fashion

    
    args:
        imgs(np.ndarray): 3-dimensional array containing one image per intensity setting (not all the 200)
    
    """

    num_sensitivities = len(sensitivity)
    
    
    # Ensure the number of sensitivity settings matches the third dimension of imgs
    if num_sensitivities != imgs.shape[2]:
        raise ValueError(f"Number of sensitivity settings ({num_sensitivities}) does not match the number of images ({imgs.shape[2]}).")
    
    # Create subplots
    cols = 3
    rows = int(np.ceil(num_sensitivities / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    
    for i in range(num_sensitivities):
        ax = axes[i]
        img = imgs[:, :, i]
        
        # Create the histogram for the image
        ax.hist(img.ravel(), bins=50, range=(0, 254), color='steelblue')
        # ax.hist(imgs[:,:,i].ravel(), bins= 50,range=(0,254))
        ax.set_title(f'Sensitivity Lvl {sensitivity[i]}')
        ax.set_xlim([0, 255])
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.grid(True)
    
    # Hide any extra subplots
    for i in range(num_sensitivities, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

    
    
        
def plot_histograms_channels(img,sensitivity):
    """
    
    Plots the histogram for each channel in a subplot (1 row, 3 cols)
    
    args:
        img(np.ndarray): The RGB image
        sensitivity(float): The gain settings of the img series
    
    """
    
    channels = ['Red', 'Green', 'Blue']
    
    # Check if the input image has three channels
    if img.shape[2] != 3:
        raise ValueError("The input image does not have three channels (RGB).")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, ax in enumerate(axes):
        channel_data = img[:, :, i]
        
        # Debug: Print channel statistics
        # print(f"Channel {channels[i]} - Min: {channel_data.min()}, Max: {channel_data.max()}, Mean: {channel_data.mean()}")
        
        ax.hist(channel_data.ravel(), bins=50, range=(0, 255), color='steelblue', alpha=0.6)
        ax.set_title(f'{channels[i]} Channel (Sensitivity: {sensitivity})')
        ax.set_xlim([0, 255])
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.grid(True)
    
    plt.suptitle(f'Histograms for Sensitivity lvl = {sensitivity}', fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
        
def plot_input_images(imgs,sensitivity):
    """
    
    The dataset consists of 1 image captured per different camera sensitivity (ISO) settings. Lets visualize a single image taken at each different sensitivity setting
    
    Hint: Use plot_with_colorbar. Use the vmax argument to have a scale to 255
    (if you don't use the vmax argument)
    
    args:
        imgs(np.ndarray): 3-dimensional array containing one image per intensity setting (not all the 200)
        sensitivity(np.ndarray): The sensitivy (gain) vector for the image database
    
    """
    num_sensitivities = len(sensitivity)
    
    # Ensure the number of sensitivity settings matches the third dimension of imgs
    if num_sensitivities != imgs.shape[2]:
        raise ValueError(f"Number of sensitivity settings ({num_sensitivities}) does not match the number of images ({imgs.shape[2]}).")
    
    # Create subplots
    cols = 3
    rows = int(np.ceil(num_sensitivities / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    for i in range(num_sensitivities):
        ax = axes[i]
        img = imgs[:, :, i]
        plt.sca(ax)
        plot_with_colorbar(img, vmax=255)
        ax.set_title(f'Sensitivity Level {sensitivity[i]}')
        ax.axis('off')
    
    # Hide any extra subplots
    for i in range(num_sensitivities, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

def plot_rgb_channel(img, sensitivity):

    "" 
    
    
    
    ""
    
    if img.shape[2] != 3:
        raise ValueError("The input image does not have three channels (RGB).")

    channels = ['Red', 'Green', 'Blue']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3):
        ax = axes[i]
        channel_img = img[:, :, i]
        im = ax.imshow(channel_img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'{channels[i]} Channel (Sensitivity: {sensitivity})')
        ax.axis('on')
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    # plt.tight_layout()
    plt.show()

def plot_images(data, sensitivity, statistic,color_channel):
    """
    this function should plot all 3 filters of your data, given a
    statistic (either mean or variance in this case!)

    args:

        data(np.ndarray): this should be the images, which are already
        filtered into a numpy array.

        statsistic(str): a string of either mean or variance (used for
        titling your graph mostly.)

    returns:

        void, but show the plots!

    """
    if statistic not in ['Mean', 'Variance', 'standard deviation']:
        raise ValueError("Statistic must be 'mean', 'variance', or 'std' (standard deviation).")

    if color_channel not in [0, 1, 2]:
        raise ValueError("Color channel must be 0 (R), 1 (G), or 2 (B).")

    if len(data.shape) != 4 or data.shape[2] != 3:
        raise ValueError("Data must be a 4-dimensional array with shape (H, W, 3, #sensitivity).")

    if len(sensitivity) != data.shape[3]:
        raise ValueError("Length of sensitivity array must match the number of sensitivity settings in the data.")

    num_sensitivities = len(sensitivity)
    cols = 3
    rows = (num_sensitivities + cols - 1) // cols  # Calculate the required number of rows for subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i in range(num_sensitivities):
        ax = axes[i]
        img_stack = data[:, :, color_channel, i]  # Extract images for the current sensitivity and color channel

        if img_stack.size == 0:
            ax.set_title(f'No data for Sensitivity {sensitivity[i]}')
            ax.axis('off')
            continue  # Skip if no data is present

        if statistic == 'Mean':
            img_stat = np.mean(img_stack, axis=0)  # Mean across the color channel
            title = 'Mean'
        elif statistic == 'Variance':
            img_stat = np.var(img_stack, axis=0)  # Variance across the color channel
            title = 'Variance'
        elif statistic == 'standard deviation':
            img_stat = np.std(img_stack, axis=0)  # Standard deviation across the color channel
            title = 'Standard Deviation'

        plt.sca(ax)
        plot_with_colorbar(img_stat, vmax=255)
        ax.set_title(f'{title} for Sensitivity {sensitivity[i]}')
        ax.axis('off')

    # Hide any extra subplots
    for i in range(num_sensitivities, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    
    
def plot_relations(means, variances, skip_pixel, sensitivity, color_idx):
    """
    this function plots the relationship between means and variance. 
    Because this data is so large, it is recommended that you skip
    some pixels to help see the pixels.

    args:
        means: contains the mean values with shape (200x300x3x6)
        variances: variance of the images (200x300x3x6)
        skip_pixel: amount of pixel skipped for visualization
        sensitivity: sensitivity array with 1x6
        color_idx: the color index (0 for red, 1 green, 2 for blue)

    returns:
        void, but show plots!
    """
    if color_idx not in [0, 1, 2]:
        raise ValueError("Color index must be 0 (Red), 1 (Green), or 2 (Blue).")

    if means.shape != variances.shape:
        raise ValueError("Means and variances arrays must have the same shape.")
    
    if len(sensitivity) != means.shape[3]:
        raise ValueError("Length of sensitivity array must match the number of sensitivity settings in the data.")

    num_sensitivities = len(sensitivity)
    color_map = ['Red', 'Green', 'Blue']
    color_name = color_map[color_idx]
    
    plt.figure(figsize=(15, 10))

    for i in range(num_sensitivities):
        mean_values = means[::skip_pixel, ::skip_pixel, color_idx, i].ravel()
        var_values = variances[::skip_pixel, ::skip_pixel, color_idx, i].ravel()

        plt.scatter(mean_values, var_values, alpha=0.5, label=f'Sensitivity {sensitivity[i]}')

    plt.xlabel(f'{color_name} Channel Mean Intensity')
    plt.ylabel(f'{color_name} Channel Variance')
    plt.title(f'{color_name} Channel: Mean vs Variance Relationship')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #raise NotImplementedError
        
def plot_mean_variance_with_linear_fit(gain,delta,means,variances,skip_points=50,color_channel=0):
    """
        this function should plot the linear fit of mean vs. variance against a scatter plot of the data used for the fitting 
        
        args:
        gain (np.ndarray): the estimated slopes of the linear fits for each color channel and camera sensitivity

        delta (np.ndarray): the estimated bias/intercept of the linear fits for each color channel and camera sensitivity

        means (np.ndarray): the means of your data in the form of 
        a numpy array that has the means of each filter.

        variances (np.ndarray): the variances of your data in the form of 
        a numpy array that has the variances of each filter.
        
        skip_points: how many points to skip so the scatter plot isn't too dense
        
        color_channel: which color channel to plot

    returns:
        void, but show plots!
    """
    if color_channel not in [0, 1, 2]:
        raise ValueError("Color index must be 0 (Red), 1 (Green), or 2 (Blue).")
    
    # Extract data for the specified color channel
    means_channel = means[:, :, color_channel]
    variances_channel = variances[:, :, color_channel]
    
    # Flatten the data arrays for easier manipulation
    means_flat = means_channel.flatten()
    variances_flat = variances_channel.flatten()
    
    # Calculate fitted values (var_fit = gain * means + delta)
    var_fit = gain[color_channel, 0] * means_flat + delta[color_channel, 0]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of means vs. variances
    plt.scatter(means_flat[::skip_points], variances_flat[::skip_points], alpha=0.5, label='Data Points')
    
    # Plot the linear fit
    plt.plot(means_flat, var_fit, color='red', linewidth=2, label='Linear Fit')
    
    # Add labels and title
    plt.xlabel('Mean')
    plt.ylabel('Variance')
    plt.title(f'Mean vs. Variance with Linear Fit (Channel: {color_channel})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_read_noise_fit(sigma_read, sigma_ADC, gain, delta, color_channel=0):
    """
        this function should plot the linear fit of read noise delta vs. gain plotted against the data used for the fitting 
        
        args:
        sigma_read (np.ndarray): the estimated gain-depdenent read noise for each color channel of the sensor 

        sigma_ADC (np.ndarray): the estimated gain-independent read noise for each color channel of the sensor

        gain (np.ndarray): the estimated slopes of the linear fits of mean vs. variance for each color channel and camera sensitivity

        delta (np.ndarray): the estimated bias/intercept of the linear fits of mean vs. variance for each color channel and camera sensitivity

        color_channel: which color channel to plot
        
    returns:
        void, but show plots!
    """
    
if color_channel not in [0, 1, 2]:
        raise ValueError("Color index must be 0 (Red), 1 (Green), or 2 (Blue).")

    num_sensitivities = len(sigma_read)
    color_map = ['Red', 'Green', 'Blue']
    color_name = color_map[color_channel]

    plt.figure(figsize=(10, 6))

    # Extract data for the specified color channel
    if isinstance(sigma_read, tuple):
        sigma_read_channel = sigma_read[color_channel]
    else:
        sigma_read_channel = sigma_read[:, color_channel]
        
    sigma_ADC_channel = sigma_ADC[:, color_channel]
    gain_channel = gain[:, color_channel]
    delta_channel = delta[:, color_channel]

    # Calculate fitted values (sigma_read = gain * sigma_ADC + delta)
    fitted_sigma_read = gain_channel * sigma_ADC_channel + delta_channel

    # Plot actual data points
    plt.scatter(sigma_ADC_channel, sigma_read_channel, alpha=0.7, label='Data Points')

    # Plot fitted line
    plt.plot(sigma_ADC_channel, fitted_sigma_read, color='red', linewidth=2, label='Fitted Line')

    # Add titles and labels
    plt.xlabel('Gain')
    plt.ylabel('Read Noise (Sigma)')
    plt.title(f'{color_name} Channel: Read Noise vs. Gain')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
