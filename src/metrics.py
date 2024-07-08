import rawpy
import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def calc_mean(imgs):
    """
    calculates the mean across all time stamps of the images with a specific filter
    args:
        imgs(np.ndarray): the images separated into rgb vals, whos means you are trying to 
        calculate.
    output:
        mean_imgs(np.ndarray): the mean value of images in relation to their bayer pattern
        filters. size should be (x dimension * y dimension * r g b)
    """
    
    
    # Calculate the mean across the time_stamps axis
    mean_imgs = np.mean(imgs, axis=3)
    
    return mean_imgs
    

def calc_var(imgs):
    """
    calculates the variance across all time stamps of the images with a specific filter
    args:
        imgs(np.ndarray): the images separated into rgb vals, whos variance you are trying to 
        calculate.
    output:
        var_imgs(np.ndarray): the variance value of images in relation to their bayer pattern
        filters. size should be (x dimension * y dimension * r g b)
    """

    var_imgs = np.var(imgs, axis=3, ddof=1)
    
    return var_imgs

def fit_linear_polynom_to_variance_mean(mean, var,th=200):
    """
    finds the polyfit between mean and variance which you calculate in the previous functions, 
    mean and var.
    
    mean(np.ndarray): the mean of the img filtered into rgb values - #(M, N, Num_channel, Num_gain)
    var(np.ndarray): the variance of the img filtered into rgb values - #(M, N, Num_channel, Num_gain)
    
    output:
          gain(nd.array): the slope of the polynomial fit. Should be of shape (Num_channel,Num_gain) for our data
          delta(nd.array): the y-intercept of the polynomial fit. Should be of shape (Num_channel,Num_gain) for our data
    """
    if mean.shape != var.shape:
        raise ValueError("Mean and variance arrays must have the same shape.")
    
    M, N, num_channels, num_gains = mean.shape
    
    gain = np.zeros((num_channels, num_gains))
    delta = np.zeros((num_channels, num_gains))
    
    for channel in range(num_channels):
        for gain_idx in range(num_gains):
            # Select mean and variance for current channel and gain index
            mean_channel_gain = mean[:, :, channel, gain_idx].flatten()
            var_channel_gain = var[:, :, channel, gain_idx].flatten()
            
            # Exclude very small variances to avoid numerical instability
            valid_indices = np.where(var_channel_gain > th)[0]
            mean_valid = mean_channel_gain[valid_indices]
            var_valid = var_channel_gain[valid_indices]
            
            if len(mean_valid) == 0 or len(var_valid) == 0:
                # Handle case where there are no valid data points after thresholding
                # You might want to log this or handle it differently based on your application
                continue
            
            # Fit a linear polynomial (degree 1) to mean (x-axis) vs. variance (y-axis)
            coeffs = np.polyfit(mean_valid, var_valid, 1)
            
            # Coeffs[0] is the slope (gain), coeffs[1] is the intercept (delta)
            gain[channel, gain_idx] = coeffs[0]
            delta[channel, gain_idx] = coeffs[1]
    
    return gain, delta
    raise NotImplementedError

def fit_linear_polynom_to_read_noise(delta, gain):
    """
    finds the polyfit between mean and variance which you calculate in the previous functions, 
    mean and var.
    
    sigma(np.ndarray): the total read noise filtered into rgb values - #(Num_Channel,Num_gain)
    gain(np.ndarray): the estimated camera gain filtered into rgb values - #(Num_Channel,Num_gain)
    
    output:
          sigma_read(np.ndarray): the slope of the linear fit - #(Num_Channel)
          sigma_ADC(np.ndarray): the y-intercept of the linear fit - #(Num_Channel)
    

    if delta.shape != gain.shape:
        raise ValueError("Delta and gain arrays must have the same shape.")
    
    num_channels = delta.shape[0]
    num_sensitivities = delta.shape[1]
    
    sigma_read = np.zeros(num_channels)
    sigma_ADC = np.zeros(num_channels)
    
    for channel in range(num_channels):
        # Fit a linear polynomial (degree 1) to mean (gain) vs. variance (delta)
        coeffs = np.polyfit(gain[channel], delta[channel], 1)
        
        # Coeffs[0] is the slope (sigma_read), coeffs[1] is the intercept (sigma_ADC)
        sigma_read[channel] = coeffs[0]
        sigma_ADC[channel] = coeffs[1]
    
    return sigma_read, sigma_ADC
    #raise NotImplementedError
    if delta.shape[0] < delta.shape[1]:
        delta = delta.T
    if gain.shape[0] < gain.shape[1]:
        gain = gain.T
    

    num_channels = delta.shape[1]
    

    sigma_read = np.zeros(num_channels)
    sigma_ADC = np.zeros(num_channels)
    

    for channel in range(num_channels):
        # Use numpy's lstsq for linear regression
        A = np.vstack([gain[:, channel], np.ones(len(gain[:, channel]))]).T
        try:
            slope, intercept = np.linalg.lstsq(A, delta[:, channel], rcond=None)[0]
        except np.linalg.LinAlgError:
            # Handle the case where lstsq fails (e.g., due to insufficient data)
            print(f"Warning: Linear regression failed for channel {channel}. Skipping.")
            continue
        
        # Store results
        sigma_read[channel] = slope
        sigma_ADC[channel] = intercept
    
    return sigma_read, sigma_ADC"""
    #raise NotImplementedError
    sigma_read = np.zeros(3)
    sigma_ADC = np.zeros(3)
    for c in range(3):
        p = np.polyfit(gain[c,:]**2, delta[c,:], 1)
        sigma_read[c] = p[0]
        sigma_ADC[c] = p[1]
    return sigma_read, sigma_ADCv

    
    
def calc_SNR_for_specific_gain(mean,var):

    """
    Calculate the SNR (mean / stddev) vs. the mean pixel intensity for a specific gain setting. You will need to bin the mean values into the range [0,255] so that you can compute SNR for a discrete set of values. 
    
    mean(np.ndarray): the mean of the img filtered into rgb values - #(M, N, Num_gain)
    var(np.ndarray): the variance of the img filtered into rgb values - #(M, N, Num_gain)
    
    output:
          SNR(np.ndarray): the computed SNR vs. mean of the captured image dataset - #(255, Num_gain)
    """
    assert mean.shape == var.shape, "Mean and variance arrays must have the same shape"
    
    # Compute the standard deviation from the variance
    stddev = np.sqrt(var)
    
    # Initialize an array to store the SNR values
    Num_gain = mean.shape[-1]
    SNR = np.zeros((256, Num_gain))
    
    # Compute the SNR for each bin in the range [0, 255]
    for gain in range(Num_gain):
        for i in range(256):
            # Select pixels with mean values in the current bin
            mask = (mean[..., gain] >= i) & (mean[..., gain] < i + 1)
            
            # Calculate the mean and standard deviation for the current bin
            bin_mean = np.mean(mean[mask, gain])
            bin_stddev = np.mean(stddev[mask, gain])
            
            # Compute the SNR (handle the case where bin_stddev is zero)
            if bin_stddev > 0:
                SNR[i, gain] = bin_mean / bin_stddev
            else:
                SNR[i, gain] = 0
    
    return SNR
    
    raise NotImplementedError