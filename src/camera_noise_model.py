import numpy as np

def simulate_noisy_images(images,gain,sigma_read,sigma_adc,fwc,num_ims=20):
    """
    simulate a set of images captured according to a camera noise model using the parameters estimated from the homework. 
    The input array should be signed integer, representing the number of photons detected at each pixel.  
        
    args:
        images(np.ndarray): (H, W, #colors) array of pixel intensities without noise
        gain(np.ndarray): (#sensitivity) array of estimated camera gains
        sigma2_read(np.ndarray): (#colors) the estimated read noise variance
        sigma2_adc(np.ndarray): (#colors) the estimated adc noise variance
        fwc: the estimated full well capactiy of the sensor
        num_ims: the number of noisy images to simulate 
    output:
        noisy_images(np.ndarray): (H, W, #colors, #images, #sensitivity)
    """
    
    H, W, num_colors = images.shape
    num_sensitivity = len(gain)

    # Initialize the noisy images array
    noisy_images = np.zeros((H, W, num_colors, num_ims, num_sensitivity))

    for s in range(num_sensitivity):
        for c in range(num_colors):
            for n in range(num_ims):
                # Apply Poisson noise
                photon_noise = np.random.poisson(images[:, :, c])
                photon_noise = np.minimum(photon_noise, fwc)
                
                # Scale by gain
                signal = photon_noise * gain[s]
                
                # Add read noise and ADC noise
                read_noise = np.random.normal(0, sigma_read[c] * gain[s], size=(H, W))
                adc_noise = np.random.normal(0, sigma_adc[c], size=(H, W))
                
                noisy_images[:, :, c, n, s] = signal + read_noise + adc_noise

    return np.uint8(noisy_images)