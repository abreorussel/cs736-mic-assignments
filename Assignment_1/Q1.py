from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

from config import *
from model import *


def display_image(image):
	plt.imshow(image)
	plt.show()
	# print(image)

def normalize_image(image):
	image = np.array(image)
	return ( image - np.min(image) ) / ( np.max(image) - np.min(image) )


def optimize(x_true ,y_observed, step_size=1e-2):
	x_estimate = y_observed.copy()
	log_posterior_values = []
	
	# Initial calculation
	log_posterior, log_posterior_grad = calculate_posterior(x_estimate, y_observed)
	log_posterior_values.append(np.sum(log_posterior))
	
	for it in range(1, iterations+1):
		# Gradient ascent: update x_estimate to maximize log posterior
		x_estimate += step_size * log_posterior_grad
		
		# Compute new posterior and gradient
		new_log_posterior, new_log_posterior_grad = calculate_posterior(x_estimate, y_observed)
		
		log_posterior_values.append(np.sum(new_log_posterior))
		
		# Compute RRMSE between current estimate and true noiseless image
		current_rrmse = rrmse(x_true, x_estimate)
		
		print(f"Iteration: {it}/{iterations} => Log Posterior: {np.sum(new_log_posterior):.4f}, RRMSE: {current_rrmse:.4f}")
		
		# Update gradient for next iteration
		log_posterior_grad = new_log_posterior_grad
	display_image(x_estimate)
	
	return x_estimate


if __name__ == "__main__":
	
	phantom = loadmat('data/assignmentImageDenoising_phantom.mat')
	imageNoiseless = phantom['imageNoiseless']
	imageNoisy = phantom['imageNoisy']
	print(f'Image Size : {imageNoisy.shape}')

	imageNoiseless, imageNoisy = normalize_image(imageNoiseless), normalize_image(imageNoisy)

	optimize(imageNoiseless, imageNoisy)





