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


def optimize(x, y, step_size=1e-2):

	log_posterior_values = list()
	log_posterior, log_posterior_grad = calculate_posterior(x, y)
	log_posterior_values.append(np.sum(log_posterior))

	x_estimate = y.copy()

	for it in range(1, iterations+1):

		x_estimate = x_estimate + step_size*log_posterior_grad

		new_log_posterior, new_log_posterior_grad = calculate_posterior(x_estimate, y)
			
		# init_log_posterior = log_posterior.copy()

		log_posterior_values.append(np.sum(new_log_posterior))

		# grad_norm = np.linalg.norm(new_log_posterior)
		# if grad_norm < 0:
		# 	print(f"Converged at iteration {it} with gradient norm {grad_norm:.6e}")
		# 	break

		#  # Adaptive step size (optional)
		# if log_posterior_values[-1] < log_posterior_values[-2]:  # Log-posterior decreased
		# 	step_size *= 0.5
		# 	print(f"Step size reduced to {step_size:.6e}")
		# else:  # Log-posterior increased
		# 	step_size *= 1.1
		# 	step_size = min(step_size, 1e-1)  # Prevent step size from growing too large

		# log_posterior = new_log_posterior

			
		print(f"Iteration: {it} / {iterations} => log posterior : {np.sum(new_log_posterior)}  , RRMSE : {rrmse(x, x_estimate)}")


def optimize_new(x_true ,y_observed, step_size=1e-2):
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

	optimize_new(imageNoiseless, imageNoisy)





