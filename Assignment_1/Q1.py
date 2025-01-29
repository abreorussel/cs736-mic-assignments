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


def optimize(x_true ,y_observed, step_size=1e-1, iterations=100, alpha=0.5, gamma=0, likelihood="gaussian", prior="huber"):

	max_threshold = 0.5
	min_threshold = 0.01
	max_step_size = 1e-1
	min_step_size = 1e-5
	
	x_estimate = y_observed.copy()
	log_posterior_values = []
	
	# Initial calculation
	log_posterior, log_posterior_grad = calculate_posterior(x_estimate, y_observed, alpha, gamma, likelihood=likelihood, prior=prior)
	log_posterior_values.append(np.sum(log_posterior))
	
	for it in range(1, iterations+1):
		x_estimate += step_size * log_posterior_grad
		
		new_log_posterior, new_log_posterior_grad = calculate_posterior(x_estimate, y_observed, alpha, gamma, likelihood=likelihood, prior=prior)

		posterior_increase = new_log_posterior - log_posterior_values[-1]
		print(f"Posterior Increase: {posterior_increase}")
		if posterior_increase > max_threshold:
			step_size = min(step_size * 1.1 , max_step_size)
			print(f"Increasing step size to : {step_size}")
		elif posterior_increase < min_threshold:
			step_size = max(step_size * 0.9 , min_step_size)
			print(f"Decreasing step size to : {step_size}")

		else:
			print(f"Keeping step size same")

		log_posterior_values.append(np.sum(new_log_posterior))
		
		current_rrmse = rrmse(x_true, x_estimate)

		print(f"Iteration: {it}/{iterations} => Log Posterior: {np.sum(new_log_posterior):.4f}, RRMSE: {current_rrmse:.4f}")
		
		log_posterior_grad = new_log_posterior_grad
	display_image(x_estimate)
	
	return x_estimate


if __name__ == "__main__":
	
	phantom = loadmat('data/assignmentImageDenoising_phantom.mat')
	imageNoiseless = phantom['imageNoiseless']
	imageNoisy = phantom['imageNoisy']
	print(f'Image Size : {imageNoisy.shape}')

	imageNoiseless, imageNoisy = normalize_image(imageNoiseless), normalize_image(imageNoisy)

	optimize(imageNoiseless, imageNoisy, alpha=0.08, gamma=0.4, likelihood="gaussian", prior="quadratic")





