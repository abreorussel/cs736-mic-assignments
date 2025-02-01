from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os

from config import *
from model import *


def display_image(image):
	plt.imshow(image)
	plt.show()
	# print(image)

def normalize_image(image):
	image = np.array(image)
	return ( image - np.min(image) ) / ( np.max(image) - np.min(image) )


def optimize(x_true ,y_observed, step_size=1e-2, iterations=150, alpha=0.5, gamma=0, likelihood="gaussian", prior="quadratic"):
	# print(f"ALpha : {alpha}")
	max_threshold = 2.5
	min_threshold = 0.01
	max_step_size = 1e-1
	min_step_size = 1e-5
	increase_factor = 1.1  # Increase step size by 5% if improvement is large\
	decrease_factor = 0.5
	
	x_estimate = y_observed.copy()
	log_posterior_values = []
	rrmse_values = list()
	
	# Initial calculation
	initial_log_posterior, log_posterior_grad = calculate_posterior(x_estimate, y_observed, alpha, gamma, likelihood=likelihood, prior=prior)
	log_posterior_values.append(initial_log_posterior)
	print("_________________________________________________________________________________")
	print(f"\nAlpha: {alpha:.4f} | Gamma: {gamma:.4f} | Prior: {prior} | Likelihood: {likelihood}")
	print(f"Initial RRMSE between Noiseless and Noisy Image: {rrmse(x_true, x_estimate):.4f}")
	
	for it in range(1, iterations+1):
		x_estimate += step_size * log_posterior_grad
		new_log_posterior, new_log_posterior_grad = calculate_posterior(x_estimate, y_observed, alpha, gamma, likelihood=likelihood, prior=prior)

		# if new_log_posterior/initial_log_posterior > 1:
		# 	step_size *= 1.1
		# else:
		# 	step_size *= 0.5

		percentage_change = (new_log_posterior - initial_log_posterior) / abs(initial_log_posterior)
		if new_log_posterior >= initial_log_posterior:
			step_size = (1 - percentage_change) * step_size
		else:
			step_size = (1 + percentage_change) * step_size

		initial_log_posterior = new_log_posterior.copy()

		log_posterior_values.append(new_log_posterior)
		
		current_rrmse = rrmse(x_true, x_estimate)

		# print(f"Iteration: {it}/{iterations} =>  RRMSE: {current_rrmse:.4f}")
		
		log_posterior_grad = new_log_posterior_grad

		if it == iterations: rrmse_values.append(current_rrmse)
		if step_size <= 1e-8 : break


	print(f"Post Denoising RRMSE:  {current_rrmse:.4f}")
	print("_________________________________________________________________________________")
	
	return x_estimate, new_log_posterior, current_rrmse


	
def grid_search(imageNoiseless, imageNoisy, gamma_start = 0, gamma_end=0, alpha_start=0, alpha_end=1, prior="quadratic", likelihood="gaussian"):
	rrmse = 100
	alpha_optimal = 0
	gamma_optimal = 0
	gamma = 0

	for alpha in np.linspace(alpha_start, alpha_end, 500):
		if prior != "quadratic":
			for gamma in np.linspace(gamma_start, gamma_end, 100):
				x_estimate, new_log_posterior, current_rrmse = optimize(imageNoiseless, imageNoisy, alpha = alpha, gamma = gamma, likelihood = likelihood, prior = prior)
				if current_rrmse < rrmse:
					rrmse = current_rrmse
					alpha_optimal = alpha
					gamma_optimal = gamma
		else:
			x_estimate, new_log_posterior, current_rrmse = optimize(imageNoiseless, imageNoisy, alpha = alpha, gamma = gamma, likelihood = likelihood, prior = prior)
			if current_rrmse < rrmse:
				rrmse = current_rrmse
				alpha_optimal = alpha
				gamma_optimal = gamma
	

	print(f"############################# Optimal Values #################################")
	print(f'prior : {prior} | alpha : {alpha_optimal} | gamma : {gamma_optimal} | log posterior : {new_log_posterior} | RRMSE : {rrmse}')


if __name__ == "__main__":
	
	brain_mri = loadmat('data/assignmentImageDenoising_brainMRIslice.mat')
	print(brain_mri.keys())
	imageNoiseless = brain_mri['brainMRIsliceOrig']
	imageNoisy = brain_mri['brainMRIsliceNoisy']
	print(f'Image Size : {imageNoisy.shape}')

	results_folder = "results/Q1"
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print(f'Folder "{results_folder}" created.')

	optimize(imageNoiseless, imageNoisy, alpha=0.142285, gamma=0.04, likelihood="gaussian", prior="quadratic")
	# grid_search(imageNoiseless, imageNoisy, gamma_start = 1, gamma_end=10, alpha_start=0, alpha_end=1, prior="huber", likelihood="gaussian")
	# grid_search(imageNoiseless, imageNoisy, gamma_start = 1, gamma_end=10, alpha_start=0, alpha_end=1, prior="huber", likelihood="gaussian")


