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


def optimize(x_true ,y_observed, step_size=1e-2, iterations=150, alpha=0.5, gamma=0, likelihood="gaussian", prior="quadratic"):
	print(prior)
	max_threshold = 2.5
	min_threshold = 0.001
	max_step_size = 1e-1
	min_step_size = 1e-5
	
	x_estimate = y_observed.copy()
	log_posterior_values = []
	rrmse_values = list()
	
	# Initial calculation
	initial_log_posterior, log_posterior_grad = calculate_posterior(x_estimate, y_observed, alpha, gamma, likelihood=likelihood, prior=prior)
	log_posterior_values.append(initial_log_posterior)
	
	for it in range(1, iterations+1):
		x_estimate += step_size * log_posterior_grad
		
		new_log_posterior, new_log_posterior_grad = calculate_posterior(x_estimate, y_observed, alpha, gamma, likelihood=likelihood, prior=prior)

		# posterior_increase = new_log_posterior - log_posterior_values[-1]
		# print(f"Posterior Increase: {posterior_increase}")
		# if posterior_increase > max_threshold:
		# 	step_size = min(step_size * 1.05 , max_step_size)
		# 	print(f"Increasing step size to : {step_size}")
		# elif posterior_increase < min_threshold:
		# 	step_size = max(step_size * 0.9 , min_step_size)
		# 	print(f"Decreasing step size to : {step_size}")

		# else:
		# 	print(f"Keeping step size same")


		if new_log_posterior/initial_log_posterior > 1:
			step_size *= 1.1
		else:
			step_size *= 0.5

		initial_log_posterior = new_log_posterior.copy()

		log_posterior_values.append(new_log_posterior)
		
		current_rrmse = rrmse(x_true, x_estimate)

		# print(f"Iteration: {it}/{iterations} => Log Posterior: {np.sum(new_log_posterior):.4f}, RRMSE: {current_rrmse:.4f}")
		
		log_posterior_grad = new_log_posterior_grad

		if it == iterations: rrmse_values.append(current_rrmse)
		if step_size <= 1e-8 : break


	print(f"Log Posterior: {new_log_posterior:.4f}, RRMSE: {current_rrmse:.4f} Alpha: {alpha} Gamma: {gamma} Prior: {prior}")
	
	return x_estimate, new_log_posterior, current_rrmse


	
def grid_search(imageNoiseless, imageNoisy, gamma_start = 0, gamma_end=0, alpha_start=0, alpha_end=1, prior="quadratic", likelihood="gaussian"):
	# stats = list()
	rrmse = 100
	alpha_optimal = 0
	gamma_optimal = 0
	gamma = 0
	for alpha in np.linspace(alpha_start, alpha_end, 10):
		if prior != "quadratic":
			for gamma in np.linspace(gamma_start, gamma_end, 10):
				x_estimate, new_log_posterior, current_rrmse = optimize(imageNoiseless, imageNoisy, alpha, gamma, likelihood, prior)
				print(current_rrmse)
				if current_rrmse < rrmse:
					rrmse = current_rrmse
					alpha_optimal = alpha
					gamma_optimal = gamma
		else:
			print(f"ELse : {prior}")
			x_estimate, new_log_posterior, current_rrmse = optimize(imageNoiseless, imageNoisy, alpha, gamma, likelihood, prior)
			# print(current_rrmse)
			if current_rrmse < rrmse:
				rrmse = current_rrmse
				alpha_optimal = alpha
				gamma_optimal = gamma

	print(f"############################# Optimal Values #################################")
	print(f'prior : {prior} | alpha : {alpha_optimal} | gamma : {gamma_optimal} | log posterior : {new_log_posterior} | RRMSE : {rrmse}')


				 

				

if __name__ == "__main__":
	
	phantom = loadmat('data/assignmentImageDenoising_phantom.mat')
	imageNoiseless = phantom['imageNoiseless']
	imageNoisy = phantom['imageNoisy']
	print(f'Image Size : {imageNoisy.shape}')

	imageNoiseless, imageNoisy = normalize_image(imageNoiseless), normalize_image(imageNoisy)

	# optimize(imageNoiseless, imageNoisy, alpha=0.0875, gamma=0.5, likelihood="gaussian", prior="quadratic")
	grid_search(imageNoiseless, imageNoisy, gamma_start = 1, gamma_end=10, alpha_start=0, alpha_end=1, prior="quadratic", likelihood="gaussian")




