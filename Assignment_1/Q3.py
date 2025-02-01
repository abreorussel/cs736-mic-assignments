from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import mat73
import os

from config import *
from model import *

def normalize_image(image):
	image = np.array(image)
	return ( image - np.min(image) ) / ( np.max(image) - np.min(image) )

def display_image(image):
	plt.figure()
	plt.imshow(image)
	plt.show()
	# print(image)

def optimize(x_true ,y_observed, step_size=1e-2, iterations=200, alpha=0.5, gamma=0, likelihood="gaussian", prior="quadratic", print_log = True):
	
	x_estimate = y_observed.copy()
	log_posterior_values = list()
	
	# Initial calculation
	initial_log_posterior, log_posterior_grad = calculate_posterior(x_estimate, y_observed, alpha, gamma, likelihood=likelihood, prior=prior)
	if print_log:
		print("_________________________________________________________________________________")
		print(f"\nAlpha: {alpha:.4f} | Gamma: {gamma:.4f} | Prior: {prior} | Likelihood: {likelihood}")
		print(f"Initial RRMSE between Noiseless and Noisy Image: {rrmse(x_true, x_estimate):.4f}")
	
	for it in range(1, iterations+1):
		x_estimate += step_size * log_posterior_grad
		new_log_posterior, new_log_posterior_grad = calculate_posterior(x_estimate, y_observed, alpha, gamma, likelihood=likelihood, prior=prior)

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

		if step_size <= 1e-8 : break

	if print_log:
		print(f"Post Denoising RRMSE:  {current_rrmse:.4f}")
		print("_________________________________________________________________________________")
	
	return x_estimate, new_log_posterior, current_rrmse, log_posterior_values


	
def grid_search(imageNoiseless, imageNoisy, gamma_start = 0, gamma_end=0, alpha_start=0, alpha_end=1, prior="quadratic", likelihood="gaussian", mode ="optimize"):
	rrmse = 100
	alpha_optimal = 0
	gamma_optimal = 0
	gamma = 0.16
	# alpha = 0.6482

	for alpha in np.linspace(alpha_start, alpha_end, 200):
	# for gamma in np.linspace(gamma_start, gamma_end, 200):
		if mode != "optimize":
			for gamma in np.linspace(gamma_start, gamma_end, 10):
				x_estimate, new_log_posterior, current_rrmse,_ = optimize(imageNoiseless, imageNoisy, alpha = alpha, gamma = gamma, likelihood = likelihood, prior = prior)
				if current_rrmse < rrmse:
					rrmse = current_rrmse
					alpha_optimal = alpha
					gamma_optimal = gamma
		else:
			x_estimate, new_log_posterior, current_rrmse,_ = optimize(imageNoiseless, imageNoisy, alpha = alpha, gamma = gamma, likelihood = likelihood, prior = prior)
			if current_rrmse < rrmse:
				rrmse = current_rrmse
				alpha_optimal = alpha
				gamma_optimal = gamma
	

	print(f"############################# Optimal Values #################################")
	print(f'prior : {prior} | alpha : {alpha_optimal} | gamma : {gamma_optimal} | log posterior : {new_log_posterior:.4f} | RRMSE : {rrmse}')

def check_nearby_parameters(alpha =0 , gamma = 0, prior ="quadratic"):
	increased_alpha = alpha
	if 1.2 * alpha < 1:
		increased_alpha = alpha * 1.2
		
	if prior in ["square-l2", "l2"]:
		_, _, rrmse_optimal, _ = optimize(imageNoiseless, imageNoisy, alpha=alpha, gamma=gamma, likelihood="gaussian", prior=prior ,print_log =False)
		_, _, rrmse1, _ = optimize(imageNoiseless, imageNoisy, alpha=increased_alpha, gamma=gamma, likelihood="gaussian", prior=prior, print_log=False)
		_, _, rrmse2, _ = optimize(imageNoiseless, imageNoisy, alpha=0.8*alpha, gamma=gamma, likelihood="gaussian", prior=prior, print_log=False)
		
		print(f"prior : {prior} | alpha : {alpha}")
		print(f"Optimal RRMSE : {rrmse_optimal} ")
		print(f"1.2 times alpha RRMSE : {rrmse1} ")
		print(f"0.8 times alpha RRMSE : {rrmse2} ")
	else:
		_, _, rrmse_optimal, _ = optimize(imageNoiseless, imageNoisy, alpha=alpha, gamma=gamma, likelihood="gaussian", prior=prior, print_log=False)
		_, _, rrmse1, _ = optimize(imageNoiseless, imageNoisy, alpha=increased_alpha, gamma=gamma, likelihood="gaussian", prior=prior, print_log=False)
		_, _, rrmse2, _ = optimize(imageNoiseless, imageNoisy, alpha=0.8*alpha, gamma=gamma, likelihood="gaussian", prior=prior, print_log=False)
		_, _, rrmse3, _ = optimize(imageNoiseless, imageNoisy, alpha=alpha, gamma=1.2*gamma, likelihood="gaussian", prior=prior, print_log=False)
		_, _, rrmse4, _ = optimize(imageNoiseless, imageNoisy, alpha=alpha, gamma=0.8*gamma, likelihood="gaussian", prior=prior, print_log=False)

		print(f"prior : {prior} | alpha : {alpha} | gamma : {gamma}")
		print(f"Optimal RRMSE : {rrmse_optimal} ")
		print(f"1.2 times alpha RRMSE : {rrmse1} ")
		print(f"0.8 times alpha RRMSE : {rrmse2} ")
		print(f"1.2 times gamma RRMSE : {rrmse3} ")
		print(f"0.8 times gamma RRMSE : {rrmse4} ")


def save_image(directory, image, filename):
	plt.figure(figsize=(8, 5))
	plt.imshow(image, cmap="jet")
	# plt.imshow((image * 255).astype(np.int32))
	plt.grid(False)
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()

def save_all_images(imageNoiseless, imageNoisy, x_estimate_squared_l2, x_estimate_l2, x_estimate_huber_reg_l1):
	save_image(results_folder, imageNoiseless,  "image_noiseless")
	save_image(results_folder, imageNoisy,  "image_noisy")
	save_image(results_folder, x_estimate_squared_l2,  "x_estimate_squared_l2")
	save_image(results_folder, x_estimate_l2,  "x_estimate_l2")
	save_image(results_folder, x_estimate_huber_reg_l1,  "x_estimate_huber_reg_l1")

def construct_graph(iterations, function_values, title, filename, directory):
	# print(function_values)
	plt.figure(figsize=(8, 5))
	plt.plot(list(range(1,iterations+1)), function_values, marker='o', linestyle='-', color='b', markersize=4)
	plt.xlabel("Iterations")
	plt.ylabel("Objective Function Value")
	plt.title(title)
	plt.grid(True)
	# plt.show()
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()

if __name__ == "__main__":
	
	microscopy = mat73.loadmat('data/assignmentImageDenoising_microscopy.mat')
	imageNoiseless = microscopy['microscopyImageOrig']
	imageNoisy = microscopy['microscopyImageNoisyScale350sigma0point06']
	print(f'Image Size : {imageNoisy.shape}')

	results_folder = "results/Q3"
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print(f'Folder "{results_folder}" created.')

	imageNoiseless, imageNoisy = normalize_image(imageNoiseless), normalize_image(imageNoisy)

	# optimize(imageNoiseless, imageNoisy, alpha=0.8, gamma=0.5, likelihood="gaussian", prior="square-l2")
	# grid_search(imageNoiseless, imageNoisy, gamma_start = 0, gamma_end=0.2, alpha_start=0, alpha_end=1, prior="square-l2", likelihood="gaussian")
	

	# To check 20% above and below optimal parameters
	# check_nearby_parameters(alpha=0.5477, gamma=0, prior="square-l2")
	# check_nearby_parameters(alpha=0.0955, gamma=0.04165632, prior="l2") 
	# check_nearby_parameters(alpha=0.4545, gamma=0.1558, prior="huber-l1")


	# Best Estimates
	# x_estimate_square_l2, new_log_posterior_square_l2, current_rrmse_square_l2, log_posterior_values_square_l2 = optimize(imageNoiseless, imageNoisy, alpha=0.5477, gamma=0, likelihood="gaussian", prior="square-l2", print_log=False)
	# x_estimate_l2, new_log_posterior_l2, current_rrmse_l2, log_posterior_values_l2 = optimize(imageNoiseless, imageNoisy, alpha=0.0955, gamma=0.04165632, likelihood="gaussian", prior="l2", print_log=False)
	# x_estimate_huber_l1, new_log_posterior_huber_l1, current_rrmse_huber_l1, log_posterior_values_huber_l1 = optimize(imageNoiseless, imageNoisy, alpha=0.4545, gamma=0.1558, likelihood="gaussian", prior="huber-l1", print_log=False)
	
	#Get all plots and respective images
	# save_all_images(imageNoiseless, imageNoisy, x_estimate_square_l2, x_estimate_l2, x_estimate_huber_l1)

	# construct_graph(iterations=200, function_values=log_posterior_values_square_l2, title="Objective Function vs Iterations : Squared L2 norm", filename="squared_l2_plot", directory=results_folder )
	# construct_graph(iterations=200, function_values=log_posterior_values_l2, title="Objective Function vs Iterations : L2 norm", filename="l2_plot", directory=results_folder )
	# construct_graph(iterations=200, function_values=log_posterior_values_huber_l1, title="Objective Function vs Iterations : Huber regularized L1", filename="huber_reg_l1_plot", directory=results_folder )


