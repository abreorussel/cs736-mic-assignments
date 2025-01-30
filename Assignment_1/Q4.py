# from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import mat73
import os

from config import *
from model import *


def display_image(image):
	plt.imshow(image)
	plt.show()
	# print(image)

def save_image(directory, image, filename):
	plt.imshow(image, cmap=None)
	# plt.imshow((image * 255).astype(np.int32))
	plt.savefig(os.path.join(directory, f'{filename}.png'))

def normalize_image(image):
	image = np.array(image)
	return ( image - np.min(image) ) / ( np.max(image) - np.min(image) )

import numpy as np

def extract_non_overlapping_patches(image, patch_size=(8, 8)):
    image_height, image_width = image.shape
    patch_height, patch_width = patch_size

    num_patches_height = image_height // patch_height
    num_patches_width = image_width // patch_width

    patches = np.zeros((num_patches_height * num_patches_width, patch_height, patch_width))

    patch_index = 0
    for i in range(0, image_height, patch_height):
        for j in range(0, image_width, patch_width):
    
            if i + patch_height <= image_height and j + patch_width <= image_width:
                patches[patch_index] = image[i:i+patch_height, j:j+patch_width]
                patch_index += 1

    patches = patches[:patch_index]
    return patches

	


def optimize(x_true ,y_observed, step_size=1e-2, iterations=100, alpha=0.5, gamma=0, likelihood="gaussian", prior="huber"):

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


		# if new_log_posterior/initial_log_posterior > 1:
		# 	step_size *= 1.1
		# else:
		# 	step_size *= 0.5

		initial_log_posterior = new_log_posterior.copy()

		log_posterior_values.append(new_log_posterior)
		
		current_rrmse = rrmse(x_true, x_estimate)

		print(f"Iteration: {it}/{iterations} => Log Posterior: {np.sum(new_log_posterior):.4f}, RRMSE: {current_rrmse:.4f}")
		
		log_posterior_grad = new_log_posterior_grad

		if it == iterations: rrmse_values.append(current_rrmse)
		if step_size <= 1e-8 : break


	print(x_estimate)
	print(f"Log Posterior: {new_log_posterior:.4f}, RRMSE: {current_rrmse:.4f} Alpha: {alpha} Gamma: {gamma} Prior: {prior}")
	print(f"Saving the denoised image to => {os.path.join("images/" , f'{prior}-denoised')}")
	save_image("images/",x_estimate, f"{prior}-denoised")
	return x_estimate

	

if __name__ == "__main__":
	
	K = 64
	chestCT = mat73.loadmat('data/assignmentImageDenoising_chestCT.mat')

	image = chestCT['imageChestCT']
	save_image("images/", image, "orig_chestCT.png")
	print(f'Image Size : {image.shape}')

	image =  normalize_image(image)

	patches = extract_non_overlapping_patches(image)
	var_patches = list()
	for patch in patches:
		var_patches.append((np.var(patch), patch))

	var_patches = sorted(var_patches, key=lambda x: x[0], reverse=True)

	columns = []
	for i in range(K):
		columns.append(var_patches[i][1].flatten())

	D =	np.column_stack(columns)

	print(D.shape)
	print(D)

	# optimize(imageNoiseless, imageNoisy, alpha=0.08, gamma=0.5, likelihood="gaussian", prior="huber-l1")

