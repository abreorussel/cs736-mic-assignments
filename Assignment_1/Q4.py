# from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import mat73
import os

from config import *
from model import *


def display_images(original, noisy, denoised, titles=["Original Image", "Simulated Noisy Image", "Denoised Image"], directory="images/", filename="all_three_images"):
	plt.figure(figsize=(12, 4))

	images = [original, noisy, denoised]
	
	for i in range(3):
		plt.subplot(1, 3, i + 1)
		plt.imshow(images[i])
		plt.title(titles[i])
		plt.axis('off')  # Hide axes

	plt.tight_layout()
	# plt.show()
	plt.savefig(os.path.join(directory, f'{filename}.png'))



def construct_graph(iterations, function_values, title, filename, directory):
	plt.figure(figsize=(8, 5))
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

def plot_histogram(coeffs, p, directory):
	plt.figure(figsize=(8, 5))
	plt.figure(figsize=(8, 5))
	plt.hist(coeffs, bins=100, log=True, range=(-0.1, 0.1), alpha=0.7)
	plt.xlabel("Coefficient Value")
	plt.ylabel("Frequency (log scale)")
	plt.title(f"Histogram of Coefficients (p={p})")
	plt.grid(True)
	plt.savefig(os.path.join(directory, f'hist_p{p}.png'))
	plt.close()


def display_image(image):
	plt.figure()
	plt.imshow(image)
	plt.show()

def save_image(directory, image, filename):
	plt.figure()
	plt.imshow(image)
	# plt.imshow((image * 255).astype(np.int32))
	plt.grid(False)
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()

def normalize_image(image):
	image = np.array(image)
	return ( image - np.min(image) ) / ( np.max(image) - np.min(image) )


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


def reconstruct_image_from_patches_matrix(patches_matrix, image_size, patch_size=(8, 8)):

	image_height, image_width = image_size
	patch_height, patch_width = patch_size

	num_patches_height = image_height // patch_height
	num_patches_width = image_width // patch_width

	reconstructed_image = np.zeros((image_height, image_width))

	patch_index = 0

	for i in range(num_patches_height):
		for j in range(num_patches_width):
			if patch_index < patches_matrix.shape[1]:  
			
				patch = patches_matrix[:, patch_index].reshape(patch_height, patch_width)
				
				
				reconstructed_image[i * patch_height : (i + 1) * patch_height, 
									j * patch_width : (j + 1) * patch_width] = patch
				
				patch_index += 1

	return reconstructed_image


def compute_gradient_D(X, D, R):

	RR_T = R @ R.T  
	XR_T = X @ R.T  

	grad_D = 2 * (D @ RR_T - XR_T) 

	return grad_D




def compute_gradient_R(X, D, R, lambda_reg, p=1):
	error = X - np.dot(D, R)

	grad_reconstruction = -2 * np.dot(D.T, error)
	grad_regularization = lambda_reg * p * np.sign(R) * (np.abs(R) + 1e-8) ** (p - 1)

	grad_R = grad_reconstruction + grad_regularization
	return grad_R




def loss_function(X, D, R, lambda_reg, p=1):

	error = X - np.dot(D, R)
	reconstruction_loss = np.sum(error ** 2) 

	sparsity_penalty = lambda_reg * np.sum(np.abs(R) ** p)

	total_loss = reconstruction_loss + sparsity_penalty
	return total_loss



def compute_norm(matrix):
	return np.sqrt(np.sum(matrix**2))



def alternate_minimization(X, D_init, R_init, lambda_reg=0.1, p=1, lr_D=0.01, lr_R=0.01, 
						   max_iter=100, tol=1e-6):
	D = D_init.copy()
	R = R_init.copy()

	loss_values = list()
	
	for i in range(max_iter):

		grad_R = compute_gradient_R(X, D, R, lambda_reg, p)
		R -= lr_R * grad_R

		grad_D = compute_gradient_D(X, D, R)
		D -= lr_D * grad_D

		norms = np.sqrt(np.sum(D**2, axis=0))  
		D = D / np.where(norms > 1, norms, 1.0).reshape(1, -1)
	
		loss = loss_function(X, D, R, lambda_reg, p)
		loss_values.append(loss)
		print(f"Iteration : {i+1} / {max_iter} => Loss : {loss}")
		
		# Compute Frobenius norm manually for convergence check
		norm_grad_D = compute_norm(grad_D)
		norm_grad_R = compute_norm(grad_R)
		
		if norm_grad_D < tol and norm_grad_R < tol:
			print(f"Converged at iteration {i}")
			break

	construct_graph(max_iter, loss_values, f"Objective function versus Iterations: p = {p}", f"dictionary_learning_p{p}",results_folder )
	coeffs = R.flatten()
	plot_histogram(coeffs, p, "images/")
	return D, R
	

def noise_addition(image, mean=0):
	intensity_range = image.max() - image.min()
	sigma = 0.1 * intensity_range
	gaussian_noise = np.random.normal(mean, sigma, image.shape)
	noisy_image = image +  gaussian_noise
	if image.dtype == np.uint8:
		noisy_image = np.clip(noisy_image, 0, 1).astype(np.uint8)
	noisy_image = normalize_image(noisy_image)
	return noisy_image


def denoise_using_D(noisy_image, D, R, max_iter=100, lambda_reg=0.2, lr_R=0.001, p= 0.8, tol=1e-6 ):

	patches = extract_non_overlapping_patches(noisy_image)
	xcols = []
	for patch in patches:
		xcols.append(patch.flatten())
	X = np.column_stack(xcols)

	loss_values = list()
	for i in range(max_iter):

		grad_R = compute_gradient_R(X, D, R, lambda_reg, p)
		R -= lr_R * grad_R
	
		loss = loss_function(X, D, R, lambda_reg, p)
		loss_values.append(loss)
		print(f"Iteration : {i+1} / {max_iter} => Loss : {loss}")
		

		norm_grad_R = compute_norm(grad_R)
		
		if norm_grad_R < tol:
			print(f"Converged at iteration {i}")
			break
	construct_graph(max_iter, loss_values, "Objective function versus Iterations", f"optimization_{p}",results_folder )

	denoised_patches = D @ R
	denoised_image = reconstruct_image_from_patches_matrix(denoised_patches, noisy_image.shape)

	return denoised_image

def reconstruct_dictionary_atoms(D, image_shape, directory, filename, grid_shape=None):

	N, M = D.shape  
	
	h, w = image_shape 
	assert N == h * w, "Incorrect image shape!"

	
	if grid_shape is None:
		grid_rows = int(np.floor(np.sqrt(M)))
		grid_cols = int(np.ceil(M / grid_rows))
	else:
		grid_rows, grid_cols = grid_shape
	
	fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))
	axes = axes.flatten() 

	for i in range(M):
		img = D[:, i].reshape(image_shape)  
		axes[i].imshow(img, cmap="gray")
		axes[i].axis("off")


	for i in range(M, len(axes)):
		axes[i].axis("off")

	plt.tight_layout()
	# plt.show()
	plt.savefig(os.path.join(directory, f'{filename}.png'))

		

if __name__ == "__main__":
	K = 64
	chestCT = mat73.loadmat('data/assignmentImageDenoising_chestCT.mat')
	image = chestCT['imageChestCT']
	results_folder = "results/Q4"

	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print(f'Folder "{results_folder}" created.')


	save_image(results_folder, image, "orig_chestCT")
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
	reconstruct_dictionary_atoms(D, (8,8), directory=results_folder,filename="initial_dictionary")
	R = np.zeros((K, len(var_patches)))
	# R = np.random.rand(K, len(var_patches))

	xcols = []

	for patch in patches:
		xcols.append(patch.flatten())
	X = np.column_stack(xcols)

	noisy_image =  noise_addition(image=normalize_image(chestCT['imageChestCT']))
	save_image(results_folder, noisy_image, "simulated-noisy-image")

	D8 = None
	# [2, 1.6, 1.2, 0.8]
	for p in [2, 1.6, 1.2, 0.8]:
		D, R = alternate_minimization(X, D, R, lambda_reg=0.2, p=p, lr_D=0.001, lr_R=0.001, 
						   max_iter=2000, tol=1e-6)
		
		reconstruct_dictionary_atoms(D, (8,8), directory=results_folder,filename=f"learnt_dictionary_p{p}")
		
		if p == 0.8:
			D8 = D


	print(f"Initital RRMSE : {rrmse(normalize_image(chestCT['imageChestCT']), noisy_image)}")

	denoised_img = denoise_using_D(noisy_image, D8, R,  max_iter=2000, lambda_reg=0.2, lr_R=0.001, p= 0.8, tol=1e-6)

	print(f"Pre denoising RRMSE : {rrmse(normalize_image(chestCT['imageChestCT']), noisy_image)}")
	print(f"Post denoising RRMSE : {rrmse(normalize_image(chestCT['imageChestCT']), denoised_img)}")
	save_image(results_folder, denoised_img, "denoised-image")
	display_images(original=image, noisy=noisy_image, denoised=denoised_img, directory=results_folder, filename="all_three_images")