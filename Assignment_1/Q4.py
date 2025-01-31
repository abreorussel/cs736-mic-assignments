# from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import mat73
import os

from config import *
from model import *


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

def plot_histogram(coeffs, p, directory):
    plt.figure(figsize=(8, 5))
    plt.hist(coeffs, bins=100, log=True, range=(-0.1, 0.1), alpha=0.7)
    plt.xlabel("Coefficient Value")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Histogram of Coefficients (p={p})")
    plt.grid(True)
    plt.savefig(os.path.join(directory, f'hist_p{p}.png'))
    plt.close()


def display_image(image):
	plt.imshow(image)
	plt.show()
	# print(image)

def save_image(directory, image, filename):
	plt.imshow(image, cmap="jet")
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


import numpy as np

def reconstruct_image_from_patches_matrix(patches_matrix, image_size, patch_size=(8, 8)):
    """
    Reconstructs an image from patches stored as columns in a matrix.

    Parameters:
        patches_matrix (np.ndarray): Array of shape (patch_height * patch_width, num_patches),
                                     where each column is a flattened patch.
        image_size (tuple): Original image size (height, width).
        patch_size (tuple): Size of each patch (patch_height, patch_width).

    Returns:
        np.ndarray: Reconstructed image.
    """
    image_height, image_width = image_size
    patch_height, patch_width = patch_size

    num_patches_height = image_height // patch_height
    num_patches_width = image_width // patch_width

    # Initialize an empty image
    reconstructed_image = np.zeros((image_height, image_width))

    patch_index = 0

    # Iterate over the original image grid
    for i in range(num_patches_height):
        for j in range(num_patches_width):
            if patch_index < patches_matrix.shape[1]:  # Ensure we don't exceed available patches
                # Reshape column vector back to patch
                patch = patches_matrix[:, patch_index].reshape(patch_height, patch_width)
                
                # Place patch into the reconstructed image
                reconstructed_image[i * patch_height : (i + 1) * patch_height, 
                                    j * patch_width : (j + 1) * patch_width] = patch
                
                patch_index += 1

    return reconstructed_image


def compute_gradient_D(X, D, R):
	"""
	Compute the gradient of ||X - D R||_2^2 with respect to D.
	
	Parameters:
		X : np.ndarray of shape (64, n)   - Input data matrix
		D : np.ndarray of shape (64, 64)  - Dictionary matrix
		R : np.ndarray of shape (64, n)   - Coefficient matrix

	Returns:
		grad_D : np.ndarray of shape (64, 64) - Gradient w.r.t D
	"""
	# Compute required matrix multiplications
	RR_T = R @ R.T  # (64, 64)
	XR_T = X @ R.T  # (64, 64)
	
	# Compute the gradient
	grad_D = 2 * (D @ RR_T - XR_T)  # (64, 64)

	return grad_D




def compute_gradient_R(X, D, R, lambda_reg, p=1):
	"""
	Computes the gradient of the loss function w.r.t R.

	Parameters:
		X : np.ndarray of shape (64, n)  - Input data matrix (patches)
		D : np.ndarray of shape (64, 64) - Dictionary matrix
		R : np.ndarray of shape (64, n)  - Coefficient matrix
		lambda_reg : float               - Regularization strength
		p : float, optional (default=1)  - Norm degree for sparsity (e.g., L1 or Lp norm)

	Returns:
		grad_R : np.ndarray of shape (64, n) - Gradient of the loss w.r.t. R
	"""
	# grad_regularization = lambda_reg * p * np.sign(R) * (np.abs(R) + 1e-8) ** (p - 1)
	# Compute reconstruction error: X - D @ R
	error = X - np.dot(D, R)

	# Gradient of reconstruction loss: -2 D^T (X - D R)
	grad_reconstruction = -2 * np.dot(D.T, error)

	# Gradient of regularization term: λ p * sign(R) * |R|^(p-1)
	# grad_regularization = lambda_reg * p * np.sign(R) * (np.abs(R) ** (p - 1))
	grad_regularization = lambda_reg * p * np.sign(R) * (np.abs(R) + 1e-8) ** (p - 1)

	# Total gradient
	grad_R = grad_reconstruction + grad_regularization
	return grad_R




def loss_function(X, D, R, lambda_reg, p=1):
	"""
	Computes the dictionary learning loss function:
		sum(||X_i - D R_i||_2^2) + λ * ||R_i||_p^p
	
	Parameters:
		X : np.ndarray of shape (64, n)  - Input data matrix (patches)
		D : np.ndarray of shape (64, 64) - Dictionary matrix
		R : np.ndarray of shape (64, n)  - Coefficient matrix
		lambda_reg : float               - Regularization strength
		p : float, optional (default=1)  - Norm degree for sparsity (e.g., L1 or Lp norm)

	Returns:
		loss : float - Total loss value
	"""
	# Compute reconstruction loss: ||X - D R||_F^2 (Frobenius norm squared)
	error = X - np.dot(D, R)  # Compute X - D @ R
	reconstruction_loss = np.sum(error ** 2)  # Squaring each element and summing

	# Compute sparsity regularization: λ * ||R||_p^p
	sparsity_penalty = lambda_reg * np.sum(np.abs(R) ** p)

	# Total loss
	total_loss = reconstruction_loss + sparsity_penalty
	return total_loss



def compute_norm(matrix):
	"""
	Computes the Frobenius norm of a matrix manually.
	Equivalent to: np.linalg.norm(matrix, 'fro')
	"""
	return np.sqrt(np.sum(matrix**2))



def alternate_minimization(X, D_init, R_init, lambda_reg=0.1, p=1, lr_D=0.01, lr_R=0.01, 
						   max_iter=100, tol=1e-6):
	"""
	Performs alternate minimization to update D and R using gradient descent.

	Parameters:
		X : np.ndarray (64, n)  - Input data matrix (patches)
		D_init : np.ndarray (64, 64) - Initial dictionary
		R_init : np.ndarray (64, n)  - Initial coefficient matrix
		lambda_reg : float - Regularization strength
		p : float - Norm degree for sparsity
		lr_D : float - Learning rate for D
		lr_R : float - Learning rate for R
		max_iter : int - Maximum number of iterations
		tol : float - Convergence tolerance

	Returns:
		D, R : Updated dictionary and coefficient matrix
	"""
	D = D_init.copy()
	R = R_init.copy()

	loss_values = list()
	
	for i in range(max_iter):
		# Update R while fixing D
		grad_R = compute_gradient_R(X, D, R, lambda_reg, p)
		R -= lr_R * grad_R

		# Update D while fixing R
		grad_D = compute_gradient_D(X, D, R)
		D -= lr_D * grad_D
		# norms = np.sqrt(np.sum(D**2, axis=0, keepdims=True))
		# Normalize columns
		# D = D/ norms

		norms = np.sqrt(np.sum(D**2, axis=0))  # shape (K,)
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

	# construct_graph(max_iter, loss_values, "Objective function versus Iterations", f"dictionary_learning_p{p}","images/" )
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
	# display_image(image=noisy_image)
	return noisy_image


def denoise_using_D(noisy_image, D, R, max_iter=100, lambda_reg=0.2, lr_R=0.001, p= 0.8, tol=1e-6 ):
	#extract the image patches and form X

	print("D and R shape", D.shape, R.shape)
	patches = extract_non_overlapping_patches(noisy_image)
	xcols = []
	for patch in patches:
		xcols.append(patch.flatten())
	X = np.column_stack(xcols)

	loss_values = list()
	for i in range(max_iter):
		# Update R while fixing D
		grad_R = compute_gradient_R(X, D, R, lambda_reg, p)
		R -= lr_R * grad_R
	
		loss = loss_function(X, D, R, lambda_reg, p)
		loss_values.append(loss)
		print(f"Iteration : {i+1} / {max_iter} => Loss : {loss}")
		
		# Compute Frobenius norm manually for convergence check
		norm_grad_R = compute_norm(grad_R)
		
		if norm_grad_R < tol:
			print(f"Converged at iteration {i}")
			break
	

	denoised_patches = D @ R
	denoised_image = reconstruct_image_from_patches_matrix(denoised_patches, noisy_image.shape)
	# denoised_image *= 255
	return denoised_image




	


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
	R = np.zeros((K, len(var_patches)))
	# R = np.random.rand(K, len(var_patches))

	print(D.shape, R.shape)

	xcols = []
	# print(len(patches))

	for patch in patches:
		xcols.append(patch.flatten())
	X = np.column_stack(xcols)

	# [2, 1.6, 1.2, 0.8]

	for p in [0.8]:
		D, R = alternate_minimization(X, D, R, lambda_reg=0.2, p=p, lr_D=0.001, lr_R=0.001, 
						   max_iter=500, tol=1e-6)

	noisy_image =  noise_addition(image=normalize_image(chestCT['imageChestCT']))
	save_image("images/",noisy_image, "noisy-Q4")

	print(rrmse(normalize_image(chestCT['imageChestCT']), noisy_image))

	denoised_img = denoise_using_D(noisy_image, D, R,  max_iter=500, lambda_reg=0.2, lr_R=0.001, p= 0.8, tol=1e-6)

	print(rrmse(normalize_image(chestCT['imageChestCT']), denoised_img))
	save_image("images/",denoised_img, "denoised-Q4")