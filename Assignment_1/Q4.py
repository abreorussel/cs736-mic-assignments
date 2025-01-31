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

	# max_threshold = 2.5
	# min_threshold = 0.001
	# max_step_size = 1e-1
	# min_step_size = 1e-5
	
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
    # Compute reconstruction error: X - D @ R
    error = X - np.dot(D, R)

    # Gradient of reconstruction loss: -2 D^T (X - D R)
    grad_reconstruction = -2 * np.dot(D.T, error)

    # Gradient of regularization term: λ p * sign(R) * |R|^(p-1)
    grad_regularization = lambda_reg * p * np.sign(R) * (np.abs(R) ** (p - 1))

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
    
    for i in range(max_iter):
        # Update R while fixing D
        grad_R = compute_gradient_R(X, D, R, lambda_reg, p)
        R -= lr_R * grad_R

        # Update D while fixing R
        grad_D = compute_gradient_D(X, D, R)
        D -= lr_D * grad_D
        norms = np.sqrt(np.sum(D**2, axis=0, keepdims=True))
		# Normalize columns
        D = D/ norms
        loss = loss_function(X, D, R, lambda_reg, p)
        print(f"Iteration : {i+1} / {max_iter} => Loss : {loss}")
        
        # Compute Frobenius norm manually for convergence check
        norm_grad_D = compute_norm(grad_D)
        norm_grad_R = compute_norm(grad_R)
        
        if norm_grad_D < tol and norm_grad_R < tol:
            print(f"Converged at iteration {i}")
            break

    return D, R
	

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

	xcols = []
	for patch in patches:
		xcols.append(patch.flatten())
	X = np.column_stack(xcols)


	print(D.shape , R.shape, X.shape)
	# print(D)

	# optimize(imageNoiseless, imageNoisy, alpha=0.08, gamma=0.5, likelihood="gaussian", prior="huber-l1")
	alternate_minimization(X, D, R, lambda_reg=0.2, p=2, lr_D=0.001, lr_R=0.01, 
                           max_iter=1500, tol=1e-6)

