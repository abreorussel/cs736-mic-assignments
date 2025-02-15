import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from skimage.data import shepp_logan_phantom
import mat73
from tqdm import tqdm

from helper import *


def constructARTMatrix(image, batch_size=500, delta_s=1, 
					   t_start=-100, t_end=100, t_delta=1, 
					   theta_start=0, theta_end=180, theta_delta=1):

	theta_values = np.arange(theta_start, theta_end, theta_delta)
	image_size = image.shape[0]
	t_values = np.arange(t_start, t_end, t_delta)
	s_values = np.arange(-image_size, image_size, delta_s)

	total_rows = len(theta_values) * len(t_values)
	total_pixels = image_size * image_size

	def batch_generator():
		for batch_start in range(0, total_rows, batch_size):
			batch_end = min(batch_start + batch_size, total_rows)
			A_batch = np.zeros((batch_end - batch_start, total_pixels), dtype=np.float32)

			for row in range(batch_start, batch_end):
				i = row // len(t_values) 
				j = row % len(t_values)   

				theta_radian = np.deg2rad(theta_values[i])
				t = t_values[j]

				x = t * np.cos(theta_radian) - s_values * np.sin(theta_radian)
				y = t * np.sin(theta_radian) + s_values * np.cos(theta_radian)

				shifted_x = np.clip(x + image_size // 2, 0, image_size - 1)
				shifted_y = np.clip(y + image_size // 2, 0, image_size - 1)

				interpolated_values = map_coordinates(image, [shifted_y, shifted_x],
													  order=1, mode='constant', cval=0)

				for v in range(len(interpolated_values)):
					pixelIdx = int(shifted_y[v]) * image_size + int(shifted_x[v])
					A_batch[row - batch_start, pixelIdx] = interpolated_values[v]

			yield A_batch 

	return batch_generator  


def myXrayIntegration(image, delta_s=1, t_start=-100, t_end=100, t_delta=1, 
					  theta_start=0, theta_end=180, theta_delta=1):

	theta_values = np.arange(theta_start, theta_end, theta_delta)
	image_size = image.shape[0]
	t_values = np.arange(t_start, t_end, t_delta)
	s_values = np.arange(-image_size, image_size, delta_s)
	radon_transform = np.zeros((len(theta_values), len(t_values)), dtype=np.float32)

	for i, theta in enumerate(theta_values):
		theta_radian = np.deg2rad(theta)
		for j, t in enumerate(t_values):
			x = t * np.cos(theta_radian) - s_values * np.sin(theta_radian)
			y = t * np.sin(theta_radian) + s_values * np.cos(theta_radian)
			interpolated_values = map_coordinates(image, 
												  [y + image_size//2, x + image_size//2],
												  order=1, mode='constant', cval=0)
			radon_transform[i, j] = np.sum(interpolated_values)
	return radon_transform

def save_radon_transform(transform, directory, filename):
	plt.figure(figsize=(8, 6))
	plt.imshow(transform, extent=[-90, 90, 180, 0], aspect='auto')
	plt.xlabel("Projection Position (t)")
	plt.ylabel("Angle (θ)")
	plt.title("Radon Transform")
	plt.colorbar(label="Intensity")
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()

def myXrayCTRadonTransform(image, directory):
	radon_transform = myXrayIntegration(image, delta_s=1, t_start=-90, t_end=91, t_delta=1, 
										 theta_start=0, theta_end=180, theta_delta=1)
	save_radon_transform(radon_transform, directory, "Q4_radon_transform")
	return radon_transform


def myART(A_batches, projections, image_size, iterations=10, lambda_val=1.0,
		  ordering='sequential', ground_truth=None):
	x = np.zeros(image_size * image_size, dtype=np.float32)
	rrmse_values = []
	
	for it in tqdm(range(iterations)):
		global_row_counter = 0
		for A_batch in A_batches():
			n_rows = A_batch.shape[0]
			indices = np.arange(n_rows)
			if ordering == 'random':
				np.random.shuffle(indices)
			for idx in indices:
				global_idx = global_row_counter + idx
				r = projections[global_idx] - np.dot(A_batch[idx, :], x)
				l2_squared = np.dot(A_batch[idx, :], A_batch[idx, :])
				if l2_squared > 0:
					x = x + lambda_val * (r / l2_squared) * A_batch[idx, :]
			global_row_counter += n_rows

		rrmse_val = rrmse(ground_truth.flatten(), x)
		rrmse_values.append(rrmse_val)
		print(f"Iteration {it+1}: RRMSE = {rrmse_val:.4f}")
	reconstructed_img = x.reshape(image_size, image_size)
	return reconstructed_img, rrmse_values

if __name__ == "__main__":
	results_folder = "results/Q4"
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print(f'Folder "{results_folder}" created.')

	chestCT = mat73.loadmat('data/assignmentMathImagingRecon_chestCT.mat')
	chestCT_img = chestCT['imageAC']
	print(f"Image Size: {chestCT_img.shape}")

	A_batches = constructARTMatrix(chestCT_img, batch_size=500, delta_s=1,
								   t_start=-90, t_end=91, t_delta=1,
								   theta_start=0, theta_end=180, theta_delta=1)

	radon_transform = myXrayCTRadonTransform(chestCT_img, results_folder)
	
	intensity_range = np.max(radon_transform) - np.min(radon_transform)
	noise_std = 0.05 * intensity_range
	noisy_radon_transform = radon_transform + noise_std * np.random.randn(*radon_transform.shape)
	noisy_projections = noisy_radon_transform.flatten()

	iterations = 5  
	ordering = 'sequential' 
	lambda_values = np.arange(0.1, 1.1, 0.1)

	plt.figure(figsize=(8, 6))
	for lambda_val in lambda_values:
		print(f"\n--- Running ART with λ = {lambda_val:.1f} ---")
		reconstructed_img, rrmse_history = myART(A_batches, noisy_projections, image_size=chestCT_img.shape[0],
									 iterations=iterations, lambda_val=lambda_val,
									 ordering=ordering, ground_truth=chestCT_img)
		plt.plot(np.arange(1, iterations+1), rrmse_history, label=f'λ = {lambda_val:.1f}')
	
	plt.xlabel("Iteration Number")
	plt.ylabel("RRMSE")
	plt.title("RRMSE vs. Iteration for Various λ Values")
	plt.legend()
	plt.grid(True)
	plt.savefig(os.path.join(results_folder, "Q4_RRMSE_vs_Iterations.png"))
	plt.show()

	plt.figure(figsize=(6, 6))
	plt.imshow(reconstructed_img, cmap='jet')
	plt.title(f"Reconstructed Image using ART (λ = {lambda_val:.1f})")
	plt.axis("off")
	plt.colorbar()
	plt.savefig(os.path.join(results_folder, "Q4_ART_reconstruction.png"))
	plt.show()