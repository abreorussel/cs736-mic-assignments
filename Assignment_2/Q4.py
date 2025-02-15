from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.transform import radon, rescale
from skimage.transform import iradon
import os
# from scipy.io import loadmat
import mat73

# def constructARTMatrix(image, delta_s=1, t_start=-100, t_end=100, t_delta=1, theta_start=0, theta_end=180, theta_delta=1):
# 	theta_values = np.arange(theta_start, theta_end, theta_delta)
# 	image_size = image.shape[0]
# 	t_values = np.arange(t_start, t_end, t_delta)
# 	s_values = np.arange(-image_size, image_size, delta_s)

# 	A = np.zeros((len(theta_values) *  len(t_values), image.shape[0] * image.shape[1]))
# 	row = 0
 
# 	for i, theta in enumerate(theta_values):
# 		theta_radian = np.deg2rad(theta)

# 		for j, t in enumerate(t_values):
# 			x = t * np.cos(theta_radian) - s_values * np.sin(theta_radian)
# 			y = t * np.sin(theta_radian) + s_values * np.cos(theta_radian)  
			
# 			interpolated_values = map_coordinates(image, [y + image_size//2, x + image_size//2], order=1, mode='constant', cval=0)

# 			shifted_y = y + image_size // 2
# 			shifted_x = x + image_size // 2

# 			for v in range(len(interpolated_values)):
				
# 				if shifted_y[v] >= 0 and shifted_x[v] >= 0:
# 					pixelIdx =  int(round(shifted_y[v])) * image_size + int(round(shifted_x[v]))
# 					if pixelIdx < image_size * image_size:
# 						A[row][pixelIdx] += interpolated_values[v]
# 			row += 1
# 	return A

# def constructARTMatrix(image, delta_s=1, t_start=-100, t_end=100, t_delta=1, theta_start=0, theta_end=180, theta_delta=1):
# 	"""
# 	Constructs the ART system matrix for CT reconstruction.
# 	"""
# 	theta_values = np.arange(theta_start, theta_end, theta_delta)
# 	image_size = image.shape[0]
# 	t_values = np.arange(t_start, t_end, t_delta)
# 	s_values = np.arange(-image_size, image_size, delta_s)

# 	A = np.zeros((len(theta_values) * len(t_values), image_size * image_size))
# 	row = 0

# 	for i, theta in enumerate(theta_values):
# 		theta_radian = np.deg2rad(theta)

# 		for j, t in enumerate(t_values):
# 			x = t * np.cos(theta_radian) - s_values * np.sin(theta_radian)
# 			y = t * np.sin(theta_radian) + s_values * np.cos(theta_radian)

# 			# Ensure coordinates are within bounds
# 			shifted_y = np.clip(y + image_size // 2, 0, image_size - 1)
# 			shifted_x = np.clip(x + image_size // 2, 0, image_size - 1)

# 			interpolated_values = map_coordinates(image, [shifted_y, shifted_x], order=1, mode='constant', cval=0)

# 			for v in range(len(interpolated_values)):
# 				pixelIdx = int(shifted_y[v]) * image_size + int(shifted_x[v])
# 				if pixelIdx < image_size * image_size:
# 					A[row][pixelIdx] += interpolated_values[v]

# 			row += 1
# 	return A

def constructARTMatrix(image, batch_size=500, delta_s=1, 
					   t_start=-100, t_end=100, t_delta=1, 
					   theta_start=0, theta_end=180, theta_delta=1):
	
	theta_values = np.arange(theta_start, theta_end, theta_delta)
	image_size = image.shape[0]
	t_values = np.arange(t_start, t_end, t_delta)
	s_values = np.arange(-image_size, image_size, delta_s)

	total_rows = len(theta_values) * len(t_values)
	total_pixels = image_size * image_size

	# Generator to yield batches instead of storing everything in memory
	def batch_generator():
		for batch_start in range(0, total_rows, batch_size):
			batch_end = min(batch_start + batch_size, total_rows)
			A_batch = np.zeros((batch_end - batch_start, total_pixels), dtype=np.float32)

			for row in range(batch_start, batch_end):
				i = row // len(t_values)  # Theta index
				j = row % len(t_values)   # t index

				theta_radian = np.deg2rad(theta_values[i])
				t = t_values[j]

				x = t * np.cos(theta_radian) - s_values * np.sin(theta_radian)
				y = t * np.sin(theta_radian) + s_values * np.cos(theta_radian)

				shifted_y = np.clip(y + image_size // 2, 0, image_size - 1)
				shifted_x = np.clip(x + image_size // 2, 0, image_size - 1)

				interpolated_values = map_coordinates(image, [shifted_y, shifted_x], order=1, mode='constant', cval=0)

				for v in range(len(interpolated_values)):
					pixelIdx = int(shifted_y[v]) * image_size + int(shifted_x[v])
					A_batch[row - batch_start, pixelIdx] = interpolated_values[v]

			yield A_batch  # Yield the current batch and free memory

	return batch_generator  # Return the generator function



def myXrayIntegration(image, delta_s=1, t_start=-100, t_end=100, t_delta=1, theta_start=0, theta_end=180, theta_delta=1):
	theta_values = np.arange(theta_start, theta_end, theta_delta)
	image_size = image.shape[0]
	t_values = np.arange(t_start, t_end, t_delta)
	s_values = np.arange(-image_size, image_size, delta_s)
	radon_transform = np.zeros((len(theta_values), len(t_values)))

	for i, theta in enumerate(theta_values):
		theta_radian = np.deg2rad(theta)

		for j, t in enumerate(t_values):
			x = t * np.cos(theta_radian) - s_values * np.sin(theta_radian)
			y = t * np.sin(theta_radian) + s_values * np.cos(theta_radian)  

			interpolated_values = map_coordinates(image, 
												  [y + image_size//2, x + image_size//2], 
												  order=1, mode='constant', cval=0)

			radon_transform[i][j] = np.sum(interpolated_values)
	return radon_transform

def save_radon_transform(transform, directory, filename):
	plt.figure(figsize=(8, 6))
	plt.imshow(transform, extent=[-90, 90, 180, 0], aspect='auto')
	plt.xlabel("Projection Position (t)")
	plt.ylabel("Angle (theta)")
	plt.title("Radon Transform")
	plt.colorbar(label="Intensity")
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()

def myXrayCTRadonTransform(image, directory):
	radon_transform = myXrayIntegration(image, t_start=-90, t_end=91, t_delta=5, theta_start=0, theta_end=180)
	save_radon_transform(radon_transform, directory, f"Q4_radon_transform")
	return radon_transform


def myART(A_batches, projections, image_size, iterations=10, lambda_val=1.0):
	x = np.zeros(image_size * image_size, dtype=np.float32)  # Initial estimate

	for _ in range(iterations):
		for A_batch in A_batches():
			residual = projections[:A_batch.shape[0]] - A_batch @ x  # Compute residual
			correction = A_batch.T @ residual  # Backprojection
			x += lambda_val * correction  # Update image

	return x.reshape(image_size, image_size)


if __name__ == "__main__":
	results_folder = "results/Q4"
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print(f'Folder "{results_folder}" created.')

	chestCT = mat73.loadmat('data/assignmentMathImagingRecon_chestCT.mat')
	chestCT_img = chestCT['imageAC']
	
	print(f"Image Size : {chestCT_img.shape}")
	  
	A_batches = constructARTMatrix(chestCT_img, batch_size=500)  # Process in batches of 500 rows
	radon_transform = myXrayCTRadonTransform(chestCT_img, "results/Q4")
	
	reconstructed_img = myART(A_batches, radon_transform.flatten(), image_size=chestCT_img.shape[0])
	
	# Save the reconstructed image
	plt.imshow(reconstructed_img)
	plt.title("Reconstructed Image using ART")
	plt.axis("off")
	plt.savefig(os.path.join(results_folder, "Q4_ART_reconstruction.png"))
	plt.show()
