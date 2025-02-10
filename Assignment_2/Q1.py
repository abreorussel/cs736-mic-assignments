from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
import os

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

def myXrayCTRadonTransform(image, directory):
	radon_transform = myXrayIntegration(image, t_start=-90, t_end=91, t_delta=5, theta_start=0, theta_end=176, theta_delta=5)
	save_radon_transform(radon_transform, directory, f"Q1b_radon_transform")

def display_image(image):
	img = plt.imshow(image)
	plt.show()

def save_radon_transform(transform, directory, filename):
	plt.figure(figsize=(8, 6))
	plt.imshow(transform, extent=[-90, 90, 180, 0], aspect='auto')
	plt.xlabel("Projection Position (t)")
	plt.ylabel("Angle (theta)")
	plt.title("Radon Transform")
	plt.colorbar(label="Intensity")
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()

def save_1d_plot(transform, theta, directory, filename):
	plt.figure(figsize=(4, 6))
	projection = transform.reshape(-1, 1)
	plt.imshow(projection, cmap='viridis', aspect='auto')
	plt.xlabel(f"theta = {theta}")
	plt.ylabel("Intensity")
	plt.colorbar()
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()

if __name__ == "__main__":
	results_folder = "results/Q1"
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print(f'Folder "{results_folder}" created.')

	phantom = shepp_logan_phantom()
	
	print(f"Image Size : {phantom.shape}")
	phantom = cv2.resize(phantom, (128, 128))
	print(f"After Resize : {phantom.shape}")

	# Q1b.
	# myXrayCTRadonTransform(phantom, results_folder)


	#Q1c.
	# delta_s = [0.5, 1, 3]
	# for ds in delta_s:
	# 	radon_transform = myXrayIntegration(phantom, delta_s=ds)
	# 	save_radon_transform(radon_transform, results_folder, f"Q1c_radon_transform_delta_s_{ds}")

	# 	radon_transform_theta_zero = myXrayIntegration(phantom, delta_s=ds, theta_start=0, theta_end=1, theta_delta=1)
	# 	save_1d_plot(radon_transform_theta_zero, 0, results_folder, f"Q1c_radon_transform_delta_s_{ds}_theta_{0}")

	# 	radon_transform_theta_ninety = myXrayIntegration(phantom, delta_s=ds, theta_start=90, theta_end=91, theta_delta=1)
	# 	save_1d_plot(radon_transform_theta_ninety, 90, results_folder, f"Q1c_radon_transform_delta_s_{ds}_theta_{90}")

	