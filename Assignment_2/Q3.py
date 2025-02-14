from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from skimage.transform import radon, iradon
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import mat73
from tqdm import tqdm

from filters import *
from helper import *

def myFilter(fft, filter, num_t_values, l_mul_factor=1):
	filter_mapping = {
		"ram_lak":ram_lak_filter,
		"shepp_logan": shepp_logan_filter,
		"cosine": cosine_filter,
		"unfiltered": unfiltered 
	}

	filtered_fft = filter_mapping[filter](fft, num_t_values, l_mul_factor)
	return filtered_fft


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

def save_rrmse_plot(thetas ,rrmse_values, directory, filename):
	plt.figure(figsize=(8, 5))
	plt.plot(thetas , rrmse_values, marker='o', linestyle='-', color='b', markersize=4)
	plt.xlabel("L values")
	plt.ylabel("RRMSE")
	plt.title("RRMSE vs L ")
	plt.grid(True)
	# plt.show()
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()


def save_image(image, directory, filename):
	plt.figure()
	plt.imshow(image)
	plt.title("Reconstructed Image")
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()


if __name__ == "__main__":
	results_folder = "results/Q3"
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print(f'Folder "{results_folder}" created.')

	chestCT = mat73.loadmat('data/assignmentMathImagingRecon_chestCT.mat')
	chestCT_img = chestCT['imageAC']
	phantom = mat73.loadmat('data/assignmentMathImagingRecon_myPhantom.mat')
	print(phantom.keys())
	phantom_img = phantom['imageMyPhantomAC']
	

	thetas = np.arange(0, 181, 1)

	optimal_theta_chestCT = 0
	optimal_theta_phantom = 0
	min_rrmse_chestCT = 1000
	min_rrmse_phantom = 1000
	rrmse_values_chestCT = []
	rrmse_values_phantom = []
	for theta in tqdm(thetas):
		new_thetas = np.arange(theta , theta + 151, 1)
		
		radon_transform = radon(image=chestCT_img, theta=new_thetas)
		# save_radon_transform(radon_transform, results_folder, f"radon")
		
		fourier_1d = np.fft.fft(radon_transform, axis=0)
		filter = "cosine"
		filtered_fft = myFilter(fourier_1d, filter, chestCT_img.shape[0] )

		filtered_backprojection = np.fft.ifft(filtered_fft, axis = 0).real 
		reconstructed_image = iradon(radon_image=filtered_backprojection, theta=new_thetas, filter_name=None)

		rrmse_value = np.round(rrmse(chestCT_img, reconstructed_image), 4)
		if rrmse_value < min_rrmse_chestCT : 
			optimal_theta_chestCT = theta
			min_rrmse_chestCT = rrmse_value
		rrmse_values_chestCT.append(rrmse_value)
	save_rrmse_plot(thetas ,rrmse_values_chestCT, results_folder, f"rrmse_chestCT")

	for theta in tqdm(thetas):
		new_thetas = np.arange(theta , theta + 151, 1)
		
		radon_transform = radon(image=phantom_img, theta=new_thetas)
		# save_radon_transform(radon_transform, results_folder, f"radon")
		
		fourier_1d = np.fft.fft(radon_transform, axis=0)
		filter = "cosine"
		filtered_fft = myFilter(fourier_1d, filter, phantom_img.shape[0])

		filtered_backprojection = np.fft.ifft(filtered_fft, axis = 0).real 
		reconstructed_image = iradon(radon_image=filtered_backprojection, theta=new_thetas, filter_name=None)

		rrmse_value = np.round(rrmse(phantom_img, reconstructed_image), 4)
		if rrmse_value < min_rrmse_phantom : 
			optimal_theta_phantom = theta
			min_rrmse_phantom = rrmse_value
		rrmse_values_phantom.append(rrmse_value)
	save_rrmse_plot(thetas ,rrmse_values_phantom, results_folder, f"rrmse_phantom")


	# Save recontructed image
	new_thetas = np.arange(optimal_theta_chestCT , optimal_theta_chestCT + 151, 1)
	radon_transform = radon(image=chestCT_img, theta=new_thetas)
	fourier_1d = np.fft.fft(radon_transform, axis=0)
	filter = "cosine"
	filtered_fft = myFilter(fourier_1d, filter, chestCT_img.shape[0])
	filtered_backprojection = np.fft.ifft(filtered_fft, axis = 0).real 
	reconstructed_image = iradon(radon_image=filtered_backprojection, theta=new_thetas, filter_name=None)
	rrmse_value = np.round(rrmse(chestCT_img, reconstructed_image), 4)
	save_image(reconstructed_image, results_folder, f"reconstructed_image_chestCT_theta_{optimal_theta_chestCT}_rrmse_{rrmse_value}")

	# Save recontructed image
	new_thetas = np.arange(optimal_theta_phantom , optimal_theta_phantom + 151, 1)
	radon_transform = radon(image=phantom_img, theta=new_thetas)
	fourier_1d = np.fft.fft(radon_transform, axis=0)
	filter = "cosine"
	filtered_fft = myFilter(fourier_1d, filter, phantom_img.shape[0])
	filtered_backprojection = np.fft.ifft(filtered_fft, axis = 0).real 
	reconstructed_image = iradon(radon_image=filtered_backprojection, theta=new_thetas, filter_name=None)
	rrmse_value = np.round(rrmse(phantom_img, reconstructed_image), 4)
	save_image(reconstructed_image, results_folder, f"reconstructed_image_phantom_theta_{optimal_theta_phantom}_rrmse_{rrmse_value}")

	