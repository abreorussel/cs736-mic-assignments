from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from skimage.transform import radon, iradon
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

from filters import *
from helper import *


def myFilter(fft, filter, l_mul_factor=1):
	filter_mapping = {
		"ram_lak":ram_lak_filter,
		"shepp_logan": shepp_logan_filter,
		"cosine": cosine_filter,
		"unfiltered": unfiltered 
	}

	filtered_fft = filter_mapping[filter](fft, 128, l_mul_factor)
	return filtered_fft


def save_radon_transform(transform, directory, filename):
	plt.figure(figsize=(8, 6))
	plt.imshow(transform, extent=[-90, 90, 180, 0], aspect='auto')
	plt.xlabel("Projection Position (t)")
	plt.ylabel("Angle (theta)")
	plt.title("Radon Transform")
	plt.colorbar(label="Intensity")
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()

def save_image(image, directory, filename):
	plt.figure()
	plt.imshow(image)
	plt.title("Reconstructed Image")
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()

def display_image(image):
	img = plt.imshow(image)
	plt.show()

def save_rrmse_plot(l_values ,rrmse_values, directory, filename):
	plt.figure(figsize=(8, 5))
	plt.plot(l_values , rrmse_values, marker='o', linestyle='-', color='b', markersize=4)
	plt.xlabel("L values")
	plt.ylabel("RRMSE")
	plt.title("RRMSE vs L ")
	plt.grid(True)
	# plt.show()
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()

if __name__ == "__main__":
	results_folder = "results/Q2"
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print(f'Folder "{results_folder}" created.')

	phantom = shepp_logan_phantom()
	
	print(f"Image Size : {phantom.shape}")
	phantom = cv2.resize(phantom, (128, 128))
	print(f"After Resize : {phantom.shape}")
	
#--------------------------------------------------------------------------------------------------------------------
# Q2.a
#---------------------------------------------------------------------------------------------------------------------
	theta = np.arange(0, 178, 3)
	radon_transform = radon(image=phantom, theta=theta)
	# save_radon_transform(radon_transform, results_folder, f"radon")
	fourier_1d = np.fft.fft(radon_transform, axis=0)
	
	filters = ["ram_lak", "shepp_logan", "cosine"]
	l_factor = [1 , 0.5]
	for filter in filters:
		for l_mul in l_factor:
			filtered_fft = myFilter(fourier_1d, filter, l_mul)

			filtered_backprojection = np.fft.ifft(filtered_fft, axis = 0).real 
			reconstructed_image = iradon(radon_image=filtered_backprojection, theta=theta, filter_name=None)
			save_image(reconstructed_image, results_folder, f"reconstructed_image_{filter}_lvalue_{l_mul}")


#--------------------------------------------------------------------------------------------------------------------
# Q2.b
#--------------------------------------------------------------------------------------------------------------------

	def getGaussianKernel(size, sigma):
		def gaussian(x, y, sigma):
			return np.exp(-(x**2 + y**2) / (2 * sigma**2))
		kernel = np.zeros((size, size))
		center = size // 2
		for i in range(size):
			for j in range(size):
				kernel[i, j] = gaussian(i - center, j - center, sigma)
		kernel /= np.sum(kernel)
		return kernel


	s0 = phantom
	size = 11
	sigma = 1.0
	kernel = getGaussianKernel(size, sigma)
	s1 = convolve2d(phantom, kernel, mode="same")

	size = 51
	sigma = 5.0
	kernel = getGaussianKernel(size, sigma)
	s5 = convolve2d(phantom, kernel, mode="same")

	save_image(s0, directory=results_folder, filename=f"unblurred")
	save_image(s1, directory=results_folder, filename=f"blurred_1")
	save_image(s5, directory=results_folder, filename=f"blurred_5")


	dict = {"phantom": s0, "blurred_1":s1, "blurred_5": s5}
	# for image in dict.keys():
	# 	radon_transform = radon(image=dict[image], theta=theta)
	# 	# save_radon_transform(radon_transform, results_folder, f"radon")
		
	# 	fourier_1d = np.fft.fft(radon_transform, axis=0)
	# 	filter = "ram_lak"
	# 	filtered_fft = myFilter(fourier_1d, filter)

	# 	filtered_backprojection = np.fft.ifft(filtered_fft, axis = 0).real 
	# 	reconstructed_image = iradon(radon_image=filtered_backprojection, theta=theta, filter_name=None)
	# 	save_image(reconstructed_image, results_folder, f"reconstructed_image_{filter}_{image}")

	# 	print(f"RRMSE for {image} : {rrmse(dict[image], reconstructed_image):.4f}")

	'''
		RRMSE for phantom : 0.6598
		RRMSE for blurred_1 : 0.6594
		RRMSE for blurred_5 : 0.7382
	'''


#--------------------------------------------------------------------------------------------------------------------
# Q2.c
#--------------------------------------------------------------------------------------------------------------------
	l_values = np.arange(1 ,51, 1)
	l_mul_factor = l_values * (1/50)
	
	rrmse_dict = {}
	
	for image in dict.keys():
		rrmse_dict[image] = []
		for mul_factor in l_mul_factor:
			radon_transform = radon(image=dict[image], theta=theta)
			# save_radon_transform(radon_transform, results_folder, f"radon")
			
			fourier_1d = np.fft.fft(radon_transform, axis=0)
			filter = "ram_lak"
			filtered_fft = myFilter(fourier_1d, filter, mul_factor)

			filtered_backprojection = np.fft.ifft(filtered_fft, axis = 0).real 
			reconstructed_image = iradon(radon_image=filtered_backprojection, theta=theta, filter_name=None)
			rrmse_dict[image].append(np.round(rrmse(dict[image], reconstructed_image), 4))

		save_rrmse_plot(l_mul_factor ,rrmse_dict[image], results_folder, f"rrmse_{image}")