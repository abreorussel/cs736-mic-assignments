from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
import mat73


def display_image(image):
	plt.figure()
	plt.imshow(image)
	plt.show()
	# print(image)


# def save_image(directory, image, filename):
# 	plt.figure(figsize=(8, 5))
# 	plt.imshow(image, cmap="jet")
# 	# plt.imshow((image * 255).astype(np.int32))
# 	plt.grid(False)
# 	plt.savefig(os.path.join(directory, f'{filename}.png'))
# 	plt.close()

# def save_all_images(imageNoiseless, imageNoisy, x_estimate_quadratic, x_estimate_huber, x_estimate_adaptive):
# 	save_image(results_folder, imageNoiseless,  "image_noiseless")
# 	save_image(results_folder, imageNoisy,  "image_noisy")
# 	save_image(results_folder, x_estimate_quadratic,  "x_estimate_quadratic")
# 	save_image(results_folder, x_estimate_huber,  "x_estimate_huber")
# 	save_image(results_folder, x_estimate_adaptive,  "x_estimate_adaptive")

# def construct_graph(iterations, function_values, title, filename, directory):
# 	# print(function_values)
# 	plt.figure(figsize=(8, 5))
# 	plt.plot(list(range(1,iterations+1)), function_values, marker='o', linestyle='-', color='b', markersize=4)
# 	plt.xlabel("Iterations")
# 	plt.ylabel("Objective Function Value")
# 	plt.title(title)
# 	plt.grid(True)
# 	# plt.show()
# 	plt.savefig(os.path.join(directory, f'{filename}.png'))
# 	plt.close()


if _name_ == "_main_":
	data = mat73.loadmat('data/assignmentSegmentBrain.mat')
	print(data.keys())
	brainImage = data["imageData"]
	brainMask = data["imageMask"]

	display_image(brainImage)
	display_image(brainMask)
	
	# results_folder = "results/Q1"
	# if not os.path.exists(results_folder):
	# 	os.makedirs(results_folder)
	# 	print(f'Folder "{results_folder}" created.')