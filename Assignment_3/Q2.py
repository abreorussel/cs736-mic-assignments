import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import mat73
from tqdm import tqdm
from numba import njit
import os
from sklearn.cluster import KMeans


def plotMemberships(U, filename, title):
	plt.figure(figsize=(15, 7))
	plt.suptitle(f'{title}')

	plt.subplot(1, 3, 1)
	plt.title('Class 1 Membership')
	plt.imshow(U[:,:, 0], cmap='gray')
	plt.axis('off')

	plt.subplot(1, 3, 2)
	plt.title('Class 2 Membership')
	plt.imshow(U[:,:, 1], cmap='gray')
	plt.axis('off')

	plt.subplot(1, 3, 3)
	plt.title('Class 3 Membership')
	plt.imshow(U[:,:, 2], cmap='gray')
	plt.axis('off')

	plt.tight_layout()
	plt.savefig(os.path.join(results_folder, f'{filename}.png'))
	plt.clf()
	plt.close()

def gaussian_log_likelihood(Y, mu, sigma):
	likelihood =  - ((Y - mu)**2) / np.square(sigma)
	return likelihood


def calc_prior(X_map, row, col, k, beta=0):
	prior = 0
	# up
	if 0 <= row - 1 < H and mask[row - 1, col]:
		up = X_map[row - 1, col]
		if up != k: prior += beta

	# down
	if 0 <= row + 1 < H and mask[row + 1, col]:
		down = X_map[row + 1, col]
		if down != k: prior += beta

	# left
	if 0 <= col - 1 < W and mask[row, col - 1]:
		left = X_map[row, col - 1]
		if left != k: prior += beta

	# right
	if 0 <= col + 1 < W and mask[row, col + 1]:
		right = X_map[row, col + 1]
		if right != k: prior += beta

	return -prior


def updateMapEstimate(Y, X_map, beta):
	X_map_new = X_map.copy()
	for row in range(H):
		for col in range(W):
			if mask[row, col]:
				max_posterior = -1000
				max_k = None
				for k in range(K):
					posterior = 0
					# log likelihood
					posterior += gaussian_log_likelihood(Y[row,col], means[k], std_devs[k])
					# prior
					posterior += calc_prior(X_map, row, col, k, beta)

					if posterior > max_posterior:
						max_posterior = posterior
						max_k = k

				X_map_new[row, col] = max_k

	return X_map_new

def calc_gaussian(y, mu, std_dev):
	return (1/(np.sqrt(2*np.pi) * std_dev)) * np.exp((-(y - mu)**2)/(2 * std_dev**2))

def updateMemberships(membership, X_map, Y, means, std_devs, beta):
	for row in range(H):
		for col in range(W):
			if mask[row, col]:
				denominator = 0.0
				for k in range(K):
					denominator += calc_gaussian(Y[row, col], means[k], std_devs[k]) * np.exp(calc_prior(X_map, row, col, k, beta))

				for k in range(K):
					numerator = calc_gaussian(Y[row, col], means[k], std_devs[k]) * np.exp(calc_prior(X_map, row, col, k, beta))
					membership[row, col, k] = numerator / (denominator + 1e-8)

	return membership

def updateMeansAndVariances(Y, membership):
	new_means = np.zeros(K)
	new_std_devs = np.zeros(K)
	for k in range(K):
		weights = membership[:, :, k]
		new_means[k] = np.sum(weights * Y) / (np.sum(weights) + 1e-8)
		new_std_devs[k] = np.sqrt(np.sum(weights * (Y - new_means[k])**2) / (np.sum(weights) + 1e-8))
	return new_means, new_std_devs

def compute_log_posterior(Y, X_map, means, std_devs):
	log_posterior = 0
	for row in range(H):
		for col in range(W):
			if mask[row, col]:
				k = X_map[row, col]
				# Log likelihood
				log_posterior += gaussian_log_likelihood(Y[row, col], means[k], std_devs[k])
				# Log prior
				log_posterior += calc_prior(X_map, row, col, k)
	return log_posterior

def construct_graph(iterations, function_values, title):
 # if np.linalg.norm(U - U_old) < threshold:
	#     print(f"Converged at iteration {iteration}")
	#     break	# print(function_values)
	plt.figure(figsize=(8, 5))
	plt.plot(list(range(1,iterations+1)), function_values, marker='o', linestyle='-', color='b', markersize=4)
	plt.xlabel("Iterations")
	plt.ylabel("Objective Function Value")
	plt.title(title)
	plt.grid(True)
	# plt.show()
	plt.savefig(os.path.join(results_folder, 'ObjectiveFn.png'))
	plt.close()


def save_image(image, directory, filename, title):
	plt.figure()
	plt.imshow(image, cmap="gray")
	plt.title(f'{title}')
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.clf()
	plt.close()


if __name__ == "__main__":
	results_folder = "results/Q2"
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print(f'Folder "{results_folder}" created.')


	# extract the data
	data = mat73.loadmat('data/assignmentSegmentBrainGmmEmMrf.mat')
	Y = data['imageData']
	mask = data['imageMask']


	#set up params
	K = 3
	means = None
	H, W = Y.shape
	max_iter = 100
	beta = 2
	membership = np.zeros((H,W,K))

	flattened_y = Y.reshape((-1,))
	flattened_mask = mask.reshape((-1,))
	valid_indices = np.where(flattened_mask == 1.0)


	pixel_values = flattened_y[valid_indices].reshape((-1,1))
	kmeans = KMeans(n_clusters=K, random_state=42)
	kmeans.fit(pixel_values)

	means = kmeans.cluster_centers_.squeeze()

	labels = kmeans.labels_
	std_devs = {}
	for cluster in range(K):
		cluster_data = pixel_values[labels == cluster]
		std_devs[cluster] = np.std(cluster_data)
		# print(f"Cluster {cluster} standard deviation: {std_devs[cluster]}")

	X_map = kmeans.predict(flattened_y.reshape((-1,1))).reshape((Y.shape))

	# c
	save_image(X_map*mask, results_folder, "initial_label_estimate", "Initial Label Image Estimate")



	for iteration in range(max_iter):
		print(f"Iteration {iteration + 1}")
		log_posterior_before = compute_log_posterior(Y, X_map, means, std_devs)
		print(f"  Log Posterior Before ICM: {log_posterior_before:.4f}")

		X_map = updateMapEstimate(Y, X_map, beta)

		log_posterior_after = compute_log_posterior(Y, X_map, means, std_devs)
		print(f"  Log Posterior After ICM: {log_posterior_after:.4f}")

		membership = updateMemberships(membership, X_map, Y, means, std_devs, beta)
		means, std_devs = updateMeansAndVariances(Y, membership)

		if abs(log_posterior_after - log_posterior_before) < 1e-8:
			print("Converged!")
			break

	save_image(Y, results_folder, "corrupted_image", "Corrupted Image")
	save_image(X_map*mask, results_folder, "optimal_label_estimate", "Optimal Label Image Estimate")
	plotMemberships(membership, "optimal_membership", "Optimal Class Membership Estimate")

	beta_new = 0
	for iteration in range(max_iter):
		print(f"Iteration {iteration + 1}")
		log_posterior_before = compute_log_posterior(Y, X_map, means, std_devs)
		print(f"  Log Posterior Before ICM: {log_posterior_before:.4f}")

		X_map = updateMapEstimate(Y, X_map, beta=0)

		log_posterior_after = compute_log_posterior(Y, X_map, means, std_devs)
		print(f"  Log Posterior After ICM: {log_posterior_after:.4f}")

		membership = updateMemberships(membership, X_map, Y, means, std_devs, beta=0)
		means, std_devs = updateMeansAndVariances(Y, membership)

		if abs(log_posterior_after - log_posterior_before) < 1e-8:
			print("Converged!")
			break

	# save_image(Y, results_folder, "corrupted_image", "Corrupted Image")
	save_image(X_map*mask, results_folder, "optimal_label_estimate_beta0", "Optimal Label Image Estimate beta=0")
	save_image(membership, results_folder, "optimal_membership_beta0", "Optimal Class Membership Estimate beta=0")
	plotMemberships(membership, "optimal_membership_beta0", "Optimal Class Membership Estimate beta=0")



