import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import mat73
from tqdm import tqdm
from numba import njit
import os


# utilities
def plotMemberships(filename, title):
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


def initialize():
	global bias_field, class_means, U, weights
	bias_field = np.ones_like(Y)
	class_means = np.linspace(np.min(Y[mask > 0]), np.max(Y[mask > 0]), K)
	class_means[0] = 0.5 
	class_means[1] = 0.25
	class_means[2] = 0.1
	# class_means *= 255
	print(f'Initial class means: {class_means}')
	x, y = np.meshgrid(np.arange(-neigh_size//2 + 1, neigh_size//2 + 1),
							np.arange(-neigh_size//2 + 1, neigh_size//2 + 1))
	weights = np.exp(-(x**2 + y**2) / (2 * sigma**2))
	weights /= np.sum(weights)

	# U = np.zeros((H, W, K))
	# U[mask == 1, :] = 1.0 / K
	alpha = 1
	U = np.zeros((H, W, K), dtype=np.float64)
	for i in range(H):
		for j in range(W):
			if mask[i, j]:
				intensity = Y[i, j]
				# Compute similarity for each class: lower difference gives higher similarity.
				# We use a negative exponential to convert differences to similarity scores.
				similarities = np.exp(-alpha * np.abs(intensity - class_means))
				# Normalize so that the memberships sum to 1.
				U[i, j, :] = similarities / (np.sum(similarities) + 1e-8)

	plotMemberships(title="Initial Membership Distributions", filename="InitialMemberships")


@njit
def calcObjectiveFn(Y, U, class_means, bias_field, weights, q, neigh_size, mask):
	objectiveFunction = 0.0
	# Loop over each center pixel j (here represented by (a,b))
	for a in range(H):
		for b in range(W):
			# Only compute if the center pixel is inside the mask
			if mask[a, b] == 1:
				# For each class k
				for k_idx in range(K):
					inner_sum = 0.0
					# Loop over the neighborhood of the center pixel
					for i in range(-neigh_size//2 + 1, neigh_size//2 + 1):
						for j in range(-neigh_size//2 + 1, neigh_size//2 + 1):
							nx = a + i
							ny = b + j
							# Check boundary conditions and neighbor mask
							if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] == 1:
								weight = weights[i + neigh_size//2 - 1, j + neigh_size//2 - 1]
								# Use center intensity (Y[a,b]) and neighbor's bias (bias_field[nx,ny])
								diff = Y[a, b] - class_means[k_idx] * bias_field[nx, ny]
								inner_sum += weight * (diff ** 2)
					# Multiply the local error by the membership (raised to q) at the center pixel
					objectiveFunction += (U[a, b, k_idx] ** q) * inner_sum
	return objectiveFunction


@njit
def updateMemberships(Y, U, class_means, bias_field, weights, q, neigh_size, mask):
	new_U = U.copy()
	for a in range(H):
		for b in range(W):
			if mask[a, b] == 1:
				# 1) Compute d_{k} for each class k at pixel (a,b)
				D_list = np.zeros(K, dtype=np.float64)
				for k_idx in range(K):
					Dkj = 0.0
					for i in range(-neigh_size//2 + 1, neigh_size//2 + 1):
						for j in range(-neigh_size//2 + 1, neigh_size//2 + 1):
							nx = a + i
							ny = b + j
							if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] == 1:
								weight = weights[i + neigh_size//2 - 1, j + neigh_size//2 - 1]
								diff = Y[a, b] - class_means[k_idx] * bias_field[nx, ny]
								Dkj += weight * (diff ** 2)
					D_list[k_idx] = Dkj

				# 2) Compute the denominator (sum of (1/d)^1/(q-1)) for all k
				denom = 0.0
				for k_idx in range(K):
					denom += (1.0 / (D_list[k_idx] + 1e-8))**(1/(q-1))

				# 3) Assign the membership
				for k_idx in range(K):
					numerator = (1.0 / (D_list[k_idx] + 1e-8))**(1/(q-1))
					new_U[a, b, k_idx] = numerator / (denom + 1e-8)
			else:
				# Outside the mask, set memberships to 0 (or keep them unchanged if desired)
				for k_idx in range(K):
					new_U[a, b, k_idx] = 0.0
	return new_U


@njit
def updateBiasField(Y, U, class_means, bias_field, weights, q, neigh_size, mask):
	b_new = np.zeros_like(Y, dtype=np.float64)
	r = neigh_size // 2

	for i in range(H):
		for j in range(W):
			if mask[i, j] == 1:
				numerator = 0.0
				denominator = 0.0
				
				# Loop over the neighborhood around (i, j)
				for di in range(-r, r + 1):
					for dj in range(-r, r + 1):
						nx = i + di
						ny = j + dj
						# Check bounds and neighbor mask
						if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] == 1:
							w_ij = weights[di + r, dj + r]
							sum_k_c  = 0.0  # sum of u^q * c_k
							sum_k_c2 = 0.0  # sum of u^q * c_k^2
							for k_idx in range(K):
								u_q = (U[nx, ny, k_idx]) ** q
								c_k = class_means[k_idx]
								sum_k_c  += u_q * c_k
								sum_k_c2 += u_q * (c_k ** 2)
							
							numerator   += w_ij * Y[nx, ny] * sum_k_c
							denominator += w_ij * sum_k_c2
				
				b_new[i, j] = numerator / (denominator + 1e-8)
			else:
				# Outside mask, set bias to 0 or keep it unchanged
				b_new[i, j] = 0.0

	return b_new


@njit
def updateClassMeans(Y, U, class_means, bias_field, weights, q, neigh_size, mask):
	new_class_means = class_means.copy()
	for k in range(K):
		# if k == 0:
		#     continue
		numerator = 0.0
		denominator = 0.0
		
		# Loop over each center pixel (a,b)
		for a in range(H):
			for b in range(W):
				# Only process center pixels that are inside the mask
				if mask[a, b] == 1:
					sum_bi  = 0.0
					sum_bi2 = 0.0
					
					# Loop over the neighborhood
					for i in range(-neigh_size//2 + 1, neigh_size//2 + 1):
						for j in range(-neigh_size//2 + 1, neigh_size//2 + 1):
							nx = a + i
							ny = b + j
							# Check bounds AND mask for the neighbor
							if (0 <= nx < H and 0 <= ny < W and mask[nx, ny] == 1):
								weight = weights[i + neigh_size//2 - 1, j + neigh_size//2 - 1]
								sum_bi  += weight * bias_field[nx, ny]
								sum_bi2 += weight * (bias_field[nx, ny] ** 2)
					
					# Accumulate into numerator and denominator
					numerator   += (U[a, b, k] ** q) * Y[a, b] * sum_bi
					denominator += (U[a, b, k] ** q) * sum_bi2
		
		# Update the class mean for class k
		new_class_means[k] = numerator / (denominator + 1e-8)
	return new_class_means

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
	results_folder = "results/Q1"
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print(f'Folder "{results_folder}" created.')

	losses = []
	bias_field = None
	class_means = None
	U = None
	weights = None
	iterations_ran = 0
	# get the data
	data = mat73.loadmat('data/assignmentSegmentBrain.mat')
	Y = data['imageData']
	# Y = Y * 255
	mask = data['imageMask']
	brain_intensities = Y[mask == 1].ravel()

	# hyperparams
	K = 3 
	q = 1.6
	neigh_size = 15
	sigma = 2.5
	max_iter = 100
	threshold = 1e-4
	H, W = Y.shape
	initialize()

	for iteration in tqdm(range(max_iter)):
		iterations_ran += 1
		loss = calcObjectiveFn(Y, U, class_means, bias_field, weights, q, neigh_size, mask)
		losses.append(loss)
		class_means = updateClassMeans(Y, U, class_means, bias_field, weights, q, neigh_size, mask)
		U_old = U.copy()
		U = updateMemberships(Y, U, class_means, bias_field, weights, q, neigh_size, mask)
		# bias_field = updateBiasField(Y, U, class_means, bias_field, weights, q, neigh_size, mask)
		bias_field = updateBiasField(Y, U, class_means, bias_field, weights, q, neigh_size, mask)
		if np.linalg.norm(U - U_old) < threshold:
			print(f"Converged at iteration {iteration}")
			break
	

	# Optimal class means: [0.59480953 0.48876253 0.26209626]
	print(f'Optimal class means: {class_means}')		
	construct_graph(iterations_ran, losses, "Objective Function v/s Iterations")

	# class_means = class_means * 255
	A = np.sum(U * class_means[None, None, :], axis=2) * mask

	# Construct residual image R
	R = (Y - A * bias_field)

	# save the neighbourhood image
	save_image(image=weights, directory=results_folder, filename="neighbourhood", title="Neigbourhood weights")

	# plotting the histogram
	brain_intensities = Y[mask == 1].ravel()
	# Plot histogram
	plt.figure(figsize=(8, 5))
	plt.hist(brain_intensities, bins=100, color='blue', edgecolor='black')
	plt.xlabel('Intensity')
	plt.ylabel('Frequency')
	plt.title('Histogram of Brain Intensities')
	plt.grid(True)
	plt.savefig(os.path.join(results_folder,'histogram.png'))
	plt.clf()
	plt.close()


	save_image(image=Y, directory=results_folder, filename="OriginalImage", title="Original Image")
	plotMemberships(title="Optimal Membership Distributions", filename="OptimalMemberships")
	save_image(image=bias_field, directory=results_folder, filename="optimalBiasField", title="Optimal Bias Field Image")
	save_image(image=A, directory=results_folder, filename="BiasRemoved", title="Bias-removed Image")
	save_image(image=R, directory=results_folder, filename="Residual", title="Residual Image")


