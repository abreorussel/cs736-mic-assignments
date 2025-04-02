# from scipy.io import loadmat
import mat73
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_pointsets(data, directory, filename):
	plt.figure(figsize=(8, 6))
	for i in range(data.shape[1]):  # Loop over each pointset
		color = np.random.rand(3,)  # Random color
		plt.plot(data[0, i], data[1, i], marker='o', linestyle='-', color=color, alpha=0.7)

	plt.xlabel("X-axis")
	plt.ylabel("Y-axis")
	plt.title("Initial 2D Pointsets of Hand Shapes")
	plt.axis("equal")  # Keep aspect ratio
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()


def plot_aligned_shapes(mean_shape, aligned_shapes, directory, filename):
	D, N, M = aligned_shapes.shape
	plt.figure(figsize=(8, 8))
	
	# Plot each aligned shape in light gray.
	for m in range(M):
		shape = aligned_shapes[:, :, m]
		plt.plot(shape[0, :], shape[1, :], 'o-', color='lightgray', alpha=0.7)
	
	# Overlay the mean shape in red with thicker line and markers.
	plt.plot(mean_shape[0, :], mean_shape[1, :], 'ro-', linewidth=2, markersize=8, label="Mean Shape")
	
	plt.title("Aligned Shapes and Computed Mean Shape")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.axis('equal')
	plt.legend()
	plt.savefig(os.path.join(directory, f'{filename}.png'))
	plt.close()




# preshapespace population
def populatePreshapeSpace(shapes):
	preshape = np.zeros(shapes.shape)
	D, N, M = shapes.shape
	for i in range(M):
		current_shape = shapes[:,:, i]
		centroid = np.sum(current_shape, axis = 1, keepdims=True) / number_of_points
		centered_shape = current_shape - centroid
	
		s = np.sqrt(np.sum(np.square(centered_shape)))
		scaled_and_centered = centered_shape / s
		preshape[:,:, i] = scaled_and_centered
	return preshape


def align_preshape(X, Y):
	A = X @ Y.T
	U, S, Vt = np.linalg.svd(A)
	R = Vt.T @ U.T
	if np.linalg.det(R) < 0:
		M = np.identity(Vt.shape[-1])
		M[-1, -1] = -1
		R = Vt.T @ M @ U.T
	return R


def compute_mean_shape(shapes, tol=1e-6, max_iter=100):

	D, N, M = shapes.shape
	
	preshapes = populatePreshapeSpace(shapes)
	# Initialize mean_shape with the first shape in preshape space.
	mean_shape = preshapes[:, :, 0].copy()
	
	for it in range(max_iter):
		aligned = np.empty_like(preshapes)
		# Align each preshape to the current mean shape using optimal rotation.
		for m in range(M):
			R = align_preshape(mean_shape, preshapes[:, :, m])
			aligned[:, :, m] = R @ preshapes[:, :, m]
		
		# Compute the new mean as the average of the aligned shapes.
		new_mean = np.mean(aligned, axis=2)  # average over the M shapes
		
		# Re-center and re-scale new_mean to project it onto preshape space.
		new_mean = new_mean - np.mean(new_mean, axis=1, keepdims=True)
		new_mean = new_mean / np.linalg.norm(new_mean, 'fro')
		
		if np.linalg.norm(new_mean - mean_shape, 'fro') < tol:
			break
		
		mean_shape = new_mean
	
	return mean_shape, aligned


if __name__ == "__main__":
	
	hands2D = mat73.loadmat('data/hands2D.mat')

	shapes = hands2D['shapes']

	results_folder = "results/Q1"
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print(f'Folder "{results_folder}" created.')
			
	plot_pointsets(shapes, results_folder, "initial_pointsets")

	no_of_pointsets = shapes.shape[-1]
	number_of_points = shapes.shape[1]

	mean_shape, aligned_shapes = compute_mean_shape(shapes)
	plot_aligned_shapes(mean_shape, aligned_shapes, results_folder, "aligned_pointsets")