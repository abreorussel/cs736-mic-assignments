import mat73
import os
import numpy as np
import matplotlib.pyplot as plt

def compute_pca(shapes, algo="cootes"):
	algorithm_mapping = {
		"cootes" : CootesCalculateMean,
		"kabsch" : compute_mean_shape_kabsch
	}
	mean_shape, aligned_shapes = algorithm_mapping[algo](shapes)
	centered_shapes = aligned_shapes - mean_shape[:, :, np.newaxis]
	covariance_matrix = np.cov(centered_shapes.reshape(2 * shapes.shape[1], shapes.shape[2]))
	eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
	idx = np.argsort(-eigenvalues)
	return eigenvalues[idx], eigenvectors[:, idx], mean_shape, aligned_shapes


def CootesAlignPoint(x1, x2):
	n = x1.shape[1]
	X1 = np.sum(x1[0, :])
	Y1 = np.sum(x1[1, :])
	X2 = np.sum(x2[0, :])
	Y2 = np.sum(x2[1, :])
	Z = np.sum(x2[0, :]**2 + x2[1, :]**2)
	C1 = np.sum(x1[0, :] * x2[0, :] + x1[1, :] * x2[1, :])
	C2 = np.sum(x1[1, :] * x2[0, :] - x1[0, :] * x2[1, :])
 
	A = np.array([
		[X2, -Y2, n, 0],
		[Y2, X2, 0, n],
		[Z, 0, X2, Y2],
		[0, Z, -Y2, X2]
	])
	
	b = np.array([X1, Y1, C1, C2])
 
	ax, ay, tx, ty = np.linalg.solve(A, b)
	s = np.sqrt(ax**2 + ay**2)
	theta = np.arctan2(ay, ax) 
 
	R = np.array([[np.cos(theta), -np.sin(theta)],
				  [np.sin(theta), np.cos(theta)]])
	
	x2_transformed = s * R @ x2 + np.array([[tx], [ty]])
	return x2_transformed
 
def CootesCalculateMean(shapes, iterations = 100 ,tol = 1e-2):
	dims, number_of_points, number_of_examples = shapes.shape
	aligned_shapes = np.zeros(shapes.shape)
	for i in range(number_of_examples):
		if i == 0:
			continue
		aligned_shapes[:,:,i] = CootesAlignPoint(shapes[:,:,0], shapes[:,:,i])
 
	mean_shape = np.mean(aligned_shapes, axis = 2)
 
	for it in range(iterations):
		centered = mean_shape - np.mean(mean_shape, axis = 1, keepdims=True)
		norm = np.sqrt(np.sum(np.square(centered)))
		mean_shape = centered / norm
 
		for i in range(number_of_examples):
			aligned_shapes[:,:,i] = CootesAlignPoint(mean_shape, shapes[:,:,i])
			
		new_mean_shape = np.mean(aligned_shapes, axis = 2)
		
		if np.linalg.norm(new_mean_shape - mean_shape) < tol:
			mean_shape = new_mean_shape
			break  
		mean_shape = new_mean_shape
		
	centered = mean_shape - np.mean(mean_shape, axis = 1, keepdims=True)
	norm = np.sqrt(np.sum(np.square(centered)))
	mean_shape = centered / norm
	return mean_shape, aligned_shapes


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


def align_preshape_kabsch(X, Y):
	A = X @ Y.T
	U, S, Vt = np.linalg.svd(A)
	R = Vt.T @ U.T
	if np.linalg.det(R) < 0:
		M = np.identity(Vt.shape[-1])
		M[-1, -1] = -1
		R = Vt.T @ M @ U.T
	return R


def compute_mean_shape_kabsch(shapes, tol=1e-6, max_iter=100):

	D, N, M = shapes.shape
	
	preshapes = populatePreshapeSpace(shapes)
	# Initialize mean_shape with the first shape in preshape space.
	mean_shape = preshapes[:, :, 0].copy()
	
	for it in range(max_iter):
		aligned = np.empty_like(preshapes)
		# Align each preshape to the current mean shape using optimal rotation.
		for m in range(M):
			R = align_preshape_kabsch(mean_shape, preshapes[:, :, m])
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

 
def plot_shape_variation(mean_shape, eigenvectors, eigenvalues, directory, filename, mode=0, k=2):
	for factor in [-k, 0, k]:
		plt.figure(figsize=(6, 6))
		variation = mean_shape + factor * np.sqrt(eigenvalues[mode]) * eigenvectors[:, mode].reshape(2, -1)
		plt.plot(variation[0, :], variation[1, :], label=f'b{mode}={factor}\sqrt{{\lambda_{mode}}}')
		plt.legend()
		plt.xlabel("X-axis")
		plt.ylabel("Y-axis")
		plt.title(f"Shape Variation Along Mode {mode} with Perturbation {factor}")
		plt.grid()
		plt.savefig(os.path.join(directory, f'{filename}_{factor}.png'))
		plt.close()
 
def plot_eigenvalues(eigenvalues, directory, filename):
	for i in range(3):
		plt.figure(figsize=(6, 4))
		plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
		plt.xlabel("Principal Mode")
		plt.ylabel("Eigenvalue (Variance)")
		plt.title(f"Variance Along Principal Mode {i+1}")
		plt.grid()
		plt.savefig(os.path.join(directory, f'{filename}_{i}.png'))
		plt.close()

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

	mean_shape, aligned_shapes = compute_mean_shape_kabsch(shapes)
	plot_aligned_shapes(mean_shape, aligned_shapes, results_folder, "aligned_pointsets_kabsch")

	mean_shape, aligned_shapes = CootesCalculateMean(shapes)
	plot_aligned_shapes(mean_shape, aligned_shapes, results_folder, "aligned_pointsets_cootes")

	algo = "cootes"
	eigenvalues, eigenvectors, mean_shape, _ = compute_pca(shapes, algo=algo)
	for mode in range(3):
		plot_shape_variation(mean_shape, eigenvectors, eigenvalues, results_folder, f"mode_{mode}_{algo}" ,mode=mode)
	plot_eigenvalues(eigenvalues, results_folder, f"eigenvalues_{algo}")