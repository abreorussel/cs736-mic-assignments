import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def optimal_transform(X, Y):
	"""
	Given two pointsets X and Y (of size N x 2), compute the optimal 
	similarity transform (scaling, rotation, translation) aligning X to Y.
	Returns:
	  s : optimal scale
	  R : 2x2 optimal rotation matrix
	  t : optimal translation vector (2,)
	Both X and Y are assumed to be already centered.
	"""
	# Compute the optimal rotation using the singular value decomposition (SVD)
	A = X.T @ Y
	U, _, Vt = np.linalg.svd(A)
	R = Vt.T @ U.T
	if np.linalg.det(R) < 0:
		M = np.identity(Vt.shape[-1])
		M[-1, -1] = -1
		R = Vt.T @ M @ U.T

	# Optimal scale is computed as:
	s = np.trace((Y.T @ (X @ R))) / np.trace(X.T @ X)
	
	# Since the shapes are centered, the translation is simply the difference of centroids.
	# (If not centered, one would compute centroids of original shapes.)
	t = np.zeros(2)  
	return s, R, t

def align_shape(shape, mean_shape):
	"""
	Align a shape (N x 2) to a mean shape (N x 2).
	We first center both shapes, then compute the optimal rotation and scale.
	Then the shape is transformed.
	"""
	# Center both shapes
	shape_centered = shape - shape.mean(axis=0)
	mean_centered = mean_shape - mean_shape.mean(axis=0)
	
	# Compute optimal transform (translation is zero now since shapes are centered)
	s, R, _ = optimal_transform(shape_centered, mean_centered)
	aligned = s * (shape_centered @ R)
	return aligned

def compute_mean_shape_squared(shapes, max_iter=50, tol=1e-5):
	"""
	Compute the mean shape by minimizing the sum of squared Procrustes distances.
	Uses an iterative procedure:
	  1. Initialize the mean shape (e.g., by averaging centered shapes).
	  2. For each shape, compute the optimal alignment to the current mean.
	  3. Update the mean shape as the (L2) average of the aligned shapes.
	  4. Normalize (center and scale) the new mean shape.
	  5. Repeat until convergence.
	"""
	num_shapes = shapes.shape[0]
	# Initialize: center each shape and average them
	aligned_shapes = []
	for i in range(num_shapes):
		centered = shapes[i] - shapes[i].mean(axis=0)
		aligned_shapes.append(centered)
	mean_shape = np.mean(aligned_shapes, axis=0)
	# Normalize mean shape (centroid zero and unit Frobenius norm)
	mean_shape = mean_shape - mean_shape.mean(axis=0)
	mean_shape = mean_shape / np.linalg.norm(mean_shape)
	
	for it in range(max_iter):
		new_aligned = []
		for i in range(num_shapes):
			new_aligned.append(align_shape(shapes[i], mean_shape))
		new_aligned = np.array(new_aligned)
		new_mean = np.mean(new_aligned, axis=0)
		new_mean = new_mean - new_mean.mean(axis=0)
		new_mean = new_mean / np.linalg.norm(new_mean)
		diff = np.linalg.norm(new_mean - mean_shape)
		# Update mean shape
		mean_shape = new_mean
		if diff < tol:
			break
	return mean_shape, new_aligned

def compute_geometric_median(points, eps=1e-5, max_iter=100):
	"""
	Compute the geometric median (L1 mean) of an array of points.
	points is an array of shape (M, D) (here, D = 2N, i.e. each shape vectorized).
	We use the Weiszfeld algorithm.
	"""
	median = np.mean(points, axis=0)
	for _ in range(max_iter):
		distances = np.linalg.norm(points - median, axis=1)
		# Avoid division by zero: add small epsilon to distances if necessary.
		distances = np.where(distances < eps, eps, distances)
		new_median = np.sum(points / distances[:, None], axis=0) / np.sum(1 / distances)
		if np.linalg.norm(new_median - median) < eps:
			return new_median
		median = new_median
	return median

def compute_mean_shape_abs(shapes, max_iter=50, tol=1e-5):
	"""
	Compute the mean shape by minimizing the sum of (unsquared) Procrustes distances.
	As before, we iteratively align shapes to the current mean.
	Instead of an L2 average, we update the mean by computing the geometric median of the aligned shapes.
	"""
	num_shapes = shapes.shape[0]
	# Initialize: center each shape and average them
	aligned_shapes = []
	for i in range(num_shapes):
		centered = shapes[i] - shapes[i].mean(axis=0)
		aligned_shapes.append(centered)
	mean_shape = np.mean(aligned_shapes, axis=0)
	mean_shape = mean_shape - mean_shape.mean(axis=0)
	mean_shape = mean_shape / np.linalg.norm(mean_shape)
	
	for it in range(max_iter):
		new_aligned = []
		for i in range(num_shapes):
			new_aligned.append(align_shape(shapes[i], mean_shape))
		new_aligned = np.array(new_aligned)
		# Vectorize the aligned shapes for geometric median computation
		vecs = new_aligned.reshape(num_shapes, -1)  # each row is a vectorized shape
		new_mean_vec = compute_geometric_median(vecs)
		new_mean = new_mean_vec.reshape(mean_shape.shape)
		new_mean = new_mean - new_mean.mean(axis=0)
		new_mean = new_mean / np.linalg.norm(new_mean)
		diff = np.linalg.norm(new_mean - mean_shape)
		mean_shape = new_mean
		if diff < tol:
			break
	return mean_shape, new_aligned

def plot_shapes(shapes, mean_shape, title, folder, filename):
	"""
	Plot original shapes, the mean shape, and aligned shapes.
	"""
	plt.figure(figsize=(8,6))
	# Plot original shapes in light gray
	for shape in shapes:
		plt.plot(shape[:,0], shape[:,1], 'o-', color='gray', alpha=0.5)
	# Plot aligned shapes in blue (if provided)
	if mean_shape is not None:
		plt.plot(mean_shape[:,0], mean_shape[:,1], 'r*-', linewidth=2, markersize=10, label='Mean Shape')
	plt.title(title)
	plt.axis('equal')
	plt.legend()
	plt.savefig(os.path.join(folder, filename))
	plt.close()

if __name__ == "__main__":
	# Create results folder if needed
	results_folder = "results/Q3"
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print(f'Folder "{results_folder}" created.')
	
	# Load dataset
	data = loadmat('data/robustShapeMean2D.mat')
	robshape = data['pointsets']  # shape: (num_shapes, num_points, 2)
	num_shapes, num_points, dim = robshape.shape
	print(f"Loaded {num_shapes} shapes, each with {num_points} points in {dim}D.")
	
	# Plot original shapes
	# (For display purposes, we plot each shape in its raw form)
	plt.figure(figsize=(8,6))
	for i in range(num_shapes):
		plt.plot(robshape[i,:,0], robshape[i,:,1], 'o-', color='gray', alpha=0.5)
	plt.title("Original Pointsets")
	plt.axis('equal')
	plt.savefig(os.path.join(results_folder, "original_shapes.png"))
	plt.close()
	
	# (a) Mean shape with sum of squared distances (L2)
	mean_shape_sq, aligned_sq = compute_mean_shape_squared(robshape)
	# Plot the estimated mean and aligned shapes
	plt.figure(figsize=(8,6))
	for shape in aligned_sq:
		plt.plot(shape[:,0], shape[:,1], 'o-', color='blue', alpha=0.5)
	plt.plot(mean_shape_sq[:,0], mean_shape_sq[:,1], 'r*-', linewidth=2, markersize=10, label='Mean (L2)')
	plt.title("Aligned Shapes and Mean (Sum of Squared Distances)")
	plt.axis('equal')
	plt.legend()
	plt.savefig(os.path.join(results_folder, "mean_shape_squared.png"))
	plt.close()
	
	# (b) Mean shape with sum of absolute distances (L1, geometric median)
	mean_shape_abs, aligned_abs = compute_mean_shape_abs(robshape)
	# Plot the estimated mean and aligned shapes
	plt.figure(figsize=(8,6))
	for shape in aligned_abs:
		plt.plot(shape[:,0], shape[:,1], 'o-', color='green', alpha=0.5)
	plt.plot(mean_shape_abs[:,0], mean_shape_abs[:,1], 'r*-', linewidth=2, markersize=10, label='Mean (L1)')
	plt.title("Aligned Shapes and Mean (Sum of Distances)")
	plt.axis('equal')
	plt.legend()
	plt.savefig(os.path.join(results_folder, "mean_shape_abs.png"))
	plt.close()
	
	print("Processing complete. Figures saved in:", results_folder)
