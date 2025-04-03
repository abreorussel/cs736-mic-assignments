import mat73
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
import math

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
	# for m in range(M):
	# 	shape = aligned_shapes[:, :, m]
	# 	plt.plot(shape[0, :], shape[1, :], 'o-', color='lightgray', alpha=0.7)

	for m in range(M):
		shape = aligned_shapes[:, :, m]
		half = shape.shape[1] // 2  # Get the halfway index

		# Close the loop by adding the first point at the end
		inner_x = np.append(shape[0, :half], shape[0, 0])
		inner_y = np.append(shape[1, :half], shape[1, 0])

		outer_x = np.append(shape[0, half:], shape[0, half])
		outer_y = np.append(shape[1, half:], shape[1, half])

		# Plot inner boundary (loop closed)
		plt.plot(inner_x, inner_y, 'o-', color='blue', alpha=0.7, label="Inner" if m == 0 else None)

		# Plot outer boundary (loop closed)
		plt.plot(outer_x, outer_y, 'o-', color='red', alpha=0.7, label="Outer" if m == 0 else None)

	
	# Overlay the mean shape in red with thicker line and markers.
	# plt.plot(mean_shape[0, :], mean_shape[1, :], 'ro-', linewidth=2, markersize=8, label="Mean Shape")
	# Get halfway index
	half = mean_shape.shape[1] // 2  

	# Close the loop for inner part
	mean_inner_x = np.append(mean_shape[0, :half], mean_shape[0, 0])
	mean_inner_y = np.append(mean_shape[1, :half], mean_shape[1, 0])

	# Close the loop for outer part
	mean_outer_x = np.append(mean_shape[0, half:], mean_shape[0, half])
	mean_outer_y = np.append(mean_shape[1, half:], mean_shape[1, half])

	# Plot inner part (using green instead of red/blue)
	plt.plot(mean_inner_x, mean_inner_y, 'go-', linewidth=2, markersize=8, label="Mean Inner Shape")

	# Plot outer part (using orange instead of red/blue)
	plt.plot(mean_outer_x, mean_outer_y, 'mo-', linewidth=2, markersize=8, label="Mean Outer Shape")

	
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
		# plt.plot(variation[0, :], variation[1, :], label=f'b{mode}={factor}\sqrt{{\lambda_{mode}}}')

		# Get halfway index
		half = variation.shape[1] // 2  

		# Close the loop for the first half
		var_inner_x = np.append(variation[0, :half], variation[0, 0])
		var_inner_y = np.append(variation[1, :half], variation[1, 0])

		# Close the loop for the second half
		var_outer_x = np.append(variation[0, half:], variation[0, half])
		var_outer_y = np.append(variation[1, half:], variation[1, half])

		# Plot first half (use cyan for distinction)
		plt.plot(var_inner_x, var_inner_y, 'co-', label=f'Inner b{mode}={factor}\sqrt{{\lambda_{mode}}}')

		# Plot second half (use purple for distinction)
		plt.plot(var_outer_x, var_outer_y, 'yo-', label=f'Outer b{mode}={factor}\sqrt{{\lambda_{mode}}}')

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

def get_line_points(center, theta, max_length):
    """
    Generate a list of (row, col) points along a ray starting from 'center',
    at angle 'theta' (radians), extending out to 'max_length'.
    """
    num_samples = int(round(max_length))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    points = []
    for d in range(num_samples + 1):
        r = int(round(center[0] + d * sin_theta))
        c = int(round(center[1] + d * cos_theta))
        points.append((r, c))
    
    return points

def find_ring_boundaries_on_ray(img, center, theta, max_length):
    """
    For a single ray from 'center' at angle 'theta', detect the contiguous white region.
    Return two boundary points for that region:
      1) The first white pixel (inner boundary).
      2) The last white pixel in that contiguous region (outer boundary).
    
    If no white region is found, returns an empty list.
    """
    line_pts = get_line_points(center, theta, max_length)
    
    in_white = False
    region_start = None
    region_end = None
    
    for i, (r, c) in enumerate(line_pts):
        if r < 0 or r >= img.shape[0] or c < 0 or c >= img.shape[1]:
            if in_white and region_start is not None:
                region_end = i - 1
            break
        
        pixel_val = img[r, c]
        if pixel_val == 255:
            if not in_white:
                in_white = True
                region_start = i
        else:
            if in_white:
                region_end = i - 1
                break
    
    if in_white and region_end is None:
        region_end = len(line_pts) - 1
    
    boundaries = []
    if region_start is not None and region_end is not None:
        inner_edge = line_pts[region_start]
        outer_edge = line_pts[region_end]
        boundaries = [inner_edge, outer_edge]
    
    return boundaries

def generate_radial_pointset(img_path, num_angles=36):
    """
    Generates a single combined pointset for a ring-shaped object.
    For each ray, extracts two points:
      - Inner boundary: first white pixel.
      - Outer boundary: last white pixel of that contiguous white region.

    The points are ordered such that all inner boundary points come first,
    followed by all outer boundary points.

    Returns:
        NumPy array of shape (2, number_of_points), where:
        - The first half of columns correspond to inner boundary points.
        - The second half correspond to outer boundary points.
    """
    # Read and binarize the image.
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read the image at: " + img_path)

    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    h, w = img_bin.shape

    # Use the image center as a reference.
    center = (h // 2, w // 2)
    max_length = math.sqrt(h**2 + w**2)

    inner_points = []
    outer_points = []

    for theta in np.linspace(0, 2 * math.pi, num_angles, endpoint=False):
        boundaries = find_ring_boundaries_on_ray(img_bin, center, theta, max_length)
        if len(boundaries) == 2:
            inner_points.append(boundaries[0])
            outer_points.append(boundaries[1])

    # Stack inner and outer points together
    if inner_points and outer_points:
        inner_array = np.array(inner_points).T  # shape (2, num_angles)
        outer_array = np.array(outer_points).T  # shape (2, num_angles)

        # Concatenate along axis 1 to form (2, num_angles * 2)
        pointset = np.hstack((inner_array, outer_array))
    else:
        pointset = np.empty((2, 0))

    return pointset


if __name__ == "__main__":
	
	image_dir = "data/anatomicalSegmentations/anatomicalSegmentations/*.png"
	image_paths = glob(image_dir)

	pointsets = [generate_radial_pointset(path, num_angles=40) for path in image_paths]

	shapes = np.stack(pointsets, axis=2)
	print("Final tensor shape (dims, num_points, num_examples):", shapes.shape)

	results_folder = "results/Q2"
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