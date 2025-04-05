import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA


def load_images_from_folder(folder_path, size=(64, 64)):
	images = []
	images_2d = []
	for filename in sorted(os.listdir(folder_path)):
		if filename.lower().endswith(('.png')):
			path = os.path.join(folder_path, filename)
			image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
			image = image.astype(np.float32) / 255.0
			images_2d.append(image)
			images.append(image.flatten())
	return np.array(images), np.array(images_2d)

def plot_image(img, title="Image", cmap='gray'):
	plt.figure()
	plt.imshow(img, cmap=cmap)
	plt.title(title)
	plt.axis('off')
	plt.savefig(os.path.join(results_folder, f'{title}.png'))
	plt.close()

def plot_eigen_spectrum(eigenvalues, title="Eigen Spectrum"):
	plt.figure()
	plt.plot(eigenvalues, marker='o')
	plt.title(title)
	plt.xlabel("Principal Component Index")
	plt.ylabel("Explained Variance")
	plt.grid(True)
	plt.savefig(os.path.join(results_folder, f'{title}.png'))
	plt.close()

def plot_modes(mean_img, modes, shape=(64,64), titles=["Mode 1", "Mode 2"]):
	plot_image(mean_img.reshape(shape), title="pca_mean")
	for i, mode in enumerate(modes[:2]):
		plot_image(mode.reshape(shape), title=f'pca_mode_{i}')

def plot_modes_of_variation(mean_img, components, eigenvalues, shape=(64, 64), n_modes=2, scale=2):
	mean_img = mean_img.reshape(-1)
	for i in range(n_modes):
		v = components[i]
		std_dev = np.sqrt(eigenvalues[i])
		variation_pos = mean_img + scale * std_dev * v
		variation_neg = mean_img - scale * std_dev * v

		plt.figure(figsize=(10, 3))
		plt.suptitle(f"PCA: Mode {i+1} of Variation Around the Mean", fontsize=14)

		plt.subplot(1, 3, 1)
		plt.imshow(variation_neg.reshape(shape), cmap='gray')
		plt.title("- Variation")
		plt.axis('off')

		plt.subplot(1, 3, 2)
		plt.imshow(mean_img.reshape(shape), cmap='gray')
		plt.title("Mean Image")
		plt.axis('off')

		plt.subplot(1, 3, 3)
		plt.imshow(variation_pos.reshape(shape), cmap='gray')
		plt.title("+ Variation")
		plt.axis('off')

		# plt.show()
		plt.savefig(os.path.join(results_folder, 'PCA_variation.png'))
		plt.close()

def perform_pca(images, n_components=10):
	pca = PCA(n_components=n_components)
	transformed = pca.fit_transform(images)
	return pca, transformed

def perform_kernel_pca(images, n_components=10, gamma=1e-4):
	kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma, fit_inverse_transform=True)
	transformed = kpca.fit_transform(images)
	preimage_mean = find_preimage_mean(images, gamma=gamma)
	return kpca, transformed, preimage_mean

def rbf_kernel(X, z, gamma):
	distances = np.sum((X - z)**2, axis=1)
	return np.exp(-gamma * distances)

def objective(z, X, gamma):
	kernels = rbf_kernel(X, z, gamma)
	return np.mean(kernels)

def compute_gradient(z, X, gamma):
	diffs = X - z
	kernels = np.exp(-gamma * np.sum(diffs**2, axis=1))
	grad = (2 * gamma / X.shape[0]) * np.sum(diffs * kernels[:, np.newaxis], axis=0)
	return grad

def find_preimage_mean(X, gamma=1e-4, lr=0.1, max_iter=100, tol=1e-6):

	z = np.zeros(X.shape[1])
	prev_obj = objective(z, X, gamma)
	# print(f"Mean Pre-image: Initial objective: {prev_obj:.6f}")
	
	for i in range(max_iter):
		grad = compute_gradient(z, X, gamma)
		grad_norm = np.linalg.norm(grad)
		z_new = z + lr * grad
		curr_obj = objective(z_new, X, gamma)
		# print(f"Mean Pre-image: Iter {i+1}: Objective = {curr_obj:.6f}, Grad norm = {grad_norm:.6f}")
		
		if np.abs(curr_obj - prev_obj) < tol:
			# print("Mean Pre-image: Converged.")
			z = z_new
			break
		z = z_new
		prev_obj = curr_obj
	return z

def objective_weighted(z, X, weights, gamma):
	kernels = np.exp(-gamma * np.sum((X - z)**2, axis=1))
	return np.dot(weights, kernels) / np.sum(weights)

def compute_gradient_weighted(z, X, weights, gamma):
	diffs = X - z
	kernels = np.exp(-gamma * np.sum(diffs**2, axis=1))
	weighted = weights * kernels
	grad = (2 * gamma / np.sum(weights)) * np.sum(diffs * weighted[:, np.newaxis], axis=0)
	return grad

def find_preimage_weighted(X, weights, gamma=1e-4, lr=0.1, max_iter=100, tol=1e-6):
	z = np.zeros(X.shape[1])  # initialize z as zeros
	prev_obj = objective_weighted(z, X, weights, gamma)
	# print(f"Weighted Pre-image: Initial objective: {prev_obj:.6f}")
	
	for i in range(max_iter):
		grad = compute_gradient_weighted(z, X, weights, gamma)
		grad_norm = np.linalg.norm(grad)
		z_new = z + lr * grad
		curr_obj = objective_weighted(z_new, X, weights, gamma)
		# print(f"Weighted Pre-image: Iter {i+1}: Objective = {curr_obj:.6f}, Grad norm = {grad_norm:.6f}")
		
		if np.abs(curr_obj - prev_obj) < tol:
			# print("Weighted Pre-image: Converged.")
			z = z_new
			break
		z = z_new
		prev_obj = curr_obj
	return z

def compute_projection_weights(kpca, image, n_components=3):
	image_trans = kpca.transform(image.reshape(1, -1))
	X_proj = kpca.transform(kpca.X_fit_)
	weights = np.dot(X_proj[:, :n_components], image_trans[0, :n_components])
	return weights


def reconstruct_pca(pca, image, n_components=3):
	coeffs = np.dot(image - pca.mean_, pca.components_[:n_components].T)
	reconstruction = pca.mean_ + np.dot(coeffs, pca.components_[:n_components])
	return reconstruction

def reconstruct_kernel_pca(kpca, image, n_components=3, gamma=1e-4):
	image_trans = kpca.transform(image.reshape(1, -1))
	if kpca.n_components > n_components:
		pad = np.zeros((1, kpca.n_components - n_components))
		image_trans = np.hstack([image_trans[:, :n_components], pad])
	weights = compute_projection_weights(kpca, image, n_components)
	preimage = find_preimage_weighted(kpca.X_fit_, weights, gamma=gamma, lr=0.1, max_iter=100)
	return preimage

if __name__ == "__main__":
	results_folder = "results/Q4"
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print(f'Folder "{results_folder}" created.')

	folder_path = "data/anatomicalSegmentations"
	images, images_2d = load_images_from_folder(folder_path)


	pca_model, pca_transformed = perform_pca(images, n_components=10)

	plot_eigen_spectrum(pca_model.explained_variance_, "pca_eigen_spectrum")

	plot_modes(pca_model.mean_, pca_model.components_)


	plot_modes_of_variation(
		mean_img=pca_model.mean_,
		components=pca_model.components_,
		eigenvalues=pca_model.explained_variance_,
		shape=(64, 64),
		n_modes=2,
		scale=2
	)

	kpca_model, kpca_transformed, preimage_mean = perform_kernel_pca(images, n_components=10, gamma=1e-4)
	plot_eigen_spectrum(kpca_model.eigenvalues_, "kpca_eigen_spectrum")
	plot_image(preimage_mean.reshape(64, 64), "kpca_mean")

	reconstructions_pca = []
	reconstructions_kpca = []

	for img in images:
		rec_pca = reconstruct_pca(pca_model, img)
		rec_kpca = reconstruct_kernel_pca(kpca_model, img, n_components=3, gamma=1e-4)
		reconstructions_pca.append(rec_pca)
		reconstructions_kpca.append(rec_kpca)

	num_examples = 1
	for i in range(num_examples):
		plt.figure(figsize=(12, 4))
		plt.suptitle(f"Reconstruction Comparison for Image #{i + 1}", fontsize=14)

		plt.subplot(1, 3, 1)
		plt.imshow(images_2d[i], cmap='gray')
		plt.title("Original Segmentation")
		plt.axis('off')

		plt.subplot(1, 3, 2)
		plt.imshow(reconstructions_pca[i].reshape(64, 64), cmap='gray')
		plt.title("PCA Reconstruction (Top 3 Modes)")
		plt.axis('off')

		plt.subplot(1, 3, 3)
		plt.imshow(reconstructions_kpca[i].reshape(64, 64), cmap='gray')
		plt.title("Kernel PCA Reconstruction (Top 3 Modes)")
		plt.axis('off')

		plt.savefig(os.path.join(results_folder, f'reconstruction_{i + 1}.png'))
		plt.close()
