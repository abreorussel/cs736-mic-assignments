from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import map_coordinates

def myXrayIntegration(image, delta_t, delta_s, theta_values = np.arange(0,180,1)):
    image_size = image.shape[0]
    t_values = np.arange(-image_size, image_size, delta_t)
    s_values = np.arange(-image_size, image_size, delta_s)
    print(len(theta_values))
    print(len(t_values))
    radon_transform = np.zeros((len(theta_values), len(t_values)))

    for i, theta in enumerate(theta_values):
        theta_radian = np.deg2rad(theta)

        for j, t in enumerate(t_values):
            x = t * np.cos(theta_radian) - s_values * np.sin(theta_radian)
            y = t * np.sin(theta_radian) + s_values * np.cos(theta_radian)  

            interpolated_values = map_coordinates(image, 
                                                  [y + image_size//2, x + image_size//2], 
                                                  order=1, mode='constant', cval=0)

            radon_transform[i][j] = np.sum(interpolated_values)
    return radon_transform


def display_image(image):
    img = plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    phantom = shepp_logan_phantom()
    
    print(f"Image Size : {phantom.shape}")
    phantom = cv2.resize(phantom, (128, 128))
    print(f"After Resize : {phantom.shape}")

    radon_transform = myXrayIntegration(phantom, 1, 1)
    plt.figure(figsize=(8, 6))
    plt.imshow(radon_transform, extent=[-90, 90, 180, 0], aspect='auto')
    plt.xlabel("Projection Position (t)")
    plt.ylabel("Angle (theta)")
    plt.title("Radon Transform (X-ray Integration)")
    plt.colorbar(label="Intensity")
    plt.savefig("radon.png")