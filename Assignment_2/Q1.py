from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
import cv2
import numpy as np


def display_image(image):
    img = plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    phantom = shepp_logan_phantom()
    
    print(f"Image Size : {phantom.shape}")
    phantom = cv2.resize(phantom, (128, 128))
    print(f"After Resize : {phantom.shape}")

    # display_image(phantom)

    size = 128
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    X, Y = np.meshgrid(x, y)

    # Plot the phantom image
    plt.figure(figsize=(6, 6))
    plt.imshow(phantom, extent=[-size//2, size//2, -size//2, size//2])
    plt.xlabel("X-axis (pixels)")
    plt.ylabel("Y-axis (pixels)")
    plt.title("Shepp-Logan Phantom (128x128) with Centered Origin")
    plt.colorbar(label="Intensity")
    plt.show()