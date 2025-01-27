from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


phantom = loadmat('data/assignmentImageDenoising_phantom.mat')


imageNoiseless = phantom['imageNoiseless']
imageNoisy = phantom['imageNoisy']


def display_image(image):
    plt.imshow(image)
    plt.show()
    print(image)

def normalize_image(image):
    return ( image - np.min(image) ) / ( np.max(image) - np.min(image) )






