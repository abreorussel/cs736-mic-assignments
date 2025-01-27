from config import *
import numpy as np

# denoising image with gaussian noise model and quadratic prior model
noisy_image = phantom['imageNoiseless']
image_noisy = phantom['imageNoisy']
