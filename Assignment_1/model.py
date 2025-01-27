import numpy as np
from config import *


def gaussian_likelihood_gradient(x, y):
    return 2*(x - y)/np.square(sigma_likelihood)

# def get_4_neighbor_values(image, i, j):
#     height, width = image.shape
#     neighbor_values = []

#     left_value = image[i, (j - 1) % width]
#     neighbor_values.append(left_value)

#     right_value = image[i, (j + 1) % width]
#     neighbor_values.append(right_value)

#     up_value = image[(i - 1) % height, j]
#     neighbor_values.append(up_value)

#     down_value = image[(i + 1) % height, j]
#     neighbor_values.append(down_value)
    
#     return neighbor_values

# def quadratic_prior_gradient(image, x, y):
#     current_pixel = image[x][y]
#     neighbors = get_4_neighbor_values(image, x, y)
#     return 2*np.sum(neighbors - current_pixel)

def rrmse(A,B):
	rrmse = np.sqrt(np.sum(np.square(A-B))/np.sum(A*2))
	return rrmse

def calc_quadratic_prior(x):
    up = np.roll(x, 1, axis=0)
    down = np.roll(x, -1, axis=0)
    right = np.roll(x, -1, axis=1)
    left = np.roll(x, 1, axis=1)

    left_prior = np.square(np.abs(left - x))
    right_prior = np.square(np.abs(right - x))
    up_prior = np.square(np.abs(up - x))
    down_prior = np.square(np.abs(down - x))

    left_prior_grad = 2 * (left - x)
    right_prior_grad = 2 * (right - x)
    up_prior_grad = 2 * (up - x)
    down_prior_grad = 2 * (down - x)

    prior = np.sum(left_prior, right_prior, up_prior, down_prior)
    prior_grad = np.sum(left_prior_grad, right_prior_grad ,up_prior_grad ,down_prior_grad)

    return prior, prior_grad

    




