import numpy as np
from config import *


################
# Likelihoods
################


def gaussian_likelihood(x, y):
	likelihood =  - np.sum(np.abs((x - y)**2) / np.square(sigma_likelihood))
	likelihood_gradient = - 2*(x - y) / np.square(sigma_likelihood)
	return likelihood, likelihood_gradient



##########
# Priors
##########

def get_clique_differences(x):
	up = np.roll(x, 1, axis=0)
	down = np.roll(x, -1, axis=0)
	right = np.roll(x, -1, axis=1)
	left = np.roll(x, 1, axis=1)

	return up, down, right, left



def calc_quadratic_prior(x, gamma=None):
	up, down, right, left = get_clique_differences(x)

	left_prior = np.sum(np.square(np.abs(left - x)))
	right_prior = np.sum(np.square(np.abs(right - x)))
	up_prior = np.sum(np.square(np.abs(up - x)))
	down_prior = np.sum(np.square(np.abs(down - x)))

	left_prior_grad = 2 * np.abs(left - x) * np.sign(left - x) 
	right_prior_grad = 2 * np.abs(right - x) * np.sign(right - x)
	up_prior_grad = 2 * np.abs(up - x) * np.sign(up - x)
	down_prior_grad = 2 * np.abs(down - x) * np.sign(down - x)

	prior = - (left_prior + right_prior + up_prior + down_prior)
	prior_grad =  (left_prior_grad + right_prior_grad + up_prior_grad + down_prior_grad)

	return prior, prior_grad



def calc_huber_prior(x, gamma=0):
	up, down, right, left = get_clique_differences(x)

	def huber(diff, gamma):
		abs_diff = np.abs(diff) 
		return np.where(abs_diff <= gamma, 0.5 * (diff ** 2), gamma * abs_diff - 0.5 * (gamma ** 2))
	
	def huber_grad(diff, gamma):
		abs_diff = np.abs(diff) 
		return np.where(abs_diff <= gamma, diff, gamma * np.sign(diff))
	
	left_prior = np.sum(huber(left - x, gamma))
	right_prior = np.sum(huber(right - x, gamma))
	up_prior = np.sum(huber(up - x, gamma))
	down_prior = np.sum(huber(down - x, gamma))

	left_prior_grad = huber_grad(left - x, gamma)
	right_prior_grad = huber_grad(right - x, gamma)
	up_prior_grad = huber_grad(up - x, gamma)
	down_prior_grad = huber_grad(down - x, gamma)

	prior = - (left_prior + right_prior + up_prior + down_prior)
	prior_grad =  (left_prior_grad + right_prior_grad + up_prior_grad + down_prior_grad)

	return prior, prior_grad

def calc_adaptive_prior(x, gamma=0):
	up, down, right, left = get_clique_differences(x)

	def adaptive(diff, gamma):
		abs_diff = np.abs(diff)
		return (gamma * abs_diff) - (gamma**2) * np.log(1 + (abs_diff / gamma))
	
	def adaptive_grad(diff, gamma):
		abs_diff = np.abs(diff)
		return  (gamma * np.sign(diff)) - (gamma / (1 + (abs_diff / gamma))) * np.sign(diff) 

	left_prior = np.sum(adaptive(left - x, gamma))
	right_prior = np.sum(adaptive(right - x, gamma))
	up_prior = np.sum(adaptive(up - x, gamma))
	down_prior = np.sum(adaptive(down - x, gamma))

	left_prior_grad = adaptive_grad(left - x, gamma)
	right_prior_grad = adaptive_grad(right - x, gamma)
	up_prior_grad = adaptive_grad(up - x, gamma)
	down_prior_grad = adaptive_grad(down - x, gamma)

	prior = - (left_prior + right_prior + up_prior + down_prior)
	prior_grad =  (left_prior_grad + right_prior_grad + up_prior_grad + down_prior_grad)
	return prior, prior_grad



def calculate_posterior(x, y, alpha=0.5, gamma = 0, likelihood="gaussian", prior="quadratic"):
	# prior, prior_grad = calc_quadratic_prior(x)
	# prior, prior_grad = calc_huber_prior(x, gamma)
	
    likelihood_mapping = {
		"gaussian": gaussian_likelihood,
    }

    prior_mapping = {
		"quadratic":calc_quadratic_prior,
		"huber": calc_huber_prior,
		"adaptive":calc_adaptive_prior
    }
	
    likelihood_fn = likelihood_mapping[likelihood]
    prior_fn = prior_mapping[prior]
	
    prior, prior_grad = prior_fn(x, gamma)
    likelihood, likelihood_grad = likelihood_fn(x, y)

    log_posterior = alpha*(prior) + (1 - alpha)*likelihood
    log_posterior_grad = alpha*(prior_grad) + (1 - alpha)*likelihood_grad
	
    print(log_posterior)

    return log_posterior, log_posterior_grad


def rrmse(A,B):
	rrmse = np.sqrt(np.sum(np.square(np.abs(A)-np.abs(B)))/np.sum(A**2))
	return rrmse

	
def grid_search(gamma_end, prior="quadratic" ,alpha_start=0, alpha_end=1, alpha_increment=0.1, gamma_start=0):
    pass

