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

	left_prior_grad = 2 * (left - x)  
	right_prior_grad = 2 * (right - x) 
	up_prior_grad = 2 * (up - x) 
	down_prior_grad = 2 * (down - x) 

	prior = - (left_prior + right_prior + up_prior + down_prior)
	prior_grad =  (left_prior_grad + right_prior_grad + up_prior_grad + down_prior_grad)

	return prior, prior_grad



def calc_huber_prior(x, gamma=0):
	up, down, right, left = get_clique_differences(x)

	def huber(diff, gamma):
		abs_diff = np.abs(diff) 
		return np.where(abs_diff <= gamma, 0.5 * (diff ** 2), (gamma * abs_diff) - 0.5 * (gamma ** 2))
	
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




def calc_square_l2_prior(x, gamma=None):

	up, down, right, left = get_clique_differences(x)

	left_prior = np.sum(np.square(left - x))
	right_prior = np.sum(np.square(right - x))
	up_prior = np.sum(np.square(up - x))
	down_prior = np.sum(np.square(down - x))

	left_prior_grad = 2 * (left - x)  
	right_prior_grad = 2 * (right - x) 
	up_prior_grad = 2 * (up - x) 
	down_prior_grad = 2 * (down - x) 

	prior = - (left_prior + right_prior + up_prior + down_prior)
	prior_grad =  (left_prior_grad + right_prior_grad + up_prior_grad + down_prior_grad)

	return prior, prior_grad


def calc_l2_prior(x, gamma=None):
	epsilon = 1e-8
	up, down, right, left = get_clique_differences(x)

	left_prior = np.sum(np.sqrt(np.square(left - x)))
	right_prior = np.sum(np.sqrt(np.square(right - x)))
	up_prior = np.sum(np.sqrt(np.square(up - x)))
	down_prior = np.sum(np.sqrt(np.square(down - x)))

	left_prior_grad = (left - x) * (1 / (np.sqrt(np.square(left - x) + epsilon)))  
	right_prior_grad = (right - x) * (1 / (np.sqrt(np.square(right - x) + epsilon))) 
	up_prior_grad = (up - x) * (1 / (np.sqrt(np.square(up - x) + epsilon))) 
	down_prior_grad = (down - x) * (1 / (np.sqrt(np.square(down - x) + epsilon))) 
	
	prior = - (left_prior + right_prior + up_prior + down_prior)
	prior_grad =  (left_prior_grad + right_prior_grad + up_prior_grad + down_prior_grad)

	return prior, prior_grad


def calc_huber_l1_prior(x, gamma=0):
	up, down, right, left = get_clique_differences(x)

	def huber(l1_norm):
		return np.where(l1_norm <= gamma, 0.5 * (l1_norm ** 2), gamma * l1_norm - 0.5 * (gamma ** 2))
	
	def huber_grad(diff, l1_norm):
		return np.where(l1_norm <= gamma, l1_norm * np.sign(diff), gamma * np.sign(diff))
	
	
	l1_norm_left = np.sum(np.abs(left - x))
	l1_norm_right = np.sum(np.abs(right - x))
	l1_norm_up = np.sum(np.abs(up - x))
	l1_norm_down = np.sum(np.abs(down - x))

	left_prior_grad = huber_grad(left - x ,l1_norm_left)
	right_prior_grad = huber_grad(right - x ,l1_norm_right)
	up_prior_grad = huber_grad(up - x ,l1_norm_up)
	down_prior_grad = huber_grad(down - x ,l1_norm_down)

	prior = - (huber(l1_norm_left) + huber(l1_norm_right) + huber(l1_norm_up) + huber(l1_norm_down))
	prior_grad =  (left_prior_grad + right_prior_grad + up_prior_grad + down_prior_grad)

	return prior, prior_grad


##############
# Posterior
##############



def calculate_posterior(x, y, alpha=0.5, gamma = 0, likelihood="gaussian", prior="quadratic"):
	# prior, prior_grad = calc_quadratic_prior(x)
	# prior, prior_grad = calc_huber_prior(x, gamma)
	
	likelihood_mapping = {
		"gaussian": gaussian_likelihood,
	}

	prior_mapping = {
		"quadratic":calc_quadratic_prior,
		"huber": calc_huber_prior,
		"adaptive":calc_adaptive_prior,
		"square-l2": calc_square_l2_prior,
		"l2": calc_l2_prior,
		"huber-l1": calc_huber_l1_prior
	}
	
	likelihood_fn = likelihood_mapping[likelihood]
	prior_fn = prior_mapping[prior]
	
	prior, prior_grad = prior_fn(x, gamma)
	
	likelihood, likelihood_grad = likelihood_fn(x, y)

	log_posterior = alpha*(prior) + (1 - alpha)*likelihood
	log_posterior_grad = alpha*(prior_grad) + (1 - alpha)*likelihood_grad
	
	# print(log_posterior)

	return log_posterior, log_posterior_grad


def rrmse(A,B):
	rrmse = np.sqrt(np.sum(np.square(np.abs(A)-np.abs(B)))/np.sum(A**2))
	return rrmse

