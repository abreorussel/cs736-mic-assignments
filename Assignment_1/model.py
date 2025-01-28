import numpy as np
from config import *


################
# Likelihoods
################


def gaussian_likelihood(x, y):
	likelihood =  - np.abs((x - y)**2) / np.square(sigma_likelihood)
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



def calc_quadratic_prior(x):
	up, down, right, left = get_clique_differences(x)

	left_prior = np.square(np.abs(left - x))
	right_prior = np.square(np.abs(right - x))
	up_prior = np.square(np.abs(up - x))
	down_prior = np.square(np.abs(down - x))

	left_prior_grad = 2 * (left - x)
	right_prior_grad = 2 * (right - x)
	up_prior_grad = 2 * (up - x)
	down_prior_grad = 2 * (down - x)

	prior = - (left_prior + right_prior + up_prior + down_prior)
	prior_grad =  (left_prior_grad + right_prior_grad + up_prior_grad + down_prior_grad)

	return prior, prior_grad



def calc_huber_prior(x):
	up, down, right, left = get_clique_differences(x)

	def huber(diff, gamma):
		abs_diff = np.abs(diff) 
		return np.where(abs_diff <= gamma, 0.5 * (diff ** 2), gamma * abs_diff - 0.5 * (gamma ** 2))
	
	def huber_grad(diff, gamma):
		abs_diff = np.abs(diff) 
		return np.where(abs_diff <= gamma, diff, gamma * np.sign(diff))
	
	left_prior = huber(left - x, gamma)
	right_prior = huber(right - x, gamma)
	up_prior = huber(up - x, gamma)
	down_prior = huber(down - x, gamma)

	left_prior_grad = huber_grad(left - x, gamma)
	right_prior_grad = huber_grad(right - x, gamma)
	up_prior_grad = huber_grad(up - x, gamma)
	down_prior_grad = huber_grad(down - x, gamma)

	prior = - (left_prior + right_prior + up_prior + down_prior)
	prior_grad =  (left_prior_grad + right_prior_grad + up_prior_grad + down_prior_grad)

	return prior, prior_grad

def calc_adaptive_prior(x):
	up, down, right, left = get_clique_differences(x)

	def adaptive(diff, gamma):
		abs_diff = np.abs(diff)
		return (gamma * abs_diff) - (gamma**2) * np.log(1 + (abs_diff / gamma))
	
	def adaptive_grad(diff, gamma):
		abs_diff = np.abs(diff)
		return  (gamma * np.sign(diff)) - (gamma / (1 + (abs_diff / gamma))) * np.sign(diff) 

	left_prior = adaptive(left - x, gamma)
	right_prior = adaptive(right - x, gamma)
	up_prior = adaptive(up - x, gamma)
	down_prior = adaptive(down - x, gamma)

	left_prior_grad = adaptive_grad(left - x, gamma)
	right_prior_grad = adaptive_grad(right - x, gamma)
	up_prior_grad = adaptive_grad(up - x, gamma)
	down_prior_grad = adaptive_grad(down - x, gamma)

	prior = - (left_prior + right_prior + up_prior + down_prior)
	prior_grad =  (left_prior_grad + right_prior_grad + up_prior_grad + down_prior_grad)
	return prior, prior_grad



		
		 

	 
	


def calculate_posterior(x, y, alpha=alpha):
	# prior, prior_grad = calc_quadratic_prior(x)
	# prior, prior_grad = calc_huber_prior(x)
	prior, prior_grad = calc_adaptive_prior(x)
	likelihood, likelihood_grad = gaussian_likelihood(x, y)

	log_posterior = alpha*(prior) + (1 - alpha)*likelihood
	log_posterior_grad = alpha*(prior_grad) + (1 - alpha)*likelihood_grad

	return log_posterior, log_posterior_grad




def rrmse(A,B):
	rrmse = np.sqrt(np.sum(np.square(np.abs(A)-np.abs(B)))/np.sum(A**2))
	return rrmse

	




