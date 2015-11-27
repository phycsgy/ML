# /usr/env/bin python
# coding=utf-8
"""This module contains most used function in neural net work"""
# Libraries
# Standard Libraries
# import json
import random
import sys
import os
# import time
# import math
# Third-party libraries
import numpy as np


class Activation(object):
	@staticmethod
	def sigmoid(inputs):
		"""sigmoid function"""
		return 1.0/(1.0 + np.exp(-1.0*inputs))

	@staticmethod
	def sigmoid_prime(inputs):
		"""derivative of the sigmoid function."""
		return np.exp(-1.0*inputs)/((1.0 + np.exp(-1.0*inputs)) * (1.0 + np.exp(-1.0*inputs)))

	@staticmethod
	def tangent(inputs):
		return (np.exp(inputs) - np.exp(-1.0*inputs))/(np.exp(inputs) + np.exp(-inputs))

	@staticmethod
	def softmax(inputs):
		return np.exp(inputs)/np.dot(np.exp(inputs).sum(axis=1), np.ones((1, len(inputs))))


class Cost(object):
	@staticmethod
	def crossentropycost(output_data, result_data):
		n_sample = len(output_data)
		# C = -[y*ln(x) + (1-y)*ln(1-x)]
		cost_vector = -1.0*result_data*np.log(output_data) \
			+ (np.ones(result_data.shape) - result_data)*np.log(output_data)
		cost = np.sum(cost_vector)/n_sample
		return cost

	@staticmethod
	def crossentropycost_prime(output_data, result_data):
		# ∂C/∂A(l) = (1-y)/(1-x) - y/x
		cost_prime = (1.0*np.ones(result_data.shape) - result_data)/(1.0*np.ones(output_data.shape) - output_data) \
			- 1.0*result_data/output_data
		return cost_prime


def accurate(outputs_data, results_data):
	return 1.0 - np.count_nonzero(np.argmax(outputs_data, axis=1) - np.argmax(results_data, axis=1)) \
		/ len(outputs_data)


def seperate(training_data):
	"""vectorlization the data: data is a list containing n
	2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
	containing the input image.  ``y`` is a 10-dimensional
	numpy.ndarray representing the unit vector corresponding to the
	correct digit for ``x`` """
	n = len(training_data)
	input_shape = (n, 784)
	result_shape = (n, 10)
	input_data = np.array([b[0] for b in training_data]).reshape(input_shape)
	if isinstance(training_data[0][1], np.ndarray):
		result_data = np.array([b[1] for b in training_data]).reshape(result_shape)
	else:
		result_data = np.array([[0]*b[1] + [1] + [0]*(9-b[1]) for b in training_data]).reshape(result_shape)
	return input_data, result_data


def weights_initialize(size, pre_layer_size):
	return np.random.randn(size, pre_layer_size)/np.sqrt(size), np.random.randn(size, 1)