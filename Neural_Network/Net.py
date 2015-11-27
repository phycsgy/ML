# /usr/env/bin python
# coding=utf-8
""" This is a general module for neuron net work.
"""
# Libraries
# Standard Libraries
# import json
import random
import copy
import sys
import os
# import time
# import math
# Third-party libraries
import numpy as np
import matplotlib.pyplot as mp
import Net_Util
import mnist_loader


class net_monitor(object):
	def __init__(self, net, training_data, validation_data):
		self.net = net
		validation_inputs_data, validation_results_data = \
			Net_Util.seperate(validation_data)

		training_inputs_data, training_results_data = \
			Net_Util.seperate(training_data)

		self.evaluation_cost = np.sum(Net_Util.Cost.crossentropycost(
			net.feedforword(validation_inputs_data), validation_results_data))

		self.evaluation_accurate = Net_Util.accurate(
			net.feedforword(training_inputs_data), training_results_data)

		self.training_cost = np.sum(Net_Util.Cost.crossentropycost(
			net.feedforword(training_inputs_data), training_results_data))

		self.training_accurate = Net_Util.accurate(
			net.feedforword(training_inputs_data), training_results_data)


class Layer(object):
	def __init__(self, size, pre_layer=None, next_layer=None):
		self.size = size
		self.pre_layer = pre_layer
		self.next_layer = next_layer
		self.inputs = None
		self.outputs = None
		if pre_layer is not None:
			# initialize the weights of layer
			self.weights, self.biases = Net_Util.weights_initialize(self.size, self.pre_layer.size)
		else:
			self.weights = None
			self.biases = None
		self.delta_weights = None
		self.delta_biases = None
		self.delta = None
		self.regular = None

	def feedforword(self, inputs, softmax=False):
		if self.pre_layer is None:  # input layer
			self.inputs = inputs
			self.outputs = inputs
		else:                       # hiden layer and output layer
			self.inputs = np.dot(self.pre_layer.feedforword(inputs), self.weights.T) \
				+ np.dot(self.biases, np.ones((1, len(inputs)))).T
			if softmax:
				self.outputs = Net_Util.Activation.softmax(self.inputs)
			else:
				self.outputs = Net_Util.Activation.sigmoid(self.inputs)
		return self.outputs

	def backpropagation(self, result_data):
		if self.pre_layer is None:       # input layer
			self.next_layer.backpropagation(result_data)
		else:
			if self.next_layer is None:  # output layer  δ(l) = ∂C/∂Z(l) = ∂C/∂A(l) * ∂A(l)/∂Z(l)
				self.delta = Net_Util.Cost.crossentropycost_prime(self.outputs, result_data) \
					* Net_Util.Activation.sigmoid_prime(self.inputs)
			else:                        # hiden layer   δ(j) = ∂C/∂Z(j) = ∂C/∂Z(j+1) * ∂Z(j+1)/∂A(j) * ∂A(j)/∂Z(j)
				self.delta = np.dot(self.next_layer.backpropagation(result_data), self.next_layer.weights) \
					* Net_Util.Activation.sigmoid_prime(self.inputs)
			# ∂C/∂B(l) = ∂C/∂Z(l) * ∂Z(l)/∂B(l) = ∂C/∂Z(l)  = δ(l)
			delta_biases_shape = (self.delta.shape[0], self.delta.shape[1], 1)
			self.delta_biases = self.delta.reshape(delta_biases_shape)
			# ∂C/∂W(j) = ∂C/∂Z(j) * ∂Z(j)/∂W(j) = δ(j) * A(j-1)
			# reshape for array multiply
			# array(12,1,10)*array(12,20,1) = array(12,20,10)
			delta_shape = (self.delta.shape[0], self.delta.shape[1], 1)
			outputs_shape = (self.pre_layer.outputs.shape[0], 1, self.pre_layer.outputs.shape[1])
			self.delta_weights = self.pre_layer.outputs.reshape(outputs_shape) * self.delta.reshape(delta_shape)

		return self.delta

	def regularization(self, lamda=0.0):
		self.regular = lamda*self.weights
		return self.regular


class Network(object):
	def __init__(self, size):
		"""size is a list of positive number.
		The length of size represent the number of network'layer
		and the value of size represent the number of neurons in each layer
		"""
		self.size = size
		self.layer_num = len(size)
		self.input_layer, self.hiden_layers, self.output_layer = self.layers_initilization(self.size)
		self.outputs = None

	def layers_initilization(self, size):
		input_layer = Layer(size[0])
		pre_layer = input_layer
		hiden_layers = []
		for i in xrange(1, self.layer_num - 1):
			hiden_layer = Layer(size[i], pre_layer)
			hiden_layers.append(hiden_layer)
			pre_layer.next_layer = hiden_layer
			pre_layer = hiden_layer
		output_layer = Layer(size[-1], pre_layer)
		pre_layer.next_layer = output_layer

		return input_layer, hiden_layers, output_layer

	def feedforword(self, inputs):
		return self.output_layer.feedforword(inputs)

	def net_training(self, training_data, eta, lamda):
		"""Training the net work with stochastic gradient descent
			eta: learning rate
			lamda: regulation rate

		"""
		input_data, result_data = Net_Util.seperate(training_data)
		# feedforword
		self.outputs = self.feedforword(input_data)
		# backpropagation
		self.input_layer.backpropagation(result_data)
		# update weights and biases
		for layer in self.hiden_layers:
			layer.weights = layer.weights - eta*(np.sum(layer.delta_weights, axis=1)/len(input_data)
				+ layer.regularization(lamda))  # regularization
			layer.biases = layer.biases - eta*np.sum(layer.delta_biases, axis=1)/len(input_data)


def SGD(training_data, batch_size, epoch, eta, lamda, validation_data):
	n_sample = len(training_data)
	net = Network([784, 30, 10])
	net_monitors = []
	for i in xrange(epoch):
		random.shuffle(training_data)
		mini_baches = [training_data[k:k+batch_size] for k in xrange(0, n_sample, batch_size)]
		for mini_batch in mini_baches:
			net.net_training(mini_batch, eta, lamda)
		net_monitors.append(copy.deepcopy(net_monitor(net, training_data, validation_data)))


if __name__ == "__main__":
	print 'Neuron Network Test'
	# load data
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	training_data = training_data[0:600]
	validation_data = validation_data[0:100]
	test_data = test_data[0:100]
	# init net
	Net = Network([784, 30, 10])
	# traning and mornitor
	SGD(training_data, 30, 30, 0.5, 0.1, validation_data)
	# visualization
