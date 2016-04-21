# /usr/env/bin python
# coding=utf-8
"""This module contains most used function in neural net work"""

import matplotlib.pyplot as plt


class Trainlog(object):

	def __init__(self, learning_method, learning_rate,
				activation_method, regularzation_method, regularzation_rate):
		self.learning_method = learning_method
		self.learning_rate = learning_rate
		self.activation_method = activation_method
		self.regularzation_method = regularzation_method
		self.regularzation_rate = regularzation_rate
		self.fig_text = "Activation method:" + activation_method \
						+ "\n Learning method:" + learning_method \
						+ "\n Learning rate:" + learning_rate \
						+ "\n Regularzation method:" + regularzation_method \
						+ "\n Regularzation rate:" + regularzation_rate
		self.training_accurate = []
		self.test_accurate = []
		self.iterations = 0

	def logging(self, training_accurate, test_accurate):
		self.training_accurate.append(training_accurate)
		self.test_accurate.append(test_accurate)
		self.iterations += 1

	def visualize(self):
		fig = plt.figure()
		train_line, test_line = plt.plot(range(1, self.iterations+1), self.training_accurate,
					range(1, self.iterations+1), self.test_accurate)
		train_line.set(linewidth=2.0, linestyle="--")
		test_line.set(linewidth=2.0, linestyle="--")
		plt.xlabel('Iterations')
		plt.ylabel('Accurate')
		plt.title('Training Results')
		ax = fig.add_axes()
		ax.text(self.fig_text, horizontalalignment='right',
				verticalalignment='bottom', transform=ax.transAxes)
		plt.show()





