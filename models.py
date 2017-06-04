"""Sequence-to-sequence model with an attention mechanism."""
import numpy as np
import tensorflow as tf

from utils.utils import *

class Model(object):
	"""Class for Seq2Seq model"""
	def __init__(self, cell_size, num_layers, lr=5e-5, cell_type='gru'):
		self.vocab_size = len(OUTPUT_CHARS)
		self.learning_rate = lr
		if cell_type == 'gru':
			cell_class = tf.nn.rnn_cell.GRUCell
		elif cell_type == 'lstm':
			cell_class = tf.nn.rnn_cell.LSTMCell
		else:
			raise ValueError("Cell type '%s' not valid"%cell_type)

		self.cell = single_cell = cell_class(cell_size)
		if num_layers > 1:
			self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

	def step(self):
		pass
		