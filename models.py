"""Sequence-to-sequence model with an attention mechanism."""
import numpy as np
import tensorflow as tf

from utils.utils import *

class Model(object):
	"""Class for Seq2Seq model"""
	def __init__(self, cell_size, num_layers, batch_size, lr=5e-5, cell_type='gru'):
		vocab_size = len(OUTPUT_CHARS)
		self.learning_rate = lr
		self.batch_size = batch_size
		self.source_vocab_size = source_vocab_size
		self.target_vocab_size = vocab_size

		if cell_type == 'gru':
			cell_class = tf.nn.rnn_cell.GRUCell
		elif cell_type == 'lstm':
			cell_class = tf.nn.rnn_cell.LSTMCell
		else:
			raise ValueError("Cell type '%s' not valid"%cell_type)

		self.cell = single_cell = cell_class(cell_size)
		if num_layers > 1:
			self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

		# The seq2seq function: we use embedding for the input and attention.
		def seq2seq_func(encoder_inputs, decoder_inputs, do_decode):
			return tf.nn.seq2seq.embedding_attention_seq2seq(
				encoder_inputs,
				decoder_inputs,
				self.cell,
				num_encoder_symbols=self.source_vocab_size,
				num_decoder_symbols=self.target_vocab_size,
				embedding_size=5,
				feed_previous=do_decode)

	def step(self, samples_batch, labels_batch, step_type):
		assert step_type in ('train','valid'), "Unrecognized step_type %s" % step_type

		print(samples_batch.shape)
		print(labels_batch.shape)
		