import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import sys
import random

from utils.utils import DataSet

tf.app.flags.DEFINE_string("dataset", None, 'Path to dataset npz.')
tf.app.flags.DEFINE_integer("batch_size", 128, 'Size of train and valid batches.')
FLAGS = tf.app.flags.FLAGS

class Model(object):
	"""Core training model"""
	def __init__(self, max_len, hidden_units, lr):
		# Placeholders
		self.input_node = tf.placeholder(tf.float32, [None, max_len, 5])
		self.labels = tf.placeholder(tf.float32, [None])

		weights = tf.Variable(tf.random_normal([hidden_units, 1]))
		biases = tf.Variable(tf.random_normal([1]))

		cell = LSTMCell(hidden_units)
		outputs, states = tf.nn.dynamic_rnn(cell,
											self.input_node,
											dtype=tf.float32,
											time_major=False)
		outputs_T = tf.transpose(outputs, [1,0,2])
		last = tf.gather(outputs_T, int(outputs_T.get_shape()[0]) - 1)
		raw_logits = tf.matmul(last, weights) + biases
		self.logits = tf.squeeze(tf.nn.sigmoid(raw_logits))
		self.loss = tf.reduce_mean(tf.square(self.labels - self.logits))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

	def step(self, input_batch, label_batch, sess, valid=False):
		if valid:
			l, pred = sess.run([self.loss, self.logits],
								   feed_dict={
								    self.input_node: input_batch,
								    self.labels: label_batch
								   })
			return l, pred
		else:
			l, pred, _ = sess.run([self.loss, self.logits, self.optimizer],
								   feed_dict={
								    self.input_node: input_batch,
								    self.labels: label_batch
								   })
			return l, pred


def main(*args):
	learning_rate = 0.001
	training_steps = 10000
	valid_step = 50
	seq_max_len = 136 # Sequence max length
	n_hidden = 256 # hidden layer num of features

	dataset = DataSet(FLAGS.dataset)
	model = Model(seq_max_len, n_hidden, learning_rate)

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		for step in range(training_steps):
			train_samples, train_labels = dataset.get_batch(FLAGS.batch_size, 'train')
			loss, prediction = model.step(train_samples, train_labels, sess)

			if (step % valid_step) == 0:
				valid_samples, valid_labels = dataset.get_batch(FLAGS.batch_size, 'valid')
				v_loss, v_prediction = model.step(valid_samples, valid_labels, sess, valid=True)
				print('*'*10+'Step '+str(step)+'*'*10)
				print("Valid loss: %s"%v_loss)
				rmse = np.sqrt(np.mean((v_prediction-valid_labels)**2))
				print("RMSE %s"%rmse)
				diff = np.absolute(v_prediction-valid_labels)
				acc = 1 - np.mean(diff / valid_labels)
				print("Accuracy %s"%acc)

if __name__ == "__main__":
    tf.app.run()
