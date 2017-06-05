import os
import sys
import numpy as np
import tensorflow as tf

from models import Model
from utils.utils import DataSet

tf.app.flags.DEFINE_string("dataset", None, 'Path to dataset npz.')
tf.app.flags.DEFINE_integer("batch_size", 32, 'Size of train and valid batches.')
FLAGS = tf.app.flags.FLAGS

# GLOBALS
TRAINING_STEPS = 100
VALID_STEP = 10
CELL_SIZE = 5
CELL_TYPE = 'gru' # gru or lstm
NUM_LAYERS = 3
BATCH_SIZE = 16

dataset = DataSet(FLAGS.dataset)

def main(*args):
	
	model = Model(CELL_SIZE, NUM_LAYERS, BATCH_SIZE)

	with tf.Session() as sess:

		tf.global_variables_initializer().run()

		for step in range(TRAINING_STEPS):

			if step % VALID_STEP == 0:
				samples_batch, labels_batch = dataset.get_batch(BATCH_SIZE, 'valid')
				model.step(samples_batch, labels_batch, 'valid')
			else:
				samples_batch, labels_batch = dataset.get_batch(BATCH_SIZE, 'train')
				model.step(samples_batch, labels_batch, 'train')

if __name__ == "__main__":
    tf.app.run()