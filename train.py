import os
import sys
import numpy as np
import tensorflow as tf

from models import BaseModel as Model
from utils.utils import DataSet

tf.app.flags.DEFINE_string("dataset", None, 'Path to dataset npz.')
tf.app.flags.DEFINE_integer("batch_size", 32, 'Size of train and valid batches.')
FLAGS = tf.app.flags.FLAGS

dataset = DataSet(FLAGS.dataset)

def main(*args):
	
	model = Model(5, 3)
	samples_batch, labels_batch = dataset.get_batch(10,'train')
	print(samples_batch.shape)
	print(labels_batch.shape)


if __name__ == "__main__":
    tf.app.run()