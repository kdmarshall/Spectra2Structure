import os
import sys
import numpy as np
import tensorflow as tf

from models import BaseModel as Model

# tf.app.flags.DEFINE_string("dataset", None, 'Path to dataset h5.')
# FLAGS = tf.app.flags.FLAGS

def main(*args):
	
	model = Model(5, 3)


if __name__ == "__main__":
    tf.app.run()