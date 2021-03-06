from mlxtend.preprocessing import one_hot
import numpy as np
import os
import sys
import random
import math

from utils.indigo.indigo import Indigo

OUTPUT_CHARS = ['#', ')', '(', '+', '-', ',', '/', '.', '1', '0',
                    '3', '2', '5', '4', '7', '6', '9', '8', ':', '=', 'A',
                    '@', 'C', 'B', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
                    'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', '[', 'Z',
                    ']', '\\',  'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
                    'h', 'l', 'o', 'n', 's', 'r', 'u', 't',
                    '*', 'EOS_ID', 'GO_ID', '<>'
                    ]

VOCAB_SIZE = len(OUTPUT_CHARS)
UNK_ID = VOCAB_SIZE - 4
EOS_ID = VOCAB_SIZE - 3
GO_ID = VOCAB_SIZE - 2
PAD_ID = VOCAB_SIZE - 1

def encode_label(label):
	label_list = list(label)
	try:
		encoded = [OUTPUT_CHARS.index(c) for c in label_list]
	except Exception as e:
		print(e)
		sys.exit(label)
	return encoded

def ohe_label(label):
	encoded_label = encode_label(label)
	label = one_hot(encoded_label, dtype='int', num_labels=VOCAB_SIZE)
	return label

def decode_label(encoded_label):
	char_indicies = np.argmax(encoded_label, axis=1)
	return char_indicies

def decode_ohe(encoded_label):
	char_indicies = decode_label(encoded_label)
	raw_str = ''.join([OUTPUT_CHARS[idx] for idx in char_indicies])
	return raw_str

def multiplet_to_ohe(multiplet):
	multiplet_map = {'S':0,
					 'D':1,
					 'T':2,
					 'Q':3}
	assert multiplet in multiplet_map.keys(),\
		   "Multiplet %s not recognized" % multiplet
	num_labels = len(multiplet_map.keys())
	val = multiplet_map[multiplet]
	return np.squeeze(one_hot(np.array([val]), dtype='int', num_labels=num_labels))

def smiles_to_mw(smiles):
	indigo = Indigo()
	mol = indigo.loadMolecule(smiles)
	return mol.molecularWeight()


class DataSet(object):
	def __init__(self, file_path, valid_size=0.2):
		self.file_path = file_path
		self.npzfile = np.load(self.file_path)
		self.files = self.npzfile.files
		self.samples = self.npzfile['samples']
		self.labels = self.npzfile['labels']
		assert self.samples.shape[0] == self.labels.shape[0], "Number of samples and labels are not equal"
		self.set_size = self.samples.shape[0]
		total_indices = list(range(self.set_size))
		random.shuffle(total_indices)
		valid_partition = math.floor(self.set_size*valid_size)
		train_indices = total_indices[:-valid_partition]
		valid_indicies = total_indices[-valid_partition:]
		self.train_samples = self.samples[train_indices]
		self.train_labels = self.labels[train_indices]
		self.valid_samples = self.samples[valid_indicies]
		self.valid_labels = self.labels[valid_indicies]

	def get_batch(self, batch_size, batch_type):
		assert batch_type in ('train','valid'), "Unrecognized batch_type %s" % batch_type

		def batch_helper(samples, labels):
			indices = np.random.randint(0, samples.shape[0], batch_size)
			samples_batch = samples[indices]
			labels_batch = labels[indices]
			return (samples_batch, labels_batch)

		if batch_type == 'train':
			samples_batch, labels_batch = batch_helper(self.train_samples, self.train_labels)
		else:
			samples_batch, labels_batch = batch_helper(self.valid_samples, self.valid_labels)
		return (samples_batch, labels_batch)
