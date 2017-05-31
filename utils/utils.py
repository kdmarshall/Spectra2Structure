from mlxtend.preprocessing import one_hot
import numpy as np
import os

OUTPUT_CHARS = ['#', ')', '(', '+', '-', ',', '/', '.', '1', '0',
                    '3', '2', '5', '4', '7', '6', '9', '8', ':', '=', 'A',
                    '@', 'C', 'B', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
                    'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', '[', 'Z',
                    ']', '\\',  'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
                    #'|', '^', ' ', # Characters that have since been excluded
                    'h', 'l', 'o', 'n', 's', 'r', 'u', 't',
                    '*', 'EOS_ID', 'GO_ID', '<>' # Special tokens
                    ]

VOCAB_SIZE = len(OUTPUT_CHARS)     
UNK_ID = VOCAB_SIZE - 4
EOS_ID = VOCAB_SIZE - 3
GO_ID = VOCAB_SIZE - 2
PAD_ID = VOCAB_SIZE - 1

def encode_label(label):
	label = list(label)
	encoded = [OUTPUT_CHARS.index(c) for c in label]
	return encoded

def ohe_label(label):
	encoded_label = encode_label(label)
	label = one_hot(encoded_label, dtype='int', num_labels=VOCAB_SIZE)
	return label

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


