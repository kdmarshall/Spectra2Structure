import os,sys
import numpy as np
import pandas as pd

from utils.utils import (multiplet_to_ohe,
					    ohe_label,
					    smiles_to_mw)

MULTIPLETS = ['S','D','T','Q']
seq_max_len = 136

def featurize(row):
	sample_list = [cell for cell in row.split('|') if cell != '']
	points = []
	for sample in sample_list:
		shift, multiplet, position = sample.split(';')
		multiplet = multiplet[-1].upper()
		if multiplet not in MULTIPLETS:
			continue
		point = {'shift':float(shift),
				 'multiplet':multiplet,
				 'position':int(position)}
		points.append(point)
	points.sort(key=lambda x: x['position'])
	data = [[point['shift']]+multiplet_to_ohe(point['multiplet']).tolist()
					for point in points]
	return data

csv_file = sys.argv[1]
spectra_to_keep = ['Spectrum 1H','Spectrum 13C']
df = pd.read_csv(csv_file, index_col=None, low_memory=False)

cols_to_featurize = []
for col in df.columns.tolist():
	for spectra in spectra_to_keep:
		if spectra in col:
			cols_to_featurize.append(col)

# print(cols_to_featurize)

data = []
for index,row in df.iterrows():
	sample = {'label':row['smiles'],
			   'samples':''}
	for col in cols_to_featurize:
		if pd.notnull(row[col]):
			sample['samples'] += row[col]

	data.append(sample)

out_df = pd.DataFrame(data,index=None)

# out_df['samples'] = out_df['samples'].apply(featurize)
# out_df['label'] = out_df['label'].apply(ohe_label)
# out_df['label'] = out_df['label'].apply(smiles_to_mw)

# samples = np.array(out_df['samples'].tolist())
# labels = np.array(out_df['label'].tolist())
samples_list = []
labels_list = []
for index, row in out_df.iterrows():
	data = featurize(row['samples'])
	if not data or len(data) == 0:
		continue
	try:
		label = smiles_to_mw(row['label'])
	except:
		continue
	samples_list.append(data)
	labels_list.append(label)

# samples = np.array(samples_list)
max_val = 0
min_val = 1e9

for sample in samples_list:
	shifts = [vals[0] for vals in sample]
	local_max = max(shifts)
	local_min = min(shifts)
	if local_max > max_val:
		max_val = local_max
	if local_min < min_val:
		min_val = local_min

def min_max_scale(val):
	return (val - min_val) / (max_val - min_val)

samples = []
for sample in samples_list:
	_sample = []
	for block in sample:
		scaled_shift = min_max_scale(block[0])
		block[0] = scaled_shift
		_sample.append(block)
	samples.append(_sample)

# print(samples)
padded_samples = []
for seq in samples:
	seq_len = len(seq)
	if seq_len < seq_max_len:
		pad_len = seq_max_len - seq_len
		padded_seq = seq + [[0]*5 for _ in range(pad_len)]
		padded_samples.append(padded_seq)
	else:
		padded_samples.append(seq)

samples = np.array(padded_samples)

# shifts = []
# for sample in samples:
# 	row = []
# 	for features in sample:
# 		row.append(features[0])
# 	shifts.append(row)

# scaled_shifts = scale(shifts)
# samples[:,:,1] = scaled_shifts

print("Samples shape {}".format(samples.shape))
# # print("Longest sequence %s"%longest_seq)
labels = np.array(labels_list)
print("Labels shape %s"%labels.shape)
out_file = 'data/scaled_features_mw_head.npz' if 'head' in csv_file else 'data/scaled_features_mw.npz'
np.savez(out_file, samples=samples, labels=labels)

