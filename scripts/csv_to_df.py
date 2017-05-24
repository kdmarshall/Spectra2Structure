import os,sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.utils import multiplet_to_ohe

MULTIPLETS = ['S','D','T','Q']

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

out_df['samples'] = out_df['samples'].apply(featurize)

out_df.to_csv('data/features.csv',index=False)

