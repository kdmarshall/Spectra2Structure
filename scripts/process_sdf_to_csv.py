import os,sys
import numpy as np
import pandas as pd

from indigo.indigo import Indigo

sdf_file = sys.argv[1]
path_segments = [seg for seg in sdf_file.split('/') if seg != '']
out_file = os.path.join('/',*path_segments[:-1],'raw_spectra_data.csv')
print(out_file)
indigo = Indigo()

data = []
headers = set(['name','smiles'])
for mol in indigo.iterateSDFile(sdf_file):
	compound_data = {}
	name = mol.name()
	compound_data['name'] = name
	# print(name)
	try:
		canon_smiles = mol.canonicalSmiles()
		smiles = canon_smiles.split('|')[0].strip()
	except Exception as e:
		# print(e)
		# smiles = mol.smiles()
		continue
	compound_data['smiles'] = smiles
	# print(canon_smiles)
	for prop in mol.iterateProperties():
		prop_name = prop.name()
		headers.add(prop_name)
		prop_data = prop.rawData()
		compound_data[prop_name] = prop_data
		# print(prop.name(), ":", prop.rawData())
	data.append(compound_data)
	# break

# print(data)
df = pd.DataFrame(data,columns=list(headers),index=None)
print(df.columns)
df.to_csv(out_file,index=False)
