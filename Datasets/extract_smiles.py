import pandas as pd
import os
from tqdm import tqdm

files = os.listdir()
for file in tqdm(files):
    if '.csv' in file:
        df = pd.read_csv(file)
        smiles = df['smiles'].tolist()

        with open(file+'.smi', 'w') as f:
            for i in smiles:
                f.write(f'{i}\n')