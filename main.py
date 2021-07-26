import torch
import numpy as np
from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
from Model.model import MFBERT
from tqdm import tqdm, trange
import os
import pickle

DEVICE = 'cpu'
BATCH_SIZE = 1
DATA_DIR = 'Data/'
TOKENIZER_DIR = 'Tokenizer/'
OUTPUT_DIR = 'Fingerprints/'


def generate_dict_from_results(results):
    smiles_fingerprint_dict = {}
    for batch in results:
        smiles = batch[0]
        res = batch[1]
        for i in range(len(smiles)):
            smiles_fingerprint_dict[smiles[i]]=res[i]
    return smiles_fingerprint_dict


if __name__ == '__main__':

    excepted = []
    excepted_counter = 0

    tokenizer = MFBERTTokenizer.from_pretrained(TOKENIZER_DIR+'Model/',
                                                dict_file = TOKENIZER_DIR+'Model/dict.txt')

    model = MFBERT().to(DEVICE)

    for DATA_FILE in tqdm(os.listdir(DATA_DIR)):
        if DATA_FILE.startswith('.'):
            continue

        OUTPUT_FILE = f'{OUTPUT_DIR}/{DATA_FILE.split(".")[0]}_fingerprints.pkl'

        with open(f'{DATA_DIR}/{DATA_FILE}','r') as f:
            data = f.read().splitlines()


        all_res = []
        for batch in trange(0,len(data), BATCH_SIZE):

            smiles_batch = data[batch:batch+BATCH_SIZE]

            # Note the padding tokens will affect the mean embedding
            inputs = tokenizer(smiles_batch, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            try:
                res = model(inputs).detach().numpy() # numpy tensor of mean embeddings/batch
                all_res.append((smiles_batch,res))
            except:
                excepted.append(smiles_batch)
                excepted_counter+=1

                print('EXCEPTION OCCURRED TOTAL:',excepted_counter)
                
        
        dres = generate_dict_from_results(all_res)

        with open(OUTPUT_FILE, 'wb') as g:
            pickle.dump(dres, g)