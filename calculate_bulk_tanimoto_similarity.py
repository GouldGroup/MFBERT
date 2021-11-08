from rdkit import Chem, DataStructs
import pickle
import glob
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import gc
import numpy as np
from pqdm.processes import pqdm

# Load benchmark fps
with open('all_rdkit-ecfc4.pkl', 'rb') as f:
    rdkit_fps = pickle.load(f)

# For each molecule in training set, compare with every other molecule in rdkit benchmark
def calculate_sim_for_query(query_fp):
        tan_sim = DataStructs.BulkTanimotoSimilarity(query_fp, rdkit_fps)
        mean_sim = np.mean(tan_sim)
        sim_std = np.std(tan_sim)
        return (mean_sim, sim_std)
    
def calc_sim(subset):
    with open(subset, 'rb') as f:
        fps = pickle.load(f)
        
    means_n_sds = pqdm(fps, calculate_sim_for_query, n_jobs=2)
    count = len(means_n_sds)
    meansum=0
    stdsum=0
    
    for i,j in means_n_sds:
        meansum+=i
        stdsum+=j
    
    fin_mean = meansum/count
    fin_std = stdsum/count
    
    return fin_mean, fin_std



        
sim_dict = {} #subset:(mean,std)
for subset in ['ABCDEFGH','ABCDEFG','ABCDEF','ABCDE','ABCD','ABC','AB']:
    print('calculating', subset)
    mean, std = calc_sim(subset+'-ecfc4.pkl')
    sim_dict[subset] = (mean, std)
    
    
with open('tanimoto_similarities.pkl', 'wb') as f:
    pickle.dump(sim_dict,f)