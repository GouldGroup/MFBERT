import os
import sys
sys.path.append('../')
import torch
import pickle
from tqdm import tqdm
from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
import numpy as np
import pickle

DEVICE = 'cpu'
ATCH_SIZE = 1
EPOCHS = 100
TOKENIZER_DIR = '../Tokenizer/'
OUTPUT_DIR = '../Fingerprints/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class SiameseMFBERTForClintox(torch.nn.Module):
    def __init__(self):
        super(SiameseMFBERTForClintox, self).__init__()
        self.l1 = None
        self.concffnn = None
    
    def forward(self, ids1, mask1, ids2, mask2):
        out1 = torch.mean(self.l1(ids1, mask1)[0], dim=1)
        out2 = torch.mean(self.l1(ids2, mask2)[0], dim=1)
        diff = out1 - out2
        res = torch.cat([out1,out2,diff],1)
        output = self.concffnn(res)
        return output

tokenizer = MFBERTTokenizer.from_pretrained(TOKENIZER_DIR+'Model/',
                                                dict_file = TOKENIZER_DIR+'Model/dict.txt')

model = torch.load('../Model/fine_tuned/Siamese/Clintox/pytorch_model.bin', map_location=DEVICE).to(DEVICE)


with open('../Datasets/data_splits/ClinTox/test.pkl', 'rb') as f:
            testdata = pickle.load(f)
        
smiles = testdata.keys()
data = {}
fps={}
results={}
for smile in tqdm(smiles):
    inputs = tokenizer(smile, return_tensors='pt')
    ids = inputs['input_ids']
    mask = inputs['attention_mask']

    with torch.no_grad():
        res,fp = model(ids, mask)
        
    results[smile] = res.detach().numpy()
    fps[smile] = fp.detach().numpy()

data['fps'] = fps
data['results'] = results

with open(f'{OUTPUT_DIR}Clintox_Siamese_preds_and_fps.pkl', 'wb') as f:
    pickle.dump(data,f)