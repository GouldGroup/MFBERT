import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import pandas as pd
import torch
from transformers import RobertaForMaskedLM
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
import numpy as np
assert torch.cuda.device_count() == 1
from rdkit import Chem

MAX_LEN = 514
TRAIN_BATCH_SIZE = 12
VALID_BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-05
TOKENIZER_DIR = 'Tokenizer/'


exclude = ['CNCC(O)c1ccc(OC(=O)C(C)(C)C)c(OC(=O)C(C)(C)C)c1']

class HIVDataset(Dataset):
    def __init__(self):
        examples = []
        with open('Datasets/data_splits/HIV/train.pkl', 'rb') as f:
            traindata = pickle.load(f)
        for smiles, label in traindata.items():
            if smiles in exclude:
                continue
            if '.' not in smiles:
                try:
                    augsmiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True)
                    examples.append((smiles, augsmiles, label))
                except:
                    print('skipping', smiles)
                    continue
            else:
                mols = [Chem.MolFromSmiles(s) for s in smiles.split(".")]
                augsmiles = ".".join([Chem.MolToSmiles(mol, doRandom = True) for mol in mols])
                examples.append((smiles, augsmiles, label))
            
            

        self.data = examples
        self.tokenizer = MFBERTTokenizer.from_pretrained(TOKENIZER_DIR+'Model/',
                                                dict_file = TOKENIZER_DIR+'Model/dict.txt')
        self.max_len = 514
        
    def __getitem__(self, idx):
        example = self.data[idx]
        s1 = example[0]
        s2 = example[1]
        target = example[2]
        inputs1 = self.tokenizer.encode_plus(
            s1,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        inputs2 = self.tokenizer.encode_plus(
            s2,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids1 = inputs1['input_ids']
        mask1 = inputs1['attention_mask']
        ids2 = inputs2['input_ids']
        mask2 = inputs2['attention_mask']
        
        return ({'input_ids':torch.tensor(ids1, dtype=torch.long), 
                             'attention_mask':torch.tensor(mask1, dtype=torch.long), 
                             'label':torch.tensor(target, dtype=torch.long)}, 
                {'input_ids':torch.tensor(ids2, dtype=torch.long), 
                             'attention_mask':torch.tensor(mask2, dtype=torch.long)})
    
    def __len__(self):
        return len(self.data)
    
class SiameseMFBERTForHIV(torch.nn.Module):
    def __init__(self):
        super(SiameseMFBERTForHIV, self).__init__()
        self.l1 = list(RobertaForMaskedLM.from_pretrained('Model/weights').children())[0]
        self.concffnn = torch.nn.Linear(768*3, 1)
    
    def forward(self, ids1, mask1, ids2, mask2):
        out1 = torch.mean(self.l1(ids1, mask1)[0], dim=1)
        out2 = torch.mean(self.l1(ids2, mask2)[0], dim=1)
        diff = out1 - out2
        res = torch.cat([out1,out2,diff],1)
        output = self.concffnn(res)
        return output
    
trainds = HIVDataset()

model = SiameseMFBERTForHIV().cuda()

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }


training_loader = DataLoader(trainds, **train_params)

# Creating the loss function and optimizer
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

curminloss = 1

def train(epoch):
    model.train()

    for _ , data in tqdm(enumerate(training_loader, 0), desc='ITERATION', total=len(training_loader)):
        ids1 = data[0]['input_ids'].cuda()
        mask1 = data[0]['attention_mask'].cuda()
        ids2 = data[1]['input_ids'].cuda()
        mask2 = data[1]['attention_mask'].cuda()
        targets = data[0]['label'].float().cuda()
        global curminloss
        outputs = model(ids1, mask1, ids2, mask2).squeeze()
        optimizer.zero_grad()
        
        
        try:
            loss = loss_function(outputs, targets)
        except:
            print(_)
            print(ids1, ids2)
        
        if _%100==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            # save best model
            if loss.item()<curminloss:
                torch.save(model, f'fine-tuned/siamese_classification/HIV_model_best_{loss.item()}_siamese.bin')
                curminloss = loss.item()
                print('saving best...')

        loss.backward()
        optimizer.step()
        
for epoch in tqdm(range(EPOCHS), desc='EPOCHS'):
    train(epoch)


torch.save(model, 'fine-tuned/siamese_classification/HIV_model_last_siamese.bin')
