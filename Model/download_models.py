import requests
from tqdm import tqdm
import zipfile
import sys
import os
import shutil

base_path = sys.path[0]+'/'

MODEL_URLS = {'1':'https://figshare.com/ndownloader/files/28937358',
          '2':'https://figshare.com/ndownloader/files/30790099',
          '3':'https://figshare.com/ndownloader/files/30784018',
          '4':'https://figshare.com/ndownloader/files/30784024',
          '5':'https://figshare.com/ndownloader/files/30784300',
          '6':'https://figshare.com/ndownloader/files/28937370',
          '7':'https://figshare.com/ndownloader/files/30784021',
          '8':'https://figshare.com/ndownloader/files/30784048',
          '9':'https://figshare.com/ndownloader/files/30784402',
          '10':'https://figshare.com/ndownloader/files/30789946',
          '11':'https://figshare.com/ndownloader/files/30784135',
          '12':'https://figshare.com/ndownloader/files/30784213'}

MODEL_NAMES = {'1':'pre-trained checkpoint',
          '2':'rdkit_featurizer',
          '3':'BBBP_featurizer',
          '4':'Clintox_featurizer',
          '5':'HIV_featurizer',
          '6':'tox21_featurizer',
          '7':'Siamese-BBBP',
          '8':'Siamese-Clintox',
          '9':'Siamese-HIV',
          '10':'Lipophilicity',
          '11':'ESOL',
          '12':'FreeSolv'}

def download_and_extract(selected):
    for model in tqdm(selected):
        print(f'Downloading {MODEL_NAMES[model]}.zip...')
        with open(base_path + MODEL_NAMES[model]+'.zip', 'wb') as f:
            f.write(requests.get(MODEL_URLS[model]).content)
        
        print(f'Extracting {MODEL_NAMES[model]}.zip...')
        with zipfile.ZipFile(base_path + f"{MODEL_NAMES[model]}.zip","r") as zipf:
            zipf.extractall(base_path)

        if model in ['3','4','5','6','7','8','9','10','11','12']:
            os.makedirs(base_path + "fine_tuned", exist_ok=True)
            shutil.move(base_path + f"{MODEL_NAMES[model]}", base_path + "fine_tuned/")

        try:
            os.remove(base_path + f"{MODEL_NAMES[model]}.zip")
        except OSError:
            pass      
    
    print("Done")
    return


if __name__=='__main__':
    model_selection = input('''Please select which model weight(s) to download (comma separated):

    0: ALL
    1: Pre-trained checkpoint (for fine-tuning)
    2: RDKit Benchmarking platform featurizer
    3: BBBP_featurizer
    4: Clintox_featurizer
    5: HIV_featurizer
    6: tox21_featurizer
    7: Siamese BBBP featurizer/predictor
    8: Siamese Clintox featurizer/predictor
    9: Siamese HIV featurizer/predictor
    10: Lipophilicity featurizer/predictor
    11: ESOL featurizer/predictor
    12: FreeSolv featurizer/predictor

''')

    if '0' in model_selection:
        selected = ['1','2','3','4','5','6','7','8','9','10','11','12']
    else:
        selected = [i.strip() for i in model_selection.split(',')]

    download_and_extract(selected)