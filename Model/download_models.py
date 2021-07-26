import requests
from tqdm import tqdm
import zipfile
import sys
import os
import shutil

base_path = sys.path[0]+'/'

MODEL_URLS = {'1':'https://ndownloader.figshare.com/files/28937358',
          '2':'https://ndownloader.figshare.com/files/28937361',
          '3':'https://ndownloader.figshare.com/files/28937364',
          '4':'https://ndownloader.figshare.com/files/28937367',
          '5':'https://ndownloader.figshare.com/files/28937370'}

MODEL_NAMES = {'1':'pre-trained',
          '2':'BBBP',
          '3':'Clintox',
          '4':'HIV',
          '5':'tox21'}

def download_and_extract(selected):
    for model in tqdm(selected):
        print(f'Downloading {MODEL_NAMES[model]}.zip...')
        with open(base_path + MODEL_NAMES[model]+'.zip', 'wb') as f:
            f.write(requests.get(MODEL_URLS[model]).content)
        
        print(f'Extracting {MODEL_NAMES[model]}.zip...')
        with zipfile.ZipFile(base_path + f"{MODEL_NAMES[model]}.zip","r") as zipf:
            zipf.extractall(base_path)

        if model in ['2','3','4','5']:
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
    1: Pre-trained
    2: BBBP_featurizer
    3: Clintox_featurizer
    4: HIV_featurizer
    5: tox21_featurizer

''')

    if '0' in model_selection:
        selected = ['1','2','3','4','5']
    else:
        selected = [i.strip() for i in model_selection.split(',')]

    download_and_extract(selected)