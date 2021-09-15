import requests
from tqdm import tqdm
import gzip
import sys
import os
import shutil

base_path = sys.path[0]+'/'

DATA_URLS = {'1':'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
          '2':'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz',
          '3':'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv',
          '4':'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
          '5':'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv',
          '6':'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
          '7':'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv'}

DATA_NAMES = {'1':'BBBP',
          '2':'Clintox',
          '3':'HIV',
          '4':'tox21',
          '5':'Lipophilicity',
          '6':'ESOL',
          '7':'FreeSolv'}

def download_and_extract(selected):
    for ds in tqdm(selected):
        print(f'Downloading {DATA_NAMES[ds]} dataset...')
        if ds not in ['2','4']:
            with open(base_path + DATA_NAMES[ds]+'.csv', 'wb') as f:
                f.write(requests.get(DATA_URLS[ds]).content)

        elif ds in ['2','4']:
            with open(base_path + DATA_NAMES[ds]+'.csv.gz', 'wb') as f:
                f.write(requests.get(DATA_URLS[ds]).content) 

            with gzip.open(base_path + DATA_NAMES[ds]+'.csv.gz', 'rb') as f_in:
                with open(base_path + DATA_NAMES[ds]+'.csv', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)   
            try:
                os.remove(base_path + DATA_NAMES[ds]+'.csv.gz')
            except OSError:
                pass      
    
    print("Done")
    return


if __name__=='__main__':
    model_selection = input('''Please select which model weight(s) to download (comma separated):

    0: ALL
    1: BBBP
    2: Clintox
    3: HIV
    4: tox21
    5: Lipophilicity
    6: ESOL
    7: FreeSolv
    
''')

    if '0' in model_selection:
        selected = ['1','2','3','4','5','6','7']
    else:
        selected = [i.strip() for i in model_selection.split(',')]

    download_and_extract(selected)