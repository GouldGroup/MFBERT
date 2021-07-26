# MFBERT
This repository contains the inference, model and tokenizer code for the Molecular Fingerprinting model MFBERT.

## Getting Started

### Requirements

To get started, install the requirements by:

`pip install -r requirements.txt`

### Quick Start

To generate the molecular fingerprints for the set of 500 sample SMILES given, simply download the pre-trained model weights with the script `Model/download_models.py`  and run `python main.py`. This will generate a `fingerprints.pkl` file in the `Fingerprints` folder which contains a dictionary of the SMILES and their fingerprint.

To generate fingerprints on your own set of SMILES, simply format your data as a txt/smi file, with each line being a different SMILES. place your  file in the Data directory (or otherwise) change:

`DATA_FILE = 'Data/<your_dataset_here>'` in `main.py` 

you can also change other parameters depending on your dataset and setup:

```
DEVICE = 'cpu'
BATCH_SIZE = 1
DATA_FILE = 'Data/sample_smiles.txt'
TOKENIZER_DIR = 'Tokenizer/'
```

Note: Increasing the batch size to values greater than 1 will give different results for the mean token embedding/fingerprint depending on the length of the longest SMILES in the batch.

## Tokenizer
The custom tokenizer class, ported from fairseq can be found in `Tokenizer/MFBERT_Tokenizer`. It can be used like any huggingface tokenizer:

```python
from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
TOKENIZER_DIR = 'Tokenizer/'

# Initialise tokenizer
tokenizer = MFBERTTokenizer.from_pretrained(TOKENIZER_DIR+'Model/', dict_file = TOKENIZER_DIR+'Model/dict.txt')

# Tokenize SMILES
inputs = tokenizer(smiles_batch, return_tensors='pt')
```

## Model and Inference
The model code can be found in `Model/model.py` and the model can be initialised like any torch model. To get the model weights, use the script provided which downloads them from figshare and places them in the correct directory. If you would like to initialise the model from scratch, set `weights_dir=''`.
```python
from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
from Model.model import MFBERT

# load tokenizer
tokenizer = MFBERTTokenizer.from_pretrained('Tokenizer/Model/', dict_file = 'Tokenizer/Model/dict.txt')

# Load model
model = MFBERT(weights_dir='Model/weights', return_attention=False, inference_method='mean')

# Tokenize SMILES
inputs = tokenizer('CN(C)CC(=O)O', return_tensors='pt')

# Retrieve fingerprint
fingerprint = model(inputs).detach().numpy()
```
## Reproducing Results:

### Virtual Screening:
Clone the [RDkit Benchmarking Platform](https://github.com/rdkit/benchmarking_platform). 

Modify `benchmarking_platform/scoring/fingerprint_lib.py` by adding a function to the `fpdict` to generate fingerprints using MFBERT.

```python
def calc_MFBERT_fp(smile):
    inputs = tokenizer(smile, return_tensors="pt")
    outputs = model(inputs) # model is an initialised MFBERT instance
    return outputs.detach().numpy()
...
# add function pointer to fpdict dictionary
fpdict['MFBERT_fp'] = calc_MFBERT_fp
```

Modify the Cosine similarity function in `benchmarking_platform/scoring/scoring_functions.py` to use NumPy and Scikit learn and add it to the similarity dictionary.

```python
from sklearn.metrics.pairwise import cosine_similarity
...
simil_dict['skcosine'] = lambda x,y: sorted(cosine_similarity(np.array([x]), np.array(y))[0], reverse=True)
```

Run the Benchmarking Platform using the newly added fingerprint and metric as instructed by the library. 

### Classification fingerprints
To reproduce the PCA+K-means classification results, download the pre-fine-tuned model weights using the script in `Model/download_models.py`, this will download and extract the weights to the correct directory. Then, use the `Datasets/download_datasets.py` script to download the datasets of your choice from MoleculeNet.

Select a featurizer from `Featurizers` based on your dataset and run the script, this should generate the fingerprints needed to perform PCA and K-means.

Once the fingerprints are generated, you can load them into the `PCA-Kmeans.ipynb` notebook to perform the classifications and visualise the data. 

## Fine-tuning
There are 4 sample fine-tuning scripts provided. Should you wish to fine-tune on your own data, simply clone and modify one of the fine-tuning scripts as suited for your data.

## Other Scripts
There is a simple script, `Datasets/extract_smiles.py` to create an smi file from a csv by extracting the smiles column such that they can be featurized.

