# Sparse Dense RAG  

This is a repository served as part of our course project. It is not intended for formal research use.  

## Environment

```shell
conda env create -f env.yaml
```

## Data Preprocessing

The data preprocessing logic lies in `dataset.py`. Once you try to load the dataset it will automatically start grabbing context from google search and do DPR filtering.  
It will take a very long time.  
If you want to skip that, try our preprocessed file.  
[testset.pkl](https://drive.google.com/file/d/1At2i6wMMIeUD8nmt0z9KqIRSQ3Gq2xhl/view?usp=drive_link)  
[trainset.pkl](https://drive.google.com/file/d/1kpqexfRl0DlR--JMo7Zj1TKK7F05Wx0Y/view?usp=drive_link)  
trainset.pkl will be automatically split into training set and validation set in `dataset.py`. Just put these 2 files under `data/trainset.pkl` and `data/testset.pkl` and you are ready to go.  

## Finetune Generator  

```bash
python train_generator.py
```

You can change number of docs used as context and whether to use weighted loss in `train_generator.py`.  

## Test 

```bash
python main.py  
```

You can change number of docs used as context in `main.py`.  

## Evaluate  

```bash
python evaluate.py
```