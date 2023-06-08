#Binary Evaluation for M01

import numpy as np
import pandas as pd
import torch
import random
import math 
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from transformers import LongformerForSequenceClassification, BigBirdForSequenceClassification
from transformers import TrainingArguments, Trainer
import time, os, pickle, glob, shutil, sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from scipy.special import softmax
from collections import Counter
from eval_metrics_m01_binary import get_metrics

seed=0
np.random.seed(seed) 
torch.manual_seed(seed) 
random.seed(seed)
tissue=sys.argv[1]
model_used = sys.argv[2]
max_tokens = int(sys.argv[3])

#Model Directory 
model_output_dir = sys.argv[5] #eg - 'model_output/n01_output/n_rs20_bigbird_8bsize_512max_tokens_08-21-2022_08h-00m/'
model_output_dir_glob = glob.glob(model_output_dir + 'checkpoint-*/')
print(model_output_dir_glob)
if len(model_output_dir_glob) == 1:
    checkpoint_dir = model_output_dir_glob[0]
else:
    print('Error: More than 1 checkpoint in directory. Check if correct directory provided.')

#Test-set Directory
conditions = 'conf_removal_none_sp_none'
input_suffix = sys.argv[4]
input_dir = '../Target_Selection/'+input_suffix+'/'
test_pickle = input_dir + 'Target_Data_'+conditions+'_'+tissue+'_test.p'
test_data = pickle.load(open(test_pickle,'rb')).sample(frac=1, random_state = seed)
num_classes=len(set(test_data['label'])) #Assumption: All classes represented in test set 
test_output_dir = model_output_dir + 'test_best_model_evaluate_tmp/'
os.makedirs(test_output_dir, exist_ok=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

#Model Type Options 
if model_used == 'clinicalbert':
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    test_model = BertForSequenceClassification.from_pretrained(checkpoint_dir, num_labels=num_classes, local_files_only=True)
elif model_used == 'longformer':
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
    test_model = LongformerForSequenceClassification.from_pretrained(checkpoint_dir,  num_labels=num_classes, local_files_only=True)
elif model_used == 'bigbird':
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-BigBird")
    test_model = BigBirdForSequenceClassification.from_pretrained(checkpoint_dir, num_labels=num_classes, local_files_only=True)
else:
    print('Model not recognized')

#Data Configuation 
X_test = list(test_data["text"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=max_tokens)
test_dataset = Dataset(X_test_tokenized)
y_true = list(test_data['label'])
test_trainer = Trainer(model=test_model, args=TrainingArguments(output_dir = test_output_dir,
                                         per_device_eval_batch_size=32,
                                         per_device_train_batch_size=32))

#Run Trained Model on Held-out Test Set 
y_pred, _, _  = test_trainer.predict(test_dataset)

au_roc_macro, f1_micro,  = get_metrics(y_pred, y_true, output_dir=model_output_dir, 
                                                                seed = seed, test_csv = input_dir + 'Target_Data_'+conditions+'_'+tissue+'_test.csv',
                                                                suffix='test_best_model_TCGA', all_subtypes = True) 

#Print Results
print('\n ','AU-ROC_macro ', round(au_roc_macro,4), 'f1_micro', round(f1_micro, 4))

