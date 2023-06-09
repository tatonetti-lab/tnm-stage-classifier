#Test best TCGA-trained model on CUIMC data 

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
from eval_metrics_multi_testset_cuimc import get_metrics
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
input_suffix = sys.argv[4]
input_dir = '../Target_Selection/'+input_suffix
test_csv = input_dir
test_data = pd.read_csv(test_csv).sample(frac=1, random_state = seed)
#New for M01
test_data = test_data[test_data['label']!= 2].copy()

num_classes=len(set(test_data['label'])) #Assumption: All classes represented in test set 
test_output_dir = model_output_dir + 'test_best_model_cuimc_evaluate_M01_tmp/'
os.makedirs(test_output_dir, exist_ok=True)
model_output_dir_cuimc = model_output_dir + 'test_cuimc/'+ input_suffix.split('/')[1].replace('.csv', '_current')+'/'
os.makedirs(model_output_dir_cuimc, exist_ok=True)


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
test_trainer = Trainer(model=test_model, args=TrainingArguments(output_dir = test_output_dir))

#Run Trained Model on CUIMC data
y_pred, _, _  = test_trainer.predict(test_dataset)
au_roc_macro, f1_micro  = get_metrics(y_pred, y_true,
                                                                output_dir=model_output_dir_cuimc, 
                                                                seed = seed, test_csv = '_M01_',
                                                                suffix='test_best_model_CUIMC', plot=True)

#Save results
csv_dir = 'ALL_Matching_CUIMC_TEST/'+ input_suffix.split('/')[0].split('_')[1] + '/'
csv_dir = csv_dir.replace('M01X','M01')
conditions_info = input_suffix.split('/')[1].replace('M01X','M01')


info_list = conditions_info.split('_') 
n_days  = info_list[2]
task  = info_list[4] 
SP = info_list[6]
performance_df = pd.DataFrame({'Task':[task],
'Days':[n_days],
'SP Filter':[SP],
'AU-ROC Weighted':[round(au_roc_weighted,4)],
'AU-ROC Macro':[round(au_roc_macro,4)],
'F1 Micro':[round(f1_micro,4)] })
performance_df.to_csv(csv_dir+conditions_info.replace('.csv','_CUIMC_PERFORMANCE.csv'))

