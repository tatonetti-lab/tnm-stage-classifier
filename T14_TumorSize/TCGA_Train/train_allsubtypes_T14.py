#Multi-Class Classification - All subtypes 

import numpy as np
import pandas as pd
import torch
import random
import math 
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from transformers import BigBirdForSequenceClassification
from transformers import TrainingArguments, Trainer
import time, os, pickle, glob, shutil, sys 
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from scipy.special import softmax
from collections import Counter 
from eval_metrics_multi_updated import get_metrics
start = time.time()
curr_datetime = datetime.now().strftime('%m-%d-%Y_%Hh-%Mm')

#Set seeds
seed=int(sys.argv[2])
np.random.seed(seed) 
torch.manual_seed(seed) 
random.seed(seed)

#Parameters 
model_used = sys.argv[3] 
batch_size = int(sys.argv[4]) 
max_tokens = int(sys.argv[5]) 
lr_type = sys.argv[6]  #'lower_lr' or 'default_lr'
if lr_type == 'lower_lr':
    learning_rate = .000005
elif lr_type == 'default_lr':
    learning_rate = .00005
else: 
    print('Error: Learning Rate Type not recognized')
print('learning rate:',lr_type)
optim_method = sys.argv[7]
if optim_method == 'f1':
    optim_type = 'eval_f1'
elif optim_method == 'roc':
    optim_type = 'eval_roc_auc'
else:
    print('Error: optim_type not recognized')
eval_metric = optim_type 

#Directories
tissue=sys.argv[1] 
conditions = 'conf_removal_none_sp_none'
input_suffix = sys.argv[9] 
input_dir = '../Target_Selection/'+input_suffix+'/'
pickle_path = input_dir + 'Target_Data_'+conditions+'_'+tissue+'.p'
input_data = pickle.load(open(pickle_path,'rb'))
num_classes=len(set(input_data['label'])) #Multi-class
output_suffix = sys.argv[10] 
root_dir = 'model_output/' + output_suffix+ '_output/' 
model_output_dir = root_dir +tissue +'_rs'+str(seed)+'_'+model_used+'_'+str(batch_size)+'bsize'+'_'+str(max_tokens)+'max_tokens_'+lr_type+'_'+optim_method+'_optim_'+str(sys.argv[8]) +'e_' +curr_datetime+ '/'
val_best_model_evaluate_dir = model_output_dir + 'val_best_model_evaluate_tmp/'
for directory in [root_dir, model_output_dir, val_best_model_evaluate_dir]:
    os.makedirs(directory, exist_ok=True)
output_file = tissue+'_output_rs'+str(seed)+'.txt'
meta_df = pd.DataFrame({'tissue':[tissue]})

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

def compute_metrics(eval_pred):
    raw_pred, labels = eval_pred
    score_pred = softmax(raw_pred, axis=1)
    binary_pred = np.argmax(raw_pred, axis=1)
    roc = roc_auc_score(labels, score_pred, multi_class='ovr', average = 'macro') 
    f1 = f1_score(labels, binary_pred, average = 'macro')
    return {"roc_auc": roc, "f1": f1} 

#Data
data=input_data.sample(frac=1, random_state=seed) #shuffles data
pd.DataFrame({'input_index_order':list(data.index)}).to_csv(model_output_dir+'input_index_order.csv',index=False) 

if model_used == 'clinicalbert':
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=num_classes)
    print('ClinicalBERT')
elif model_used == 'bigbird':
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-BigBird")
    model = BigBirdForSequenceClassification.from_pretrained("yikuan8/Clinical-BigBird", num_labels=num_classes)
    print('Bigbird model number of parameters:',model.num_parameters())
else:
    print('Model not recognized')


#Input Data
X = list(data["text"])
y = list(data["label"])
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.1765, random_state=seed) #val_set = 15% of overall dataset (70-15-15 train/val/test split)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=max_tokens)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=max_tokens)
train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)
pos_prop_train = Counter(y_train).most_common()
pos_prop_val = Counter(y_val).most_common()

#Number of steps dependent on train set size
logging_steps = math.ceil((len(y_train) / batch_size)/5) #logs 5x/epoch
save_steps = math.ceil((len(y_train) / batch_size)/10) #10x/epoch
eval_steps = math.ceil((len(y_train) / batch_size)/10) #10x/epoch

#Training Parameters
training_args_dict = {'save_strategy':'steps',
                      'save_steps':save_steps, 
                      'save_total_limit':2,
                      'num_train_epochs':int(sys.argv[8]), 
                      'logging_steps':logging_steps, 
                      'per_device_train_batch_size':batch_size,
                      'per_device_eval_batch_size':batch_size, 
                      'evaluation_strategy':'steps',
                      'eval_steps':eval_steps, 
                      'load_best_model_at_end':True,
                      'metric_for_best_model':eval_metric,
                      'learning_rate':learning_rate,
                      'learning_rate_type':lr_type,
                      'num_classes':num_classes,
                      'optim_type':optim_type} 

#Fine-tuning pretrained model
training_args = TrainingArguments(model_output_dir, 
                                  report_to=None,
                                  seed=0, 
                                  save_strategy = training_args_dict['save_strategy'],
                                  save_steps = training_args_dict['save_steps'], 
                                  save_total_limit = training_args_dict['save_total_limit'], #Deletes all but last X checkpoints - sequentially/chronologically
                                  num_train_epochs = training_args_dict['num_train_epochs'], 
                                  logging_steps = training_args_dict['logging_steps'], #logs training_loss every X steps 
                                  per_device_train_batch_size = training_args_dict['per_device_train_batch_size'], 
				                          per_device_eval_batch_size = training_args_dict['per_device_eval_batch_size'], 
				                          evaluation_strategy = training_args_dict['evaluation_strategy'],
                                  eval_steps = training_args_dict['eval_steps'],
                                  load_best_model_at_end = training_args_dict['load_best_model_at_end'], 
                                  metric_for_best_model = training_args_dict['metric_for_best_model'],
                                  learning_rate = training_args_dict['learning_rate'])   

print(training_args_dict) #record all training parameters in output file for later reference

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=val_dataset, 
                  compute_metrics=compute_metrics)

#Train
trainer.train()

#Track Run-Time (seconds)
end = time.time()
runtime = round((end - start)/60,3)
print('\nElapsed Time: ', runtime, 'Minutes')

#Save Performance Data
history_list = trainer.state.log_history
pickle.dump(history_list, open(model_output_dir+'state_log_history.p','wb'))

#Training - Performance 
train_info = [a for a in history_list if 'loss' in a.keys()]
train_info_dict = {'step':[a['step'] for a in train_info],
                  'training_loss':[a['loss'] for a in train_info],
                  'epoch':[a['epoch'] for a in train_info],
                  'learning_rate':[a['learning_rate'] for a in train_info]}
train_info_df=pd.DataFrame(train_info_dict)

#Evaluation - Performance
e_info = [a for a in history_list if 'eval_loss' in a.keys()]
e_info_dict = {key:[a[key] for a in e_info] for key in e_info[0].keys()}
e_info_df=pd.DataFrame(e_info_dict)
train_info_df.to_csv(model_output_dir+'train_history.csv',index=False)
e_info_df.to_csv(model_output_dir+'eval_history.csv',index=False)
history_df = e_info_df.merge(train_info_df, on=['step','epoch'],how='outer')
history_df.sort_values(by='step',inplace=True)
history_df.to_csv(model_output_dir+'full_history.csv',index=False)

#Identify Best_model 
model_checkpoints = glob.glob(model_output_dir+'checkpoint*')
checkpoint_steps = [int(a.split('-')[-1]) for a in model_checkpoints]
checkpoint_data = [a for a in history_list if ((eval_metric in a.keys()) and (a['step'] in checkpoint_steps))]
eval_full_list = [a[eval_metric] for a in checkpoint_data]
best_checkpoint_steps = [a['step'] for a in checkpoint_data if a[eval_metric] == max(eval_full_list)]

#If multiple checkpoints have the same eval_metric, choose the chronologically later one (trained on more data)
best_model_checkpoint = [a for a in model_checkpoints if str(max(best_checkpoint_steps)) == a.split('-')[-1]][0]
print(best_model_checkpoint)

#Save best_model metadata 
info_for_test_evaluation = {'best_model_checkpoint':best_model_checkpoint,
                            'model_output_dir':model_output_dir}
pickle.dump(info_for_test_evaluation, open(model_output_dir+'info_for_test_evaluation.p','wb')) 

#Print Best Model metrics - Performance on validation set 
print('Validation - Best Model Performance')

#Load best model 
if model_used == 'clinicalbert':
    best_model = BertForSequenceClassification.from_pretrained(best_model_checkpoint, num_labels=num_classes, local_files_only=True)
    print('ClinicalBERT eval')
elif model_used == 'bigbird':
    best_model = BigBirdForSequenceClassification.from_pretrained(best_model_checkpoint, num_labels=num_classes, local_files_only=True)
    print('Bigbird eval')
else:
    print('Model not recognized')

#Evaluate best-model on validation dataset 
best_trainer = Trainer(model=best_model, args=TrainingArguments(output_dir = val_best_model_evaluate_dir))
y_pred, _, _  = best_trainer.predict(Dataset(X_val_tokenized))
au_roc_macro, pred_pos_dict, actual_pos_dict, f1_micro, class_specific_auroc_dict  = get_metrics(y_pred, y_val, 
                                                                                          output_dir=model_output_dir, 
                                                                                          seed = seed, trainval_csv = input_dir + 'Target_Data_'+conditions+'_'+tissue+'.csv', 
                                                                                          suffix='val_best_model', all_subtypes = True, plot=True)
print(class_specific_auroc_dict)

#Update meta_df
meta_df['runtime_min'] = [runtime]
#best model steps
meta_df['best_model_steps'] = [best_model_checkpoint.split('-')[-1]]
meta_df['best_model_dir'] = [best_model_checkpoint]
meta_df['pos_prop_train'] = [pos_prop_train]
meta_df['pos_prop_val'] = [pos_prop_val]
meta_df['save_steps'] = [training_args_dict['save_steps']]
meta_df['num_train_epochs'] = [training_args_dict['num_train_epochs']]
meta_df['logging_steps'] = [training_args_dict['logging_steps']]
meta_df['per_device_train_batch_size'] = [training_args_dict['per_device_train_batch_size']]
meta_df['per_device_eval_batch_size'] = [training_args_dict['per_device_eval_batch_size']]
meta_df['eval_steps'] = [training_args_dict['eval_steps']]
meta_df['learning_rate_type'] = [lr_type]
meta_df['learning_rate'] = [learning_rate]
meta_df['val_au_roc_bestmodel_macro'] = [au_roc_macro]
meta_df['f1_micro'] = [f1_micro] 
meta_df['random_seed'] = [seed]
meta_df['n_pred_pos'] = [pred_pos_dict] 
meta_df['n_actual_pos'] = [actual_pos_dict] 
meta_df['num_classes']=[num_classes]
meta_df['optim_type']=optim_type
for class_spec_roc in list(class_specific_auroc_dict.keys()):
    meta_df['auroc_'+str(class_spec_roc)] = [class_specific_auroc_dict[class_spec_roc]] 

#Save meta_df to csv
meta_df.to_csv(model_output_dir+tissue+'_rs'+str(seed)+'_meta_df.csv',index=False)

#Delete checkpoint that is not the best model (just the last checkpoint) 
non_best_model_checkpoint = [a for a in model_checkpoints if str(max(best_checkpoint_steps)) != a.split('-')[-1]][0]
shutil.rmtree(non_best_model_checkpoint)
print('Non-Best-Model checkpoint deleted')
