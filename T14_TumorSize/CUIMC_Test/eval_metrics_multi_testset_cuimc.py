#For evaluating best-performing model on held-out test set (multi-class) 

from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score 
from itertools import cycle
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns 
import pickle 
import math

data_class_dict = {'N03':{0:'N0', 1: 'N1', 2:'N2', 3:'N3'},
                   'M01':{0:'M0', 1:'M1'},
                   'T14': {0:'T1', 1:'T2', 2:'T3', 3:'T4'}
                   }

def get_class_dict(test_csv):
    return_dict = {}
    for data_key in list(data_class_dict.keys()):    
        if '_' + data_key + '_' in test_csv: 
            return_dict = data_class_dict[data_key]
    if len(return_dict) == 0:
        print('\nError: Dataset not recognized, class_dict error')
    return return_dict 

def get_metrics(raw_pred, y_true, output_dir, seed, test_csv, suffix='', plot=True):

    class_dict = get_class_dict(test_csv)
    print('\nclass_dict:',class_dict)
    
    if len(set(y_true)) != len(set(class_dict.values())): 
        print('\nError: test set mismatch with class_dict')
        print(set(y_true))
        print(set(class_dict.values()))	

    y_pred = np.argmax(raw_pred, axis=1)
    f1_micro = f1_score(y_true, y_pred, average='micro')


    n_classes = len(set(class_dict.values()))
    if n_classes == 2:
       #Binary case
        softmax_array = softmax(raw_pred, axis=1) #compute softmax per row
        au_roc_macro = roc_auc_score(y_true, softmax_array[:,1])
    else: 
        softmax_array = softmax(raw_pred, axis=1) #compute softmax per row
        au_roc_macro = roc_auc_score(y_true, softmax_array, multi_class='ovr', average = 'macro')

    raw_df = pd.DataFrame({'y_pred':y_pred,
                            'y_true':y_true})
    for class_label in range(raw_pred.shape[1]): #number of columns in raw_pred
        raw_df['raw_class_'+str(class_label)] = list(raw_pred[:,class_label])
        raw_df['softmax_class_'+str(class_label)] = list(softmax_array[:,class_label])    
    raw_df.to_csv(output_dir+'raw_predictions_'+suffix+'.csv', index=False)

    #Class-specific AU-ROC (ovr)
    classes = list(range(raw_pred.shape[1]))
    bin_true = label_binarize(y_true, classes=classes)
    class_specific_auroc_dict, class_specific_support_dict = {i:[''] for i in classes}, {i:[''] for i in classes} #Instantiate dict 
    if len(classes) > 2:
        for class_i in classes:
            class_specific_auroc_dict[class_i] = round(roc_auc_score(bin_true[:,class_i], raw_df['softmax_class_'+str(class_i)]),4) 
            class_specific_support_dict[class_i] = sum(bin_true[:,class_i])

    eval_df = pd.DataFrame({'AUROC_macro':[au_roc_macro]})
    eval_df['f1_micro'] = f1_micro

    for class_i in classes:
        eval_df['auroc_'+class_dict[class_i]] = class_specific_auroc_dict[class_i] 
        eval_df['support_'+class_dict[class_i]] = class_specific_support_dict[class_i] 
    pred_pos_dict, actual_pos_dict = {},{}
    for class_label in range(raw_pred.shape[1]):
        npp = len([a for a in y_pred if a == class_label])
        ap = len([a for a in y_true if a == class_label])
        pred_pos_dict[class_label] = npp
        actual_pos_dict[class_label] = ap
        eval_df['n_pred_pos_class_'+str(class_label)] = npp
        eval_df['n_actual_pos_class_'+str(class_label)] = ap
    
    
    eval_df.to_csv(output_dir+'eval_metrics_'+suffix+'.csv', index=False)

    #Calculate and save confusion matrix
    cm=confusion_matrix(y_true, y_pred)
    cm_df=pd.DataFrame(cm, columns=[str(a)+'_predicted' for a in list(set(y_true))])
    cm_df['col']=[str(a)+'_actual' for a in list(set(y_true))]
    cm_df=cm_df[['col']+[str(a)+'_predicted' for a in list(set(y_true))]]
    cm_df.to_csv(output_dir+'c_matrix_'+suffix+'.csv', index=False)
    
    if plot: 
 
        #ROC curve - Across all subtypes (in aggregate) 
        sns.set(rc={'figure.figsize':(7,7)})   
        new_style = {'grid': False}
        plt.rc('axes', **new_style)    
        plt.figure()
        colors = cycle(['lightblue','orange','lightgreen','pink','khaki','chocolate'])#'darkorange','sienna']) 
        fpr, tpr = {},{}
        for class_i in classes:
            fpr[class_i], tpr[class_i], _ = roc_curve(bin_true[:, class_i], raw_df['softmax_class_'+str(class_i)])
   
        for i, color in zip(range(len(classes)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label="{0} (AUC: {1:0.2f})".format(class_dict[i], class_specific_auroc_dict[i]),)

        plt.title('Class-Specific ROC')
        plt.legend(loc = 'lower right')
        plt.xlim([-.02, 1.0])
        plt.ylim([-0.02, 1.02])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.plot([0, 1], [0, 1], "--", lw=2, color='grey')
        plt.savefig(output_dir+'roc_curve_'+suffix+'.png',dpi=1200, facecolor='w')
        plt.close()

    return au_roc_macro, pred_pos_dict, actual_pos_dict, f1_micro, class_specific_auroc_dict

