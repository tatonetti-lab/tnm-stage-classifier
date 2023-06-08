#Binary for M01 Target

from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns 
import pickle 
import math
from sklearn.model_selection import train_test_split

def get_metrics(raw_pred, y_true, output_dir,  seed, test_csv, suffix='', all_subtypes=False, plot=True):

    print(y_true)
    class_dict = {0:'M0', 1:'M1'}
    y_pred = np.argmax(raw_pred, axis=1)
    softmax_array = softmax(raw_pred, axis=1) #compute softmax per row
    au_roc_macro = roc_auc_score(y_true, softmax_array[:,1])
    f1_micro = f1_score(y_true, y_pred, average='micro')
    raw_df = pd.DataFrame({'y_pred':y_pred,
                            'y_true':y_true})

    

    eval_df = pd.DataFrame({'AUROC_macro':[au_roc_macro],
                              'f1_micro':[f1_micro]})
    eval_df.to_csv(output_dir+'eval_metrics_'+suffix+'.csv', index=False)

    #Calculate and save confusion matrix
    cm=confusion_matrix(y_true, y_pred)
    cm_df=pd.DataFrame(cm, columns=[str(a)+'_predicted' for a in list(set(y_true))])
    cm_df['col']=[str(a)+'_actual' for a in list(set(y_true))]
    cm_df=cm_df[['col']+[str(a)+'_predicted' for a in list(set(y_true))]]
    cm_df.to_csv(output_dir+'c_matrix_'+suffix+'.csv', index=False)

    if plot == True: 

        #ROC curve - Across all subtypes (in aggregate) 
        sns.set(rc={'figure.figsize':(7,7)})   
        new_style = {'grid': False}
        plt.rc('axes', **new_style)    
        plt.figure()
        colors = ['lightblue','orange','lightgreen','pink','khaki','chocolate'] 
        fpr, tpr, _ = roc_curve(y_true, softmax_array[:,1])
   
        plt.plot(fpr, tpr, color=colors[0], lw=2,
             label="{0} (AUC: {1:0.2f})".format('M1',au_roc_macro),)

        plt.title('ROC Curve')
        plt.legend(loc = 'lower right')
        plt.xlim([-.02, 1.0])
        plt.ylim([-0.02, 1.02])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.plot([0, 1], [0, 1], "--", lw=2, color='grey')
        plt.savefig(output_dir+'roc_curve_'+suffix+'.png',dpi=1200, facecolor='w')
        plt.close()

        #Plot across all subtypes
        full_df = pickle.load(open('../Target_Selection/Target_Data_conf_removal_none_sp_none_binary.p','rb')) #This file is necessary - contains all patients
        test_df = pd.read_csv(test_csv)
        test_data=test_df.sample(frac=1, random_state=seed)
        X = test_data[['patient_filename','text']]
        y = list(test_data["label"])
        eval_patients = list(X['patient_filename']) #this is in the same order
        df1=full_df[full_df['patient_filename'].isin(eval_patients)]
        df1.reset_index(inplace=True, drop=True)
        df2=raw_df #this is in the same order
        df2.reset_index(inplace=True, drop=True)
        concat_df = pd.concat([df1,df2], axis=1)
        #testing
        print(min(concat_df.index), max(concat_df.index))
	
        if all_subtypes == True:   
            for threshold_support in [3,5]: 
                all_types =  list(set(concat_df['type'])) 
                all_types = sorted(all_types)
                max_n_images = 1
                side = math.ceil(max_n_images**.5)
                sns.set_style('dark')
                fig, axs = plt.subplots(side, side, squeeze=False, figsize=(side*4,side*4))
                fig.suptitle('ROC Curves across Cancer Types')

                i,j=0,0
                i_used, j_used = [],[]
                for class_i in [1]: #class = 1

                    #subtype-color assignments
                    cm = plt.get_cmap('ocean')  #nipy_spectral #gist_rainbox
                    color_dict = {}
                    count = 0
                    for subtype in all_types:
                        color_dict[subtype] = cm(count*10)
                        count+=1

                    i_used.append(i)
                    j_used.append(j)
                    class_types = []
                    for subtype in all_types:
                        subtype_index = list(concat_df[concat_df['type']==subtype].index)
                        subtype_bin = [y_true[i] for i in subtype_index] #changed                       
                        if sum(subtype_bin) >= threshold_support:
                            class_types.append(subtype)


                    fpr, tpr = {},{}
                    tmp_class_specific_auroc_dict = {}
                    for subtype in class_types:
                        subtype_df = concat_df[concat_df['type']==subtype].copy()
                        subtype_df.reset_index(inplace=True, drop=True)
                        subtype_index = list(concat_df[concat_df['type']==subtype].index)
                        print(subtype, subtype_index)
                        subtype_bin = [y_true[i] for i in subtype_index] 
                        softmax_subtype = [softmax_array[:,1][i] for i in subtype_index]
                        fpr[subtype], tpr[subtype], _ = roc_curve(subtype_bin, softmax_subtype) ##class_i = 1 
                        tmp_class_specific_auroc_dict[subtype] = roc_auc_score(subtype_bin, softmax_subtype) 

                    linestyle_types = ['-','--', '-.']  #['solid', 'dashed', 'dashdot', 'dotted']
                    linestyle_types = linestyle_types * len(all_types)
                    count = 0
                    for subtype in all_types:
                        if subtype in class_types:
                            axs[i,j].plot(fpr[subtype], tpr[subtype], linestyle_types[count],
                                     color = color_dict[subtype],
                                     lw=.5, 
                                     alpha=1, 
                                     label="{0} (AUC: {1:0.2f})".format(subtype, tmp_class_specific_auroc_dict[subtype]))
                            count+=1
                    axs[i,j].set_title(class_dict[class_i])
                    axs[i,j].legend(loc = 'lower right', fontsize=5)
                    axs[i,j].set_xlim([-.02, 1.0])
                    axs[i,j].set_ylim([-0.02, 1.02])
                    axs[i,j].plot([0, 1], [0, 1], "--", lw=1.5, color='grey')

                    j+=1
                    if j > side-1:
                        j=0
                        i+=1

                for ax in axs.flat:
                    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')

                used_list = [(i_used[a],j_used[a]) for a in range(len(i_used))]
                for i in range(side):
                    for j in range(side):
                        if (i,j) not in used_list:
                            fig.delaxes(axs[i][j])

                fig.tight_layout()
                plt.savefig(output_dir+'roc_curves_across_cancertypes_ALL_classes_threshold'+str(threshold_support)+'_'+suffix+'.png',dpi=900, facecolor='w')
                plt.close()

    
    return  au_roc_macro, f1_micro

