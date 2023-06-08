#Evaluation for multi-class training, TCGA Report Set 
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

def get_metrics(raw_pred, y_true, output_dir, seed, trainval_csv, suffix='', all_subtypes=False, plot=True):

    class_dict = {0:'N0', 1:'N1', 2:'N2', 3:'N3'}

    y_pred = np.argmax(raw_pred, axis=1)
    f1_micro = f1_score(y_true, y_pred, average='micro')

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
    class_specific_auroc_dict, class_specific_support_dict = {}, {} 
    
    for class_i in classes:
        class_specific_auroc_dict[class_i] = round(roc_auc_score(bin_true[:,class_i], raw_df['softmax_class_'+str(class_i)]),4) 
        class_specific_support_dict[class_i] = sum(bin_true[:,class_i])
    
    eval_df = pd.DataFrame({'AUROC_macro':[au_roc_macro]})

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
    
    eval_df['f1_micro'] = f1_micro
    eval_df.to_csv(output_dir+'eval_metrics_'+suffix+'.csv', index=False)

    #Calculate and save confusion matrix
    cm=confusion_matrix(y_true, y_pred)
    cm_df=pd.DataFrame(cm, columns=[str(a)+'_predicted' for a in list(set(y_true))])
    cm_df['col']=[str(a)+'_actual' for a in list(set(y_true))]
    cm_df=cm_df[['col']+[str(a)+'_predicted' for a in list(set(y_true))]]
    cm_df.to_csv(output_dir+'c_matrix.csv', index=False)
    
    if all_subtypes == True:
            
        full_df = pickle.load(open('../Target_Selection/Target_Data_conf_removal_none_sp_none_binary.p','rb')) #This file is necessary - contains all patients
        classes = list(range(len(set(raw_df['y_true'])))) 

        trainval_df = pd.read_csv(trainval_csv) #eg  - 'Target_Data_conf_removal_none_sp_none_m.csv'
        trainval_data=trainval_df.sample(frac=1, random_state=seed) 
        X = trainval_data[['patient_filename','text']] 
        y = list(trainval_data["label"])
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.1765, random_state=seed) 

        eval_patients = list(X_val['patient_filename']) #this is in the same order 
        df1=full_df[full_df['patient_filename'].isin(eval_patients)] 
        df1.reset_index(inplace=True, drop=True)
        df2=raw_df #this is in the same order
        df2.reset_index(inplace=True, drop=True)
        concat_df = pd.concat([df1,df2], axis=1)
        
        eval_df_all = pd.DataFrame({'type':['all'],
                        'f1_micro':[f1_micro],
                        'f1_macro':[f1_macro],
                        'roc_macro':[au_roc_macro],
                        'roc_weighted':[au_roc_weighted],
                        'f1_weighted':[f1_weighted],
                        'support':[concat_df.shape[0]]})
        for class_i in classes:
            eval_df_all['roc_'+class_dict[class_i]] = class_specific_auroc_dict[class_i] 
            eval_df_all['support_'+class_dict[class_i]] = class_specific_support_dict[class_i] 

        subtype_dict, support_subtype_dict, sub_class_specific_auroc_dict = {}, {}, {}
        type_list, f1micro_list, f1macro_list, rocmacro_list, rocweighted_list, f1weighted_list, support_list = [], [], [], [], [], [], []
        for subtype in list(set(concat_df['type'])):

            subtype_index = list(concat_df[concat_df['type']==subtype].index)
            subtype_bin = bin_true[subtype_index,:]

            classes_subtype, support_subtype = [],{}
            for class_i in classes: 
                support_subtype[class_i] = sum(subtype_bin[:,class_i])
                if sum(subtype_bin[:,class_i]) >= 5:
                    classes_subtype.append(class_i)
            subtype_dict[subtype] = classes_subtype
            support_subtype_dict[subtype] = support_subtype 

            #if len(classes_subtype) > 0:
            if len(classes_subtype) == len(set(classes)):  #Include only subtypes with support >=5 for all classes 
                type_list.append(subtype)
                subtype_df = concat_df[concat_df['type']==subtype].copy()
                subtype_df.reset_index(inplace=True, drop=True)
                support_list.append(subtype_df.shape[0])
                sub_pred = subtype_df['y_pred']
                sub_true = subtype_df['y_true'] 
                cm = confusion_matrix(sub_true, sub_pred)
                cm_df = pd.DataFrame(cm, columns = [str(a)+'_predicted' for a in classes])#list(set(sub_true))])
                cm_df['col']=[str(a)+'_actual' for a in classes]#list(set(sub_true))]
                cm_df = cm_df[['col']+[str(a)+'_predicted' for a in classes]]#list(set(sub_true))]]
                cm_df.to_csv(output_dir+subtype+'_c_matrix.csv', index=False)

                f1micro_list.append(f1_score(sub_true, sub_pred, average='micro'))
                f1macro_list.append(f1_score(sub_true, sub_pred, average='macro'))
                f1weighted_list.append(f1_score(sub_true, sub_pred, average='weighted'))

                softmax_cols_relevant = ['softmax_class_' + str(i) for i in classes] #all classes included
                sub_softmax_array = subtype_df[softmax_cols_relevant]
                rocweighted_list.append(roc_auc_score(sub_true, sub_softmax_array, multi_class='ovr', average = 'weighted'))
                rocmacro_list.append(roc_auc_score(sub_true, sub_softmax_array, multi_class='ovr', average = 'macro'))

                for class_i in classes:
                    class_roc = roc_auc_score(subtype_bin[:,class_i], sub_softmax_array['softmax_class_'+str(class_i)])
                    if class_i in sub_class_specific_auroc_dict.keys():
                        sub_class_specific_auroc_dict[class_i].append(class_roc)
                    else:
                        sub_class_specific_auroc_dict[class_i] = [class_roc]

        print('Number of subtypes with sufficient support in validation set:',str(len(type_list)), type_list) 

        meta_subtype_df = pd.DataFrame({'subtype':list(subtype_dict.keys()),
                'n_classes_valset':[len(a) for a in list(subtype_dict.values())],
                 'classes_list': list(subtype_dict.values()),
                 'support_by_class':[support_subtype_dict[subtype] for subtype in list(subtype_dict.keys())]})
        meta_subtype_df.sort_values(by='subtype', inplace=True)
        meta_subtype_df.to_csv(output_dir+'classes_per_subtype.csv', index=False)


        if len(type_list) > 0: 
            eval_subtypes_df = pd.DataFrame({'type':type_list,
                                                  'f1_micro':f1micro_list,
                                                  'roc_macro':rocmacro_list,
                                                  'support':support_list})
            for class_i in classes:
                eval_subtypes_df['roc_'+class_dict[class_i]] = sub_class_specific_auroc_dict[class_i]

            eval_subtypes_df.sort_values(by='type', inplace=True) #Alphabetical ordering

            for class_i in classes:
                eval_subtypes_df['support_'+class_dict[class_i]] = [support_subtype_dict[subtype][class_i] for subtype in eval_subtypes_df['type']] 

            #Merge dfs
            all_merged_df = pd.concat([eval_df_all,eval_subtypes_df], axis=0, ignore_index=True)
            all_merged_df.to_csv(output_dir+'full_merged_eval_df.csv', index=False) 
            all_merged_df.round(3).to_csv(output_dir+'full_merged_eval_df_rounded.csv', index=False)

            #Create and save additional df with all subtypes (with at least 1 eligible target class)
            additional_types = [subtype for subtype in list(set(concat_df['type'])) if (len(subtype_dict[subtype]) > 0) & (subtype not in list(eval_subtypes_df['type']))]
            additional_types = sorted(additional_types)
            additional_support = []
            additional_class_specific_auroc_dict = {}
            for subtype in additional_types: 
                subtype_df = concat_df[concat_df['type']==subtype].copy()
                subtype_df.reset_index(inplace=True, drop=True)
                subtype_index = list(concat_df[concat_df['type']==subtype].index)
                subtype_bin = bin_true[subtype_index,:]
                additional_support.append(subtype_df.shape[0])
                for class_i in classes:
                    if class_i in subtype_dict[subtype]: #only classes with enough support/subtype
                        class_roc = roc_auc_score(subtype_bin[:,class_i], subtype_df['softmax_class_'+str(class_i)])
                    else:
                        class_roc = 'NA' 
                    if class_i in additional_class_specific_auroc_dict.keys():
                        additional_class_specific_auroc_dict[class_i].append(class_roc)
                    else:
                        additional_class_specific_auroc_dict[class_i] = [class_roc]


            additional_df = pd.DataFrame({'type':additional_types, 'f1_micro':['NA']*len(additional_types),
                                           'roc_macro':['NA']*len(additional_types),
                                           'support':additional_support})
            for class_i in classes:
                additional_df['roc_'+class_dict[class_i]] = additional_class_specific_auroc_dict[class_i]
            
            for class_i in classes:
                additional_df['support_'+class_dict[class_i]] = [support_subtype_dict[subtype][class_i] for subtype in additional_df['type']] 

            additional_merged_df = pd.concat([all_merged_df,additional_df], axis=0, ignore_index=True)
            #Round columns (including mixed-type)
            for col in additional_merged_df:
                additional_merged_df[col] = [round(a,3) if type(a) != str else a for a in additional_merged_df[col]]
            additional_merged_df.to_csv(output_dir+'additional_merged_eval_df_rounded.csv', index=False)
         
            #Plot individual-subtype performance metrics
            subtypes_plot_df = all_merged_df[all_merged_df['type']!='all']

            #Plot microF1 across subtypes
            new_style = {'grid': False}
            plt.rc('axes', **new_style)
            subtypes_plot_df.plot.bar(x='type',y=['f1_micro'] ,figsize=(5, 5),width=.7, fontsize=12)
            plt.ylabel('Micro-F1',fontsize=12)
            plt.xlabel('Cancer Type',fontsize=12)
            plt.title('Micro-F1 across Cancer Types',fontsize=12)
            plt.legend([],[], frameon=False)
            plt.savefig(output_dir+'f1_across_tissues.png',dpi=900, bbox_inches='tight')
            plt.close()

            max_n_images = len(type_list)
            side = math.ceil(max_n_images**.5)
            new_style = {'grid': False}
            plt.rc('axes', **new_style)
            fig, axs = plt.subplots(side, side, squeeze=False, figsize=(side*4,side*4)) 
            fig.suptitle('ROC Curves across Cancer Types')

            i,j=0,0
            i_used, j_used = [],[]
            for typ_ind in range(len(type_list)): #Only plot subtypes that have >=5 validation samples/class for all classes
                i_used.append(i)
                j_used.append(j)
                subtype = list(eval_subtypes_df['type'])[typ_ind] #Alphabetical order 
                subtype_df = concat_df[concat_df['type']==subtype].copy()
                subtype_df.reset_index(inplace=True, drop=True)
                subtype_index = list(concat_df[concat_df['type']==subtype].index)
                subtype_bin = bin_true[subtype_index,:]

                subtype_roc_df = all_merged_df[all_merged_df['type']==subtype].copy()
                tmp_class_specific_auroc_dict={}
                for class_i in classes:
                    tmp_class_specific_auroc_dict[class_i] = list(subtype_roc_df['roc_'+class_dict[class_i]])[0] 

                colors = cycle(['lightblue','orange','lightgreen','pink','khaki','chocolate'])#'darkorange','sienna'])
                fpr, tpr = {},{}
                for class_i in classes:
                    fpr[class_i], tpr[class_i], _ = roc_curve(subtype_bin[:, class_i], subtype_df['softmax_class_'+str(class_i)])

                for class_i, color in zip(range(len(classes)), colors):
                    axs[i,j].plot(fpr[class_i], tpr[class_i], color=color, lw=2,
                             label="{0} (AUC: {1:0.2f})".format(class_dict[class_i], tmp_class_specific_auroc_dict[class_i]))

                axs[i,j].set_title(subtype)
                axs[i,j].legend(loc = 'lower right', fontsize=5) 
                axs[i,j].set_xlim([-.02, 1.0])
                axs[i,j].set_ylim([-0.02, 1.02])
                axs[i,j].plot([0, 1], [0, 1], "--", lw=2, color='grey')

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
                        fig.delaxes(axs[i][j]) #Remove empty plots
 
            fig.tight_layout()  
            plt.savefig(output_dir+'roc_curves_across_cancertypes.png',dpi=1500, facecolor='w') #Increased DPI 
            plt.close()

            #Plot all subtypes with any eligible target classes in validation set (still leaving out classes with <5 sample support)
            all_types = list(set(concat_df['type']))
            all_types = sorted(all_types)
            all_types = [subtype for subtype in all_types if len(subtype_dict[subtype]) > 0]
            max_n_images = len(all_types)
            side = math.ceil(max_n_images**.5)
            new_style = {'grid': False}
            plt.rc('axes', **new_style)
            fig, axs = plt.subplots(side, side, squeeze=False, figsize=(side*4,side*4)) 
            fig.suptitle('ROC Curves across Cancer Types')

            i,j=0,0
            i_used, j_used = [],[]
            for typ_ind in range(len(all_types)): #Only plot subtypes that have >=5 validation samples/class for all classes
                i_used.append(i)
                j_used.append(j)
                subtype = all_types[typ_ind] 
                subtype_df = concat_df[concat_df['type']==subtype].copy()
                subtype_df.reset_index(inplace=True, drop=True)
                subtype_index = list(concat_df[concat_df['type']==subtype].index)
                subtype_bin = bin_true[subtype_index,:]

                tmp_class_specific_auroc_dict = {}
                colors = cycle(['lightblue','orange','lightgreen','pink','khaki','chocolate'])#'darkorange','sienna'])
                fpr, tpr = {},{}
                for class_i in subtype_dict[subtype]: #only classes with enough support/subtype 
                    fpr[class_i], tpr[class_i], _ = roc_curve(subtype_bin[:, class_i], subtype_df['softmax_class_'+str(class_i)])
                    tmp_class_specific_auroc_dict[class_i] = roc_auc_score(subtype_bin[:,class_i], subtype_df['softmax_class_'+str(class_i)])
                for class_i, color in zip(subtype_dict[subtype], colors):
                    axs[i,j].plot(fpr[class_i], tpr[class_i], color=color, lw=2,
                             label="{0} (AUC: {1:0.2f})".format(class_dict[class_i], tmp_class_specific_auroc_dict[class_i]))

                axs[i,j].set_title(subtype)
                axs[i,j].legend(loc = 'lower right', fontsize=5) 
                axs[i,j].set_xlim([-.02, 1.0])
                axs[i,j].set_ylim([-0.02, 1.02])
                axs[i,j].plot([0, 1], [0, 1], "--", lw=2, color='grey')

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
                        fig.delaxes(axs[i][j]) #Remove empty plots

            fig.tight_layout() 
            plt.savefig(output_dir+'roc_curves_across_cancertypes_ALL_subtypes.png',dpi=900, facecolor='w') 
            plt.close()

            #Plot roc curves for target classes across all eligible subtypes (still leaving out classes with <threshold sample support)
            threshold_support_list = [5,10]
            for threshold_support in threshold_support_list: 

                all_types = list(set(additional_merged_df['type']))
                all_types = [a for a in all_types if a != 'all']
                all_types = sorted(all_types)
                max_n_images = len(classes) 
                side = math.ceil(max_n_images**.5)
                sns.set_style('dark')
                fig, axs = plt.subplots(side, side, squeeze=False, figsize=(side*4,side*4))     
                fig.suptitle('ROC Curves across Cancer Types')

                i,j=0,0
                i_used, j_used = [],[]
                for class_i in classes:
  
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
                        #print(list(meta_subtype_df[meta_subtype_df['subtype']==subtype]['support_by_class'])[0])#
                        dt = list(meta_subtype_df[meta_subtype_df['subtype']==subtype]['support_by_class'])[0]
                        if dt[class_i] >= threshold_support:
                            class_types.append(subtype)


                    fpr, tpr = {},{}
                    tmp_class_specific_auroc_dict = {}
                    for subtype in class_types:
                        subtype_df = concat_df[concat_df['type']==subtype].copy()
                        subtype_df.reset_index(inplace=True, drop=True)
                        subtype_index = list(concat_df[concat_df['type']==subtype].index)
                        subtype_bin = bin_true[subtype_index,:]
                        fpr[subtype], tpr[subtype], _ = roc_curve(subtype_bin[:, class_i], subtype_df['softmax_class_'+str(class_i)])
                        tmp_class_specific_auroc_dict[subtype] = roc_auc_score(subtype_bin[:,class_i], subtype_df['softmax_class_'+str(class_i)])

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
                            fig.delaxes(axs[i][j]) #Remove empty plots

                fig.tight_layout() 
                plt.savefig(output_dir+'roc_curves_across_cancertypes_ALL_classes_threshold'+str(threshold_support)+'.png',dpi=900, facecolor='w') 
                plt.close()

    if plot: 
 
        #ROC curve - all subtypes at once
        sns.set(rc={'figure.figsize':(7,7)})   
        new_style = {'grid': False}
        plt.rc('axes', **new_style) #removes grid from sns background (still gray)   
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

