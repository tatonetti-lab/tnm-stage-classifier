1. TNM models have been trained and are now posted on huggingface: 

https://huggingface.co/jkefeli/CancerStage_Classifier_T
https://huggingface.co/jkefeli/CancerStage_Classifier_N
https://huggingface.co/jkefeli/CancerStage_Classifier_M

We have included a small dataset, the T14 TCGA pathology report held-out test set, to demonstrate the utility and ease-of-use of the trained models. Please see T14_test.csv.

2. System Requirements

The following python package versions were used in model training: 

numpy==1.19.5
pandas==1.2.4
scikit-learn==0.24.2
scipy==1.6.3
seaborn==0.11.2
transformers==4.12.5
torch==1.7.1

To run the trained model on new data, depending on the size of the target dataset, a GPU cluster would be faster but is not required. To re-train the models (i.e., reproduce the work in this study) would require a GPU cluster to run the included shell scripts. 

3. Demo

To run the T14 model on the example TCGA T14 held-out test set, use the Demo_Code jupyter notebook included in this directory.  

The approximate run-time on a modern laptop is 2 hours.

The expected output AU-ROC is 0.9451. 

4. Instructions for downstream use 

To run the software on an external dataset, follow the same instructions as given for the demo. Replace the input parameters test_data with data formatted the same way as T14_test.csv (with 1 column of labels and 1 column of text). For N or M models, replace the model "jkefeli/CancerStage_Classifier_T" with the relevant N or M model and num_classes with the relevant number of classes for each parameter (4 classes for N03, 2 classes for M01). 


