CUDA_VISIBLE_DEVICES=1 nohup python3 train_allsubtypes_N03.py n 0 bigbird 4 1024 lower_lr roc 30 n01_data_multiclass  n03 > n03_gridsearch/n03_output_rs0_BB_4_1024_30e_lowerLR_roc.txt 2>&1 &