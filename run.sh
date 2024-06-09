CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -W ignore lucid.py --CLS 2 --BSZ 112 --DATA_DIR ./data/selected_ct_data --SET_TYPE train --USE_TEXT False > without_text_EGFR.out 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -W ignore lucid.py --CLS 2 --BSZ 112 --DATA_DIR ./data/selected_ct_data --SET_TYPE train --USE_TEXT True > with_text_EGFR.out 
