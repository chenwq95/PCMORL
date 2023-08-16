#!/bin/bash


# CUDA_VISIBLE_DEVICES=0 python -u main_unified_test.py --folder ./experiments_paraphrase_ssd/mscoco_var_xlan_diverse_momle/ --eval_multi_num 5 --results_path gen_captions/mscoco_var_xlan_diverse_momle_pred5_1.0.json --multi_objective 1 --preference_diversity_weight 1.0
# CUDA_VISIBLE_DEVICES=0 python -u main_unified_test.py --folder ./experiments_paraphrase_ssd/$1/  --results_path gen_captions/gen_$1.json --multi_objective 1 --preference_diversity_weight 0.0

CUDA_VISIBLE_DEVICES=0 python -u main_unified_test.py --folder ./experiments_paraphrase_ssd/$1/ --eval_multi_num 5 --results_path gen_captions/$1_pred5_$2.json --multi_objective 1 --preference_diversity_weight $2
CUDA_VISIBLE_DEVICES=0 python -u main_unified_test_multi.py --folder ./experiments_paraphrase_ssd/$1/ --results_path gen_captions/$1_pred5_$2.json --preference_diversity_weight $2

