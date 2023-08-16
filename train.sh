

CUDA_VISIBLE_DEVICES=0 nohup python -u main_unified.py --folder ./experiments_paraphrase_ssd/mscoco_var_xlan_diverse_momle_lambda_2.0 --multi_objective 1 > temp_train_mscoco_var_xlan_diverse_momle_lambda_2.0.out &
CUDA_VISIBLE_DEVICES=0 nohup python -u main_unified.py --folder ./experiments_paraphrase_ssd/mscoco_var_xlan_diverse_momle_morl_lambda_2.0 --multi_objective 1 --resume 1 > temp_train_mscoco_var_xlan_diverse_momle_morl_lambda_2.0.out &
