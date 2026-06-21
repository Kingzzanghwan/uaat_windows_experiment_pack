@echo off
set PYTHONPATH=%CD%\src
python experiments\train_eval_policy.py --csv runs\e2_mvtec_ad\features.csv --out_dir runs\e2_mvtec_ad\policy --target_coverage 0.80 --c_wrong 10 --c_defer 1 --group_col category
