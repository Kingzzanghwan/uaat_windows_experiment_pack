@echo off
set PYTHONPATH=%CD%\src
python experiments\train_eval_policy.py --csv runs\e3_iwildcam\features.csv --out_dir runs\e3_iwildcam\policy --target_coverage 0.80 --c_wrong 5 --c_defer 1 --group_col wilds_split
