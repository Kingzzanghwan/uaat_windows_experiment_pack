@echo off
set PYTHONPATH=%CD%\src
python experiments\train_eval_policy.py --csv runs\e1_cifar10c\features.csv --out_dir runs\e1_cifar10c\policy --target_coverage 0.80 --c_wrong 5 --c_defer 1 --group_col severity
