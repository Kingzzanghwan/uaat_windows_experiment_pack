@echo off
set PYTHONPATH=%CD%\src
python experiments\e1_train_cifar.py --data_dir data --out_dir runs\e1_cifar_base --epochs 50 --batch_size 128 --workers 0
