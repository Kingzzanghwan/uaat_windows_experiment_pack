@echo off
set PYTHONPATH=%CD%\src
python experiments\e3_train_extract_iwildcam.py --data_dir data --out_dir runs\e3_iwildcam --out_csv runs\e3_iwildcam\features.csv --epochs 3 --batch_size 64 --workers 0 --image_size 224
