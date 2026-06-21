@echo off
set PYTHONPATH=%CD%\src
REM Change --cifar10c_dir to the folder where you extracted CIFAR-10-C.tar
python experiments\e1_extract_cifar10c.py --cifar10c_dir data\CIFAR-10-C --ckpt runs\e1_cifar_base\cifar10_resnet18.pt --out_csv runs\e1_cifar10c\features.csv --corruptions paper5 --severities 1,2,3,4,5 --max_per_severity 0 --tta 4 --batch_size 256 --workers 0
