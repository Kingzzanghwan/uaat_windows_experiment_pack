@echo off
set PYTHONPATH=%CD%\src
REM Change --root to data\mvtec_loco_ad for LOCO. For a quick test, use --categories bottle or one category.
python experiments\e2_extract_mvtec_knn.py --root data\mvtec_ad --out_csv runs\e2_mvtec_ad\features.csv --categories all --batch_size 64 --workers 0 --image_size 224
