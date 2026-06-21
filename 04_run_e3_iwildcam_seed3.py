from pathlib import Path
import sys
from seed3_utils import SEEDS, project_root, add_seed_if_supported, run_cmd, detect_group_col, run_policy_suite

# E3 iWildCam main cost setting.
MAIN_C_WRONG = 5

root = project_root()
script = r"experiments\e3_train_extract_iwildcam.py"

for seed in SEEDS:
    base = Path(r"runs_seed3\e3_iwildcam") / f"seed_{seed:02d}"
    features_csv = base / "features.csv"
    train_out = base / "train_extract"
    base.mkdir(parents=True, exist_ok=True)

    if not features_csv.exists():
        cmd = [
            sys.executable,
            script,
            "--data_dir", "data",
            "--out_dir", str(train_out),
            "--out_csv", str(features_csv),
            "--epochs", "3",
            "--batch_size", "32",
            "--workers", "0",
            "--image_size", "224",
        ]
        cmd = add_seed_if_supported(cmd, script, seed)
        run_cmd(cmd, seed=seed)
    else:
        print(f"Already exists: {features_csv}")

    group_col = detect_group_col(features_csv, preferred="wilds_split")
    run_policy_suite(base, features_csv, seed=seed, group_col=group_col, main_c_wrong=MAIN_C_WRONG)
