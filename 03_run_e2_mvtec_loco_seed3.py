from pathlib import Path
import sys
from seed3_utils import SEEDS, project_root, find_existing, add_seed_if_supported, run_cmd, detect_group_col, run_policy_suite

# LOCO AD main cost setting.
MAIN_C_WRONG = 10

root = project_root()
script = r"experiments\e2_extract_mvtec_knn.py"
loco_root = find_existing([
    r"data\mvtec_loco_ad",
    r"data\MVTec_LOCO_AD",
    r"data\mvtec_loco",
])

for seed in SEEDS:
    base = Path(r"runs_seed3\e2_mvtec_loco") / f"seed_{seed:02d}"
    features_csv = base / "features.csv"
    base.mkdir(parents=True, exist_ok=True)

    if not features_csv.exists():
        cmd = [
            sys.executable,
            script,
            "--root", str(loco_root),
            "--out_csv", str(features_csv),
            "--categories", "all",
            "--batch_size", "64",
            "--workers", "0",
            "--image_size", "224",
        ]
        cmd = add_seed_if_supported(cmd, script, seed)
        run_cmd(cmd, seed=seed)
    else:
        print(f"Already exists: {features_csv}")

    group_col = detect_group_col(features_csv, preferred="category")
    run_policy_suite(base, features_csv, seed=seed, group_col=group_col, main_c_wrong=MAIN_C_WRONG)
