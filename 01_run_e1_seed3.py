from pathlib import Path
import sys
from seed3_utils import SEEDS, project_root, find_existing, add_seed_if_supported, add_if_supported, run_cmd, detect_group_col, run_policy_suite

# E1 main cost setting. 필요하면 10으로 바꿔도 된다.
MAIN_C_WRONG = 5

root = project_root()
script = r"experiments\e1_extract_cifar10c.py"
cifar10c_dir = find_existing([r"data\CIFAR-10-C", r"data\cifar10c", r"data\CIFAR10-C"])
ckpt = find_existing([r"runs\e1_cifar_base\cifar10_resnet18.pt"])

for seed in SEEDS:
    base = Path(r"runs_seed3\e1_cifar10c_all") / f"seed_{seed:02d}"
    features_csv = base / "features.csv"
    base.mkdir(parents=True, exist_ok=True)

    if not features_csv.exists():
        cmd = [
            sys.executable,
            script,
            "--cifar10c_dir", str(cifar10c_dir),
            "--ckpt", str(ckpt),
            "--out_csv", str(features_csv),
            "--corruptions", "all",
            "--severities", "1,2,3,4,5",
            "--max_per_severity", "0",
            "--tta", "4",
            "--batch_size", "256",
            "--workers", "0",
        ]
        cmd = add_if_supported(cmd, script, "--image_size", "224")
        cmd = add_seed_if_supported(cmd, script, seed)
        run_cmd(cmd, seed=seed)
    else:
        print(f"Already exists: {features_csv}")

    group_col = detect_group_col(features_csv, preferred="corruption")
    run_policy_suite(base, features_csv, seed=seed, group_col=group_col, main_c_wrong=MAIN_C_WRONG)
