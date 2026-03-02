"""
Run all ablation experiments sequentially and report pass/fail.
Usage: cd model && python experiment/run_ablations.py
"""

import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

EXPERIMENTS = [
    "ablation_image_size.py",
    "ablation_augmentation.py",
    "ablation_hog_vs_cnn.py",
    "ablation_optimizer.py",
]


def main():
    results = {}

    for script in EXPERIMENTS:
        script_path = _SCRIPT_DIR / script
        name = script.replace(".py", "").replace("ablation_", "")
        print(f"\n{'#'*60}")
        print(f"# Running: {script}")
        print(f"{'#'*60}\n")

        ret = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(_SCRIPT_DIR.parent),  # run from model/ so imports work
        )
        results[name] = "PASS" if ret.returncode == 0 else "FAIL"

    # Summary
    print(f"\n{'='*60}")
    print("  Ablation Suite Summary")
    print(f"{'='*60}")
    print(f"  {'Experiment':<25} {'Status':>10}")
    print(f"  {'-'*35}")
    for name, status in results.items():
        print(f"  {name:<25} {status:>10}")
    print()

    # Exit with failure if any experiment failed
    if any(s == "FAIL" for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
