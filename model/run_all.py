"""
Train and evaluate all models sequentially.
Usage: python run_all.py [--dl_only] [--svm_only] [--eval_only] [--skip_eval]
"""

import argparse
import subprocess
import sys
from pathlib import Path


DL_MODELS = ["resnet50", "efficientnet_b2", "vit_b_16", "convnext_tiny"]
ALL_MODELS = DL_MODELS + ["svm_resnet_features"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dl_only", action="store_true", help="Only train deep learning models")
    parser.add_argument("--svm_only", action="store_true", help="Only train SVM")
    parser.add_argument("--eval_only", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation after training")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # --- Training ---
    if not args.eval_only:
        if not args.svm_only:
            for name in DL_MODELS:
                print(f"\n{'='*60}\nTraining {name}\n{'='*60}")
                ret = subprocess.run([sys.executable, str(script_dir / "train.py"), "--model", name])
                if ret.returncode != 0:
                    print(f"Failed: {name}")
                    return ret.returncode

        if not args.dl_only:
            print(f"\n{'='*60}\nTraining SVM (ResNet-18 features)\n{'='*60}")
            ret = subprocess.run([sys.executable, str(script_dir / "train.py"), "--model", "svm_resnet_features"])
            if ret.returncode != 0:
                return ret.returncode

    # --- Evaluation ---
    if not args.skip_eval:
        models_to_eval = ALL_MODELS
        if args.dl_only:
            models_to_eval = DL_MODELS
        elif args.svm_only:
            models_to_eval = ["svm_resnet_features"]

        for name in models_to_eval:
            print(f"\n{'='*60}\nEvaluating {name}\n{'='*60}")
            ret = subprocess.run([sys.executable, str(script_dir / "evaluate.py"), "--model", name])
            if ret.returncode != 0:
                print(f"Evaluation failed: {name}")

    print("\nAll done.")


if __name__ == "__main__":
    sys.exit(main())
