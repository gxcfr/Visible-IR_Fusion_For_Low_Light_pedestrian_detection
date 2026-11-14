"""
CLI Entry Point: LLVIP Dataset Audit and Inspection
Usage:
    python scripts/inspect_dataset.py --path <path_to_llvip>
"""

import sys
import argparse
from pathlib import Path
import yaml

# Add root directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.dataset_inspector import LLVIPDatasetInspector


def main():
    parser = argparse.ArgumentParser(description="Audit and validate the LLVIP dataset.")
    parser.add_argument("--path", type=str, default="", help="Path to extracted LLVIP dataset directory")
    args = parser.parse_args()

    dataset_path = args.path
    if not dataset_path:
        config_path = ROOT_DIR / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                dataset_path = cfg.get("dataset", {}).get("root_path", "")

    if not dataset_path or not Path(dataset_path).exists():
        print("=" * 60)
        print("ERROR: Dataset path is not specified or does not exist.")
        print(f"Provided path: '{dataset_path}'")
        print("Please provide a valid path via --path <path_to_llvip>")
        print("or configure dataset.root_path in config.yaml.")
        print("=" * 60)
        sys.exit(1)

    print(f"Starting LLVIP Dataset Audit on: {dataset_path}")
    output_metrics = ROOT_DIR / "results" / "metrics"
    inspector = LLVIPDatasetInspector(dataset_path, output_dir=str(output_metrics))
    summary, df_manifest = inspector.audit_dataset()

    print("\n" + "=" * 60)
    print("DATASET AUDIT SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"{k:35s}: {v}")
    print("=" * 60)
    print(f"Manifest written to: {output_metrics / 'dataset_pairing_manifest.csv'}")
    print(f"Audit report saved:  {output_metrics / 'dataset_audit.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
