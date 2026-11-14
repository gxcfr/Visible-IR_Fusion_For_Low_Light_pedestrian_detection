"""
Unit Tests for LLVIP Dataset Integrity, Pairing, and Annotations
"""

import os
from pathlib import Path
import pytest
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def config():
    cfg_path = ROOT_DIR / "config.yaml"
    assert cfg_path.exists(), "config.yaml not found"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def dataset_root(config):
    root = Path(config["dataset"]["root_path"])
    assert root.exists(), f"Configured dataset path does not exist: {root}"
    return root


def test_dataset_folder_structure(dataset_root):
    """Verify expected subdirectories exist in LLVIP."""
    assert (dataset_root / "Annotations").exists(), "Annotations directory missing"
    assert (dataset_root / "infrared").exists(), "infrared directory missing"
    assert (dataset_root / "visible").exists(), "visible directory missing"
    assert (dataset_root / "infrared" / "train").exists(), "infrared/train missing"
    assert (dataset_root / "infrared" / "test").exists(), "infrared/test missing"
    assert (dataset_root / "visible" / "train").exists(), "visible/train missing"
    assert (dataset_root / "visible" / "test").exists(), "visible/test missing"


def test_pairing_manifest_exists():
    """Verify that pairing manifest was generated."""
    manifest_path = ROOT_DIR / "results" / "metrics" / "dataset_pairing_manifest.csv"
    if not manifest_path.exists():
        pytest.skip("Manifest not yet generated (audit in progress)")

    df = pd.read_csv(manifest_path)
    assert len(df) == 15488, f"Expected 15,488 pairs, found {len(df)}"
    assert (df["pairing_status"] == "PAIRED").all(), "Found un-paired samples in manifest"


def test_single_class_integrity():
    """Ensure no unintended classes exist beyond 'person'."""
    audit_path = ROOT_DIR / "results" / "metrics" / "dataset_audit.json"
    if not audit_path.exists():
        pytest.skip("Audit report not yet generated")

    import json
    with open(audit_path, "r") as f:
        data = json.load(f)

    classes = data.get("class_distribution", {})
    assert set(classes.keys()) == {"person"}, f"Unexpected classes found: {classes.keys()}"
