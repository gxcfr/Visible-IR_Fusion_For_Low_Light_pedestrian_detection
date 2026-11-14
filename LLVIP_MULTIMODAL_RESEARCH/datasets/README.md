# LLVIP Dataset Directory Guide

This directory holds configuration files, documentation, and manifest pointers for the LLVIP dataset.

## Dataset Integrity Policy
- **DO NOT** download duplicate copies of LLVIP.
- **DO NOT** commit raw image pairs or large zip archives to git.
- The raw dataset remains at its external local directory and is linked via configuration (`dataset_config.yaml`).

## Expected Structure
Once extracted, LLVIP is expected to follow or resemble:
```
LLVIP/
    Annotations/
    infrared/
        train/
        test/
    visible/
        train/
        test/
```
The exact layout will be verified by the dataset audit script (`scripts/inspect_dataset.py`) during Milestone 2.
