# -*- coding: utf-8 -*-
"""
General Project Utilities
This module contains common utility functions that are used across different
parts of the SNN-IDS project. This includes functionalities like file operations,
artifact archiving, etc.

It also includes global seeding utilities to ensure reproducibility across
Python's random module, NumPy, and TensorFlow when available.
"""

import os
import random
import numpy as np
from typing import Optional
import zipfile
from typing import List

def set_global_seed(seed: int) -> None:
    """Sets global RNG seeds for reproducibility across libraries.

    This function sets:
    - PYTHONHASHSEED environment variable
    - Python's built-in random seed
    - NumPy random seed
    - TensorFlow random seed (if TensorFlow is installed)

    Args:
        seed: The seed value to use.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf  # lazy import
    except Exception:
        tf = None

    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        try:
            tf.random.set_seed(seed)
            # Best-effort determinism toggles (may reduce performance)
            os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        except Exception:
            pass

def zip_artifacts(directories_to_zip: List[str], zip_filename: str):
    """
    Creates a ZIP archive of all files within the specified directories.
    This is used to package all benchmark results (logs, models, visualizations)
    into a single file for easy storage and sharing.

    Args:
        directories_to_zip: A list of directory paths to include in the archive.
        zip_filename: The name of the output ZIP file.
    """
    print(f"\n📦 Creating ZIP archive: {zip_filename}")
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for directory in directories_to_zip:
                if not os.path.isdir(directory):
                    print(f"  ⚠️ Directory '{directory}' does not exist, skipping.")
                    continue

                for root, _, files in os.walk(directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Create a relative path for the files inside the zip
                        arcname = os.path.relpath(file_path, start=os.path.dirname(directory))
                        zipf.write(file_path, arcname)

        print(f"✅ Archive created successfully.")
    except Exception as e:
        print(f"❌ Error while creating ZIP archive: {e}")
