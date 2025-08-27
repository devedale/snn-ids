# -*- coding: utf-8 -*-
"""
General Project Utilities
This module contains common utility functions that are used across different
parts of the SNN-IDS project. This includes functionalities like file operations,
artifact archiving, etc.
"""

import os
import zipfile
from typing import List

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
