import os
import sys
import json
import time
import importlib
import numpy as np
import pandas as pd
import tensorflow as tf

# Add the project root to the Python path to allow for module imports
sys.path.append(os.path.abspath('.'))

# Import our custom modules
from preprocessing import process
from training import train

# Reload our modules to ensure the latest code is used
importlib.reload(process)
importlib.reload(train)

# Now, we can safely import the functions we need from our reloaded modules
from training.train import train_and_evaluate

print("Libraries imported and custom modules reloaded successfully.")

# Define a configuration for a quick smoke test
smoke_test_config = {
    "PREPROCESSING_CONFIG": {
        "sample_size": None
    }
}

# Run the preprocessing function
X_processed, y_processed = process.preprocess_data(config_override=smoke_test_config)

print("Preprocessing complete")


start_time = time.time()

# Pass the preprocessed data directly to the training function
training_log, best_model_path = train_and_evaluate(
    X=X_processed, 
    y=y_processed, 
)

end_time = time.time()
print(f"\n--- TRAINING COMPLETE in {(end_time - start_time):.2f} seconds ---")
print(f"Best model saved to: {best_model_path}")

