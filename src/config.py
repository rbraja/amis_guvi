import os

# Project Root Folder
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Folders
DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

# Model Paths
SENTIMENT_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sentiment_rf.pkl"
)

CATEGORY_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "category_classifier.pkl"
)

PRIORITY_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "priority_model.pkl"
)

RESOLUTION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "resolution_model.pkl"
)

