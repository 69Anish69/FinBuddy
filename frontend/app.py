import os

# Load DB credentials
DB_USER = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# EDA features
EDA_FEATURES = os.getenv("EDA_FEATURES", "").split(",")

# ML model config
HYPERPARAM_C = float(os.getenv("HYPERPARAM_C", 1.0))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.25))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

EXPECTED_ACCURACY = float(os.getenv("EXPECTED_ACCURACY", 0.80))
EPOCHS = int(os.getenv("EPOCHS", 10))

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "default_experiment")
EXPERIMENT_VERSION = os.getenv("EXPERIMENT_VERSION", "v0.1")
