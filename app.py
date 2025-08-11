import os
import logging

# -------------------------------
# Setup logging (INFO level)
# -------------------------------
os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists

logging.basicConfig(
    filename='logs/app.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("✅ Application started")

# -------------------------------
# Load DB credentials (from env)
# -------------------------------
DB_USER = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Log non-sensitive DB info
logging.info(f"Database Host: {DB_HOST}")
logging.info(f"Database Port: {DB_PORT}")

# -------------------------------
# Load EDA features
# -------------------------------
EDA_FEATURES = os.getenv("EDA_FEATURES", "").split(",")
logging.info(f"EDA Features: {EDA_FEATURES}")

# -------------------------------
# Load ML hyperparameters
# -------------------------------
HYPERPARAM_C = float(os.getenv("HYPERPARAM_C", 1.0))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.25))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
EXPECTED_ACCURACY = float(os.getenv("EXPECTED_ACCURACY", 0.80))
EPOCHS = int(os.getenv("EPOCHS", 10))

logging.info(f"Hyperparameter C: {HYPERPARAM_C}")
logging.info(f"Test Size: {TEST_SIZE}, Random State: {RANDOM_STATE}")
logging.info(f"Expected Accuracy: {EXPECTED_ACCURACY}")
logging.info(f"Number of Epochs: {EPOCHS}")

# -------------------------------
# Experiment Metadata
# -------------------------------
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "default_experiment")
EXPERIMENT_VERSION = os.getenv("EXPERIMENT_VERSION", "v0.1")

logging.info(f"Experiment: {EXPERIMENT_NAME}, Version: {EXPERIMENT_VERSION}")

# -------------------------------
# You can continue your app logic here:
# e.g., connect to DB, load data, train model
# -------------------------------

def run_app():
    logging.info("🔄 Application logic starts here...")
    # Example stub:
    print("✅ App is running with secure config!")
    logging.info("✅ App is running successfully.")

DB_PASSWORD = "thisisahardcodedpassword123"

if __name__ == "__main__":
    run_app()
